"""ProjectAirSim scene RPC server.

这个服务延续 UAV-ON 的设计：Python 进程作为“地图启动管家”，外部通过
msgpack-rpc 请求它打开或关闭 UE 打包地图。

和原 AirSim 1.x server 的区别：

1. 不再生成 settings.json，也不再使用 ApiServerPort。
2. 每个 ProjectAirSim 场景使用一对端口：topic_port 和 service_port。
3. 启动 UE 脚本时传入 -topicsport=<port> 和 -servicesport=<port>。
4. server 只负责管理地图进程；无人机控制、取图、传感器读取由后续
   ProjectAirSim client adapter 完成。
"""

import argparse
import copy
from dataclasses import dataclass, field
import errno
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid

from AerialDojo.runtime_config import (
    DEFAULT_SERVER_CONFIG_FILE,
    load_runtime_config,
    server_settings,
)

try:
    import msgpackrpc
except ImportError:
    msgpackrpc = None
else:
    try:
        from ._msgpackrpc_compat import patch_msgpackrpc_encoding
    except ImportError:
        from _msgpackrpc_compat import patch_msgpackrpc_encoding

    patch_msgpackrpc_encoding()


PROJECTAIRSIM_ENDPOINT_SCHEMA = "projectairsim_topic_service_v1"

RUNTIME_CONTRACT = {
    "simulator": "ProjectAirSim",
    "endpoint_schema": PROJECTAIRSIM_ENDPOINT_SCHEMA,
    "endpoint_ports_per_scene": 2,
    "topic_port_argument": "-topicsport",
    "service_port_argument": "-servicesport",
    "default_topic_port": 8989,
    "default_service_port": 8990,
}


env_exec_path_dict = {
    "CityPark": {
        "exec_path": "CityPark",
        "bash_name": "AirSimFantasyCity",
    },
    "FantasyCity": {
        # 兼容旧任务中使用的 FantasyCity 场景名。
        "exec_path": "N_Island_0",
        "bash_name": "N_Island_0",
    },
    "N_Island_0": {
        "exec_path": "N_Island_0",
        "bash_name": "N_Island_0",
    },
    "F_Tunnel_0": {
        "exec_path": "F_Tunnel_0",
        "bash_name": "F_Tunnel_0",
    },
    "HongKongStreet": {
        "bash_name": "AirSimFantasyCity",
        "exec_path": "HongKongStreet",
    },
}


@dataclass
class ServerConfig:
    """ProjectAirSim server 启动配置。"""

    host: str = "127.0.0.1"
    port: int = 36000
    root_path: str = "~/maps"
    gpus: list = field(default_factory=lambda: [0])
    endpoint_base_port: int = 0
    launch_wait: float = 10.0
    log_dir: str = ""
    dry_run: bool = False
    render_offscreen: bool = True
    no_sound: bool = True
    no_vsync: bool = True
    low_graphics: bool = False
    low_graphics_width: int = 640
    low_graphics_height: int = 360
    texture_pool_size_mb: int = 1024
    scene_launch_interval: float = 0.0
    extra_ue_args: list = field(default_factory=list)

    @property
    def first_endpoint_port(self) -> int:
        """第一个 ProjectAirSim topic 端口。"""
        if self.endpoint_base_port > 0:
            return int(self.endpoint_base_port)
        return int(self.port) + 100

    @property
    def resolved_log_dir(self) -> Path:
        """UE 启动日志目录。"""
        if self.log_dir:
            return Path(self.log_dir).expanduser().resolve()
        return Path(__file__).resolve().parent / "logs"


def _now() -> str:
    return str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def _decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def pid_exists(pid) -> bool:
    """检查 PID 是否存在。"""
    if pid is None or not isinstance(pid, int) or pid < 0:
        return False

    try:
        os.kill(pid, 0)
    except OSError as error:
        if error.errno == errno.ESRCH:
            return False
        if error.errno == errno.EPERM:
            return True
        raise
    return True


def FromPortGetPid(port: int):
    """查询当前用户下，占用某个 TCP 端口的进程 PID。"""
    try:
        result = subprocess.run(
            ["fuser", "-n", "tcp", str(int(port))],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception as error:
        print("{}\tFromPortGetPid\t{}".format(_now(), error))
        return None

    for token in result.stdout.split():
        try:
            pid = int(token)
            if os.stat("/proc/{}".format(pid)).st_uid != os.geteuid():
                continue
            return pid
        except (OSError, ValueError):
            continue
    return None


def KillPid(pid) -> None:
    """强制杀掉一个进程。"""
    if pid is None or not isinstance(pid, int):
        return

    while pid_exists(pid):
        try:
            print("pid {} is killed".format(pid))
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
        time.sleep(0.5)


def KillPorts(ports) -> None:
    """按 topic/service 端口批量关闭场景进程。"""
    threads = []

    def _kill_port(port):
        pid = FromPortGetPid(port)
        KillPid(pid)

    for port in ports:
        if port is None:
            continue
        thread = threading.Thread(target=_kill_port, args=(int(port),), daemon=True)
        threads.append(thread)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def port_is_occupied(port: int, host: str = "127.0.0.1") -> bool:
    """返回 host:port 是否已经被监听进程占用。"""
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        return FromPortGetPid(port) is not None
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host, int(port)))
    except OSError as error:
        if error.errno in (errno.EADDRINUSE, errno.EACCES):
            return True
        raise
    finally:
        probe.close()
    return False


def _scene_candidates(root_path, scene_name):
    """Search only map launchers, excluding Engine utility shell scripts."""
    root = Path(root_path).expanduser()
    if Path(scene_name).name != scene_name or scene_name in (".", ".."):
        raise ValueError("invalid scene name: {}".format(scene_name))
    candidates = []
    for parent in (root, root / "ID_ENVS", root / "OOD_ENVS"):
        candidate = parent / scene_name / (scene_name + ".sh")
        if candidate.is_file():
            candidates.append(candidate.resolve())
    # Also accept root_path pointing to a single map directory.
    if root.name == scene_name and (root / (scene_name + ".sh")).is_file():
        candidates.append((root / (scene_name + ".sh")).resolve())
    return sorted(set(candidates))


def discover_scene_scripts(root_path: str) -> dict:
    """Discover flat map folders and published ID_ENVS/OOD_ENVS maps."""
    root = Path(root_path).expanduser()
    names = set(env_exec_path_dict)
    for parent in (root, root / "ID_ENVS", root / "OOD_ENVS"):
        if parent.is_dir():
            names.update(path.name for path in parent.iterdir()
                         if path.is_dir() and (path / (path.name + ".sh")).is_file())
    if (root / (root.name + ".sh")).is_file():
        names.add(root.name)
    scenes = {}
    for scene_name in sorted(names):
        try:
            script = resolve_scene_executable(root_path, scene_name)
            scenes[scene_name] = {"scene_id": scene_name, "script_path": str(script),
                                  "exists": True, "map_dir": str(script.parent)}
        except (FileNotFoundError, KeyError):
            info = env_exec_path_dict.get(scene_name, {})
            script = root / info.get("exec_path", scene_name) / (info.get("bash_name", scene_name) + ".sh")
            scenes[scene_name] = {"scene_id": scene_name, "script_path": str(script),
                                  "exists": False, "map_dir": str(script.parent)}
    return scenes


def resolve_scene_executable(root_path: str, scene_id) -> Path:
    """Resolve published map names first, then retain legacy aliases."""
    scene_name = _decode_text(scene_id)
    if scene_name.lower() == "none":
        return None
    candidates = _scene_candidates(root_path, scene_name)
    if len(candidates) > 1:
        raise ValueError("ambiguous map launchers for {}: {}".format(scene_name, candidates))
    if candidates:
        return candidates[0]
    env_info = env_exec_path_dict.get(scene_name)
    if env_info is None:
        for map_name, candidate in env_exec_path_dict.items():
            if scene_name.startswith(map_name):
                env_info = candidate
                break
    if env_info:
        root = Path(root_path).expanduser()
        for parent in (root, root / "ID_ENVS", root / "OOD_ENVS"):
            script = parent / env_info["exec_path"] / (env_info["bash_name"] + ".sh")
            if script.is_file():
                return script.resolve()
    raise FileNotFoundError("scene launcher not found under {}: {}.sh".format(root_path, scene_name))


def build_projectairsim_launch_command(
    script_path: Path,
    gpu_id: int,
    topic_port: int,
    service_port: int,
    config: ServerConfig,
) -> list:
    """构造启动 ProjectAirSim UE 程序的命令。"""
    command = ["bash", str(script_path)]
    if config.render_offscreen:
        command.append("-RenderOffScreen")
    if config.no_sound:
        command.append("-nosound")
    if config.no_vsync:
        command.append("-NoVSync")
    command.extend(
        [
            "-GraphicsAdapter={}".format(int(gpu_id)),
            "-topicsport={}".format(int(topic_port)),
            "-servicesport={}".format(int(service_port)),
        ]
    )
    if config.low_graphics:
        command.extend(build_low_graphics_args(config))
    command.extend(config.extra_ue_args)
    return command


def build_low_graphics_args(config: ServerConfig) -> list:
    """构造更省显存的 UE 渲染参数。

    这些参数主要用于一张 8GB 左右显存的显卡同时打开两个地图时降低崩溃概率。
    不能使用 -nullrhi，因为那会让相机图像也渲染不出来。
    """
    width = max(320, int(config.low_graphics_width))
    height = max(240, int(config.low_graphics_height))
    texture_pool_size = max(256, int(config.texture_pool_size_mb))
    cvars = [
        "r.RayTracing=0",
        "r.RayTracing.ForceAllRayTracingEffects=0",
        "r.Lumen.Reflections.Allow=0",
        "r.Lumen.DiffuseIndirect.Allow=0",
        "r.Shadow.Virtual.Enable=0",
        "r.ShadowQuality=0",
        "r.Streaming.PoolSize={}".format(texture_pool_size),
        "r.MipMapLODBias=2",
        "r.SkeletalMeshLODBias=2",
        "r.StaticMeshLODDistanceScale=2",
        "r.ScreenPercentage=40",
        "r.PostProcessAAQuality=0",
        "r.MotionBlurQuality=0",
        "r.BloomQuality=0",
        "r.EyeAdaptationQuality=0",
        "r.AmbientOcclusionLevels=0",
        "r.SSR.Quality=0",
        "r.VolumetricFog=0",
        "sg.ViewDistanceQuality=0",
        "sg.AntiAliasingQuality=0",
        "sg.ShadowQuality=0",
        "sg.GlobalIlluminationQuality=0",
        "sg.ReflectionQuality=0",
        "sg.PostProcessQuality=0",
        "sg.TextureQuality=0",
        "sg.EffectsQuality=0",
        "sg.FoliageQuality=0",
        "sg.ShadingQuality=0",
    ]
    return [
        "-NoRayTracing",
        "-noraytracing",
        "-ResX={}".format(width),
        "-ResY={}".format(height),
        "-ForceRes",
        "-ExecCmds={}".format(";".join(cvars)),
    ]


class EventHandler(object):
    """msgpack-rpc 暴露的 ProjectAirSim 场景管理接口。"""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.scene_endpoint_ports = [
            (
                config.first_endpoint_port + 2 * index,
                config.first_endpoint_port + 2 * index + 1,
            )
            for index in range(1000)
        ]
        self.scene_processes = {}
        self.scene_log_files = {}
        self.scene_manifest = {}
        self.endpoint_to_scene = {}
        # 保留场景槽位的请求顺序（包括 disabled 槽位），用于后续客户端在
        # scene/GPU 完全一致时复用仍然存活的 UE 进程。
        self.active_scene_requests = []
        self.active_launch_records = []
        self.manifest_generation = 0
        self.server_instance_id = uuid.uuid4().hex
        self.port_interference_events = []
        self.scene_lock = threading.RLock()

    def ping(self) -> bool:
        """RPC 接口：健康检查。"""
        return True

    def list_available_scenes(self) -> dict:
        """RPC 接口：查看当前 server 能解析哪些 scene_id。"""
        return discover_scene_scripts(self.config.root_path)

    def get_runtime_contract(self) -> dict:
        """RPC 接口：返回 ProjectAirSim 端点协议。"""
        return copy.deepcopy(RUNTIME_CONTRACT)

    def get_scene_manifest(self) -> dict:
        """RPC 接口：返回当前打开的场景清单。"""
        with self.scene_lock:
            return {
                "server_instance_id": self.server_instance_id,
                "generation": self.manifest_generation,
                "dry_run": bool(self.config.dry_run),
                "scenes": [
                    self._scene_manifest_with_process_state(topic_port)
                    for topic_port in sorted(self.scene_manifest)
                ],
            }

    def get_scene_process_status(self, topic_port: int) -> dict:
        """RPC 接口：按 topic_port 返回 UE 进程是否还活着。"""
        with self.scene_lock:
            return self._scene_manifest_with_process_state(int(topic_port))

    def get_port_diagnostics(self) -> list:
        """RPC 接口：返回端口占用/冲突记录。"""
        return copy.deepcopy(self.port_interference_events)

    def validate_scene_lease(self, topic_port: int, lease_id: str) -> bool:
        """RPC 接口：确认 topic_port 上的场景是否仍是原来的 lease。"""
        with self.scene_lock:
            manifest = self.scene_manifest.get(int(topic_port))
            # msgpack-rpc-python 在新 msgpack 兼容模式下可能把字符串参数
            # 解成 bytes；这里统一转回普通文本再比较。
            return bool(
                manifest
                and manifest["lease_id"] == _decode_text(lease_id)
            )

    def _record_port_interference(self, port, reason, owner_pid=None):
        event = {
            "time": time.time(),
            "port": int(port),
            "reason": str(reason),
            "owner_pid": owner_pid,
        }
        self.port_interference_events.append(event)
        self.port_interference_events = self.port_interference_events[-100:]
        print("PORT_INTERFERENCE {}".format(json.dumps(event, sort_keys=True)))

    def _scene_manifest_with_process_state(self, topic_port: int) -> dict:
        """给 manifest 补充实时进程状态。"""
        topic_port = int(topic_port)
        manifest = self.scene_manifest.get(topic_port)
        if manifest is None:
            return {}

        item = copy.deepcopy(manifest)
        process = self.scene_processes.get(topic_port)
        if process is not None:
            returncode = process.poll()
            item["process_running"] = returncode is None
            item["process_returncode"] = returncode
        else:
            process_pid = item.get("process_pid")
            item["process_running"] = pid_exists(process_pid)
            item["process_returncode"] = None
        return item

    def _choose_endpoint_pairs(self, scene_count: int) -> list:
        """为每个真实场景选择一对空闲 topic/service 端口。"""
        chosen = []
        index = 0
        while len(chosen) < int(scene_count):
            if index >= len(self.scene_endpoint_ports):
                raise RuntimeError("no uncontested ProjectAirSim endpoint is available")
            topic_port, service_port = self.scene_endpoint_ports[index]
            index += 1

            busy_ports = []
            for port in (topic_port, service_port):
                owner_pid = FromPortGetPid(port)
                if port_is_occupied(port):
                    busy_ports.append(port)
                    self._record_port_interference(
                        port,
                        "candidate ProjectAirSim port is already occupied",
                        owner_pid,
                    )
            if busy_ports:
                continue
            chosen.append((topic_port, service_port))
        return chosen

    def _close_scene_process(self, topic_port: int):
        process = self.scene_processes.pop(int(topic_port), None)
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=3)
            except Exception:
                try:
                    if process.poll() is None:
                        process.kill()
                except Exception:
                    pass

        log_file = self.scene_log_files.pop(int(topic_port), None)
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass

    def _stop_scene_by_topic_port(self, topic_port: int):
        manifest = self.scene_manifest.get(int(topic_port))
        if manifest is None:
            return

        self._close_scene_process(int(topic_port))
        KillPorts([manifest.get("topic_port"), manifest.get("service_port")])
        self.scene_manifest.pop(int(topic_port), None)
        self.endpoint_to_scene.pop(int(topic_port), None)

    def _close_scenes_locked(self, increment_generation=True) -> bool:
        print("{}\tSTART close_scenes".format(_now()))
        try:
            for topic_port in list(self.scene_manifest):
                self._stop_scene_by_topic_port(int(topic_port))
            self.active_scene_requests = []
            self.active_launch_records = []
            if increment_generation:
                self.manifest_generation += 1
            result = True
        except Exception as error:
            print(error)
            result = False
        print("{}\tEND close_scenes".format(_now()))
        return result

    def close_scenes(self, ip: str) -> bool:
        """RPC 接口：关闭当前 server 管理的所有 ProjectAirSim 场景。"""
        with self.scene_lock:
            return self._close_scenes_locked(increment_generation=True)

    def _normalize_scene_requests(self, scen_id_gpu_list: list) -> list:
        """把 scene 请求转换成明确的 ``(scene_id, gpu_id)`` 映射。

        UAV-ON 兼容请求 ``[scene_id, gpu_id]`` 会保留显式 GPU；只传
        scene id（字符串或单元素列表）时，按 server 的 GPU 池轮询分配。
        """
        gpu_pool = list(self.config.gpus) or [0]
        requests = []
        for index, item in enumerate(scen_id_gpu_list):
            if isinstance(item, (str, bytes)):
                scene_id = item
                requested_gpu_id = None
            elif isinstance(item, (list, tuple)) and item:
                scene_id = item[0]
                requested_gpu_id = item[1] if len(item) > 1 else None
            else:
                raise ValueError(
                    "scene request must be a scene id or [scene_id, gpu_id], "
                    "got: {!r}".format(item)
                )

            if requested_gpu_id is None:
                gpu_id = int(gpu_pool[index % len(gpu_pool)])
            else:
                gpu_id = int(requested_gpu_id)
            if gpu_id < 0:
                raise ValueError("GPU id must be non-negative, got: {}".format(gpu_id))
            requests.append((_decode_text(scene_id), gpu_id))
        return requests

    def _make_log_path(self, scene_name: str, topic_port: int) -> Path:
        log_dir = self.config.resolved_log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_scene_name = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in scene_name
        )
        return log_dir / "{}_{}.log".format(safe_scene_name, int(topic_port))

    def _launch_scene(
        self,
        scene_name: str,
        gpu_id: int,
        topic_port: int,
        service_port: int,
        ip: str,
        generation: int,
    ) -> dict:
        script_path = resolve_scene_executable(self.config.root_path, scene_name)
        command = build_projectairsim_launch_command(
            script_path=script_path,
            gpu_id=gpu_id,
            topic_port=topic_port,
            service_port=service_port,
            config=self.config,
        )
        log_path = self._make_log_path(scene_name, topic_port)
        manifest_item = {
            "endpoint_schema": PROJECTAIRSIM_ENDPOINT_SCHEMA,
            "scene_id": scene_name,
            "gpu_id": int(gpu_id),
            "topic_port": int(topic_port),
            "service_port": int(service_port),
            "ip": str(ip),
            "lease_id": uuid.uuid4().hex,
            "generation": int(generation),
            "server_instance_id": self.server_instance_id,
            "server_pid": os.getpid(),
            "script_path": str(script_path),
            "command": shlex.join(command),
            "log_path": str(log_path),
            "dry_run": bool(self.config.dry_run),
            "process_started": False,
            "process_pid": None,
            "process_running": False,
            "process_returncode": None,
            "reused": False,
        }

        if self.config.dry_run:
            print("DRY_RUN {}".format(shlex.join(command)))
        else:
            print(shlex.join(command))
            log_file = open(str(log_path), "ab")
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(script_path.parent),
            )
            self.scene_processes[int(topic_port)] = process
            self.scene_log_files[int(topic_port)] = log_file
            manifest_item["process_started"] = True
            manifest_item["process_pid"] = int(process.pid)
            manifest_item["process_running"] = process.poll() is None

        self.scene_manifest[int(topic_port)] = manifest_item
        self.endpoint_to_scene[int(topic_port)] = (scene_name, int(gpu_id))
        return copy.deepcopy(manifest_item)

    def _disabled_scene_record(self, scene_name: str, gpu_id: int, ip: str, generation: int):
        return {
            "endpoint_schema": PROJECTAIRSIM_ENDPOINT_SCHEMA,
            "scene_id": scene_name,
            "gpu_id": int(gpu_id),
            "topic_port": None,
            "service_port": None,
            "ip": str(ip),
            "lease_id": None,
            "generation": int(generation),
            "server_instance_id": self.server_instance_id,
            "server_pid": os.getpid(),
            "script_path": None,
            "command": "",
            "log_path": None,
            "dry_run": bool(self.config.dry_run),
            "process_started": False,
            "process_pid": None,
            "process_running": False,
            "process_returncode": None,
            "reused": False,
        }

    def _open_scenes(self, ip: str, scen_id_gpu_list: list):
        """内部接口：关闭旧场景，然后按请求打开一批新场景。"""
        print("{}\t关闭场景中".format(_now()))
        self._close_scenes_locked(increment_generation=False)
        print("{}\t已关闭所有场景".format(_now()))

        scene_requests = self._normalize_scene_requests(scen_id_gpu_list)
        print("scene/GPU assignments: {}".format(scene_requests))
        enabled_requests = [
            item for item in scene_requests if item[0].lower() != "none"
        ]
        endpoints = iter(self._choose_endpoint_pairs(len(enabled_requests)))
        generation = self.manifest_generation + 1
        launch_records = []

        enabled_launch_index = 0
        for scene_name, gpu_id in scene_requests:
            if scene_name.lower() == "none":
                launch_records.append(
                    self._disabled_scene_record(scene_name, gpu_id, ip, generation)
                )
                continue

            topic_port, service_port = next(endpoints)
            launch_records.append(
                self._launch_scene(
                    scene_name=scene_name,
                    gpu_id=gpu_id,
                    topic_port=topic_port,
                    service_port=service_port,
                    ip=ip,
                    generation=generation,
                )
            )
            enabled_launch_index += 1
            if (
                enabled_launch_index < len(enabled_requests)
                and not self.config.dry_run
                and self.config.scene_launch_interval > 0
            ):
                time.sleep(float(self.config.scene_launch_interval))

        self.manifest_generation = generation
        self.active_scene_requests = list(scene_requests)
        self.active_launch_records = copy.deepcopy(launch_records)
        if launch_records and not self.config.dry_run and self.config.launch_wait > 0:
            time.sleep(float(self.config.launch_wait))

        print("finished", ip)
        return True, {
            "ip": str(ip),
            "endpoint_schema": PROJECTAIRSIM_ENDPOINT_SCHEMA,
            "generation": self.manifest_generation,
            "reused_all": False,
            "endpoints": launch_records,
        }

    def _can_reuse_scenes(self, scene_requests: list) -> bool:
        """确认当前槽位与新请求一致，且所有真实 UE 进程仍然存活。"""
        if list(scene_requests) != list(self.active_scene_requests):
            return False
        if len(self.active_launch_records) != len(scene_requests):
            return False

        for request, record in zip(scene_requests, self.active_launch_records):
            scene_name, gpu_id = request
            if (
                str(record.get("scene_id")) != str(scene_name)
                or int(record.get("gpu_id", -1)) != int(gpu_id)
            ):
                return False
            if str(scene_name).lower() == "none":
                continue

            topic_port = record.get("topic_port")
            if topic_port is None:
                return False
            status = self._scene_manifest_with_process_state(int(topic_port))
            if not status:
                return False
            if (
                str(status.get("scene_id")) != str(scene_name)
                or int(status.get("gpu_id", -1)) != int(gpu_id)
            ):
                return False
            if not self.config.dry_run and not status.get("process_running"):
                return False
        return True

    def _reuse_scenes(self, ip: str, scene_requests: list):
        """沿用 UE 进程和端口，并签发新 lease 交给新的顺序客户端。"""
        generation = self.manifest_generation + 1
        launch_records = []
        for (scene_name, gpu_id), existing_record in zip(
            scene_requests,
            self.active_launch_records,
        ):
            if str(scene_name).lower() == "none":
                record = self._disabled_scene_record(
                    scene_name,
                    gpu_id,
                    ip,
                    generation,
                )
                record["reused"] = True
                launch_records.append(record)
                continue

            topic_port = int(existing_record["topic_port"])
            manifest = self.scene_manifest[topic_port]
            manifest["ip"] = str(ip)
            manifest["lease_id"] = uuid.uuid4().hex
            manifest["generation"] = int(generation)
            manifest["reused"] = True
            launch_records.append(
                self._scene_manifest_with_process_state(topic_port)
            )

        self.manifest_generation = generation
        self.active_scene_requests = list(scene_requests)
        self.active_launch_records = copy.deepcopy(launch_records)
        print("reusing live scene/GPU assignments: {}".format(scene_requests))
        return True, {
            "ip": str(ip),
            "endpoint_schema": PROJECTAIRSIM_ENDPOINT_SCHEMA,
            "generation": self.manifest_generation,
            "reused_all": True,
            "endpoints": launch_records,
        }

    def acquire_scenes(self, ip: str, scen_id_gpu_list: list):
        """RPC 接口：优先复用完全匹配的存活场景，否则重新打开。"""
        with self.scene_lock:
            print("{}\tSTART acquire_scenes".format(_now()))
            try:
                if isinstance(ip, bytes):
                    ip = ip.decode("utf-8")
                scene_requests = self._normalize_scene_requests(scen_id_gpu_list)
                if self._can_reuse_scenes(scene_requests):
                    result = self._reuse_scenes(ip, scene_requests)
                else:
                    result = self._open_scenes(ip, scen_id_gpu_list)
            except Exception as error:
                print(error)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tracebacks = "".join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback)
                )
                print("traceback:", tracebacks)
                result = False, None
            print("{}\tEND acquire_scenes".format(_now()))
            return result

    def reopen_scenes(self, ip: str, scen_id_gpu_list: list):
        """RPC 接口：重启一批 ProjectAirSim 场景。

        参数示例：
            ip = "127.0.0.1"
            scen_id_gpu_list = [["FantasyCity", 0], ["HongKongStreet", 0]]
        """
        with self.scene_lock:
            print("{}\tSTART reopen_scenes".format(_now()))
            try:
                if isinstance(ip, bytes):
                    ip = ip.decode("utf-8")
                result = self._open_scenes(ip, scen_id_gpu_list)
            except Exception as error:
                print(error)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tracebacks = "".join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback)
                )
                print("traceback:", tracebacks)
                result = False, None
            print("{}\tEND reopen_scenes".format(_now()))
            return result

    def reopen_scene_from_port(self, topic_port: int):
        """RPC 接口：按 topic_port 重启单个 ProjectAirSim 场景。"""
        with self.scene_lock:
            topic_port = int(topic_port)
            manifest = self.scene_manifest.get(topic_port)
            if manifest is None:
                return False, None

            scene_name = manifest["scene_id"]
            gpu_id = int(manifest["gpu_id"])
            service_port = int(manifest["service_port"])
            ip = manifest["ip"]
            self._stop_scene_by_topic_port(topic_port)

            generation = self.manifest_generation + 1
            try:
                manifest_item = self._launch_scene(
                    scene_name=scene_name,
                    gpu_id=gpu_id,
                    topic_port=topic_port,
                    service_port=service_port,
                    ip=ip,
                    generation=generation,
                )
                self.manifest_generation = generation
                return True, manifest_item
            except Exception as error:
                print(error)
                return False, None


def serve_background(server, daemon=False):
    """后台线程启动 msgpack-rpc server。"""
    def _start_server(server_obj):
        try:
            server_obj.start()
        finally:
            server_obj.close()

    thread = threading.Thread(target=_start_server, args=(server,))
    thread.daemon = daemon
    thread.start()
    return thread


def stop_rpc_server(server, thread, join_timeout=5.0):
    """停止 msgpack-rpc 后台线程，避免 Ctrl+C 后卡在 threading shutdown。"""
    loop = getattr(getattr(server, "_loop", None), "_ioloop", None)
    if loop is not None:
        try:
            loop.add_callback(loop.stop)
        except Exception:
            pass
    try:
        server.stop()
    except Exception:
        pass
    if thread is not None and thread.is_alive():
        thread.join(timeout=float(join_timeout))
    try:
        server.close()
    except Exception:
        pass
    if thread is not None and thread.is_alive():
        print(
            "{}\tRPC server thread is still stopping; process will exit".format(
                _now()
            )
        )


def serve(config: ServerConfig, daemon=False):
    """创建并监听 ProjectAirSim RPC server。"""
    if msgpackrpc is None:
        raise RuntimeError(
            "msgpackrpc is not installed. Install msgpack-rpc-python before "
            "starting the RPC server."
        )
    handler = EventHandler(config)
    server = msgpackrpc.Server(handler)
    addr = msgpackrpc.Address(config.host, int(config.port))
    server.listen(addr)
    thread = serve_background(server, daemon)
    return addr, server, thread, handler


def _parse_gpus(gpus: str) -> list:
    parsed = []
    for gpu in str(gpus).split(","):
        gpu = gpu.strip()
        if gpu:
            parsed.append(int(gpu))
    return parsed or [0]


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_SERVER_CONFIG_FILE),
        help=(
            "Read server port, GPU pool and display settings from "
            "config/server_config.yaml."
        ),
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=36000, help="RPC service port")
    parser.add_argument("--root_path", type=str, default="~/maps")
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--dataset_root", default=None, help="Override dataset_root in a recording configuration.")
    parser.add_argument(
        "--endpoint_base_port",
        type=int,
        default=0,
        help="first ProjectAirSim topic port; default is RPC port + 100",
    )
    parser.add_argument("--launch_wait", type=float, default=10.0)
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="print launch commands without starting UE",
    )
    parser.add_argument("--no_render_offscreen", action="store_true")
    parser.add_argument("--enable_sound", action="store_true")
    parser.add_argument("--enable_vsync", action="store_true")
    parser.add_argument(
        "--low_graphics",
        action="store_true",
        help="use lower UE render quality to reduce VRAM usage",
    )
    parser.add_argument(
        "--low_graphics_width",
        type=int,
        default=640,
        help="render width used with --low_graphics",
    )
    parser.add_argument(
        "--low_graphics_height",
        type=int,
        default=360,
        help="render height used with --low_graphics",
    )
    parser.add_argument(
        "--texture_pool_size_mb",
        type=int,
        default=1024,
        help="UE r.Streaming.PoolSize used with --low_graphics",
    )
    parser.add_argument(
        "--scene_launch_interval",
        type=float,
        default=0.0,
        help="seconds to wait between launching two UE scene processes",
    )
    parser.add_argument(
        "--extra_ue_arg",
        action="append",
        default=[],
        help="extra UE/ProjectAirSim argument; can be repeated",
    )
    return parser.parse_args(argv)


def config_from_args(parsed_args) -> ServerConfig:
    settings = {}
    if parsed_args.config:
        payload = load_runtime_config(parsed_args.config)
        if getattr(parsed_args, "dataset_root", None):
            import yaml
            from AerialDojo.trajectory_recording.published_dataset import resolve_config
            raw = yaml.safe_load(Path(parsed_args.config).expanduser().read_text(encoding="utf-8"))
            payload = resolve_config(raw, parsed_args.dataset_root)
        settings = server_settings(payload)

    def configured(name, fallback):
        value = settings.get(name, fallback)
        return fallback if value is None else value

    low_graphics = bool(configured("low_graphics", parsed_args.low_graphics))
    scene_launch_interval = float(
        configured("scene_launch_interval", parsed_args.scene_launch_interval)
    )
    if low_graphics and scene_launch_interval <= 0:
        scene_launch_interval = 5.0
    show_game = bool(
        configured("show_game", parsed_args.no_render_offscreen)
    )
    extra_ue_args = list(configured("extra_ue_args", []))
    if show_game:
        if bool(configured("fullscreen", False)):
            extra_ue_args.append("-fullscreen")
        else:
            extra_ue_args.extend(
                [
                    "-windowed",
                    "-ResX={}".format(int(configured("window_width", 1280))),
                    "-ResY={}".format(int(configured("window_height", 720))),
                ]
            )
    extra_ue_args.extend(parsed_args.extra_ue_arg)

    configured_gpus = settings.get("gpus")
    gpus = (
        _parse_gpus(parsed_args.gpus) if parsed_args.gpus is not None
        else ([int(gpu_id) for gpu_id in configured_gpus] if configured_gpus is not None else [0])
    )
    return ServerConfig(
        host=str(configured("host", parsed_args.host)),
        port=int(configured("port", parsed_args.port)),
        root_path=str(configured("root_path", parsed_args.root_path)),
        gpus=gpus,
        endpoint_base_port=int(
            configured("endpoint_base_port", parsed_args.endpoint_base_port)
        ),
        launch_wait=float(configured("launch_wait", parsed_args.launch_wait)),
        log_dir=str(configured("log_dir", parsed_args.log_dir)),
        dry_run=bool(parsed_args.dry_run),
        render_offscreen=not show_game,
        no_sound=bool(configured("no_sound", not parsed_args.enable_sound)),
        no_vsync=bool(configured("no_vsync", not parsed_args.enable_vsync)),
        low_graphics=low_graphics,
        low_graphics_width=int(
            configured("low_graphics_width", parsed_args.low_graphics_width)
        ),
        low_graphics_height=int(
            configured("low_graphics_height", parsed_args.low_graphics_height)
        ),
        texture_pool_size_mb=int(
            configured("texture_pool_size_mb", parsed_args.texture_pool_size_mb)
        ),
        scene_launch_interval=scene_launch_interval,
        extra_ue_args=extra_ue_args,
    )


def main():
    args = _parse_args()
    config = config_from_args(args)
    print("PROJECT_ROOT_DIR", Path(__file__).resolve().parent.parent)
    print(
        "ProjectAirSim server config {}".format(
            json.dumps(config.__dict__, sort_keys=True, ensure_ascii=False)
        )
    )

    addr, server, thread, handler = serve(config)
    print("start listening \t{}:{}".format(addr._host, addr._port))
    try:
        while thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("{}\tKeyboardInterrupt: closing scenes".format(_now()))
        try:
            handler.close_scenes(config.host)
        except Exception:
            traceback.print_exc()
    finally:
        stop_rpc_server(server, thread)


if __name__ == "__main__":
    main()
