"""AerialDojo 使用的 ProjectAirSim 客户端适配层。

这个文件承担两层工作：

1. 通过 msgpack-rpc 请求 ProjectAirSimSimulatorServerTool 打开/关闭 UE 地图。
2. 使用 ProjectAirSimClient、World 和 Drone 连接每个地图进程，创建无人机，
   并向上层提供姿态、动作、图像和传感器接口。

公共方法尽量保持 UAV-ON 旧 Client 的形状，便于上层代码逐步迁移：
run_call、closeScenes、setPoses、move_to_next_pose、getImageResponses、
getSensorInfo。

注意：这里没有导入 AirSim 1.x 的 ``airsim`` 包。输入姿态可以是普通字典、
ProjectAirSim 字典，也兼容具有 position/orientation 与 x_val/y_val/z_val
属性的旧姿态对象。
"""

import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import commentjson
import numpy as np

try:
    import msgpackrpc
except ImportError as error:
    msgpackrpc = None
    _MSGPACKRPC_IMPORT_ERROR = error
else:
    _MSGPACKRPC_IMPORT_ERROR = None
    try:
        from ._msgpackrpc_compat import patch_msgpackrpc_encoding
    except ImportError:
        from _msgpackrpc_compat import patch_msgpackrpc_encoding

    patch_msgpackrpc_encoding()

try:
    from projectairsim import Drone, ProjectAirSimClient, World
    from projectairsim.drone import YawControlMode
    from projectairsim.types import ImageType, Pose, Quaternion, Vector3
    from projectairsim.utils import unpack_image
except ImportError as error:
    Drone = None
    ProjectAirSimClient = None
    World = None
    YawControlMode = None
    ImageType = None
    Pose = None
    Quaternion = None
    Vector3 = None
    unpack_image = None
    _PROJECTAIRSIM_IMPORT_ERROR = error
else:
    _PROJECTAIRSIM_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)

PROJECTAIRSIM_ENDPOINT_SCHEMA = "projectairsim_topic_service_v1"
DEFAULT_SCENE_CONFIG = "scene_collision.jsonc"
DEFAULT_DRONE_NAME = "Drone1"
DEFAULT_CAMERAS = ("0", "1", "2", "3")
PROJECTAIRSIM_CAMERA_NAME_ALIASES = {
    "0": ("FrontCamera", "Camera0"),
    "1": ("LeftCamera", "Camera1"),
    "2": ("RightCamera", "Camera2"),
    "3": ("DownCamera", "Camera3"),
    "front": ("FrontCamera", "Camera0"),
    "left": ("LeftCamera", "Camera1"),
    "right": ("RightCamera", "Camera2"),
    "down": ("DownCamera", "Camera3"),
}
DEPTH_MODES = ("uint8", "float32_m")
RGB_MODES = ("png", "raw")
COLLISION_EVENT_WAIT_SECONDS = 0.12


def _scene_config_uses_nonphysics(
    sim_config_path: str,
    scene_config_name: str,
) -> bool:
    """从配置内容判断 physics-type，文件读取失败时兼容旧文件名规则。"""
    scene_name_lower = str(scene_config_name).lower()
    fallback = (
        "nonphysics" in scene_name_lower
        or "non-physics" in scene_name_lower
    )
    try:
        config_dir = Path(sim_config_path)
        with (config_dir / scene_config_name).open(
            "r", encoding="utf-8-sig"
        ) as stream:
            scene_config = commentjson.load(stream)
        robot_actor = next(
            actor
            for actor in scene_config.get("actors", [])
            if actor.get("type") == "robot"
        )
        robot_config = robot_actor.get("robot-config")
        if isinstance(robot_config, str):
            with (config_dir / robot_config).open(
                "r", encoding="utf-8-sig"
            ) as stream:
                robot_config = commentjson.load(stream)
        return str(robot_config.get("physics-type", "")) == "non-physics"
    except Exception as error:
        LOGGER.warning(
            "无法从 %s 判断 physics-type，退回文件名判断: %s",
            scene_config_name,
            error,
        )
        return fallback


def convert_depth_metres(depth_metres, depth_mode="uint8"):
    """Encode ProjectAirSim DepthPerspective values whose unit is metres."""
    depth_mode = str(depth_mode)
    if depth_mode not in DEPTH_MODES:
        raise ValueError(
            "depth_mode must be one of {}, got {}".format(
                DEPTH_MODES, depth_mode
            )
        )
    depth_metres = np.asarray(depth_metres, dtype=np.float32)
    if depth_mode == "float32_m":
        return depth_metres
    return (
        np.clip(depth_metres, 0.0, 100.0) / 100.0 * 255.0
    ).astype(np.uint8)


def convert_depth_mm(depth_mm, depth_mode="uint8"):
    """Convert an explicitly millimetre-valued array for legacy callers."""
    depth_metres = np.asarray(depth_mm, dtype=np.float32) / 1000.0
    return convert_depth_metres(depth_metres, depth_mode=depth_mode)


POSITION_TOLERANCE_METRES = 5e-2
YAW_TOLERANCE_DEGREES = 5e-2
HORIZONTAL_STEP_SIZES = (1.0, 3.0, 5.0)
VERTICAL_STEP_SIZES = (1.0, 2.0)
ROTATION_DEGREES = 15.0


def _decode_rpc_value(value):
    """递归解码 msgpack-rpc 返回的 bytes 键和值。"""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, list):
        return [_decode_rpc_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode_rpc_value(item) for item in value)
    if isinstance(value, dict):
        return {
            _decode_rpc_value(key): _decode_rpc_value(item)
            for key, item in value.items()
        }
    return value


def _as_float(value) -> float:
    """兼容 numpy 数值以及普通 Python 数值。"""
    return float(value)


def _vector_component(vector, key: str) -> float:
    """从字典或 AirSim 风格对象中读取向量分量。"""
    if isinstance(vector, dict):
        if key in vector:
            return _as_float(vector[key])
        legacy_key = "{}_val".format(key)
        if legacy_key in vector:
            return _as_float(vector[legacy_key])
    if hasattr(vector, key):
        return _as_float(getattr(vector, key))
    legacy_key = "{}_val".format(key)
    if hasattr(vector, legacy_key):
        return _as_float(getattr(vector, legacy_key))
    raise TypeError("向量缺少分量 {}".format(key))


def _quaternion_component(quaternion, key: str) -> float:
    """从字典或对象中读取四元数分量。"""
    return _vector_component(quaternion, key)


def _extract_pose(pose) -> Tuple[np.ndarray, np.ndarray]:
    """统一提取 NED 位置 [x,y,z] 和四元数 [x,y,z,w]。"""
    if isinstance(pose, dict):
        position = pose.get("position", pose.get("translation"))
        orientation = pose.get("orientation", pose.get("rotation"))
    else:
        position = getattr(pose, "position", None)
        orientation = getattr(pose, "orientation", None)

    if position is None or orientation is None:
        raise TypeError(
            "姿态必须包含 position/orientation 或 translation/rotation"
        )

    position_array = np.array(
        [_vector_component(position, axis) for axis in ("x", "y", "z")],
        dtype=np.float64,
    )
    quaternion_array = np.array(
        [
            _quaternion_component(orientation, axis)
            for axis in ("x", "y", "z", "w")
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quaternion_array))
    if norm <= 1e-12:
        raise ValueError("四元数长度不能为 0")
    return position_array, quaternion_array / norm


def make_pose(
    position: Sequence[float],
    orientation_xyzw: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    """创建 ProjectAirSim ground-truth kinematics 使用的姿态字典。"""
    return {
        "position": {
            "x": _as_float(position[0]),
            "y": _as_float(position[1]),
            "z": _as_float(position[2]),
        },
        "orientation": {
            "w": _as_float(orientation_xyzw[3]),
            "x": _as_float(orientation_xyzw[0]),
            "y": _as_float(orientation_xyzw[1]),
            "z": _as_float(orientation_xyzw[2]),
        },
    }


def _yaw_radians(quaternion_xyzw: Sequence[float]) -> float:
    """四元数转换为 NED yaw，返回弧度。"""
    x, y, z, w = map(float, quaternion_xyzw)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


def _yaw_degrees(quaternion_xyzw: Sequence[float]) -> float:
    return math.degrees(_yaw_radians(quaternion_xyzw))


def _quaternion_to_rpy_degrees(
    quaternion_xyzw: Sequence[float],
) -> Tuple[float, float, float]:
    """把归一化 [x,y,z,w] 四元数转换成 roll/pitch/yaw 角度。"""
    x, y, z, w = map(float, quaternion_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("四元数长度不能为 0")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sin_pitch) \
        if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)

    yaw = _yaw_radians((x, y, z, w))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def _format_config_vector(values: Sequence[float]) -> str:
    return " ".join("{:.17g}".format(float(value)) for value in values)


def _build_no_sweep_reset_config(
    scene_config: Dict[str, Any],
    drone_name: str,
    target_pose,
) -> Dict[str, Any]:
    """生成直接在 episode 起点出生的运行时 scene 配置。"""
    runtime_config = copy.deepcopy(scene_config)
    target_position, target_quaternion = _extract_pose(target_pose)
    target_rpy_degrees = _quaternion_to_rpy_degrees(target_quaternion)

    robot_actor = next(
        (
            actor
            for actor in runtime_config.get("actors", [])
            if actor.get("type") == "robot"
            and actor.get("name") == str(drone_name)
        ),
        None,
    )
    if robot_actor is None:
        raise ValueError(
            "scene config 中找不到机器人 {!r}".format(drone_name)
        )

    robot_actor["origin"] = {
        "xyz": _format_config_vector(target_position),
        "rpy-deg": _format_config_vector(target_rpy_degrees),
    }

    # World 已将 robot-config 文件展开为字典。这里再强制开启根 Link
    # 碰撞，避免运行时误选到无碰撞配置后静默穿墙。
    robot_config = robot_actor.get("robot-config")
    if not isinstance(robot_config, dict):
        raise ValueError("robot-config 尚未展开为字典，无法重载 episode")
    links = robot_config.get("links", [])
    if not links:
        raise ValueError("robot config 没有 links")
    root_link = next(
        (link for link in links if link.get("name") == "Frame"),
        links[0],
    )
    root_link.setdefault("collision", {})["enabled"] = True
    return runtime_config


def _quaternion_from_yaw(yaw_radians: float) -> np.ndarray:
    """生成仅含 yaw 旋转的 [x,y,z,w] 四元数。"""
    half_yaw = float(yaw_radians) * 0.5
    return np.array(
        [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)],
        dtype=np.float64,
    )


def _angular_error_degrees(actual: float, target: float) -> float:
    return abs((float(actual) - float(target) + 180.0) % 360.0 - 180.0)


def _pose_errors(
    actual_position: Sequence[float],
    actual_quaternion: Sequence[float],
    target_position: Sequence[float],
    target_quaternion: Sequence[float],
) -> Tuple[float, float]:
    position_error = float(
        np.linalg.norm(
            np.asarray(actual_position, dtype=np.float64)
            - np.asarray(target_position, dtype=np.float64)
        )
    )
    yaw_error = _angular_error_degrees(
        _yaw_degrees(actual_quaternion),
        _yaw_degrees(target_quaternion),
    )
    return position_error, yaw_error


def _nearest_discrete(value: float, choices: Iterable[float]) -> float:
    return min(
        choices,
        key=lambda choice: (abs(float(choice) - float(value)), float(choice)),
    )


def _discretize_target_pose(current_pose, requested_pose, fly_type: str):
    """把上层请求约束到 UAV-ON 的离散动作空间。"""
    current_position, current_quaternion = _extract_pose(current_pose)
    requested_position, requested_quaternion = _extract_pose(requested_pose)
    delta = requested_position - current_position
    horizontal_distance = float(np.linalg.norm(delta[:2]))
    vertical_distance = abs(float(delta[2]))

    if fly_type in ("move", "move_horizontal", "move_vertical"):
        target_position = current_position.copy()
        if fly_type == "move_horizontal":
            intended_axis = "horizontal"
        elif fly_type == "move_vertical":
            intended_axis = "vertical"
        elif horizontal_distance > POSITION_TOLERANCE_METRES and (
            horizontal_distance >= vertical_distance
            or vertical_distance <= POSITION_TOLERANCE_METRES
        ):
            intended_axis = "horizontal"
        elif vertical_distance > POSITION_TOLERANCE_METRES:
            intended_axis = "vertical"
        else:
            intended_axis = "stationary"

        if (
            intended_axis == "horizontal"
            and horizontal_distance > POSITION_TOLERANCE_METRES
        ):
            applied_step = _nearest_discrete(
                horizontal_distance,
                HORIZONTAL_STEP_SIZES,
            )
            target_position[:2] += delta[:2] / horizontal_distance * applied_step
            action_kind = "horizontal"
        elif (
            intended_axis == "vertical"
            and vertical_distance > POSITION_TOLERANCE_METRES
        ):
            applied_step = _nearest_discrete(
                vertical_distance,
                VERTICAL_STEP_SIZES,
            )
            target_position[2] += math.copysign(applied_step, delta[2])
            action_kind = "vertical"
        else:
            applied_step = 0.0
            action_kind = "stationary"
        target_quaternion = requested_quaternion
    elif fly_type == "rotate":
        requested_delta = (
            _yaw_degrees(requested_quaternion)
            - _yaw_degrees(current_quaternion)
            + 180.0
        ) % 360.0 - 180.0
        direction = -1.0 if requested_delta < 0.0 else 1.0
        target_yaw = math.radians(
            _yaw_degrees(current_quaternion)
            + direction * ROTATION_DEGREES
        )
        target_position = current_position.copy()
        target_quaternion = _quaternion_from_yaw(target_yaw)
        applied_step = ROTATION_DEGREES
        action_kind = "rotation"
    else:
        target_position = current_position.copy()
        target_quaternion = current_quaternion.copy()
        applied_step = 0.0
        action_kind = "stop"

    return make_pose(target_position, target_quaternion), {
        "action_kind": action_kind,
        "requested_horizontal_step": horizontal_distance,
        "requested_vertical_step": vertical_distance,
        "applied_step": float(applied_step),
    }


def _vector_to_list(vector: Optional[Dict]) -> List[float]:
    if not vector:
        return [0.0, 0.0, 0.0]
    return [_vector_component(vector, axis) for axis in ("x", "y", "z")]


def _quaternion_to_list(quaternion: Optional[Dict]) -> List[float]:
    if not quaternion:
        return [0.0, 0.0, 0.0, 1.0]
    return [
        _quaternion_component(quaternion, axis)
        for axis in ("x", "y", "z", "w")
    ]


def _rotation_matrix(quaternion_xyzw: Sequence[float]) -> List[List[float]]:
    """把 [x,y,z,w] 四元数转换为 3x3 旋转矩阵。"""
    q1, q2, q3, q0 = map(float, quaternion_xyzw)
    return [
        [
            1 - 2 * (q2 * q2 + q3 * q3),
            2 * (q1 * q2 - q3 * q0),
            2 * (q1 * q3 + q2 * q0),
        ],
        [
            2 * (q1 * q2 + q3 * q0),
            1 - 2 * (q1 * q1 + q3 * q3),
            2 * (q2 * q3 - q1 * q0),
        ],
        [
            2 * (q1 * q3 - q2 * q0),
            2 * (q2 * q3 + q1 * q0),
            1 - 2 * (q1 * q1 + q2 * q2),
        ],
    ]


def _wait_projectairsim_task(async_method, *args, **kwargs):
    """在同步训练代码中执行 ProjectAirSim 的两层异步任务。"""

    async def _runner():
        task = await async_method(*args, **kwargs)
        return await task

    return asyncio.run(_runner())


@dataclass
class SceneConnection:
    """一个 UE 场景端点及其 ProjectAirSim 对象。"""

    endpoint: Dict[str, Any]
    client: Any
    world: Any
    drone: Any
    machine_index: int
    scene_index: int
    latest_collision: Optional[Dict[str, Any]] = None
    actual_pose_override: Optional[Dict[str, Any]] = None
    collision_sequence: int = 0
    collision_lock: threading.RLock = field(default_factory=threading.RLock)
    operation_lock: threading.RLock = field(default_factory=threading.RLock)

    def on_collision(self, _topic, message) -> None:
        """ProjectAirSim 只在发生碰撞时发布消息，因此保存最近一次事件。"""
        collision_info = _decode_rpc_value(copy.deepcopy(message))
        if not isinstance(collision_info, dict):
            collision_info = {"raw_message": collision_info}
        # ProjectAirSim 的 CollisionInfo topic 只在 has_collided=true 时发布，
        # 但当前 C++ CollisionInfoMessage 序列化结果没有 has_collided 字段。
        # 收到该 topic 本身就代表碰撞，不能用缺失字段的默认 False 覆盖它。
        collision_info.setdefault("has_collided", True)
        with self.collision_lock:
            self.latest_collision = collision_info
            self.collision_sequence += 1

    def collision_snapshot(self) -> Tuple[int, Optional[Dict[str, Any]]]:
        with self.collision_lock:
            return self.collision_sequence, copy.deepcopy(self.latest_collision)

    def clear_collision(self) -> None:
        with self.collision_lock:
            self.latest_collision = None

    def set_actual_pose_override(self, pose: Optional[Dict[str, Any]]) -> None:
        """碰撞后保存 UE 命中位置，修正 NonPhysics ground truth 偏差。"""
        with self.collision_lock:
            self.actual_pose_override = copy.deepcopy(pose)

    def actual_pose_override_snapshot(self) -> Optional[Dict[str, Any]]:
        with self.collision_lock:
            return copy.deepcopy(self.actual_pose_override)


class ProjectAirSimSimulatorClientTool:
    """管理一组机器上的多个 ProjectAirSim 场景。"""

    def __init__(
        self,
        machines_info,
        sim_config_path: Optional[str] = None,
        scene_config_name: str = DEFAULT_SCENE_CONFIG,
        drone_name: str = DEFAULT_DRONE_NAME,
    ) -> None:
        self.machines_info = copy.deepcopy(machines_info)
        default_config_path = (
            Path(__file__).resolve().parents[2] / "config" / "sim_config"
        )
        self.sim_config_path = str(
            Path(sim_config_path or default_config_path).expanduser().resolve()
        )
        self.scene_config_name = str(scene_config_name)
        self.uses_nonphysics_config = _scene_config_uses_nonphysics(
            self.sim_config_path,
            self.scene_config_name,
        )
        self.drone_name = str(drone_name)

        self.socket_clients = []
        self.scene_connections = self._empty_scene_matrix()
        self.projectairsim_clients = self._empty_scene_matrix()
        self.worlds = self._empty_scene_matrix()
        self.drones = self._empty_scene_matrix()
        # 兼容旧代码可能读取的属性名；内容现在是 ProjectAirSimClient。
        self.airsim_clients = self.projectairsim_clients

        self.endpoints_by_machine = self._empty_scene_matrix()
        self.scene_leases = self._empty_scene_matrix()
        self.last_verified_poses = self._empty_scene_matrix()
        self.pose_diagnostics = self._empty_scene_matrix()
        self.task_generations = [
            [0 for _ in item["open_scenes"]]
            for item in self.machines_info
        ]
        self.port_diagnostics = [[] for _ in self.machines_info]
        self.runtime_contract = {}
        self.projectairsim_endpoints = []

        # 保留 UAV-ON 原有字段，避免尚未迁移的上层代码访问时报错。
        self.airsim_ports = []
        self.airsim_ports_by_machine = self._empty_scene_matrix()
        self.airsim_ip = "127.0.0.1"
        self.objects_name_cnt = [
            [0 for _ in item["open_scenes"]]
            for item in self.machines_info
        ]
        self._init_check()

    def _empty_scene_matrix(self):
        return [
            [None for _ in item["open_scenes"]]
            for item in self.machines_info
        ]

    def _init_check(self) -> None:
        ips = [str(item["MACHINE_IP"]) for item in self.machines_info]
        if len(ips) != len(set(ips)):
            raise ValueError("MACHINE_IP 不能重复")
        for machine in self.machines_info:
            if len(machine["open_scenes"]) != len(machine["gpus"]):
                raise ValueError("open_scenes 和 gpus 数量必须一致")

    @staticmethod
    def _require_dependencies() -> None:
        if msgpackrpc is None:
            raise RuntimeError(
                "缺少 msgpack-rpc-python，无法连接场景管理 Server"
            ) from _MSGPACKRPC_IMPORT_ERROR
        if ProjectAirSimClient is None:
            raise RuntimeError(
                "缺少 ProjectAirSim Python Client。请先从 ProjectAirSim "
                "仓库安装 client/python/projectairsim。"
            ) from _PROJECTAIRSIM_IMPORT_ERROR

    @staticmethod
    def _confirm_socket_connection(socket_client) -> bool:
        try:
            return bool(socket_client.call("ping"))
        except Exception as error:
            LOGGER.error(
                "无法连接场景管理 Server %s:%s: %s",
                socket_client.address._host,
                socket_client.address._port,
                error,
            )
            return False

    def _close_socket_connections(self) -> None:
        for socket_client in self.socket_clients:
            try:
                socket_client.close()
            except Exception:
                pass
        self.socket_clients = []

    @staticmethod
    def _safe_disconnect(client) -> None:
        if client is None:
            return
        # ProjectAirSimClient.disconnect 会先尝试 /Sim/Unsubscribe。连接只完成
        # 一半、或者 UE 进程即将关闭时，这个请求会刷很多 warning；这里直接
        # 关 NNG socket，作为场景重试/退出时的清理动作更稳。
        try:
            client.state = False
        except Exception:
            pass
        thread = getattr(client, "recv_topic_thread", None)
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass
        closed_any_socket = False
        for socket_name in ("socket_topics", "socket_services"):
            socket_object = getattr(client, socket_name, None)
            if socket_object is not None:
                try:
                    socket_object.close()
                    closed_any_socket = True
                except Exception:
                    pass
        if closed_any_socket:
            return
        try:
            client.disconnect()
            return
        except Exception:
            pass

    def _close_projectairsim_connections(self) -> None:
        for row in self.scene_connections:
            for connection in row:
                if connection is None:
                    continue
                with connection.operation_lock:
                    try:
                        connection.drone.cancel_last_task()
                    except Exception:
                        pass
                    try:
                        connection.drone.disarm()
                    except Exception:
                        pass
                    try:
                        connection.drone.disable_api_control()
                    except Exception:
                        pass
                    self._safe_disconnect(connection.client)

        self.scene_connections = self._empty_scene_matrix()
        self.projectairsim_clients = self._empty_scene_matrix()
        self.worlds = self._empty_scene_matrix()
        self.drones = self._empty_scene_matrix()
        self.airsim_clients = self.projectairsim_clients

    def disconnectClients(self) -> None:
        """只断开 Python 客户端连接，不关闭 server 管理的 UE 场景。"""
        self._close_projectairsim_connections()
        self._close_socket_connections()

    @staticmethod
    def _validate_runtime_contract(contract: Dict[str, Any]) -> None:
        if contract.get("simulator") != "ProjectAirSim":
            raise RuntimeError(
                "Server 不是 ProjectAirSim: {}".format(contract)
            )
        if contract.get("endpoint_schema") != PROJECTAIRSIM_ENDPOINT_SCHEMA:
            raise RuntimeError(
                "Client/Server endpoint 协议不一致: {}".format(contract)
            )
        if int(contract.get("endpoint_ports_per_scene", 0)) != 2:
            raise RuntimeError("ProjectAirSim 每个场景必须提供两个端口")

    def _machine_config_value(
        self,
        machine_index: int,
        key: str,
        default,
    ):
        return self.machines_info[machine_index].get(key, default)

    def _get_scene_process_status(
        self,
        machine_index: int,
        endpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        """从管理 server 查询某个 ProjectAirSim UE 进程是否还活着。"""
        socket_client = None
        if 0 <= machine_index < len(self.socket_clients):
            socket_client = self.socket_clients[machine_index]
        if socket_client is None or endpoint.get("topic_port") is None:
            return {}
        try:
            return _decode_rpc_value(
                socket_client.call(
                    "get_scene_process_status",
                    int(endpoint["topic_port"]),
                )
            )
        except Exception as error:
            return {"status_query_error": str(error)}

    @staticmethod
    def _scene_process_exited(status: Dict[str, Any]) -> bool:
        if not status:
            return False
        if not bool(status.get("process_started", False)):
            return False
        return status.get("process_running") is False

    @staticmethod
    def _format_scene_process_failure(
        endpoint: Dict[str, Any],
        status: Dict[str, Any],
        last_error,
    ) -> str:
        log_path = status.get("log_path") or endpoint.get("log_path")
        return (
            "UE 场景进程已退出，ProjectAirSim 端口未建立；"
            "scene={scene} topic={topic} service={service} pid={pid} "
            "returncode={returncode} log={log} last_error={error}"
        ).format(
            scene=endpoint.get("scene_id"),
            topic=endpoint.get("topic_port"),
            service=endpoint.get("service_port"),
            pid=status.get("process_pid"),
            returncode=status.get("process_returncode"),
            log=log_path,
            error=last_error,
        )

    def _connect_endpoint(
        self,
        machine_index: int,
        scene_index: int,
        endpoint: Dict[str, Any],
        connection_timeout: float,
    ) -> SceneConnection:
        """连接 topic/service 端口，加载配置并创建 Drone1。"""
        deadline = time.time() + max(1.0, float(connection_timeout))
        last_error = None
        attempt = 0

        while time.time() < deadline:
            client = None
            attempt += 1
            try:
                client = ProjectAirSimClient(
                    address=str(endpoint["ip"]),
                    port_topics=int(endpoint["topic_port"]),
                    port_services=int(endpoint["service_port"]),
                )
                client.connect()

                config_path = self._machine_config_value(
                    machine_index,
                    "SIM_CONFIG_PATH",
                    self.sim_config_path,
                )
                config_name = self._machine_config_value(
                    machine_index,
                    "SCENE_CONFIG",
                    self.scene_config_name,
                )
                load_delay = float(
                    self._machine_config_value(
                        machine_index,
                        "SCENE_LOAD_DELAY",
                        1.0,
                    )
                )
                world = World(
                    client,
                    str(config_name),
                    delay_after_load_sec=load_delay,
                    sim_config_path=str(config_path),
                )
                drone_name = self._machine_config_value(
                    machine_index,
                    "DRONE_NAME",
                    self.drone_name,
                )
                drone = Drone(client, world, str(drone_name))
                connection = SceneConnection(
                    endpoint=copy.deepcopy(endpoint),
                    client=client,
                    world=world,
                    drone=drone,
                    machine_index=machine_index,
                    scene_index=scene_index,
                )
                client.subscribe(
                    drone.robot_info["collision_info"],
                    connection.on_collision,
                )
                enable_api_control_on_connect = bool(
                    self._machine_config_value(
                        machine_index,
                        "ENABLE_API_CONTROL_ON_CONNECT",
                        not self.uses_nonphysics_config,
                    )
                )
                if enable_api_control_on_connect:
                    if not drone.enable_api_control():
                        raise RuntimeError("Drone.enable_api_control 返回 False")
                arm_on_connect = bool(
                    self._machine_config_value(
                        machine_index,
                        "ARM_ON_CONNECT",
                        not self.uses_nonphysics_config,
                    )
                )
                if arm_on_connect:
                    if not drone.arm():
                        raise RuntimeError("Drone.arm 返回 False")
                # 和旧 Client 一样先让飞控进入飞行/悬停状态。随后 setPoses
                # 会把无人机传送到 episode 起点并清零速度。
                takeoff_on_connect = bool(
                    self._machine_config_value(
                        machine_index,
                        "TAKEOFF_ON_CONNECT",
                        not self.uses_nonphysics_config,
                    )
                )
                if takeoff_on_connect:
                    _wait_projectairsim_task(
                        drone.takeoff_async,
                        timeout_sec=20,
                    )
                    _wait_projectairsim_task(drone.hover_async)
                # 用一次同步状态请求确认 service 端口和 Drone 都已可用。
                drone.get_ground_truth_kinematics()
                return connection
            except (FileNotFoundError, ValueError, TypeError):
                self._safe_disconnect(client)
                raise
            except Exception as error:
                last_error = error
                self._safe_disconnect(client)
                status = {}
                if attempt == 1 or attempt % 5 == 0:
                    status = self._get_scene_process_status(
                        machine_index,
                        endpoint,
                    )
                    LOGGER.warning(
                        "连接 ProjectAirSim 未完成，scene=%s topic=%s "
                        "service=%s attempt=%s process_running=%s "
                        "returncode=%s error=%s",
                        endpoint.get("scene_id"),
                        endpoint.get("topic_port"),
                        endpoint.get("service_port"),
                        attempt,
                        status.get("process_running"),
                        status.get("process_returncode"),
                        error,
                    )
                    if self._scene_process_exited(status):
                        raise RuntimeError(
                            self._format_scene_process_failure(
                                endpoint,
                                status,
                                last_error,
                            )
                        ) from error
                time.sleep(1.0)

        status = self._get_scene_process_status(machine_index, endpoint)
        if self._scene_process_exited(status):
            raise RuntimeError(
                self._format_scene_process_failure(endpoint, status, last_error)
            )
        raise RuntimeError(
            "连接 ProjectAirSim 失败，scene={} topic={} service={} error={}".format(
                endpoint.get("scene_id"),
                endpoint.get("topic_port"),
                endpoint.get("service_port"),
                last_error,
            )
        )

    def _open_machine(
        self,
        machine_index: int,
        socket_client,
        connection_timeout: float,
        reuse_existing: bool = False,
    ):
        machine = self.machines_info[machine_index]
        requests = list(zip(machine["open_scenes"], machine["gpus"]))
        LOGGER.info(
            "请求机器 %s:%s 打开场景 %s",
            socket_client.address._host,
            socket_client.address._port,
            requests,
        )
        scene_rpc = "acquire_scenes" if reuse_existing else "reopen_scenes"
        result = _decode_rpc_value(
            socket_client.call(
                scene_rpc,
                socket_client.address._host,
                requests,
            )
        )
        if not result or not result[0]:
            raise RuntimeError("Server 打开场景失败: {}".format(result))

        payload = result[1]
        if not isinstance(payload, dict):
            raise RuntimeError(
                "Server 仍在返回 AirSim 1.x 端口列表，请启动新的 "
                "ProjectAirSimSimulatorServerTool.py"
            )
        if payload.get("endpoint_schema") != PROJECTAIRSIM_ENDPOINT_SCHEMA:
            raise RuntimeError("未知 endpoint payload: {}".format(payload))
        LOGGER.info(
            "机器 %s:%s 场景获取完成，reused_all=%s",
            socket_client.address._host,
            socket_client.address._port,
            bool(payload.get("reused_all", False)),
        )

        endpoints = list(payload.get("endpoints", []))
        if len(endpoints) != len(machine["open_scenes"]):
            raise RuntimeError("Server 返回的场景数量不正确")

        contract = _decode_rpc_value(
            socket_client.call("get_runtime_contract")
        )
        self._validate_runtime_contract(contract)
        manifest = _decode_rpc_value(
            socket_client.call("get_scene_manifest")
        )
        diagnostics = _decode_rpc_value(
            socket_client.call("get_port_diagnostics")
        )

        connections = []
        for scene_index, endpoint in enumerate(endpoints):
            expected_scene = machine["open_scenes"][scene_index]
            disabled = expected_scene is None or str(expected_scene).lower() == "none"
            if disabled:
                connections.append(None)
                continue
            if str(endpoint.get("scene_id")) != str(expected_scene):
                raise RuntimeError(
                    "场景错位：请求 {}，Server 返回 {}".format(
                        expected_scene,
                        endpoint.get("scene_id"),
                    )
                )
            if endpoint.get("dry_run") or not endpoint.get("process_started"):
                raise RuntimeError(
                    "Server 处于 dry-run 或 UE 进程未启动: {}".format(endpoint)
                )
            connection = self._connect_endpoint(
                machine_index,
                scene_index,
                endpoint,
                connection_timeout,
            )
            connections.append(connection)

        return {
            "contract": contract,
            "manifest": manifest,
            "diagnostics": diagnostics,
            "endpoints": endpoints,
            "connections": connections,
        }

    def run_call(
        self,
        airsim_timeout: int = 300,
        reuse_existing: bool = False,
    ) -> None:
        """打开所有场景并建立 ProjectAirSim 连接。

        ``airsim_timeout`` 参数名为兼容旧调用保留，现在表示等待 ProjectAirSim
        topic/service 端点就绪的最长秒数。

        ``reuse_existing=True`` 时，server 会复用地图、GPU、槽位顺序完全一致
        且仍然存活的 UE 进程；否则保持原有的强制重启行为。
        """
        self._require_dependencies()
        self._close_projectairsim_connections()
        self._close_socket_connections()

        self.socket_clients = [
            msgpackrpc.Client(
                msgpackrpc.Address(
                    machine["MACHINE_IP"],
                    machine["SOCKET_PORT"],
                ),
                timeout=max(300, int(airsim_timeout)),
            )
            for machine in self.machines_info
        ]
        for socket_client in self.socket_clients:
            if not self._confirm_socket_connection(socket_client):
                self._close_socket_connections()
                raise RuntimeError("无法建立场景管理 RPC 连接")

        started_at = time.time()
        machine_results = [None for _ in self.machines_info]
        errors = []
        with ThreadPoolExecutor(
            max_workers=max(1, len(self.machines_info))
        ) as executor:
            future_map = {
                executor.submit(
                    self._open_machine,
                    machine_index,
                    self.socket_clients[machine_index],
                    float(airsim_timeout),
                    bool(reuse_existing),
                ): machine_index
                for machine_index in range(len(self.machines_info))
            }
            for future in as_completed(future_map):
                machine_index = future_map[future]
                try:
                    machine_results[machine_index] = future.result()
                except Exception as error:
                    errors.append(
                        "machine {}: {}".format(machine_index, error)
                    )

        if errors:
            self._close_projectairsim_connections()
            self._close_socket_connections()
            raise RuntimeError("；".join(errors))

        contracts = [result["contract"] for result in machine_results]
        if any(contract != contracts[0] for contract in contracts[1:]):
            raise RuntimeError("多台 Server 的 runtime contract 不一致")
        self.runtime_contract = contracts[0] if contracts else {}

        for machine_index, result in enumerate(machine_results):
            self.port_diagnostics[machine_index] = result["diagnostics"]
            for scene_index, connection in enumerate(result["connections"]):
                endpoint = result["endpoints"][scene_index]
                self.endpoints_by_machine[machine_index][scene_index] = endpoint
                if connection is None:
                    continue
                self.scene_connections[machine_index][scene_index] = connection
                self.projectairsim_clients[machine_index][scene_index] = \
                    connection.client
                self.worlds[machine_index][scene_index] = connection.world
                self.drones[machine_index][scene_index] = connection.drone
                self.scene_leases[machine_index][scene_index] = {
                    "topic_port": int(endpoint["topic_port"]),
                    "service_port": int(endpoint["service_port"]),
                    "lease_id": str(endpoint["lease_id"]),
                    "server_instance_id": str(endpoint["server_instance_id"]),
                    "scene_id": str(endpoint["scene_id"]),
                }

        self.projectairsim_endpoints = [
            (
                str(endpoint["ip"]),
                int(endpoint["topic_port"]),
                int(endpoint["service_port"]),
            )
            for row in self.endpoints_by_machine
            for endpoint in row
            if endpoint and endpoint.get("topic_port") is not None
        ]
        if len(self.projectairsim_endpoints) != len(
            set(self.projectairsim_endpoints)
        ):
            raise RuntimeError("ProjectAirSim endpoint 被重复分配")

        # 兼容旧诊断字段：airsim_ports 对应 topic 端口。
        self.airsim_ports_by_machine = [
            [
                int(endpoint["topic_port"])
                if endpoint and endpoint.get("topic_port") is not None
                else None
                for endpoint in row
            ]
            for row in self.endpoints_by_machine
        ]
        self.airsim_ports = [
            port
            for row in self.airsim_ports_by_machine
            for port in row
            if port is not None
        ]
        if self.projectairsim_endpoints:
            self.airsim_ip = self.projectairsim_endpoints[0][0]

        self.validate_runtime_integrity()
        self._close_socket_connections()
        LOGGER.info(
            "全部 ProjectAirSim 场景连接完成，耗时 %.2f 秒",
            time.time() - started_at,
        )

    def connect_existing_scene(
        self,
        machine_index: int,
        scene_index: int,
        endpoint: Dict[str, Any],
        connection_timeout: float = 300.0,
    ) -> SceneConnection:
        """Connect one process to one existing server-managed scene lease."""
        self._require_dependencies()
        machine_index = int(machine_index)
        scene_index = int(scene_index)
        try:
            expected_scene = self.machines_info[machine_index]["open_scenes"][
                scene_index
            ]
        except IndexError as error:
            raise IndexError(
                "invalid machine/scene index: {}/{}".format(
                    machine_index, scene_index
                )
            ) from error

        endpoint = _decode_rpc_value(copy.deepcopy(endpoint))
        if endpoint.get("endpoint_schema") not in (
            None,
            PROJECTAIRSIM_ENDPOINT_SCHEMA,
        ):
            raise RuntimeError("unknown scene endpoint: {}".format(endpoint))
        if str(endpoint.get("scene_id")) != str(expected_scene):
            raise RuntimeError(
                "scene endpoint mismatch: expected {}, got {}".format(
                    expected_scene, endpoint.get("scene_id")
                )
            )

        machine = self.machines_info[machine_index]
        socket_client = msgpackrpc.Client(
            msgpackrpc.Address(
                machine["MACHINE_IP"],
                machine["SOCKET_PORT"],
            ),
            timeout=max(300, int(connection_timeout)),
        )
        self.socket_clients = [None for _ in self.machines_info]
        self.socket_clients[machine_index] = socket_client
        try:
            if not self._confirm_socket_connection(socket_client):
                raise RuntimeError("cannot connect to scene management server")
            contract = _decode_rpc_value(
                socket_client.call("get_runtime_contract")
            )
            self._validate_runtime_contract(contract)
            lease_id = endpoint.get("lease_id")
            if not lease_id or not socket_client.call(
                "validate_scene_lease",
                int(endpoint["topic_port"]),
                str(lease_id),
            ):
                raise RuntimeError(
                    "scene lease is no longer valid: scene={} topic={}".format(
                        endpoint.get("scene_id"), endpoint.get("topic_port")
                    )
                )
            connection = self._connect_endpoint(
                machine_index,
                scene_index,
                endpoint,
                float(connection_timeout),
            )
        finally:
            self._close_socket_connections()

        self.runtime_contract = contract
        self.scene_connections[machine_index][scene_index] = connection
        self.projectairsim_clients[machine_index][scene_index] = connection.client
        self.worlds[machine_index][scene_index] = connection.world
        self.drones[machine_index][scene_index] = connection.drone
        self.endpoints_by_machine[machine_index][scene_index] = endpoint
        self.scene_leases[machine_index][scene_index] = {
            "topic_port": int(endpoint["topic_port"]),
            "service_port": int(endpoint["service_port"]),
            "lease_id": str(endpoint["lease_id"]),
            "server_instance_id": str(endpoint["server_instance_id"]),
            "scene_id": str(endpoint["scene_id"]),
        }
        self.projectairsim_endpoints = [
            (
                str(endpoint["ip"]),
                int(endpoint["topic_port"]),
                int(endpoint["service_port"]),
            )
        ]
        self.airsim_ports_by_machine[machine_index][scene_index] = int(
            endpoint["topic_port"]
        )
        self.airsim_ports = [int(endpoint["topic_port"])]
        self.airsim_ip = str(endpoint["ip"])
        self.validate_runtime_integrity()
        return connection

    def validate_runtime_integrity(self) -> bool:
        """确认每个 topic 端口仍属于启动时拿到的 scene lease。"""
        self._require_dependencies()
        for machine_index, machine in enumerate(self.machines_info):
            active_leases = [
                lease
                for lease in self.scene_leases[machine_index]
                if lease is not None
            ]
            if not active_leases:
                continue
            socket_client = msgpackrpc.Client(
                msgpackrpc.Address(
                    machine["MACHINE_IP"],
                    machine["SOCKET_PORT"],
                ),
                timeout=30,
            )
            try:
                for lease in active_leases:
                    valid = socket_client.call(
                        "validate_scene_lease",
                        lease["topic_port"],
                        lease["lease_id"],
                    )
                    if not valid:
                        raise RuntimeError(
                            "场景 lease 已变化，scene={} topic_port={}".format(
                                lease["scene_id"],
                                lease["topic_port"],
                            )
                        )
            finally:
                socket_client.close()
        return True

    def closeScenes(self) -> bool:
        """断开无人机 Client，并通知所有 Server 关闭 UE 场景。"""
        self._require_dependencies()
        self._close_projectairsim_connections()
        success = True
        socket_clients = []
        try:
            socket_clients = [
                msgpackrpc.Client(
                    msgpackrpc.Address(
                        machine["MACHINE_IP"],
                        machine["SOCKET_PORT"],
                    ),
                    timeout=300,
                )
                for machine in self.machines_info
            ]
            for socket_client in socket_clients:
                if not self._confirm_socket_connection(socket_client):
                    success = False
                    continue
                result = socket_client.call(
                    "close_scenes",
                    socket_client.address._host,
                )
                success = bool(result) and success
        finally:
            for socket_client in socket_clients:
                try:
                    socket_client.close()
                except Exception:
                    pass
            self._close_socket_connections()

        self.endpoints_by_machine = self._empty_scene_matrix()
        self.scene_leases = self._empty_scene_matrix()
        self.airsim_ports_by_machine = self._empty_scene_matrix()
        self.airsim_ports = []
        self.projectairsim_endpoints = []
        return success

    def _set_connection_pose(
        self,
        connection: SceneConnection,
        target_pose,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """直接写 ground-truth kinematics，用于 episode 初始位置和精确收敛。"""
        target_position, target_quaternion = _extract_pose(target_pose)
        if self.uses_nonphysics_config:
            # NonPhysics Drone 只提供 SetPose，不注册 CancelLastTask 或
            # SetGroundTruthKinematics。reset_kinematics=True 会在闪现时清零速度。
            pose = Pose({
                "translation": Vector3({
                    "x": float(target_position[0]),
                    "y": float(target_position[1]),
                    "z": float(target_position[2]),
                }),
                "rotation": Quaternion({
                    "w": float(target_quaternion[3]),
                    "x": float(target_quaternion[0]),
                    "y": float(target_quaternion[1]),
                    "z": float(target_quaternion[2]),
                }),
                "frame_id": "DEFAULT_FRAME",
            })
            if not connection.drone.set_pose(pose, reset_kinematics=True):
                raise RuntimeError("NonPhysics Drone.set_pose 返回 False")
        else:
            connection.drone.cancel_last_task()
            kinematics = _decode_rpc_value(
                connection.drone.get_ground_truth_kinematics()
            )
            kinematics["pose"] = make_pose(target_position, target_quaternion)
            zero_vector = {"x": 0.0, "y": 0.0, "z": 0.0}
            kinematics.setdefault("twist", {})
            kinematics["twist"]["linear"] = copy.deepcopy(zero_vector)
            kinematics["twist"]["angular"] = copy.deepcopy(zero_vector)
            kinematics.setdefault("accels", {})
            kinematics["accels"]["linear"] = copy.deepcopy(zero_vector)
            kinematics["accels"]["angular"] = copy.deepcopy(zero_vector)
            if not connection.drone.set_ground_truth_kinematics(kinematics):
                raise RuntimeError("set_ground_truth_kinematics 返回 False")

        actual = _decode_rpc_value(
            connection.drone.get_ground_truth_kinematics()
        )
        actual_position = _vector_to_list(actual["pose"]["position"])
        actual_quaternion = _quaternion_to_list(
            actual["pose"]["orientation"]
        )
        position_error, yaw_error = _pose_errors(
            actual_position,
            actual_quaternion,
            target_position,
            target_quaternion,
        )
        diagnostics = {
            "position_error_metres": position_error,
            "yaw_error_degrees": yaw_error,
            "within_tolerance": (
                position_error <= POSITION_TOLERANCE_METRES
                and yaw_error <= YAW_TOLERANCE_DEGREES
            ),
        }
        return actual, diagnostics

    def _parallel_scene_call(self, worker):
        """对所有有效场景并行执行操作，并保持 machines/scenes 的嵌套结构。"""
        results = self._empty_scene_matrix()
        futures = {}
        scene_count = sum(
            connection is not None
            for row in self.scene_connections
            for connection in row
        )
        with ThreadPoolExecutor(max_workers=max(1, scene_count)) as executor:
            for machine_index, row in enumerate(self.scene_connections):
                for scene_index, connection in enumerate(row):
                    if connection is None:
                        continue
                    future = executor.submit(
                        worker,
                        machine_index,
                        scene_index,
                        connection,
                    )
                    futures[future] = (machine_index, scene_index)

            errors = []
            for future in as_completed(futures):
                machine_index, scene_index = futures[future]
                try:
                    results[machine_index][scene_index] = future.result()
                except Exception as error:
                    errors.append(
                        "machine {} scene {}: {}".format(
                            machine_index,
                            scene_index,
                            error,
                        )
                    )
            if errors:
                raise RuntimeError("；".join(errors))
        return results

    def setPoses(self, poses: list) -> bool:
        """把每架无人机放到 episode 起点，并清零速度与加速度。"""
        self.validate_runtime_integrity()

        def _set_pose(machine_index, scene_index, connection):
            with connection.operation_lock:
                self.task_generations[machine_index][scene_index] += 1
                connection.clear_collision()
                actual, diagnostics = self._set_connection_pose(
                    connection,
                    poses[machine_index][scene_index],
                )
                diagnostics["correction_kind"] = "start"
                diagnostics["task_generation"] = \
                    self.task_generations[machine_index][scene_index]
                self.last_verified_poses[machine_index][scene_index] = \
                    copy.deepcopy(actual["pose"])
                self.pose_diagnostics[machine_index][scene_index] = \
                    copy.deepcopy(diagnostics)
                return diagnostics

        try:
            self._parallel_scene_call(_set_pose)
            return True
        except Exception as error:
            LOGGER.error("setPoses 失败: %s", error)
            return False

    def resetPosesNoSweep(self, poses: list) -> bool:
        """重载 robot actor，使其直接在 episode 起点出生且不做 sweep。

        Server 管理的 UE 进程和 UnrealNative 地图保持不变；这里只通过
        ``World.load_scene`` 重建 Project AirSim scene/Drone，并恢复碰撞订阅。
        """
        self.validate_runtime_integrity()

        def _reset_pose(machine_index, scene_index, connection):
            with connection.operation_lock:
                target_pose = poses[machine_index][scene_index]
                runtime_config = _build_no_sweep_reset_config(
                    connection.world.get_configuration(),
                    self._machine_config_value(
                        machine_index,
                        "DRONE_NAME",
                        self.drone_name,
                    ),
                    target_pose,
                )
                load_delay = float(
                    self._machine_config_value(
                        machine_index,
                        "SCENE_LOAD_DELAY",
                        1.0,
                    )
                )

                self.task_generations[machine_index][scene_index] += 1
                connection.clear_collision()
                connection.set_actual_pose_override(None)
                connection.world.load_scene(
                    runtime_config,
                    delay_after_load_sec=load_delay,
                )

                drone_name = self._machine_config_value(
                    machine_index,
                    "DRONE_NAME",
                    self.drone_name,
                )
                drone = Drone(
                    connection.client,
                    connection.world,
                    str(drone_name),
                )
                connection.drone = drone
                connection.clear_collision()
                connection.set_actual_pose_override(None)
                connection.client.subscribe(
                    drone.robot_info["collision_info"],
                    connection.on_collision,
                )
                self.worlds[machine_index][scene_index] = connection.world
                self.drones[machine_index][scene_index] = drone

                actual = _decode_rpc_value(
                    drone.get_ground_truth_kinematics()
                )
                target_position, target_quaternion = _extract_pose(target_pose)
                actual_position = _vector_to_list(actual["pose"]["position"])
                actual_quaternion = _quaternion_to_list(
                    actual["pose"]["orientation"]
                )
                position_error, yaw_error = _pose_errors(
                    actual_position,
                    actual_quaternion,
                    target_position,
                    target_quaternion,
                )
                diagnostics = {
                    "correction_kind": "episode_reset_no_sweep",
                    "task_generation": self.task_generations[
                        machine_index
                    ][scene_index],
                    "position_error_metres": position_error,
                    "yaw_error_degrees": yaw_error,
                    "within_tolerance": (
                        position_error <= POSITION_TOLERANCE_METRES
                        and yaw_error <= YAW_TOLERANCE_DEGREES
                    ),
                    "collision": False,
                }
                self.last_verified_poses[machine_index][scene_index] = \
                    copy.deepcopy(actual["pose"])
                self.pose_diagnostics[machine_index][scene_index] = \
                    copy.deepcopy(diagnostics)
                return diagnostics

        try:
            self._parallel_scene_call(_reset_pose)
            return True
        except Exception as error:
            LOGGER.error("resetPosesNoSweep 失败: %s", error)
            return False

    def movePosesWithSweep(self, poses: list):
        """用 NonPhysics SetPose 执行离散动作，并返回本次 sweep 碰撞。"""
        self.validate_runtime_integrity()

        def _move_pose(machine_index, scene_index, connection):
            with connection.operation_lock:
                target_pose = poses[machine_index][scene_index]
                target_position, target_quaternion = _extract_pose(target_pose)
                collision_sequence, _ = connection.collision_snapshot()
                connection.clear_collision()
                connection.set_actual_pose_override(None)

                actual, diagnostics = self._set_connection_pose(
                    connection,
                    target_pose,
                )

                # SetPose 服务先更新 core_sim，UE 在后续 Tick 执行 sweep。
                # 等待最多几个渲染帧，让碰撞 topic 有时间到达接收线程。
                deadline = time.time() + COLLISION_EVENT_WAIT_SECONDS
                collision_info = None
                while time.time() < deadline:
                    new_sequence, latest_collision = \
                        connection.collision_snapshot()
                    if new_sequence > collision_sequence:
                        collision_info = latest_collision
                        break
                    time.sleep(0.01)

                # 碰撞 topic 只会在碰撞发生时发布；兼容没有序列化
                # has_collided 字段的 ProjectAirSim 版本。
                collision = collision_info is not None
                if collision:
                    # NonPhysics 当前不会把 sweep 后停在墙前的 UE 位姿回写到
                    # ground truth；CollisionInfo.position 是 Hit.Location，作为
                    # 本回合终止前的实际位置保存并暴露给环境层。
                    collision_position_value = collision_info.get("position")
                    collision_position = (
                        _vector_to_list(collision_position_value)
                        if collision_position_value is not None
                        else _vector_to_list(actual["pose"]["position"])
                    )
                    corrected_pose = make_pose(
                        collision_position,
                        target_quaternion,
                    )
                    connection.set_actual_pose_override(corrected_pose)
                    actual = copy.deepcopy(actual)
                    actual["pose"] = copy.deepcopy(corrected_pose)
                    position_error, yaw_error = _pose_errors(
                        collision_position,
                        target_quaternion,
                        target_position,
                        target_quaternion,
                    )
                    diagnostics.update({
                        "position_error_metres": position_error,
                        "yaw_error_degrees": yaw_error,
                        "within_tolerance": False,
                    })

                diagnostics.update({
                    "correction_kind": "teleport_sweep",
                    "task_generation": self.task_generations[
                        machine_index
                    ][scene_index],
                    "collision": collision,
                    "collision_info": copy.deepcopy(collision_info),
                })
                self.last_verified_poses[machine_index][scene_index] = \
                    copy.deepcopy(actual["pose"])
                self.pose_diagnostics[machine_index][scene_index] = \
                    copy.deepcopy(diagnostics)
                sensor_info = self._sensor_info(
                    connection,
                    collision_info,
                )
                return {
                    "states": [{"sensors": sensor_info}],
                    "collision": collision,
                    "pose_diagnostics": diagnostics,
                    "actual_pose": copy.deepcopy(actual["pose"]),
                }

        try:
            return self._parallel_scene_call(_move_pose)
        except Exception as error:
            LOGGER.error("movePosesWithSweep 失败: %s", error)
            return None

    def _move_one(
        self,
        machine_index: int,
        scene_index: int,
        connection: SceneConnection,
        requested_pose,
        fly_type: str,
    ) -> Dict[str, Any]:
        with connection.operation_lock:
            generation = self.task_generations[machine_index][scene_index]
            current_kinematics = _decode_rpc_value(
                connection.drone.get_ground_truth_kinematics()
            )
            target_pose, action_diagnostics = _discretize_target_pose(
                current_kinematics["pose"],
                requested_pose,
                str(fly_type),
            )
            target_position, target_quaternion = _extract_pose(target_pose)
            collision_sequence, _ = connection.collision_snapshot()

            is_move_action = str(fly_type) in (
                "move",
                "move_horizontal",
                "move_vertical",
            )
            if is_move_action:
                distance = float(
                    np.linalg.norm(
                        target_position
                        - np.asarray(
                            _vector_to_list(
                                current_kinematics["pose"]["position"]
                            )
                        )
                    )
                )
                if distance > POSITION_TOLERANCE_METRES:
                    velocity = max(1.0, min(5.0, distance))
                    _wait_projectairsim_task(
                        connection.drone.move_to_position_async,
                        north=float(target_position[0]),
                        east=float(target_position[1]),
                        down=float(target_position[2]),
                        velocity=velocity,
                        timeout_sec=max(5.0, distance / velocity + 5.0),
                        yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
                        yaw_is_rate=False,
                        yaw=_yaw_radians(target_quaternion),
                    )
            elif fly_type == "rotate":
                _wait_projectairsim_task(
                    connection.drone.rotate_to_yaw_async,
                    yaw=_yaw_radians(target_quaternion),
                    timeout_sec=10.0,
                    margin=math.radians(1.0),
                )
            else:
                connection.drone.cancel_last_task()

            # 给 topic 接收线程一点时间提交同一仿真步中的碰撞事件。
            time.sleep(0.03)
            new_collision_sequence, collision_info = \
                connection.collision_snapshot()
            collision = new_collision_sequence > collision_sequence

            # 没有碰撞时将最终状态精确落在离散目标上，减小控制器停止误差。
            if (is_move_action or fly_type == "rotate") and not collision:
                actual_kinematics, correction = self._set_connection_pose(
                    connection,
                    target_pose,
                )
            else:
                actual_kinematics = _decode_rpc_value(
                    connection.drone.get_ground_truth_kinematics()
                )
                actual_position = _vector_to_list(
                    actual_kinematics["pose"]["position"]
                )
                actual_quaternion = _quaternion_to_list(
                    actual_kinematics["pose"]["orientation"]
                )
                position_error, yaw_error = _pose_errors(
                    actual_position,
                    actual_quaternion,
                    target_position,
                    target_quaternion,
                )
                correction = {
                    "position_error_metres": position_error,
                    "yaw_error_degrees": yaw_error,
                    "within_tolerance": False,
                }

            if generation != self.task_generations[machine_index][scene_index]:
                raise RuntimeError("动作执行期间 episode generation 发生变化")

            diagnostics = {
                **action_diagnostics,
                **correction,
                "correction_kind": str(fly_type),
                "task_generation": generation,
                "collision": bool(collision),
            }
            self.last_verified_poses[machine_index][scene_index] = \
                copy.deepcopy(actual_kinematics["pose"])
            self.pose_diagnostics[machine_index][scene_index] = \
                copy.deepcopy(diagnostics)
            sensor_info = self._sensor_info(connection, collision_info)
            return {
                "states": [{"sensors": sensor_info}],
                "collision": bool(collision),
                "pose_diagnostics": diagnostics,
                "actual_pose": copy.deepcopy(actual_kinematics["pose"]),
            }

    def move_to_next_pose(self, poses_list: list, fly_types: list):
        """并行执行一批离散移动/旋转动作。"""
        self.validate_runtime_integrity()

        def _move(machine_index, scene_index, connection):
            return self._move_one(
                machine_index,
                scene_index,
                connection,
                poses_list[machine_index][scene_index],
                fly_types[machine_index][scene_index],
            )

        try:
            return self._parallel_scene_call(_move)
        except Exception as error:
            LOGGER.error("move_to_next_pose 失败: %s", error)
            return None

    @staticmethod
    def _decode_image_pair(
        images,
        camera_name: str,
        projectairsim_camera_name: str,
        depth_mode: str = "uint8",
        rgb_mode: str = "png",
        return_timing: bool = False,
    ):
        total_started = time.perf_counter()
        validation_started = total_started
        if ImageType.SCENE not in images or \
                ImageType.DEPTH_PERSPECTIVE not in images:
            raise RuntimeError(
                "相机 {}({}) 未返回 RGB/DepthPerspective".format(
                    camera_name,
                    projectairsim_camera_name,
                )
            )

        validation_ms = (
            time.perf_counter() - validation_started
        ) * 1000.0
        rgb_unpack_started = time.perf_counter()
        rgb_array = unpack_image(images[ImageType.SCENE])
        rgb_unpack_ms = (
            time.perf_counter() - rgb_unpack_started
        ) * 1000.0
        if rgb_array is None or rgb_array.size == 0:
            raise RuntimeError(
                "相机 {}({}) RGB 为空".format(
                    camera_name,
                    projectairsim_camera_name,
                )
            )
        rgb_mode = str(rgb_mode)
        rgb_prepare_started = time.perf_counter()
        if rgb_mode == "raw":
            rgb_payload = np.ascontiguousarray(rgb_array, dtype=np.uint8)
        elif rgb_mode == "png":
            encoded_ok, encoded_rgb = cv2.imencode(".png", rgb_array)
            if not encoded_ok:
                raise RuntimeError(
                    "相机 {}({}) RGB PNG 编码失败".format(
                        camera_name,
                        projectairsim_camera_name,
                    )
                )
            rgb_payload = encoded_rgb.tobytes()
        else:
            raise ValueError(
                "rgb_mode must be one of {}, got {}".format(
                    RGB_MODES, rgb_mode
                )
            )
        rgb_prepare_ms = (
            time.perf_counter() - rgb_prepare_started
        ) * 1000.0

        # UE 将 DepthPerspective 从米换算为毫米后，通过 16UC1 传输。
        # 客户端统一除以 1000 恢复为米；环境观测仍可请求 uint8 编码，
        # float32_m 模式保留以米为单位的深度。
        depth_unpack_started = time.perf_counter()
        depth_mm = unpack_image(images[ImageType.DEPTH_PERSPECTIVE])
        depth_unpack_ms = (
            time.perf_counter() - depth_unpack_started
        ) * 1000.0
        depth_convert_started = time.perf_counter()
        depth = convert_depth_mm(
            depth_mm,
            depth_mode=depth_mode,
        )
        depth_convert_ms = (
            time.perf_counter() - depth_convert_started
        ) * 1000.0
        if not return_timing:
            return rgb_payload, depth
        return rgb_payload, depth, {
            "validation_ms": validation_ms,
            "rgb_unpack_ms": rgb_unpack_ms,
            "rgb_prepare_ms": rgb_prepare_ms,
            "depth_unpack_ms": depth_unpack_ms,
            "depth_convert_ms": depth_convert_ms,
            "processing_ms": (
                time.perf_counter() - total_started
            ) * 1000.0,
        }

    @staticmethod
    def _normalise_image_type_keys(images):
        """Match Drone.get_images by converting JSON string keys to integers."""
        normalised = dict(images)
        for key in list(normalised):
            if isinstance(key, str):
                try:
                    image_type = int(key)
                except ValueError:
                    continue
                normalised[image_type] = normalised.pop(key)
        return normalised

    @classmethod
    def _image_pair(
        cls,
        connection: SceneConnection,
        camera_name: str,
        depth_mode: str = "uint8",
        rgb_mode: str = "png",
    ):
        projectairsim_camera_names = PROJECTAIRSIM_CAMERA_NAME_ALIASES.get(
            str(camera_name).lower(),
            (str(camera_name),),
        )
        last_error = None
        images = None
        projectairsim_camera_name = None
        for candidate in projectairsim_camera_names:
            try:
                images = connection.drone.get_images(
                    candidate,
                    [ImageType.SCENE, ImageType.DEPTH_PERSPECTIVE],
                )
                projectairsim_camera_name = candidate
                break
            except Exception as error:
                last_error = error
        if images is None:
            raise RuntimeError(
                "相机 {} 不可用，候选名={}，最后错误={}".format(
                    camera_name,
                    list(projectairsim_camera_names),
                    last_error,
                )
            )
        return cls._decode_image_pair(
            images,
            camera_name,
            projectairsim_camera_name,
            depth_mode,
            rgb_mode,
        )

    @classmethod
    async def _request_image_with_timing(cls, client, request):
        """Expose client-side phases around one ProjectAirSim NNG request."""
        del cls
        total_started = time.perf_counter()
        pack_started = time.perf_counter()
        request_packed = client.preprocess_request(request)
        request_pack_ms = (time.perf_counter() - pack_started) * 1000.0

        context_started = time.perf_counter()
        context = client.socket_services.new_context()
        context_ms = (time.perf_counter() - context_started) * 1000.0

        send_started = time.perf_counter()
        await context.asend(request_packed)
        request_send_ms = (time.perf_counter() - send_started) * 1000.0

        response_started = time.perf_counter()
        response = await context.arecv_msg()
        response_wait_ms = (time.perf_counter() - response_started) * 1000.0
        response_bytes = len(response.bytes)

        decode_started = time.perf_counter()
        result = client.postprocess_response(response)
        response_decode_ms = (
            time.perf_counter() - decode_started
        ) * 1000.0
        return result, {
            "request_pack_ms": request_pack_ms,
            "context_ms": context_ms,
            "request_send_ms": request_send_ms,
            "response_wait_ms": response_wait_ms,
            "response_decode_ms": response_decode_ms,
            "response_bytes": int(response_bytes),
            "rpc_ms": (time.perf_counter() - total_started) * 1000.0,
        }

    @classmethod
    async def _image_pair_async(
        cls,
        connection: SceneConnection,
        camera_name: str,
        depth_mode: str = "uint8",
        return_timing: bool = False,
        rgb_mode: str = "png",
    ):
        """Request one camera using an independent NNG service context."""
        total_started = time.perf_counter()
        projectairsim_camera_names = PROJECTAIRSIM_CAMERA_NAME_ALIASES.get(
            str(camera_name).lower(),
            (str(camera_name),),
        )
        last_error = None
        images = None
        projectairsim_camera_name = None
        request_timing = {
            "request_prepare_ms": 0.0,
            "request_pack_ms": 0.0,
            "context_ms": 0.0,
            "request_send_ms": 0.0,
            "response_wait_ms": 0.0,
            "response_decode_ms": 0.0,
            "response_bytes": 0,
            "rpc_ms": 0.0,
        }
        attempts = 0
        for candidate in projectairsim_camera_names:
            attempts += 1
            request_prepare_started = time.perf_counter()
            request = {
                "method": "{}/{}/GetImages".format(
                    connection.drone.sensors_topic,
                    candidate,
                ),
                "params": {
                    "image_type_ids": [
                        ImageType.SCENE,
                        ImageType.DEPTH_PERSPECTIVE,
                    ]
                },
                "version": 1.0,
            }
            request_timing["request_prepare_ms"] += (
                time.perf_counter() - request_prepare_started
            ) * 1000.0
            try:
                if return_timing:
                    result, attempt_timing = (
                        await cls._request_image_with_timing(
                            connection.client, request
                        )
                    )
                    for name in (
                        "request_pack_ms",
                        "context_ms",
                        "request_send_ms",
                        "response_wait_ms",
                        "response_decode_ms",
                        "response_bytes",
                        "rpc_ms",
                    ):
                        request_timing[name] += attempt_timing[name]
                else:
                    result_list = []
                    response_task = await connection.client.request_async(
                        request,
                        result_list.append,
                    )
                    await response_task
                    if not result_list:
                        raise RuntimeError("ProjectAirSim 异步请求未返回结果")
                    result = result_list[0]
                images = cls._normalise_image_type_keys(result)
                projectairsim_camera_name = candidate
                break
            except Exception as error:
                last_error = error
        if images is None:
            raise RuntimeError(
                "相机 {} 不可用，候选名={}，最后错误={}".format(
                    camera_name,
                    list(projectairsim_camera_names),
                    last_error,
                )
            )
        image_pair = cls._decode_image_pair(
            images,
            camera_name,
            projectairsim_camera_name,
            depth_mode,
            rgb_mode,
            return_timing=return_timing,
        )
        if not return_timing:
            return image_pair

        rgb_payload, depth_image, processing_timing = image_pair
        ended_at = time.perf_counter()
        timing = {
            "camera": str(camera_name),
            "projectairsim_camera": str(projectairsim_camera_name),
            "attempts": int(attempts),
            "total_ms": (time.perf_counter() - total_started) * 1000.0,
            "rgb_bytes": int(
                getattr(rgb_payload, "nbytes", len(rgb_payload))
            ),
            "depth_bytes": int(getattr(depth_image, "nbytes", 0)),
            "_started_perf_s": total_started,
            "_ended_perf_s": ended_at,
        }
        timing.update(request_timing)
        timing.update(processing_timing)
        return rgb_payload, depth_image, timing

    def getImageResponses(
        self,
        cameras: Sequence[str] = DEFAULT_CAMERAS,
        poses=None,
        depth_mode: str = "uint8",
        rgb_mode: str = "png",
    ):
        """获取指定编码的 RGB，以及指定编码和几何类型的深度图。"""
        del poses  # 兼容旧签名；姿态由 setPoses/move_to_next_pose 管理。
        if str(depth_mode) not in DEPTH_MODES:
            raise ValueError(
                "depth_mode must be one of {}, got {}".format(
                    DEPTH_MODES, depth_mode
                )
            )
        if str(rgb_mode) not in RGB_MODES:
            raise ValueError(
                "rgb_mode must be one of {}, got {}".format(
                    RGB_MODES, rgb_mode
                )
            )
        self.validate_runtime_integrity()
        camera_names = [str(camera) for camera in cameras]

        def _get_images(machine_index, scene_index, connection):
            return self._get_scene_images(
                machine_index,
                scene_index,
                connection,
                camera_names,
                str(depth_mode),
                rgb_mode=str(rgb_mode),
            )

        try:
            return self._parallel_scene_call(_get_images)
        except Exception as error:
            LOGGER.error("getImageResponses 失败: %s", error)
            return None

    def _get_scene_images(
        self,
        machine_index: int,
        scene_index: int,
        connection: SceneConnection,
        camera_names: Sequence[str],
        depth_mode: str,
        return_timings: bool = False,
        rgb_mode: str = "png",
    ):
        """Read one scene's cameras concurrently under its operation lock."""
        total_started = time.perf_counter()
        lock_started = time.perf_counter()
        with connection.operation_lock:
            lock_wait_ms = (time.perf_counter() - lock_started) * 1000.0
            camera_names = [str(camera_name) for camera_name in camera_names]

            async def _read_all_cameras():
                if return_timings:
                    return await asyncio.gather(
                        *(
                            self._image_pair_async(
                                connection,
                                camera_name,
                                depth_mode=depth_mode,
                                return_timing=True,
                                rgb_mode=rgb_mode,
                            )
                            for camera_name in camera_names
                        )
                    )
                return await asyncio.gather(
                    *(
                        self._image_pair_async(
                            connection,
                            camera_name,
                            depth_mode=depth_mode,
                            rgb_mode=rgb_mode,
                        )
                        for camera_name in camera_names
                    )
                )

            camera_batch_started = time.perf_counter()
            image_pairs = asyncio.run(_read_all_cameras())
            camera_batch_ms = (
                time.perf_counter() - camera_batch_started
            ) * 1000.0
            if return_timings:
                camera_timings = [pair[2] for pair in image_pairs]
                for timing in camera_timings:
                    started_at = timing.pop(
                        "_started_perf_s", camera_batch_started
                    )
                    ended_at = timing.pop(
                        "_ended_perf_s",
                        started_at
                        + float(timing.get("total_ms", 0.0)) / 1000.0,
                    )
                    timing["start_offset_ms"] = (
                        started_at - camera_batch_started
                    ) * 1000.0
                    timing["end_offset_ms"] = (
                        ended_at - camera_batch_started
                    ) * 1000.0
                image_pairs = [pair[:2] for pair in image_pairs]
            else:
                camera_timings = []
            rgb_images = [pair[0] for pair in image_pairs]
            depth_images = [pair[1] for pair in image_pairs]
            if return_timings:
                return rgb_images, depth_images, {
                    "lock_wait_ms": lock_wait_ms,
                    "camera_batch_ms": camera_batch_ms,
                    "total_ms": (
                        time.perf_counter() - total_started
                    ) * 1000.0,
                    "cameras": camera_timings,
                }
            return rgb_images, depth_images

    def getSceneImageResponses(
        self,
        machine_index: int,
        scene_index: int,
        cameras: Sequence[str] = DEFAULT_CAMERAS,
        depth_mode: str = "uint8",
        return_timings: bool = False,
        rgb_mode: str = "png",
    ):
        """Get images from one scene without waiting for any other scene."""
        if str(depth_mode) not in DEPTH_MODES:
            raise ValueError(
                "depth_mode must be one of {}, got {}".format(
                    DEPTH_MODES, depth_mode
                )
            )
        if str(rgb_mode) not in RGB_MODES:
            raise ValueError(
                "rgb_mode must be one of {}, got {}".format(
                    RGB_MODES, rgb_mode
                )
            )
        machine_index = int(machine_index)
        scene_index = int(scene_index)
        try:
            connection = self.scene_connections[machine_index][scene_index]
        except IndexError as error:
            raise IndexError(
                "invalid machine/scene index: {}/{}".format(
                    machine_index, scene_index
                )
            ) from error
        if connection is None:
            raise RuntimeError(
                "machine {} scene {} is not connected".format(
                    machine_index, scene_index
                )
            )
        try:
            if not return_timings:
                return self._get_scene_images(
                    machine_index,
                    scene_index,
                    connection,
                    [str(camera) for camera in cameras],
                    str(depth_mode),
                    rgb_mode=str(rgb_mode),
                )
            return self._get_scene_images(
                machine_index,
                scene_index,
                connection,
                [str(camera) for camera in cameras],
                str(depth_mode),
                return_timings=bool(return_timings),
                rgb_mode=str(rgb_mode),
            )
        except Exception as error:
            LOGGER.error(
                "getSceneImageResponses 失败: machine=%s scene=%s error=%s",
                machine_index,
                scene_index,
                error,
            )
            return None

    @staticmethod
    def _sensor_info(
        connection: SceneConnection,
        collision_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """整理成 UAV-ON 上层代码原先读取的 state/imu 字段。"""
        kinematics = _decode_rpc_value(
            connection.drone.get_ground_truth_kinematics()
        )
        imu = _decode_rpc_value(
            connection.drone.get_imu_data("IMU1")
        )
        try:
            gps = _decode_rpc_value(
                connection.drone.get_ground_truth_geo_location()
            )
            gps_location = [
                float(gps.get("latitude", 0.0)),
                float(gps.get("longitude", 0.0)),
                float(gps.get("altitude", 0.0)),
            ]
        except Exception:
            gps_location = [0.0, 0.0, 0.0]

        _, latest_collision = connection.collision_snapshot()
        collision_info = collision_override or latest_collision or {}
        pose = kinematics.get("pose", {})
        pose_override = connection.actual_pose_override_snapshot()
        if pose_override is not None:
            pose = pose_override
        twist = kinematics.get("twist", {})
        accels = kinematics.get("accels", {})
        orientation = _quaternion_to_list(pose.get("orientation"))

        state_info = {
            "collision": {
                "has_collided": bool(
                    collision_info.get("has_collided", False)
                ),
                "object_name": str(
                    collision_info.get("object_name", "")
                ),
            },
            "gps_location": gps_location,
            "timestamp": int(kinematics.get("time_stamp", 0)),
            "position": _vector_to_list(pose.get("position")),
            "linear_velocity": _vector_to_list(twist.get("linear")),
            "linear_acceleration": _vector_to_list(accels.get("linear")),
            "orientation": orientation,
            "angular_velocity": _vector_to_list(twist.get("angular")),
            "angular_acceleration": _vector_to_list(accels.get("angular")),
        }

        imu_orientation = _quaternion_to_list(imu.get("orientation"))
        imu_info = {
            "time_stamp": int(imu.get("time_stamp", 0)),
            "rotation": _rotation_matrix(imu_orientation),
            "orientation": imu_orientation,
            "linear_acceleration": _vector_to_list(
                imu.get("linear_acceleration")
            ),
            "angular_velocity": _vector_to_list(
                imu.get("angular_velocity")
            ),
        }
        return {"state": state_info, "imu": imu_info}

    def getSensorInfo(self):
        """读取每架无人机的 ground truth、GPS、碰撞和 IMU。"""
        self.validate_runtime_integrity()

        def _get_sensor(_machine_index, _scene_index, connection):
            with connection.operation_lock:
                return {"sensors": self._sensor_info(connection)}

        try:
            return self._parallel_scene_call(_get_sensor)
        except Exception as error:
            LOGGER.error("getSensorInfo 失败: %s", error)
            return None

    def getDiagnostics(self) -> Dict[str, Any]:
        """返回端口、lease、姿态和最近碰撞诊断。"""
        collisions = self._empty_scene_matrix()
        for machine_index, row in enumerate(self.scene_connections):
            for scene_index, connection in enumerate(row):
                if connection is not None:
                    collisions[machine_index][scene_index] = \
                        connection.collision_snapshot()[1]
        return {
            "runtime_contract": copy.deepcopy(self.runtime_contract),
            "endpoints": copy.deepcopy(self.endpoints_by_machine),
            "scene_leases": copy.deepcopy(self.scene_leases),
            "pose": copy.deepcopy(self.pose_diagnostics),
            "ports": copy.deepcopy(self.port_diagnostics),
            "task_generations": copy.deepcopy(self.task_generations),
            "collisions": collisions,
        }


__all__ = [
    "ProjectAirSimSimulatorClientTool",
    "make_pose",
]
