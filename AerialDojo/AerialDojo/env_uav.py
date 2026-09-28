"""AerialDojo V1 的轻量级 ProjectAirSim 无人机环境。

这个文件是“环境层”，不是底层通信层。它负责：

1. 读取任务 JSON。
2. 根据任务里的地图名启动对应 UE/ProjectAirSim 场景。
3. 把无人机放到任务起点。
4. 执行动作，例如 forward、rotr、ascend。
5. 读取 RGB、Depth、IMU、位姿等观测。
6. 维护 done、success、collision、trajectory 等 episode 状态。

"""

from dataclasses import dataclass, field
import copy
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .env_utils_uav import (
    ActionSettings,
    DEFAULT_ACTION_SETTINGS,
    distance_to_goal,
    next_pose_from_action,
    pose_from_lists,
    pose_to_flat_list,
)
from .projectairsim_plugin import ProjectAirSimSimulatorClientTool


DEFAULT_SCENE_ALIASES: Dict[str, str] = {}


@dataclass
class UAVEnvConfig:
    """SimpleUAVEnv 的运行配置。

    训练/测试脚本一般只需要改 batch_size、cameras、motion_mode 和 server_port。
    """

    # 场景管理 server 的地址；对应 ProjectAirSimSimulatorServerTool.py。
    server_host: str = "127.0.0.1"
    # 场景管理 server 的 RPC 端口，不是 ProjectAirSim topic/service 端口。
    server_port: int = 36000
    # server 启动 UE .sh 时使用的 GPU 编号。
    gpu_id: int = 0
    # 一次 reset 打开的 episode 数。当前轻量环境默认单机多场景。
    batch_size: int = 1
    # 单个 episode 最多执行多少步，超过后 done=True。
    max_steps: int = 50
    # 判断是否进入目标范围的欧氏距离阈值。
    success_distance: float = 1.0
    # client 等待 UE/ProjectAirSim topic/service 就绪的最长秒数。
    connect_timeout: int = 300
    # 要读取的相机编号。"0".."3" 会优先映射到
    # FrontCamera/LeftCamera/RightCamera/DownCamera，也兼容旧 Camera0..Camera3。
    cameras: Tuple[str, ...] = ("0", "1", "2", "3")
    # teleport: 离散更新到目标 pose；碰撞版 NonPhysics 会 sweep 整段路径。
    # command: 调用 ProjectAirSim 飞控命令，适合后续调真实运动和碰撞。
    motion_mode: str = "teleport"
    # 任务 JSON 里的地图名到 server 支持的 scene_id 的映射。
    scene_aliases: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SCENE_ALIASES)
    )
    # 离散动作的步长和转角设置，定义在 env_utils_uav.py。
    action_settings: ActionSettings = DEFAULT_ACTION_SETTINGS
    # 可选的多 GPU 池。非空时覆盖 gpu_id，并按 batch 顺序轮询分配场景；
    # 保持在字段末尾，兼容已有的 UAVEnvConfig 位置参数。
    gpu_ids: Tuple[int, ...] = ()
    # RGB 返回格式。png 返回压缩字节，raw 返回解码后的 uint8 数组。
    rgb_mode: str = "png"
    # 深度返回格式。uint8 使用 UAV-ON 的 0~100 米量化方式；
    # float32_m 直接返回以米为单位的 float32 数组。
    depth_mode: str = "float32_m"
    # 相邻任务的场景布局相同时，保留 UE 进程并只重置无人机和回合状态。
    reuse_scenes: bool = True
    # 同一批 UE 场景连续使用的最大回合数，与 UAV-ON 的默认值一致。
    max_scene_reuses: int = 5000
    # 是否在进入成功半径时立即结束。CLIP-H 评测应设为 False，等待策略 Stop。
    terminate_on_success: bool = True
    # ProjectAirSim World 使用的场景配置。
    scene_config_name: str = "scene_collision.jsonc"
    # teleport 动作完成后、采集下一轮感知前的等待时间。图形化演示时可让
    # UE 渲染窗口和 RGB-D 捕获稳定在新的位姿。
    teleport_settle_seconds: float = 0.0


@dataclass
class EpisodeState:
    """一个 episode 在环境层维护的状态。

    它不直接控制 ProjectAirSim，只保存上层训练/评测需要的状态记录。
    """

    # 当前 episode 对应的任务信息。
    task: Dict[str, Any]
    # 当前姿态，格式为 [x, y, z, qx, qy, qz, qw]。
    pose: List[float]
    # 当前执行了多少个动作。
    step: int = 0
    # episode 是否结束。
    done: bool = False
    # 最近动作或观测是否检测到碰撞。
    collision: bool = False
    # 是否按当前终止规则成功完成任务。
    success: bool = False
    # 当前位姿是否位于成功半径内，只表示几何关系，不代表策略已正确停止。
    within_success_radius: bool = False
    # 本回合是否曾进入成功半径，对应 UAV-ON 的 oracle_success。
    oracle_success: bool = False
    # 因 Stop、碰撞或到达目标而正常终止。
    terminated: bool = False
    # 因最大步数等外部限制而截断。
    truncated: bool = False
    # stop、collision、success_radius 或 max_steps；未结束时为空字符串。
    termination_reason: str = ""
    # 累计移动距离。
    move_distance: float = 0.0
    # 轨迹历史，保留位置、四元数、动作和到目标距离。
    trajectory: List[Dict[str, Any]] = field(default_factory=list)

    def append_trajectory(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
        distance: float,
        action: str = "reset",
    ) -> None:
        """向 trajectory 追加一条轨迹记录。"""
        self.trajectory.append(
            {
                "sensors": {
                    "state": {
                        "position": [float(v) for v in position[:3]],
                        "quaternionr": [float(v) for v in quaternion[:4]],
                    }
                },
                "action": str(action),
                "move_distance": round(float(self.move_distance), 4),
                "distance_to_target": round(float(distance), 4),
            }
        )


def load_tasks(task_file: str) -> List[Dict[str, Any]]:
    """读取任务 JSON，并统一整理成环境内部使用的字段格式。"""
    path = Path(task_file).expanduser().resolve()
    raw_tasks = json.loads(path.read_text())
    if not isinstance(raw_tasks, list):
        raise ValueError("task file must contain a JSON list")
    return [_normalize_task(item, path.parent) for item in raw_tasks]


def _normalize_task(task: Dict[str, Any], task_dir: Path) -> Dict[str, Any]:
    """把不同来源的任务字段整理成 SimpleUAVEnv 统一读取的格式。"""
    item = copy.deepcopy(task)
    start_pose = item.get("start_pose", {})
    goal_pose = item.get("goal_pose", {})

    # 支持 UAV-ON 原始嵌套格式，也支持用户后续自定义的扁平格式。
    start_position = start_pose.get("start_position") or item.get("start_position")
    start_quaternion = (
        start_pose.get("start_quaternionr")
        or item.get("start_quaternionr")
        or item.get("start_orientation")
    )
    goal_position = (
        goal_pose.get("goal_position")
        or item.get("object_position")
        or item.get("pose")
    )
    if start_position is None or start_quaternion is None:
        raise ValueError("task is missing start_pose fields: {}".format(item))
    if goal_position is None:
        raise ValueError("task is missing goal position: {}".format(item))

    # 下面这些字段是 SimpleUAVEnv 后续 reset/step/observe 统一依赖的字段。
    item["task_dir"] = str(task_dir)
    item["task_id"] = str(item.get("task_id", item.get("episode_id", "")))
    item["map_name"] = str(item["map_name"])
    item["start_position"] = [float(v) for v in start_position[:3]]
    item["start_quaternionr"] = [float(v) for v in start_quaternion[:4]]
    item["goal_position"] = goal_position
    item["description"] = str(item.get("description", item.get("Landmark", "")))
    return item


class SimpleUAVEnv:
    """基于 ProjectAirSim 的最小批量无人机环境。

    常用生命周期：

    env = SimpleUAVEnv(task_file, config)
    obs = env.reset()
    obs, rewards, dones, infos = env.step(["forward"])
    env.close()
    """

    def __init__(
        self,
        task_file: Optional[str] = None,
        tasks: Optional[List[Dict[str, Any]]] = None,
        config: Optional[UAVEnvConfig] = None,
    ) -> None:
        """初始化环境对象，但不会立即启动 UE 场景。

        真正打开地图发生在 reset()，这样创建对象本身比较轻。
        """
        self.config = config or UAVEnvConfig()
        if tasks is None:
            if task_file is None:
                raise ValueError("task_file or tasks must be provided")
            self.tasks = load_tasks(task_file)
        else:
            self.tasks = [_normalize_task(task, Path.cwd()) for task in tasks]
        # cursor 用于默认顺序取 batch；每次 reset 会向后推进。
        self.cursor = 0
        # 当前正在执行的一批任务。
        self.batch: List[Dict[str, Any]] = []
        # 当前 batch 内每个 episode 的状态。
        self.states: List[EpisodeState] = []
        # 传给 projectairsim_plugin 的机器/场景描述。
        self.machines_info: List[Dict[str, Any]] = []
        # 底层 ProjectAirSim client 适配器，reset() 后才会创建。
        self.simulator_tool: Optional[ProjectAirSimSimulatorClientTool] = None
        # 当前已启动场景的布局签名，用于判断下一批任务能否软重置。
        self._scene_layout_signature: Optional[Tuple[Any, ...]] = None
        # 当前这批 UE 场景已经服务过的回合数。
        self._scene_reuse_count = 0

    def close(self, close_scenes: bool = True) -> None:
        """关闭或断开当前 ProjectAirSim 场景。

        close_scenes=True：通知 server 关闭 UE 地图进程。
        close_scenes=False：只断开 Python client，保留 UE 窗口继续显示。
        """
        if self.simulator_tool is not None:
            try:
                if close_scenes:
                    self.simulator_tool.closeScenes()
                else:
                    self.simulator_tool.disconnectClients()
            finally:
                self.simulator_tool = None
                self._scene_layout_signature = None
                self._scene_reuse_count = 0
        else:
            self._scene_layout_signature = None
            self._scene_reuse_count = 0

    def reset(
        self,
        batch: Optional[List[Dict[str, Any]]] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        """开启一个新 batch 的 episode，并返回初始观测。

        reset() 是环境最重要的入口，它会：
        1. 选择任务 batch。
        2. 根据 map_name 生成 machines_info。
        3. 场景布局变化时打开 UE；布局不变时复用现有 UE。
        4. 把无人机放到每个任务起点。
        5. 读取并返回初始观测。
        """
        selected_batch = self._select_batch(batch=batch, indices=indices)
        selected_machines_info = self._build_machines_info(selected_batch)
        selected_signature = self._machines_signature(selected_machines_info)
        max_scene_reuses = int(self.config.max_scene_reuses)
        can_reuse_scenes = bool(
            self.config.reuse_scenes
            and max_scene_reuses > 0
            and self.simulator_tool is not None
            and self._scene_layout_signature == selected_signature
            and self._scene_reuse_count < max_scene_reuses
        )

        self.batch = selected_batch
        self.machines_info = selected_machines_info
        if can_reuse_scenes:
            # 参考 UAV-ON 的 changeToNewTask：地图不变时保留 UE，只重置位姿。
            self._scene_reuse_count += 1
        else:
            # 地图、GPU 或端口布局变化时，仍执行完整的场景切换。
            self.close()
            self.batch = selected_batch
            self.machines_info = selected_machines_info
            self._open_current_scenes(selected_signature)

        # 把任务 JSON 中的起点 [position, quaternion] 转成 ProjectAirSim 姿态字典。
        start_poses = [
            pose_from_lists(task["start_position"], task["start_quaternionr"])
            for task in self.batch
        ]
        reset_pose_method = (
            self.simulator_tool.resetPosesNoSweep
            if (
                str(self.config.motion_mode) == "teleport"
                and self.simulator_tool.uses_nonphysics_config
            )
            else self.simulator_tool.setPoses
        )
        if not reset_pose_method(self._nest(start_poses)):
            if not can_reuse_scenes:
                raise RuntimeError("failed to set initial poses")
            # 复用连接失效时自动退回完整重启，避免一个旧 UE 拖垮后续任务。
            self.close()
            self.batch = selected_batch
            self.machines_info = selected_machines_info
            self._open_current_scenes(selected_signature)
            reset_pose_method = (
                self.simulator_tool.resetPosesNoSweep
                if (
                    str(self.config.motion_mode) == "teleport"
                    and self.simulator_tool.uses_nonphysics_config
                )
                else self.simulator_tool.setPoses
            )
            if not reset_pose_method(self._nest(start_poses)):
                raise RuntimeError("failed to set initial poses after reopening scenes")

        # 初始化每个 episode 的环境层状态和轨迹。
        self.states = []
        for task in self.batch:
            pose = task["start_position"] + task["start_quaternionr"]
            state = EpisodeState(task=copy.deepcopy(task), pose=pose)
            distance = distance_to_goal(task["start_position"], task["goal_position"])
            state.within_success_radius = (
                distance <= float(self.config.success_distance)
            )
            state.oracle_success = state.within_success_radius
            state.success = bool(
                self.config.terminate_on_success
                and state.within_success_radius
            )
            state.append_trajectory(
                task["start_position"],
                task["start_quaternionr"],
                distance,
                action="reset",
            )
            self.states.append(state)
        return self.observe()

    def observe(self, include_images: bool = True) -> List[Dict[str, Any]]:
        """读取当前观测。

        include_images=True 时会读取 RGB/depth；如果只想快速读状态，可设为 False。
        返回值是 list，每个元素对应 batch 中一个 episode。
        """
        self._require_running()
        # 先读位姿、碰撞、IMU 等传感器状态。
        sensor_results = self.simulator_tool.getSensorInfo()
        # 再按配置读取相机图像。图像较耗时，所以支持关闭。
        image_results = (
            self.simulator_tool.getImageResponses(
                cameras=self.config.cameras,
                rgb_mode=self.config.rgb_mode,
                depth_mode=self.config.depth_mode,
            )
            if include_images
            else None
        )
        observations = []
        for index, nested_index in enumerate(self._nested_indices()):
            machine_index, scene_index = nested_index
            sensors = sensor_results[machine_index][scene_index]["sensors"]
            state_info = sensors["state"]
            # ProjectAirSim 返回的 orientation 会被整理成 [qx,qy,qz,qw]。
            position = [float(v) for v in state_info["position"]]
            quaternion = [float(v) for v in state_info["orientation"]]
            # 环境层同步最新 pose，后续 step 会基于它计算下一步目标。
            self.states[index].pose = position + quaternion
            if not self.states[index].done:
                self.states[index].collision = bool(
                    state_info.get("collision", {}).get("has_collided", False)
                )
            # 每次观测都重新计算几何距离；是否成功停止由 step 收尾阶段决定。
            distance = distance_to_goal(
                position,
                self.states[index].task["goal_position"],
            )
            within_success_radius = (
                distance <= float(self.config.success_distance)
            )
            if not self.states[index].done:
                self.states[index].within_success_radius = within_success_radius
                self.states[index].oracle_success = bool(
                    self.states[index].oracle_success or within_success_radius
                )
                if self.config.terminate_on_success:
                    self.states[index].success = bool(
                        within_success_radius
                        and not self.states[index].collision
                    )
            rgb_images, depth_images = (None, None)
            if image_results is not None:
                rgb_images, depth_images = image_results[machine_index][scene_index]
            observations.append(
                self._make_observation(
                    index=index,
                    sensors=sensors,
                    rgb_images=rgb_images,
                    depth_images=depth_images,
                    distance=distance,
                )
            )
        return observations

    def step(
        self,
        actions: Sequence[str],
        step_sizes: Optional[Sequence[float]] = None,
        is_fixed: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[float], List[bool], List[Dict[str, Any]]]:
        """执行一批动作，并返回 ``obs, rewards, dones, infos``。

        actions 的长度必须等于当前 batch 大小。例如 batch_size=1 时：
        env.step(["forward"])

        默认 motion_mode="teleport"，动作会被转换成目标 pose；碰撞版
        NonPhysics 会对这段离散移动执行 sweep。motion_mode="command" 调用飞控命令。
        """
        self._require_running()
        if isinstance(actions, str):
            actions = [actions]
        if len(actions) != len(self.states):
            raise ValueError("actions length must match current batch")
        if step_sizes is None:
            step_sizes = [None for _ in actions]
        if len(step_sizes) != len(actions):
            raise ValueError("step_sizes length must match actions length")

        target_poses = []
        fly_types = []
        effective_actions = []
        previous_positions = [state.pose[:3] for state in self.states]
        for index, action in enumerate(actions):
            # 已结束的 episode 不再移动，统一转成 stop。
            if self.states[index].done:
                action = "stop"
            effective_actions.append(action)
            # 把离散动作转换成下一步目标姿态和飞行类型。
            target_pose, fly_type = next_pose_from_action(
                self.states[index].pose,
                action,
                step_size=step_sizes[index],
                is_fixed=is_fixed,
                settings=self.config.action_settings,
            )
            target_poses.append(target_pose)
            fly_types.append(fly_type)

        motion_mode = str(self.config.motion_mode)
        if motion_mode == "teleport":
            # NonPhysics 离散动作仍是一帧 pose 更新，但根 Link 开启碰撞时
            # UE 会 sweep 当前位姿到目标位姿之间的整段路径。
            results = self.simulator_tool.movePosesWithSweep(
                self._nest(target_poses)
            )
            if results is None:
                raise RuntimeError("movePosesWithSweep failed during teleport step")

            forced_collisions = []
            diagnostics = []
            for machine_index, scene_index in self._nested_indices():
                result = results[machine_index][scene_index]
                forced_collisions.append(bool(result.get("collision", False)))
                diagnostics.append(result.get("pose_diagnostics", {}))

            settle_seconds = max(
                0.0, float(self.config.teleport_settle_seconds)
            )
            if settle_seconds > 0.0 and any(
                action != "stop" for action in effective_actions
            ):
                time.sleep(settle_seconds)

            # 碰撞时底层会用 CollisionInfo.position 覆盖 NonPhysics 未回写的
            # ground-truth 目标位置，使轨迹、目标距离和画面位置保持一致。
            observations = self.observe()
            for diagnostics_item, fly_type in zip(diagnostics, fly_types):
                diagnostics_item.setdefault("motion_mode", motion_mode)
                diagnostics_item.setdefault("fly_type", fly_type)
            return self._finish_observed_step(
                effective_actions,
                previous_positions,
                observations,
                diagnostics,
                forced_collisions=forced_collisions,
            )

        if motion_mode != "command":
            raise ValueError("unknown motion_mode: {}".format(motion_mode))

        # 真实飞控模式：调用 ProjectAirSim 的 move/rotate 命令。
        # 该模式会受物理、飞控和碰撞影响，后续需要更细调试。
        results = self.simulator_tool.move_to_next_pose(
            poses_list=self._nest(target_poses),
            fly_types=self._nest(fly_types),
        )
        if results is None:
            raise RuntimeError("move_to_next_pose failed")

        forced_collisions = []
        diagnostics = []
        for machine_index, scene_index in self._nested_indices():
            # command 模式下，底层动作结果里会带碰撞和姿态诊断。
            result = results[machine_index][scene_index]
            forced_collisions.append(bool(result.get("collision", False)))
            diagnostics.append(result.get("pose_diagnostics", {}))
        observations = self.observe()
        return self._finish_observed_step(
            effective_actions,
            previous_positions,
            observations,
            diagnostics,
            forced_collisions=forced_collisions,
        )

    def step_gymnasium(
        self,
        actions: Sequence[str],
        step_sizes: Optional[Sequence[float]] = None,
        is_fixed: bool = True,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[float],
        List[bool],
        List[bool],
        List[Dict[str, Any]],
    ]:
        """执行动作，并分别返回 terminated 和 truncated。

        这个接口区分策略/碰撞终止与最大步数截断。
        """
        observations, rewards, _, infos = self.step(
            actions=actions,
            step_sizes=step_sizes,
            is_fixed=is_fixed,
        )
        terminated = [bool(info["terminated"]) for info in infos]
        truncated = [bool(info["truncated"]) for info in infos]
        return observations, rewards, terminated, truncated, infos

    def _finish_observed_step(
        self,
        actions: Sequence[str],
        previous_positions: Sequence[Sequence[float]],
        observations: List[Dict[str, Any]],
        diagnostics: Sequence[Dict[str, Any]],
        forced_collisions: Optional[Sequence[bool]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[float], List[bool], List[Dict[str, Any]]]:
        """统一收尾 step 结果，更新状态并组织 Gym 风格返回值。"""
        rewards = []
        dones = []
        infos = []
        for index, observation in enumerate(observations):
            action = str(actions[index])
            state = self.states[index]
            position = [float(v) for v in observation["pose"][:3]]
            quaternion = [float(v) for v in observation["pose"][3:]]
            distance = float(observation["distance_to_goal"])
            # 批量环境中其他回合仍可能继续；已经结束的回合保持最终结果不变。
            if state.done:
                rewards.append(1.0 if state.success else 0.0)
                dones.append(True)
                infos.append(
                    self._make_step_info(state, distance, diagnostics[index])
                )
                self._sync_observation_state(observation, state)
                continue
            if forced_collisions is not None:
                state.collision = bool(
                    state.collision or forced_collisions[index]
                )
            # Stop 不增加步数；已经结束的 episode 也不再增加步数。
            if action != "stop" and not state.done:
                state.step += 1
            # 这里累计的是环境层离散动作后的直线距离。
            state.move_distance += distance_to_goal(
                position,
                previous_positions[index],
            )
            within_success_radius = (
                distance <= float(self.config.success_distance)
            )
            state.within_success_radius = within_success_radius
            state.oracle_success = bool(
                state.oracle_success or within_success_radius
            )

            # 参考 UAV-ON：碰撞优先失败；Stop 后才按距离判断策略成功；
            # 最大步数是截断。默认模式仍可在进入成功半径时直接终止。
            stopped = action == "stop"
            if state.collision:
                state.success = False
                state.oracle_success = False
                state.terminated = True
                state.truncated = False
                state.termination_reason = "collision"
            elif stopped:
                state.success = within_success_radius
                state.terminated = True
                state.truncated = False
                state.termination_reason = "stop"
            elif self.config.terminate_on_success and within_success_radius:
                state.success = True
                state.terminated = True
                state.truncated = False
                state.termination_reason = "success_radius"
            elif state.step >= int(self.config.max_steps):
                state.success = False
                state.terminated = False
                state.truncated = True
                state.termination_reason = "max_steps"
            else:
                state.success = False
                state.terminated = False
                state.truncated = False
                state.termination_reason = ""
            state.done = bool(state.terminated or state.truncated)
            # 每一步都写入 trajectory，便于后续保存轨迹或给模型看历史。
            state.append_trajectory(
                position,
                quaternion,
                distance,
                action=action,
            )
            # 当前 reward 很简单：成功为 1，否则为 0。后面可以换成 shaping reward。
            reward = 1.0 if state.success else 0.0
            rewards.append(reward)
            dones.append(state.done)
            infos.append(
                self._make_step_info(state, distance, diagnostics[index])
            )
            # observe() 先生成了 observation；这里把 step 后更新的 done/reward
            # 相关状态补回 observation，保持返回值一致。
            self._sync_observation_state(observation, state)
        return observations, rewards, dones, infos

    @staticmethod
    def _make_step_info(
        state: EpisodeState,
        distance: float,
        diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """把回合终止状态整理成 step 返回的诊断信息。"""
        return {
            "task_id": state.task["task_id"],
            "success": state.success,
            "collision": state.collision,
            "within_success_radius": state.within_success_radius,
            "oracle_success": state.oracle_success,
            "terminated": state.terminated,
            "truncated": state.truncated,
            "termination_reason": state.termination_reason,
            "distance_to_goal": float(distance),
            "move_distance": state.move_distance,
            "pose_diagnostics": diagnostics,
        }

    @staticmethod
    def _sync_observation_state(
        observation: Dict[str, Any],
        state: EpisodeState,
    ) -> None:
        """把 step 更新后的回合状态同步回已经生成的观测字典。"""
        observation.update(
            {
                "done": state.done,
                "success": state.success,
                "collision": state.collision,
                "within_success_radius": state.within_success_radius,
                "oracle_success": state.oracle_success,
                "terminated": state.terminated,
                "truncated": state.truncated,
                "termination_reason": state.termination_reason,
                "step": state.step,
                "move_distance": state.move_distance,
                "trajectory": copy.deepcopy(state.trajectory),
            }
        )

    def _select_batch(
        self,
        batch: Optional[List[Dict[str, Any]]],
        indices: Optional[Sequence[int]],
    ) -> List[Dict[str, Any]]:
        """选择本次 reset 使用的任务 batch。"""
        if batch is not None:
            # 用户手动传入任务列表时，直接使用这些任务。
            return [_normalize_task(task, Path.cwd()) for task in batch]
        if indices is not None:
            # 用户传入索引时，从已加载任务中按索引取。
            return [copy.deepcopy(self.tasks[int(index)]) for index in indices]

        # 默认模式：按 cursor 顺序取 batch_size 个任务，到末尾后循环。
        selected = []
        for _ in range(int(self.config.batch_size)):
            selected.append(copy.deepcopy(self.tasks[self.cursor % len(self.tasks)]))
            self.cursor += 1
        return selected

    def _scene_id(self, map_name: str) -> str:
        """把任务里的 map_name 转成 server 认识的 scene_id。"""
        return self.config.scene_aliases.get(str(map_name), str(map_name))

    def _build_machines_info(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """根据任务 batch 生成 projectairsim_plugin 需要的 machines_info。"""
        scenes = [self._scene_id(task["map_name"]) for task in batch]
        configured_gpu_ids = tuple(self.config.gpu_ids or (self.config.gpu_id,))
        gpu_ids = [int(gpu_id) for gpu_id in configured_gpu_ids]
        if any(gpu_id < 0 for gpu_id in gpu_ids):
            raise ValueError("gpu_id/gpu_ids must contain non-negative integers")
        scene_gpu_ids = [
            gpu_ids[index % len(gpu_ids)]
            for index in range(len(scenes))
        ]
        return [
            {
                "MACHINE_IP": self.config.server_host,
                "SOCKET_PORT": int(self.config.server_port),
                "MAX_SCENE_NUM": len(scenes),
                "open_scenes": scenes,
                "gpus": scene_gpu_ids,
            }
        ]

    @staticmethod
    def _machines_signature(
        machines_info: Sequence[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        """生成场景布局签名，任务目标不同但布局相同时仍允许复用。"""
        return tuple(
            (
                str(machine["MACHINE_IP"]),
                int(machine["SOCKET_PORT"]),
                tuple(str(scene) for scene in machine["open_scenes"]),
                tuple(int(gpu_id) for gpu_id in machine["gpus"]),
            )
            for machine in machines_info
        )

    def _open_current_scenes(self, signature: Tuple[Any, ...]) -> None:
        """按当前 machines_info 打开场景，并记录后续软重置所需状态。"""
        self.simulator_tool = ProjectAirSimSimulatorClientTool(
            self.machines_info,
            scene_config_name=self.config.scene_config_name,
        )
        try:
            self.simulator_tool.run_call(
                airsim_timeout=self.config.connect_timeout
            )
        except Exception:
            # 启动中途失败时尽量清理已经创建的连接和场景。
            try:
                self.close()
            except Exception:
                self.simulator_tool = None
                self._scene_layout_signature = None
                self._scene_reuse_count = 0
            raise
        self._scene_layout_signature = signature
        self._scene_reuse_count = 1

    def _nest(self, values: Sequence[Any]) -> List[List[Any]]:
        """把一维 batch 列表整理成 [machine][scene] 的嵌套结构。"""
        nested = []
        index = 0
        for machine in self.machines_info:
            scene_count = len(machine["open_scenes"])
            nested.append(list(values[index:index + scene_count]))
            index += scene_count
        return nested

    def _nested_indices(self) -> Iterable[Tuple[int, int]]:
        """遍历当前 batch 中每个 episode 对应的 machine/scene 下标。"""
        for machine_index, machine in enumerate(self.machines_info):
            for scene_index, _ in enumerate(machine["open_scenes"]):
                yield machine_index, scene_index

    def _make_observation(
        self,
        index: int,
        sensors: Dict[str, Any],
        rgb_images: Optional[List[bytes]],
        depth_images: Optional[List[Any]],
        distance: float,
    ) -> Dict[str, Any]:
        """把底层传感器结果整理成上层更容易使用的 observation dict。"""
        state = self.states[index]
        return {
            "task": copy.deepcopy(state.task),
            "state": copy.deepcopy(sensors["state"]),
            "imu": copy.deepcopy(sensors.get("imu", {})),
            "pose": pose_to_flat_list(state.pose),
            "rgb": rgb_images,
            "depth": depth_images,
            "done": state.done,
            "success": state.success,
            "collision": state.collision,
            "within_success_radius": state.within_success_radius,
            "oracle_success": state.oracle_success,
            "terminated": state.terminated,
            "truncated": state.truncated,
            "termination_reason": state.termination_reason,
            "step": state.step,
            "move_distance": state.move_distance,
            "distance_to_goal": float(distance),
            "trajectory": copy.deepcopy(state.trajectory),
        }

    def _require_running(self) -> None:
        """确保 reset() 已经成功运行，避免未连接时调用 observe/step。"""
        if self.simulator_tool is None:
            raise RuntimeError("environment is not running; call reset() first")
