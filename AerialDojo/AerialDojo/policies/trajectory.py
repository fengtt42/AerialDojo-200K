"""忽略感知内容、按预计算轨迹顺序返回动作的示例策略。"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..env_utils_uav import wrap_pi, yaw_from_quaternion
from .base import NavigationPolicy, PolicyAction
from .factory import PolicyFactory


ACTION_MAP = {
    "forward": PolicyAction.MoveForward,
    "left": PolicyAction.MoveLeft,
    "right": PolicyAction.MoveRight,
    "ascend": PolicyAction.MoveUp,
    "descend": PolicyAction.MoveDown,
    "rotl": PolicyAction.TurnLeft,
    "rotr": PolicyAction.TurnRight,
    "stop": PolicyAction.Stop,
}

TURN_RADIANS = math.radians(15.0)


@PolicyFactory.register("trajectory")
class TrajectoryPolicy(NavigationPolicy):
    """从 JSONL 读取轨迹，并逐轮回放其中的 ``actions``。

    未指定 ``trajectory_id`` 时，根据 observation.task 中的起点和终点
    object id 自动选择轨迹。指定后则始终使用该 id 对应的轨迹。
    """

    def __init__(
        self,
        trajectory_file: str,
        trajectory_id: Optional[str] = None,
    ) -> None:
        self.trajectory_file = Path(trajectory_file).expanduser().resolve()
        self.trajectory_id = (
            str(trajectory_id) if trajectory_id is not None else None
        )
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._by_objects: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._actions: List[PolicyAction] = []
        self._next_action_index = 0
        self._load_trajectories()

    def _load_trajectories(self) -> None:
        if not self.trajectory_file.is_file():
            raise FileNotFoundError(
                "trajectory file does not exist: {}".format(
                    self.trajectory_file
                )
            )

        with self.trajectory_file.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                line = line.strip()
                if not line:
                    continue
                trajectory = json.loads(line)
                trajectory_key = str(trajectory.get("trajectory_id", ""))
                if not trajectory_key:
                    raise ValueError(
                        "trajectory line {} is missing trajectory_id".format(
                            line_number
                        )
                    )
                self._validate_actions(trajectory, line_number)
                self._by_id[trajectory_key] = trajectory
                object_key = (
                    str(trajectory.get("start_object_id", "")),
                    str(trajectory.get("goal_object_id", "")),
                )
                self._by_objects[object_key] = trajectory

    @staticmethod
    def _validate_actions(
        trajectory: Mapping[str, Any], line_number: int
    ) -> None:
        actions = trajectory.get("actions")
        if not isinstance(actions, list):
            raise ValueError(
                "trajectory line {} has no actions list".format(line_number)
            )
        unknown = [str(action) for action in actions if action not in ACTION_MAP]
        if unknown:
            raise ValueError(
                "trajectory line {} contains unsupported actions: {}".format(
                    line_number, ", ".join(unknown)
                )
            )

    @staticmethod
    def _task_object_id(task: Mapping[str, Any], prefix: str) -> str:
        for key in (
            "{}_object_name".format(prefix),
            "{}_object_id".format(prefix),
        ):
            value = task.get(key)
            if value is not None:
                return str(value)
        raise KeyError("task is missing {} object id".format(prefix))

    def _select_trajectory(
        self, observation: Mapping[str, Any]
    ) -> Dict[str, Any]:
        if self.trajectory_id is not None:
            trajectory = self._by_id.get(self.trajectory_id)
            if trajectory is None:
                raise KeyError(
                    "trajectory_id not found: {}".format(self.trajectory_id)
                )
            return trajectory

        task = observation.get("task", {})
        task_trajectory_id = task.get("trajectory_id")
        if task_trajectory_id is not None:
            trajectory = self._by_id.get(str(task_trajectory_id))
            if trajectory is not None:
                return trajectory

        object_key = (
            self._task_object_id(task, "start"),
            self._task_object_id(task, "goal"),
        )
        trajectory = self._by_objects.get(object_key)
        if trajectory is None:
            raise KeyError(
                "trajectory not found for start={} goal={}".format(*object_key)
            )
        return trajectory

    def reset(self, observation: Mapping[str, Any]) -> None:
        """选择轨迹、对齐规划起始朝向，并重置动作游标。"""
        trajectory = self._select_trajectory(observation)
        alignment_actions = self._initial_alignment_actions(
            observation, trajectory
        )
        trajectory_actions = [
            ACTION_MAP[action] for action in trajectory["actions"]
        ]
        self._actions = alignment_actions + trajectory_actions
        self._next_action_index = 0

    @staticmethod
    def _trajectory_start_step(
        trajectory: Mapping[str, Any]
    ) -> Optional[Mapping[str, Any]]:
        steps = trajectory.get("steps")
        if not isinstance(steps, list) or not steps:
            return None
        start_step = steps[0]
        return start_step if isinstance(start_step, Mapping) else None

    def _initial_alignment_actions(
        self,
        observation: Mapping[str, Any],
        trajectory: Mapping[str, Any],
    ) -> List[PolicyAction]:
        """用 15 度转向把任务初始朝向对齐到轨迹 ``steps[0]``。"""
        start_step = self._trajectory_start_step(trajectory)
        if start_step is None:
            return []

        current_pose = observation.get("pose")
        start_position = start_step.get("position_m")
        quaternion_wxyz = start_step.get("quaternion_wxyz")
        if not isinstance(current_pose, (list, tuple)) or len(current_pose) < 7:
            raise ValueError("observation pose must contain xyz + xyzw")
        if not isinstance(start_position, (list, tuple)) or len(start_position) < 3:
            raise ValueError("trajectory steps[0] is missing position_m")
        if not isinstance(quaternion_wxyz, (list, tuple)) or len(
            quaternion_wxyz
        ) < 4:
            raise ValueError("trajectory steps[0] is missing quaternion_wxyz")

        position_error = math.sqrt(
            sum(
                (float(current_pose[index]) - float(start_position[index])) ** 2
                for index in range(3)
            )
        )
        if position_error > 0.25:
            raise ValueError(
                "trajectory {} starts {:.3f} m away from task start".format(
                    trajectory.get("trajectory_id", ""), position_error
                )
            )

        w, x, y, z = [float(value) for value in quaternion_wxyz[:4]]
        desired_yaw = yaw_from_quaternion((x, y, z, w))
        current_yaw = yaw_from_quaternion(current_pose[3:7])
        yaw_delta = wrap_pi(desired_yaw - current_yaw)
        turn_count = int(round(abs(yaw_delta) / TURN_RADIANS))
        if turn_count == 0:
            return []
        turn_action = (
            PolicyAction.TurnLeft
            if yaw_delta < 0.0
            else PolicyAction.TurnRight
        )
        return [turn_action] * turn_count

    def forward(self, observation: Mapping[str, Any]) -> PolicyAction:
        """忽略本轮图像和深度，返回轨迹中的下一个动作。"""
        del observation
        if self._next_action_index >= len(self._actions):
            return PolicyAction.Stop
        action = self._actions[self._next_action_index]
        self._next_action_index += 1
        return action
