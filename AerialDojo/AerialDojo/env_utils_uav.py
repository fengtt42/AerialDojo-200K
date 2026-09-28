"""Small ProjectAirSim-native UAV pose and action helpers."""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence, Tuple

from .projectairsim_plugin import make_pose


PoseDict = Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class ActionSettings:
    """Discrete action sizes used by the lightweight environment."""

    forward_step: float = 1.0
    lateral_step: float = 1.0
    vertical_step: float = 1.0
    turn_degrees: float = 15.0


DEFAULT_ACTION_SETTINGS = ActionSettings()
VALID_ACTIONS = (
    "forward",
    "backward",
    "left",
    "right",
    "ascend",
    "descend",
    "rotl",
    "rotr",
    "stop",
)


def _component(value: Any, name: str) -> float:
    if isinstance(value, dict):
        return float(value[name])
    if hasattr(value, name):
        return float(getattr(value, name))
    return float(getattr(value, name + "_val"))


def normalize_quaternion(quaternion_xyzw: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, z, w = [float(item) for item in quaternion_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    return x / norm, y / norm, z / norm, w / norm


def pose_to_lists(pose: Any) -> Tuple[list, list]:
    """Return ``([x, y, z], [qx, qy, qz, qw])`` from dict/list/object poses."""
    if isinstance(pose, (list, tuple)):
        if len(pose) != 7:
            raise ValueError("list pose must contain 7 values")
        return [float(v) for v in pose[:3]], list(normalize_quaternion(pose[3:]))

    if isinstance(pose, dict):
        position = pose.get("position", pose.get("translation"))
        orientation = pose.get("orientation", pose.get("rotation"))
    else:
        position = getattr(pose, "position", None)
        orientation = getattr(pose, "orientation", None)

    if position is None or orientation is None:
        raise TypeError("pose must contain position/orientation")

    xyz = [_component(position, axis) for axis in ("x", "y", "z")]
    xyzw = normalize_quaternion(
        [_component(orientation, axis) for axis in ("x", "y", "z", "w")]
    )
    return xyz, list(xyzw)


def pose_to_flat_list(pose: Any) -> list:
    position, quaternion = pose_to_lists(pose)
    return position + quaternion


def pose_from_lists(position: Sequence[float], quaternion_xyzw: Sequence[float]) -> PoseDict:
    return make_pose(position, normalize_quaternion(quaternion_xyzw))


def yaw_from_quaternion(quaternion_xyzw: Sequence[float]) -> float:
    """Return yaw in radians from an ``[x, y, z, w]`` quaternion."""
    x, y, z, w = normalize_quaternion(quaternion_xyzw)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


def quaternion_from_yaw(yaw_radians: float) -> Tuple[float, float, float, float]:
    half = float(yaw_radians) * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def quaternion_from_euler(
    roll_radians: float = 0.0,
    pitch_radians: float = 0.0,
    yaw_radians: float = 0.0,
) -> Tuple[float, float, float, float]:
    """把 roll/pitch/yaw 欧拉角转换成 [qx, qy, qz, qw] 四元数。

    ProjectAirSim 使用 NED 坐标。这里采用常见的 ZYX 顺序：
    yaw -> pitch -> roll。
    """
    cr = math.cos(float(roll_radians) * 0.5)
    sr = math.sin(float(roll_radians) * 0.5)
    cp = math.cos(float(pitch_radians) * 0.5)
    sp = math.sin(float(pitch_radians) * 0.5)
    cy = math.cos(float(yaw_radians) * 0.5)
    sy = math.sin(float(yaw_radians) * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return normalize_quaternion((qx, qy, qz, qw))


def quaternion_from_euler_degrees(
    roll_degrees: float = 0.0,
    pitch_degrees: float = 0.0,
    yaw_degrees: float = 0.0,
) -> Tuple[float, float, float, float]:
    """角度制欧拉角转四元数，便于脚本和人工标注使用。"""
    return quaternion_from_euler(
        roll_radians=math.radians(float(roll_degrees)),
        pitch_radians=math.radians(float(pitch_degrees)),
        yaw_radians=math.radians(float(yaw_degrees)),
    )


def pose_from_xyz_euler_degrees(
    x: float,
    y: float,
    z: float,
    pitch_degrees: float = 0.0,
    yaw_degrees: float = 0.0,
    roll_degrees: float = 0.0,
) -> PoseDict:
    """用 xyz + pitch/yaw/roll 角度创建 ProjectAirSim 姿态。"""
    quaternion = quaternion_from_euler_degrees(
        roll_degrees=roll_degrees,
        pitch_degrees=pitch_degrees,
        yaw_degrees=yaw_degrees,
    )
    return pose_from_lists((x, y, z), quaternion)


def wrap_pi(angle_radians: float) -> float:
    return (float(angle_radians) + math.pi) % (2.0 * math.pi) - math.pi


def distance_to_goal(position: Sequence[float], goal_position: Any) -> float:
    """Distance to one goal point or the nearest point in a point list."""
    px, py, pz = [float(v) for v in position[:3]]
    if (
        isinstance(goal_position, (list, tuple))
        and goal_position
        and isinstance(goal_position[0], (list, tuple))
    ):
        return min(distance_to_goal(position, item) for item in goal_position)
    gx, gy, gz = [float(v) for v in goal_position[:3]]
    return math.sqrt((px - gx) ** 2 + (py - gy) ** 2 + (pz - gz) ** 2)


def next_pose_from_action(
    current_pose: Any,
    action: str,
    step_size: Optional[float] = None,
    is_fixed: bool = True,
    settings: ActionSettings = DEFAULT_ACTION_SETTINGS,
) -> Tuple[PoseDict, str]:
    """Compute the target pose and ProjectAirSim fly type for one action."""
    if action not in VALID_ACTIONS:
        raise ValueError("unknown action: {}".format(action))

    position, quaternion = pose_to_lists(current_pose)
    yaw = yaw_from_quaternion(quaternion)
    target = list(position)
    target_quaternion = list(quaternion)
    fly_type = "move"

    def _step(default_value: float) -> float:
        return float(default_value if is_fixed or step_size is None else step_size)

    if action == "forward":
        step = _step(settings.forward_step)
        target[0] += math.cos(yaw) * step
        target[1] += math.sin(yaw) * step
        fly_type = "move_horizontal"
    elif action == "backward":
        step = _step(settings.forward_step)
        target[0] -= math.cos(yaw) * step
        target[1] -= math.sin(yaw) * step
        fly_type = "move_horizontal"
    elif action == "left":
        step = _step(settings.lateral_step)
        target[0] -= math.cos(yaw + math.pi / 2.0) * step
        target[1] -= math.sin(yaw + math.pi / 2.0) * step
        fly_type = "move_horizontal"
    elif action == "right":
        step = _step(settings.lateral_step)
        target[0] += math.cos(yaw + math.pi / 2.0) * step
        target[1] += math.sin(yaw + math.pi / 2.0) * step
        fly_type = "move_horizontal"
    elif action == "ascend":
        target[2] -= _step(settings.vertical_step)
        fly_type = "move_vertical"
    elif action == "descend":
        target[2] += _step(settings.vertical_step)
        fly_type = "move_vertical"
    elif action == "rotl":
        degrees = _step(settings.turn_degrees)
        target_quaternion = list(quaternion_from_yaw(wrap_pi(yaw - math.radians(degrees))))
        fly_type = "rotate"
    elif action == "rotr":
        degrees = _step(settings.turn_degrees)
        target_quaternion = list(quaternion_from_yaw(wrap_pi(yaw + math.radians(degrees))))
        fly_type = "rotate"
    else:
        fly_type = "stop"

    return pose_from_lists(target, target_quaternion), fly_type
