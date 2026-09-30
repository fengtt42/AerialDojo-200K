#!/usr/bin/env python
"""Replay one precomputed A* trajectory in ProjectAirSim without recording."""

import argparse
import json
import math
from pathlib import Path
import sys
import time
import warnings

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings(
    "ignore",
    message="Python 3.8 is no longer supported by the Python core team.*",
)

DEFAULT_TRAJECTORY_FILE = (
    REPO_ROOT.parent / "TrajectoryDATA" / "ID_TRAINS"
    / "1_BaseTasks" / "N_Island_0_B_Train" / "0.json"
)
DEFAULT_PLUGIN_CONFIG_DIR = (
    REPO_ROOT / "config" / "sim_config"
)
DEFAULT_SCENE_CONFIG = "scene_aerialdojo_drone_nonphysics.jsonc"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay one A* path by directly setting the drone pose."
    )
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=38000)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--scene",
        default="N_Island_0",
        help="Scene id known by ProjectAirSimSimulatorServerTool.py.",
    )
    parser.add_argument(
        "--trajectory_file",
        default=str(DEFAULT_TRAJECTORY_FILE),
        help="Path to one current Step3 JSON or a legacy trajectories.jsonl.",
    )
    parser.add_argument(
        "--sim_config_path",
        default=str(DEFAULT_PLUGIN_CONFIG_DIR),
        help="ProjectAirSim config directory.",
    )
    parser.add_argument(
        "--scene_config_name",
        default=DEFAULT_SCENE_CONFIG,
        help="ProjectAirSim scene config loaded by World.",
    )
    parser.add_argument(
        "--trajectory_id",
        default="",
        help="Example: 0_0__10. If empty, --trajectory_index is used.",
    )
    parser.add_argument(
        "--start_object_id",
        type=int,
        default=0,
        help="Legacy numeric shortcut used with --goal_object_id.",
    )
    parser.add_argument(
        "--goal_object_id",
        type=int,
        default=0,
        help="Optional shortcut used with --start_object_id.",
    )
    parser.add_argument(
        "--trajectory_index",
        type=int,
        default=0,
        help="Zero-based line index in trajectories.jsonl.",
    )
    parser.add_argument("--connect_timeout", type=int, default=240)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="0 means replay all remaining steps.",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Wait for Enter before moving to the next step.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the selected trajectory and do not connect to UE.",
    )
    parser.add_argument(
        "--keep_scenes",
        action="store_true",
        help="Disconnect Python client but leave the UE scene open at the end.",
    )
    parser.add_argument(
        "--force_reopen_scenes",
        action="store_true",
        help="Restart UE even when the server has the same live scene and GPU.",
    )
    parser.add_argument(
        "--takeoff_on_connect",
        action="store_true",
        help="Run takeoff/hover after connecting. Not needed for pose replay.",
    )
    parser.add_argument(
        "--pose_api",
        choices=["set_pose", "ground_truth"],
        default="set_pose",
        help=(
            "set_pose matches the original A* replay script and is best for "
            "non-physics data capture. ground_truth uses our plugin wrapper."
        ),
    )
    parser.add_argument(
        "--no_pause_between_steps",
        action="store_true",
        help="Keep physics running between steps. The drone may fall while waiting.",
    )
    parser.add_argument(
        "--render_steps",
        type=int,
        default=30,
        help="When paused, advance this many sim ticks after each pose update.",
    )
    parser.add_argument(
        "--image_render_steps",
        type=int,
        default=80,
        help=(
            "When paused, let the sim run this many ticks right before "
            "requesting preview images so cameras can publish a fresh frame."
        ),
    )
    parser.add_argument(
        "--image_retries",
        type=int,
        default=2,
        help="Retry preview image requests after advancing extra sim ticks.",
    )
    parser.add_argument(
        "--position_offset",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help=(
            "Temporary replay offset in metres. In NED, negative DZ raises "
            "the drone. Example: --position_offset 0 0 -3"
        ),
    )
    parser.add_argument(
        "--preview_dir",
        default="",
        help="Optional directory for RGB and depth images from the drone camera.",
    )
    parser.add_argument(
        "--preview_cameras",
        nargs="+",
        default=["FrontCamera"],
        help="Camera ids used with --preview_dir.",
    )
    parser.add_argument(
        "--no_color",
        action="store_true",
        help="Disable ANSI colors in terminal output.",
    )
    parser.add_argument(
        "--force_color",
        action="store_true",
        help="Force ANSI colors even when stdout is not detected as a TTY.",
    )
    return parser.parse_args()


def normalize_quaternion(values):
    q = [float(item) for item in values]
    norm = math.sqrt(sum(item * item for item in q))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    return [item / norm for item in q]


def wxyz_to_xyzw(quaternion_wxyz):
    w, x, y, z = normalize_quaternion(quaternion_wxyz)
    return [x, y, z, w]


def normalize_recording_trajectory(trajectory, source="<trajectory>"):
    """Convert a current Step3 trajectory to the recorder's step contract."""
    if not isinstance(trajectory, dict):
        raise ValueError("trajectory must be a JSON object: {}".format(source))
    if isinstance(trajectory.get("steps"), list):
        return trajectory

    public_steps = trajectory.get("trajectory")
    if not isinstance(public_steps, list) or not public_steps:
        raise ValueError(
            "trajectory has neither non-empty 'steps' nor 'trajectory': {}".format(
                source
            )
        )
    normalized_steps = []
    for step_index, step in enumerate(public_steps):
        if not isinstance(step, dict):
            raise ValueError(
                "trajectory step {} must be an object: {}".format(
                    step_index, source
                )
            )
        position = step.get("position_ned_m")
        quaternion_xyzw = step.get("quaternion_xyzw")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(
                "trajectory step {} has invalid position_ned_m: {}".format(
                    step_index, source
                )
            )
        if not isinstance(quaternion_xyzw, list) or len(quaternion_xyzw) != 4:
            raise ValueError(
                "trajectory step {} has invalid quaternion_xyzw: {}".format(
                    step_index, source
                )
            )
        x, y, z, w = normalize_quaternion(quaternion_xyzw)
        normalized_steps.append({
            "step_index": int(step_index),
            "action": str(step.get("action") or ""),
            "position_m": [float(value) for value in position],
            "quaternion_wxyz": [w, x, y, z],
        })

    normalized = dict(trajectory)
    normalized["steps"] = normalized_steps
    normalized.setdefault("trajectory_id", str(trajectory.get("task_id") or ""))
    goal = trajectory.get("goal")
    if isinstance(goal, dict):
        normalized.setdefault("goal_object_name", str(goal.get("Object_name") or ""))
    return normalized


def requested_trajectory_id(args):
    if args.trajectory_id:
        return str(args.trajectory_id)
    if args.start_object_id and args.goal_object_id:
        return "{}_to_{}".format(int(args.start_object_id), int(args.goal_object_id))
    return ""


def load_trajectory(path, trajectory_id="", trajectory_index=0):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("trajectory file does not exist: {}".format(path))

    if path.suffix.lower() == ".json":
        item = normalize_recording_trajectory(
            json.loads(path.read_text(encoding="utf-8")), path
        )
        identifiers = {
            str(value)
            for value in (item.get("trajectory_id"), item.get("task_id"))
            if value not in (None, "")
        }
        if trajectory_id and str(trajectory_id) not in identifiers:
            raise ValueError("trajectory id not found: {}".format(trajectory_id))
        if int(trajectory_index) not in (0,) and not trajectory_id:
            raise IndexError("trajectory index out of range: {}".format(
                trajectory_index
            ))
        return item, 0, 1

    with path.open("r", encoding="utf-8") as handle:
        data_index = -1
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            data_index += 1
            item = normalize_recording_trajectory(
                json.loads(line), "{}:{}".format(path, line_number)
            )
            if trajectory_id:
                if (
                    item.get("trajectory_id") == trajectory_id
                    or item.get("task_id") == trajectory_id
                ):
                    return item, data_index, line_number
            elif data_index == int(trajectory_index):
                return item, data_index, line_number

    if trajectory_id:
        raise ValueError("trajectory id not found: {}".format(trajectory_id))
    raise IndexError("trajectory index out of range: {}".format(trajectory_index))


def select_steps(steps, start_step=0, max_steps=0):
    if not steps:
        raise ValueError("trajectory has no steps")
    start_step = max(0, int(start_step))
    if start_step >= len(steps):
        raise IndexError("start_step out of range: {}".format(start_step))
    selected = list(steps[start_step:])
    if int(max_steps) > 0:
        selected = selected[: int(max_steps)]
    return selected


def import_projectairsim_plugin():
    from AerialDojo.projectairsim_plugin import (  # noqa: E402
        ProjectAirSimSimulatorClientTool,
        make_pose,
    )
    from projectairsim.types import Pose, Quaternion, Vector3  # noqa: E402

    return ProjectAirSimSimulatorClientTool, make_pose, Pose, Quaternion, Vector3


def make_step_pose(step, make_pose):
    return make_pose(
        step["position_m"],
        wxyz_to_xyzw(step["quaternion_wxyz"]),
    )


def make_step_set_pose(step, Pose, Quaternion, Vector3):
    quaternion_wxyz = normalize_quaternion(step["quaternion_wxyz"])
    position = step["position_m"]
    return Pose({
        "translation": Vector3({
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        }),
        "rotation": Quaternion({
            "w": float(quaternion_wxyz[0]),
            "x": float(quaternion_wxyz[1]),
            "y": float(quaternion_wxyz[2]),
            "z": float(quaternion_wxyz[3]),
        }),
        "frame_id": "DEFAULT_FRAME",
    })


def step_with_position_offset(step, offset):
    adjusted = dict(step)
    adjusted["position_m"] = [
        float(step["position_m"][index]) + float(offset[index])
        for index in range(3)
    ]
    return adjusted


def flat_pose_from_sensor(sensor):
    state = sensor["sensors"]["state"]
    return [
        float(value) for value in state["position"]
    ] + [
        float(value) for value in state["orientation"]
    ]


def _component(value, name):
    if isinstance(value, dict):
        return float(value[name])
    if hasattr(value, name):
        return float(getattr(value, name))
    return float(getattr(value, name + "_val"))


def flat_pose_from_projectairsim_pose(pose):
    translation = pose.get("translation", pose.get("position"))
    rotation = pose.get("rotation", pose.get("orientation"))
    return [
        _component(translation, "x"),
        _component(translation, "y"),
        _component(translation, "z"),
        _component(rotation, "x"),
        _component(rotation, "y"),
        _component(rotation, "z"),
        _component(rotation, "w"),
    ]


def collision_from_sensor(sensor):
    state = sensor["sensors"]["state"]
    return bool(state.get("collision", {}).get("has_collided", False))


def read_sensor_pose(client):
    try:
        sensors = client.getSensorInfo()
    except Exception as error:
        print("  warning: getSensorInfo failed: {}".format(error))
        return None, None
    if not sensors:
        return None, None
    return flat_pose_from_sensor(sensors[0][0]), collision_from_sensor(sensors[0][0])


def first_drone(client):
    if not client.drones or not client.drones[0] or client.drones[0][0] is None:
        raise RuntimeError("ProjectAirSim Drone is not connected")
    return client.drones[0][0]


def read_direct_pose(client):
    pose = first_drone(client).get_ground_truth_pose()
    return flat_pose_from_projectairsim_pose(pose)


def first_world(client):
    """Return the first connected World object for this single-scene script."""
    if not client.worlds or not client.worlds[0] or client.worlds[0][0] is None:
        raise RuntimeError("ProjectAirSim World is not connected")
    return client.worlds[0][0]


def pause_world(client):
    world = first_world(client)
    try:
        result = world.pause()
        print("sim paused: {}".format(result))
    except Exception as error:
        print("warning: failed to pause sim: {}".format(error))


def advance_paused_world(client, step_count):
    if int(step_count) <= 0:
        return
    world = first_world(client)
    try:
        world.continue_for_n_steps(int(step_count), wait_until_complete=True)
    except Exception as error:
        print("warning: failed to advance paused sim: {}".format(error))


def advance_paused_world_async(client, step_count):
    if int(step_count) <= 0:
        return
    world = first_world(client)
    try:
        world.continue_for_n_steps(int(step_count), wait_until_complete=False)
    except Exception as error:
        print("warning: failed to start async sim advance: {}".format(error))


ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


def should_use_color(args):
    return bool(args.force_color) or (
        not bool(args.no_color) and sys.stdout.isatty()
    )


def color_text(text, *styles, use_color=True):
    if not use_color:
        return str(text)
    prefix = "".join(ANSI[style] for style in styles if style in ANSI)
    return "{}{}{}".format(prefix, text, ANSI["reset"])


def rounded_list(values, digits=4):
    return [round(float(value), int(digits)) for value in values]


def format_float(value, digits=4):
    return "{:.{}f}".format(float(value), int(digits))


def format_vec(values, digits=4):
    return "[{}]".format(
        ", ".join(format_float(value, digits) for value in values)
    )


def ned_position_to_ue_cm(position_ned):
    return [
        float(position_ned[0]) * 100.0,
        float(position_ned[1]) * 100.0,
        -float(position_ned[2]) * 100.0,
    ]


def position_error(target_position, actual_pose):
    if actual_pose is None:
        return None
    return math.sqrt(
        sum(
            (
                float(target_position[index])
                - float(actual_pose[index])
            ) ** 2
            for index in range(3)
        )
    )


def collision_label(collision, use_color):
    if collision is None:
        return color_text("未知", "yellow", use_color=use_color)
    if bool(collision):
        return color_text("是", "red", "bold", use_color=use_color)
    return color_text("否", "green", use_color=use_color)


def print_separator(use_color):
    print(color_text("\n" + "=" * 72, "cyan", use_color=use_color))


def print_trajectory_summary(
    trajectory,
    data_index,
    line_number,
    step_count,
    use_color,
):
    print_separator(use_color)
    print(color_text("已选择轨迹", "cyan", "bold", use_color=use_color))
    print("  jsonl_index        : {}".format(data_index))
    print("  jsonl_line         : {}".format(line_number))
    print("  trajectory_id      : {}".format(trajectory.get("trajectory_id")))
    print("  task_id            : {}".format(trajectory.get("task_id")))
    print("  start_object_id    : {}".format(trajectory.get("start_object_id")))
    print("  goal_object_id     : {}".format(trajectory.get("goal_object_id")))
    print("  coordinate_system  : {}".format(trajectory.get("coordinate_system")))
    print("  actions            : {}".format(len(trajectory.get("actions", []))))
    print("  steps              : {}".format(step_count))
    print("  goal_distance_m    : {}".format(trajectory.get("goal_distance_m")))
    print("  minimum_clearance_cm: {}".format(
        trajectory.get("minimum_clearance_cm")
    ))


def print_step_target(step, use_color, prefix="设置目标"):
    print_separator(use_color)
    print(
        "{}  step={}  action={}".format(
            color_text(prefix, "magenta", "bold", use_color=use_color),
            color_text(int(step["step_index"]), "cyan", "bold", use_color=use_color),
            color_text(step.get("action", ""), "yellow", "bold", use_color=use_color),
        )
    )
    print(
        "  {} {}".format(
            color_text("目标位置 NED(m)      :", "yellow", use_color=use_color),
            color_text(
                format_vec(step["position_m"], digits=4),
                "yellow",
                "bold",
                use_color=use_color,
            ),
        )
    )
    print(
        "  {} {}".format(
            color_text("目标位置 UE(cm)      :", "yellow", use_color=use_color),
            color_text(
                format_vec(ned_position_to_ue_cm(step["position_m"]), digits=2),
                "yellow",
                "bold",
                use_color=use_color,
            ),
        )
    )
    print(
        "  {} {}".format(
            color_text("目标姿态 quat[wxyz]  :", "yellow", use_color=use_color),
            color_text(
                format_vec(step["quaternion_wxyz"], digits=6),
                "yellow",
                use_color=use_color,
            ),
        )
    )


def print_pose_readback(label, step, actual_pose, collision, use_color):
    target_position = step["position_m"]
    error = position_error(target_position, actual_pose)
    error_text = "未知" if error is None else "{} m".format(format_float(error, 5))
    error_color = "green" if error is not None and error < 0.01 else "yellow"

    print(
        "  {} {}".format(
            color_text(label, "green", "bold", use_color=use_color),
            color_text(
                format_vec(actual_pose[:3], digits=4),
                "green",
                "bold",
                use_color=use_color,
            ),
        )
    )
    print(
        "  {} {}".format(
            color_text("当前位置 UE(cm)      :", "green", use_color=use_color),
            color_text(
                format_vec(ned_position_to_ue_cm(actual_pose[:3]), digits=2),
                "green",
                "bold",
                use_color=use_color,
            ),
        )
    )
    print(
        "  {} {}".format(
            color_text("当前位置姿态 quat[xyzw]:", "green", use_color=use_color),
            color_text(
                format_vec(actual_pose[3:], digits=6),
                "green",
                use_color=use_color,
            ),
        )
    )
    print(
        "  {} {}    {} {}".format(
            color_text("与目标位置误差      :", error_color, use_color=use_color),
            color_text(error_text, error_color, "bold", use_color=use_color),
            color_text("碰撞:", "red" if collision else "green", use_color=use_color),
            collision_label(collision, use_color),
        )
    )


def print_direct_pose(actual_pose, use_color):
    print(
        "  {} {}".format(
            color_text("直接读取 pose[xyzw] :", "blue", use_color=use_color),
            color_text(
                format_vec(actual_pose, digits=4),
                "blue",
                use_color=use_color,
            ),
        )
    )


def print_step(prefix, step, actual_pose=None, collision=None, use_color=False):
    if prefix == "dry_run":
        print_step_target(step, use_color, prefix="dry run")
        return
    print_step_target(step, use_color, prefix=prefix)
    if actual_pose is not None:
        print_pose_readback("当前位置 NED(m)      :", step, actual_pose, collision, use_color)


def save_preview_images(
    client,
    preview_dir,
    step,
    cameras,
    use_color,
    image_render_steps=80,
    image_retries=2,
):
    if not preview_dir:
        return
    output_dir = Path(preview_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = output_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    saved_rgb_paths = []
    saved_depth_paths = []
    missing_cameras = []
    attempts = max(1, int(image_retries) + 1)

    for camera in [str(item) for item in cameras]:
        images = None
        for attempt in range(1, attempts + 1):
            # ProjectAirSim GetImages 会等待“请求之后”的新相机帧。
            # 多相机一次性请求容易被最慢的相机拖住，所以这里逐相机请求。
            advance_paused_world_async(client, int(image_render_steps))
            images = client.getImageResponses(
                cameras=(camera,),
                depth_mode="float32_m",
            )
            if images:
                break
            if attempt < attempts:
                print(color_text(
                    "  {} retry {}/{} after advancing sim ticks".format(
                        camera,
                        attempt,
                        attempts - 1,
                    ),
                    "yellow",
                    use_color=use_color,
                ))

        if not images:
            missing_cameras.append(camera)
            continue

        rgb_images, depth_images = images[0][0]
        if not rgb_images or not depth_images or depth_images[0] is None:
            missing_cameras.append(camera)
            continue

        filename = "step_{:06d}_camera_{}.png".format(
            int(step["step_index"]),
            camera,
        )
        rgb_path = output_dir / filename
        rgb_path.write_bytes(rgb_images[0])
        saved_rgb_paths.append(str(rgb_path))

        depth_metres = np.asarray(depth_images[0], dtype=np.float32)
        if depth_metres.ndim != 2:
            raise RuntimeError(
                "recorded depth must be a single-channel matrix, got shape={}".format(
                    depth_metres.shape,
                )
            )
        depth_path = depth_dir / filename.replace(".png", ".npy")
        np.save(depth_path, depth_metres, allow_pickle=False)
        saved_depth_paths.append(str(depth_path))

    if missing_cameras:
        print(color_text(
            "  warning: preview cameras failed: {}".format(missing_cameras),
            "yellow",
            use_color=use_color,
        ))
    if not saved_rgb_paths:
        print(color_text(
            "  warning: preview image request returned empty result",
            "yellow",
            use_color=use_color,
        ))
        return
    print(
        "  {} {}".format(
            color_text("预览图片:", "blue", "bold", use_color=use_color),
            color_text(saved_rgb_paths, "blue", use_color=use_color),
        )
    )
    print(
        "  {} {}".format(
            color_text("深度图片:", "blue", "bold", use_color=use_color),
            color_text(saved_depth_paths, "blue", use_color=use_color),
        )
    )


def print_offset_note(position_offset, use_color):
    if any(abs(float(value)) > 1e-12 for value in position_offset):
        print(color_text(
            "using temporary position_offset={}".format(
                [float(value) for value in position_offset]
            ),
            "yellow",
            use_color=use_color,
        ))
        print(color_text(
            "note: ProjectAirSim uses NED, so negative z means higher.",
            "yellow",
            use_color=use_color,
        ))


def main():
    args = parse_args()
    use_color = should_use_color(args)
    trajectory_id = requested_trajectory_id(args)
    trajectory, data_index, line_number = load_trajectory(
        args.trajectory_file,
        trajectory_id=trajectory_id,
        trajectory_index=args.trajectory_index,
    )
    steps = select_steps(
        trajectory.get("steps", []),
        start_step=args.start_step,
        max_steps=args.max_steps,
    )

    print_trajectory_summary(
        trajectory,
        data_index=data_index,
        line_number=line_number,
        step_count=len(trajectory.get("steps", [])),
        use_color=use_color,
    )
    print(
        "{} {}  {} {}".format(
            color_text("回放步数:", "cyan", "bold", use_color=use_color),
            color_text(len(steps), "cyan", "bold", use_color=use_color),
            color_text("起始 step_index:", "cyan", use_color=use_color),
            color_text(steps[0]["step_index"], "cyan", "bold", use_color=use_color),
        )
    )

    print_offset_note(args.position_offset, use_color)

    if args.dry_run:
        for raw_step in steps:
            step = step_with_position_offset(raw_step, args.position_offset)
            print_step("dry_run", step, use_color=use_color)
        return

    (
        ProjectAirSimSimulatorClientTool,
        make_pose,
        Pose,
        Quaternion,
        Vector3,
    ) = import_projectairsim_plugin()
    machines_info = [
        {
            "MACHINE_IP": args.server_host,
            "SOCKET_PORT": int(args.server_port),
            "MAX_SCENE_NUM": 1,
            "open_scenes": [args.scene],
            "gpus": [int(args.gpu_id)],
            "TAKEOFF_ON_CONNECT": bool(args.takeoff_on_connect),
            "ENABLE_API_CONTROL_ON_CONNECT": bool(args.takeoff_on_connect),
            "ARM_ON_CONNECT": bool(args.takeoff_on_connect),
        }
    ]

    client = ProjectAirSimSimulatorClientTool(
        machines_info,
        sim_config_path=args.sim_config_path,
        scene_config_name=args.scene_config_name,
    )
    try:
        print_separator(use_color)
        print("{} {}".format(
            color_text("打开地图:", "cyan", "bold", use_color=use_color),
            color_text(args.scene, "cyan", "bold", use_color=use_color),
        ))
        print("  sim_config_path   : {}".format(args.sim_config_path))
        print("  scene_config_name : {}".format(args.scene_config_name))
        print("  pose_api          : {}".format(args.pose_api))
        client.run_call(
            airsim_timeout=int(args.connect_timeout),
            reuse_existing=not bool(args.force_reopen_scenes),
        )
        pause_between_steps = not bool(args.no_pause_between_steps)
        if pause_between_steps:
            pause_world(client)

        for replay_index, raw_step in enumerate(steps):
            step = step_with_position_offset(raw_step, args.position_offset)
            if replay_index > 0:
                if args.manual:
                    input("press Enter to move to step {}...".format(
                        step["step_index"]
                    ))
                else:
                    time.sleep(max(0.0, float(args.interval)))

            print_step_target(step, use_color, prefix="设置目标")
            if args.pose_api == "set_pose":
                ok = first_drone(client).set_pose(
                    make_step_set_pose(step, Pose, Quaternion, Vector3),
                    reset_kinematics=True,
                )
                if not ok:
                    raise RuntimeError("Drone.set_pose failed at step {}".format(
                        step["step_index"]
                    ))
            else:
                if not client.setPoses([[make_step_pose(step, make_pose)]]):
                    raise RuntimeError("setPoses failed at step {}".format(
                        step["step_index"]
                    ))
            immediate_pose, immediate_collision = read_sensor_pose(client)
            if immediate_pose is not None:
                print_pose_readback(
                    "立即回读位置 NED(m)  :",
                    step,
                    immediate_pose,
                    immediate_collision,
                    use_color,
                )
            if args.pose_api == "set_pose":
                print_direct_pose(read_direct_pose(client), use_color)

            if pause_between_steps:
                advance_paused_world(client, args.render_steps)

            actual_pose, collision = read_sensor_pose(client)
            if actual_pose is None:
                if args.pose_api == "set_pose":
                    actual_pose = read_direct_pose(client)
                    collision = None
                else:
                    print(color_text(
                        "  warning: getSensorInfo returned empty result",
                        "yellow",
                        use_color=use_color,
                    ))
                    continue
            print_pose_readback(
                "渲染后位置 NED(m)    :",
                step,
                actual_pose,
                collision,
                use_color,
            )
            save_preview_images(
                client,
                args.preview_dir,
                step,
                args.preview_cameras,
                use_color,
                image_render_steps=args.image_render_steps,
                image_retries=args.image_retries,
            )

        print_separator(use_color)
        print("{} {}".format(
            color_text("回放完成:", "green", "bold", use_color=use_color),
            color_text(
                trajectory.get("trajectory_id"),
                "green",
                "bold",
                use_color=use_color,
            ),
        ))
    finally:
        if args.keep_scenes:
            client.disconnectClients()
            print(color_text(
                "disconnected client; UE scene is kept open",
                "blue",
                use_color=use_color,
            ))
        else:
            client.closeScenes()
            print(color_text("closed scene", "blue", use_color=use_color))


if __name__ == "__main__":
    main()
