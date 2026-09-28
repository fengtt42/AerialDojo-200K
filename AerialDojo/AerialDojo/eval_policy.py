#!/usr/bin/env python
"""使用可扩展策略接口进行在线无人机导航评测。

评测循环每轮把 RGB、深度、位姿、IMU、任务和历史状态组成的完整
observation 交给策略 ``forward``，再执行其返回的动作。内置轨迹回放和
CLIP-H 两种策略，也可以通过工厂接入其他算法。
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping


if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from AerialDojo.env_uav import SimpleUAVEnv, UAVEnvConfig
    from AerialDojo.env_utils_uav import ActionSettings
    from AerialDojo.runtime_config import (
        DEFAULT_SERVER_CONFIG_FILE,
        load_runtime_config,
        server_settings,
    )
    from AerialDojo.policies import (
        PolicyAction,
        PolicyFactory,
    )
else:
    from .env_uav import SimpleUAVEnv, UAVEnvConfig
    from .env_utils_uav import ActionSettings
    from .runtime_config import (
        DEFAULT_SERVER_CONFIG_FILE,
        load_runtime_config,
        server_settings,
    )
    from .policies import (
        PolicyAction,
        PolicyFactory,
    )


POLICY_SETTINGS = PolicyFactory.load_config()
DEFAULT_CAMERAS = tuple(
    str(camera)
    for camera in POLICY_SETTINGS.get("cameras", ("0", "1", "2", "3"))
)
configured_policy_modules = POLICY_SETTINGS.get("policy_module", "")
if isinstance(configured_policy_modules, str):
    DEFAULT_POLICY_MODULES = (
        [configured_policy_modules] if configured_policy_modules else []
    )
elif isinstance(configured_policy_modules, list):
    DEFAULT_POLICY_MODULES = [
        str(module) for module in configured_policy_modules if str(module).strip()
    ]
else:
    raise ValueError("policy_config.yaml 'policy_module' must be a string or list")
SERVER_SETTINGS = server_settings(load_runtime_config(DEFAULT_SERVER_CONFIG_FILE))


def _load_policy_config(value: Any) -> Dict[str, Any]:
    """读取 Python 字典、内联 JSON 或 ``@file.json`` 策略参数。"""
    if isinstance(value, Mapping):
        return dict(value)
    raw_value = str(value or "").strip()
    if not raw_value:
        return {}
    if raw_value.startswith("@"):
        config_path = Path(raw_value[1:]).expanduser().resolve()
        raw_value = config_path.read_text(encoding="utf-8")
    config = json.loads(raw_value)
    if not isinstance(config, dict):
        raise ValueError("--policy-config must contain a JSON object")
    return config


def create_policy(args):
    """导入扩展模块，并通过工厂创建本次评测使用的策略。"""
    PolicyFactory.import_modules(args.policy_module)
    policy_config = _load_policy_config(args.policy_config)
    return PolicyFactory.create(args.policy, **policy_config)


def _format_step_log(
    episode_index: int,
    observation: Dict[str, Any],
    action: PolicyAction,
    distance_to_goal: float,
) -> str:
    """生成统一的逐步日志。"""
    step_log = (
        "episode={} step={} action={} distance_to_goal_m={:.2f}".format(
            episode_index,
            observation["step"],
            action.name,
            float(distance_to_goal),
        )
    )
    if observation.get("collision", False):
        return "COLLISION {} terminated=true reason=collision".format(step_log)
    return step_log


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="config/policy_config.yaml")
    config_args, _ = config_parser.parse_known_args(argv)
    policy_settings = PolicyFactory.load_config(config_args.config)
    configured_server = server_settings(
        load_runtime_config(
            policy_settings.get("server_config", DEFAULT_SERVER_CONFIG_FILE)
        )
    )

    parser = argparse.ArgumentParser(
        description="使用 AerialDojo 可扩展策略接口进行在线导航评测。"
    )
    parser.add_argument(
        "--config",
        default=config_args.config,
        help="在线策略 YAML；命令行参数会覆盖其中的值。",
    )
    parser.add_argument("--task-file", default=policy_settings["task_file"])
    parser.add_argument(
        "--server-host", default=str(configured_server.get("host", "127.0.0.1"))
    )
    parser.add_argument(
        "--server-port", type=int, default=int(configured_server["port"])
    )
    parser.add_argument("--gpu-id", type=int, default=int(policy_settings["gpu_id"]))
    parser.add_argument(
        "--max-actions", type=int, default=int(policy_settings["max_actions"])
    )
    parser.add_argument(
        "--success-distance",
        type=float,
        default=float(policy_settings["success_distance"]),
    )
    parser.add_argument(
        "--motion-mode",
        choices=("command", "teleport"),
        default=policy_settings["motion_mode"],
    )
    parser.add_argument(
        "--scene-config-name",
        default=policy_settings["scene_config"],
        help=(
            "ProjectAirSim 场景配置；默认使用 NonPhysics 四相机体积碰撞无人机。"
        ),
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(policy_settings.get("cameras", DEFAULT_CAMERAS)),
        help="传给策略的相机编号，默认 0 1 2 3。",
    )
    parser.add_argument(
        "--rgb-mode",
        choices=("png", "raw"),
        default=policy_settings.get("rgb_mode", "png"),
        help="RGB 感知格式：压缩 PNG bytes 或解码后的 uint8 数组。",
    )
    parser.add_argument(
        "--depth-mode",
        choices=("uint8", "float32_m"),
        default=policy_settings["depth_mode"],
        help="深度感知格式；默认返回以米为单位的 float32 数组。",
    )
    parser.add_argument(
        "--policy",
        default=policy_settings["policy"],
        help="工厂注册名或 module:Class。",
    )
    parser.add_argument(
        "--policy-module",
        action="append",
        default=(
            [policy_settings["policy_module"]]
            if isinstance(policy_settings.get("policy_module"), str)
            and policy_settings.get("policy_module")
            else list(policy_settings.get("policy_module", []))
        ),
        help="运行前导入的策略注册模块；可重复指定。",
    )
    parser.add_argument(
        "--policy-config",
        default=dict(policy_settings.get("policy_config", {})),
        help=(
            "策略构造参数 JSON，或 @file.json；默认读取 "
            "config/policy_config.yaml 的 policy_config。"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(policy_settings["episodes"]),
        help="要评测的任务数量；0 表示评测任务文件中的全部任务。",
    )
    parser.add_argument(
        "--output-jsonl",
        default=policy_settings["output"],
        help="可选的 JSONL 输出路径，每个任务写入一条评测摘要。",
    )
    parser.add_argument(
        "--teleport-settle-seconds",
        type=float,
        default=float(policy_settings["teleport_settle_seconds"]),
        help="闪现动作后等待多少秒再采集下一轮 RGB-D。",
    )
    return parser.parse_args(argv)


def evaluate(args) -> List[Dict[str, Any]]:
    action_settings = ActionSettings(
        forward_step=1.0,
        lateral_step=1.0,
        vertical_step=1.0,
        turn_degrees=15.0,
    )
    config = UAVEnvConfig(
        server_host=args.server_host,
        server_port=args.server_port,
        gpu_id=args.gpu_id,
        batch_size=1,
        max_steps=args.max_actions,
        success_distance=args.success_distance,
        cameras=tuple(str(camera) for camera in args.cameras),
        motion_mode=args.motion_mode,
        action_settings=action_settings,
        rgb_mode=args.rgb_mode,
        depth_mode=args.depth_mode,
        reuse_scenes=True,
        max_scene_reuses=5000,
        # 与 UAV-ON 一致：进入成功半径只记录，必须由策略主动 Stop 才结束。
        terminate_on_success=False,
        # NonPhysics 机体：reset 直接出生，teleport 动作使用体积 sweep。
        scene_config_name=args.scene_config_name,
        teleport_settle_seconds=args.teleport_settle_seconds,
    )
    env = SimpleUAVEnv(task_file=args.task_file, config=config)
    policy = create_policy(args)
    episode_count = len(env.tasks) if args.episodes <= 0 else min(
        int(args.episodes), len(env.tasks)
    )
    summaries = []
    try:
        for episode_index in range(episode_count):
            observation = env.reset(indices=[episode_index])[0]
            policy.reset(observation)
            total_reward = 0.0
            stopped_by_policy = False
            last_info: Dict[str, Any] = {
                "success": observation["success"],
                "collision": observation["collision"],
                "within_success_radius": observation["within_success_radius"],
                "oracle_success": observation["oracle_success"],
                "terminated": observation["terminated"],
                "truncated": observation["truncated"],
                "termination_reason": observation["termination_reason"],
                "distance_to_goal": observation["distance_to_goal"],
                "move_distance": observation["move_distance"],
            }

            while not observation["done"]:
                # observation 包含 rgb、depth、pose、imu、task、trajectory 等
                # 全部感知和回合数据；策略只负责 forward 并给出下一步动作。
                action = policy.forward(observation)
                if not isinstance(action, PolicyAction):
                    raise TypeError("policy forward() must return PolicyAction")
                stopped_by_policy = action is PolicyAction.Stop
                step_result = env.step_gymnasium(
                    [action.value],
                    step_sizes=[None],
                    is_fixed=True,
                )
                observations, rewards, terminated, truncated, infos = step_result
                observation = observations[0]
                total_reward += float(rewards[0])
                last_info = infos[0]
                print(
                    _format_step_log(
                        episode_index,
                        observation,
                        action,
                        last_info["distance_to_goal"],
                    ),
                    flush=True,
                )
                if terminated[0] or truncated[0]:
                    break

            summary = {
                "episode_index": episode_index,
                "task_id": observation["task"]["task_id"],
                "policy": str(args.policy),
                "steps": observation["step"],
                "success": bool(last_info["success"]),
                "collision": bool(last_info["collision"]),
                "within_success_radius": bool(
                    last_info["within_success_radius"]
                ),
                "oracle_success": bool(last_info["oracle_success"]),
                "terminated": bool(last_info["terminated"]),
                "truncated": bool(last_info["truncated"]),
                "termination_reason": str(last_info["termination_reason"]),
                "stopped_by_policy": bool(stopped_by_policy),
                "distance_to_goal_m": float(last_info["distance_to_goal"]),
                "move_distance_m": float(last_info["move_distance"]),
                "total_reward": total_reward,
            }
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        env.close()

    if args.output_jsonl:
        output_path = Path(args.output_jsonl).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "".join(
                json.dumps(item, ensure_ascii=False) + "\n" for item in summaries
            ),
            encoding="utf-8",
        )
    return summaries


def main(argv=None) -> int:
    args = parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
