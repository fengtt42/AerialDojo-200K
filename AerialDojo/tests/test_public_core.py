import unittest
import json
import tempfile
from pathlib import Path

import commentjson
import numpy as np

from AerialDojo.env_uav import EpisodeState, SimpleUAVEnv, UAVEnvConfig, load_tasks
from AerialDojo.env_utils_uav import (
    ActionSettings,
    distance_to_goal,
    next_pose_from_action,
    pose_to_flat_list,
)
from AerialDojo.eval_policy import parse_args
from AerialDojo.policies import (
    NavigationPolicy,
    PolicyAction,
    PolicyFactory,
    TrajectoryPolicy,
)
from AerialDojo.projectairsim_plugin.ProjectAirSimSimulatorClientTool import (
    SceneConnection,
    convert_depth_metres,
)
from AerialDojo.projectairsim_plugin.ProjectAirSimSimulatorServerTool import (
    _parse_args as parse_server_args,
    config_from_args,
)
from AerialDojo.runtime_config import load_runtime_config, server_settings


ROOT = Path(__file__).resolve().parents[1]


class PublicCoreTests(unittest.TestCase):
    def test_default_configs_are_self_contained(self):
        settings = server_settings(load_runtime_config())
        self.assertEqual(settings["port"], 36000)
        self.assertEqual(settings["gpus"], [0])
        policy_args = parse_args([])
        self.assertEqual(policy_args.policy, "trajectory")
        self.assertEqual(
            policy_args.task_file,
            "example/TASKS/SemanticOGS/Seen/BaseTasks/Testset/N_Island_0.json",
        )

        server = config_from_args(parse_server_args([]))
        self.assertEqual(server.root_path, "~/maps")
        self.assertTrue(server.render_offscreen)
        self.assertNotIn("-fullscreen", server.extra_ue_args)

    def test_action_enum_and_factory(self):
        @PolicyFactory.register("unit_test_policy", replace=True)
        class UnitTestPolicy(NavigationPolicy):
            def forward(self, observation):
                return PolicyAction.Stop

        self.assertIs(PolicyFactory.create("unit_test_policy").forward({}), PolicyAction.Stop)
        self.assertEqual(PolicyAction.MoveForward.value, "forward")

    def test_trajectory_selects_the_first_task(self):
        # The public code can be distributed without the example dataset.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        task_file = Path(temporary.name) / "Task.json"
        trajectory_file = Path(temporary.name) / "trajectories.jsonl"
        task_file.write_text(json.dumps([{
            "episode_id": "0", "map_name": "Map_0",
            "start_object_name": "start_actor", "goal_object_name": "goal_actor",
            "start_pose": {"start_position": [0, 0, 0], "start_quaternionr": [0, 0, 0, 1]},
            "goal_pose": {"goal_position": [2, 0, 0]},
        }]))
        trajectory_file.write_text(json.dumps({
            "trajectory_id": "example_route", "start_object_id": "start_actor",
            "goal_object_id": "goal_actor", "actions": ["forward", "forward", "stop"],
            "steps": [{"position_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]}],
        }) + "\n")
        task = load_tasks(str(task_file))[0]
        observation = {
            "task": task,
            "pose": task["start_position"] + task["start_quaternionr"],
        }
        policy = TrajectoryPolicy(str(trajectory_file))
        policy.reset(observation)
        self.assertGreater(len(policy._actions), 0)
        pose = list(observation["pose"])
        for _ in range(300):
            observation["pose"] = pose
            action = policy.forward(observation)
            if action is PolicyAction.Stop:
                break
            target_pose, _ = next_pose_from_action(
                pose,
                action.value,
                settings=ActionSettings(1.0, 1.0, 1.0, 15.0),
            )
            pose = pose_to_flat_list(target_pose)
        else:
            self.fail("trajectory did not stop within 300 actions")

        self.assertLessEqual(
            distance_to_goal(pose[:3], task["goal_position"]),
            1.0,
        )

    def test_depth_stays_float32_metres(self):
        depth = convert_depth_metres([[1.25, 2.5]], depth_mode="float32_m")
        self.assertEqual(depth.dtype, np.float32)
        np.testing.assert_allclose(depth, [[1.25, 2.5]])

    def test_collision_topic_implies_collision(self):
        connection = SceneConnection({}, None, None, None, 0, 0)
        connection.on_collision("/collision", {"object_name": "Wall"})
        sequence, collision = connection.collision_snapshot()
        self.assertEqual(sequence, 1)
        self.assertTrue(collision["has_collided"])

    def test_collision_terminates_episode(self):
        env = SimpleUAVEnv.__new__(SimpleUAVEnv)
        env.config = UAVEnvConfig(success_distance=1.0, max_steps=10)
        task = {
            "task_id": "collision-test",
            "goal_position": [5.0, 0.0, 0.0],
        }
        env.states = [EpisodeState(task=task, pose=[0, 0, 0, 0, 0, 0, 1])]
        observation = {
            "pose": [0.5, 0, 0, 0, 0, 0, 1],
            "distance_to_goal": 4.5,
        }
        observations, rewards, dones, infos = env._finish_observed_step(
            ["forward"],
            [[0, 0, 0]],
            [observation],
            [{}],
            forced_collisions=[True],
        )
        self.assertTrue(dones[0])
        self.assertEqual(rewards[0], 0.0)
        self.assertTrue(infos[0]["terminated"])
        self.assertFalse(infos[0]["truncated"])
        self.assertEqual(infos[0]["termination_reason"], "collision")
        self.assertTrue(observations[0]["collision"])

    def test_collision_robot_config(self):
        config_path = ROOT / "config" / "sim_config" / "collision.jsonc"
        payload = commentjson.loads(config_path.read_text(encoding="utf-8"))
        frame = next(link for link in payload["links"] if link["name"] == "Frame")
        self.assertTrue(frame["collision"]["enabled"])


if __name__ == "__main__":
    unittest.main()
