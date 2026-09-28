import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

import commentjson
import cv2
import numpy as np
import yaml

from AerialDojo.trajectory_recording import multi_scene_astar_record as recorder
from AerialDojo.trajectory_recording import rebuild_collected_index as rebuild
from AerialDojo.trajectory_recording.published_dataset import file_lock, resolve_config
from AerialDojo.projectairsim_plugin.ProjectAirSimSimulatorServerTool import (
    _parse_args, config_from_args, discover_scene_scripts, resolve_scene_executable,
)


ROOT = Path(__file__).resolve().parents[1]


class PublishedRecordingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "record_jobs.yaml"
        self.config.write_text(yaml.safe_dump({
            "dataset_root": str(self.root),
            "server": {"port": 38000},
            "task_source": {"layout": "published", "map_name": "Map_0"},
            "gpus": {0: {"scenes": [{"scene": "{map_name}", "scene_index": 0}]}},
        }))

    def write_partition(self, split="IID_TRAINS", category="1_BaseTasks", suffix="B", task_type="base_task"):
        is_train = split.endswith("TRAINS")
        folder = "Map_0_{}_{}".format(suffix, "Train" if is_train else "Test")
        relative = Path(split) / category / folder
        task_path = self.root / "SemanticOGS" / relative / "Task.json"
        task_path.parent.mkdir(parents=True)
        tasks = []
        # Episode numbers deliberately differ from list indices and planner ids.
        for episode_id in ("42", "7"):
            tasks.append({
                "episode_id": episode_id, "map_name": "Map_0", "task": task_type,
                "used-in-train": int(is_train), "description": "A stone tower",
                "Landmark": "wooden pier", "goal_object_name": "TowerActor",
                "start_pose": {"start_position": [0, 0, -3], "start_quaternion_xyzw": [0, 0, 0, 1]},
            })
            trajectory_path = self.root / "TrajectoryDATA" / relative / (episode_id + ".json")
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(json.dumps({
                "schema_version": "hybrid_astar_trajectory_v6", "map_name": "Map_0",
                "task_id": "planner_" + episode_id, "goal": {"Object_name": "TowerActor"},
                "trajectory": [
                    {"action": "start", "position_ned_m": [0, 0, -3], "quaternion_xyzw": [0, 0, 0, 1]},
                    {"action": "forward", "position_ned_m": [1, 0, -3], "quaternion_xyzw": [0, 0, 0, 1]},
                    {"action": "rotr", "position_ned_m": [1, 0, -3], "quaternion_xyzw": [0, 0, 0.5, 0.8660254037844386]},
                ],
            }))
        task_path.write_text(json.dumps(tasks))
        return task_path

    def args(self, *extra):
        return recorder.parse_args(["--job_file", str(self.config), *extra])

    def completed_jobs(self, args):
        jobs = recorder.load_recording_jobs(args)
        for job in jobs:
            for recording in job["recordings"]:
                recording["collected_steps"] = [
                    {"frame": step["step_index"], "action": step["action"]}
                    for step in recording["steps"]
                ]
        return jobs

    def test_episode_file_binding_ned_quaternion_and_action_alignment(self):
        self.write_partition()
        args = self.args("--episode_ids", "7", "--gpus", "2", "4")
        jobs = recorder.load_recording_jobs(args)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["gpu_id"], 2)
        recording = jobs[0]["recordings"][0]
        self.assertEqual(Path(recording["trajectory_file"]).name, "7.json")
        self.assertEqual(recording["trajectory_lookup_id"], "planner_7")
        self.assertEqual(recording["output_dir"], self.root / "VideoRECORD/IID_TRAINS/1_BaseTasks/Map_0_B/7")
        self.assertEqual([s["action"] for s in recording["steps"]], ["forward", "rotr", "stop"])
        self.assertEqual(recording["steps"][0]["position_m"], [0, 0, -3])
        for actual, expected in zip(recording["steps"][-1]["quaternion_wxyz"], [0.8660254037844386, 0, 0, 0.5]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(recording["source_task"]["Landmark"], "wooden pier")
        self.assertIn("start_pose", recording["source_task"]["task_payload"])

    def test_all_partition_names_and_no_episode_collisions(self):
        for split in ("IID_TRAINS", "IID_TESTS", "OOD_TRAINS", "OOD_TESTS"):
            for category, suffix, task_type in (
                ("1_BaseTasks", "B", "base_task"), ("2_StandardTasks", "S", "standard_task"),
                ("3_LongHorizonTasks", "L", "long_task"),
            ):
                self.write_partition(split, category, suffix, task_type)
        jobs = recorder.load_recording_jobs(self.args("--gpus", "2", "4"))
        recordings = [recording for job in jobs for recording in job["recordings"]]
        self.assertEqual(len(recordings), 24)
        self.assertEqual(len({item["trajectory_id"] for item in recordings}), 24)
        self.assertEqual(len({item["output_dir"] for item in recordings}), 24)
        self.assertEqual([len(job["recordings"]) for job in jobs], [12, 12])

    def test_filters_limit_and_cli_frame_override(self):
        self.write_partition()
        self.write_partition("IID_TESTS")
        jobs = recorder.load_recording_jobs(self.args("--splits", "IID_TESTS", "--tasks", "base", "--limit", "1", "--max_steps", "2"))
        recording = jobs[0]["recordings"][0]
        self.assertEqual(len(jobs[0]["recordings"]), 1)
        self.assertEqual(recording["source_task"]["split"], "test")
        self.assertEqual([s["action"] for s in recording["steps"]], ["forward", "stop"])

    def test_dry_run_does_not_create_output_or_connect_to_simulator(self):
        self.write_partition()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(recorder.main(["--job_file", str(self.config), "--resume", "--dry_run"]), 0)
        self.assertFalse((self.root / "VideoRECORD").exists())

    def test_missing_trajectory_is_reported_before_any_output(self):
        self.write_partition()
        path = self.root / "TrajectoryDATA/IID_TRAINS/1_BaseTasks/Map_0_B_Train/7.json"
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "7.json"):
            recorder.load_recording_jobs(self.args())
        self.assertFalse((self.root / "VideoRECORD").exists())

    def test_incremental_indexes_preserve_previous_batches_and_source_ids(self):
        self.write_partition()
        self.write_partition("IID_TESTS")
        for split in ("IID_TRAINS", "IID_TESTS"):
            for episode_id in ("7", "42"):
                args = self.args("--splits", split, "--episode_ids", episode_id)
                jobs = self.completed_jobs(args)
                recorder.write_partition_collected_indexes(args, jobs)
                recorder.write_collected_index(args, jobs)
        root = self.root / "VideoRECORD"
        root_items = [json.loads(line) for line in (root / "collected_episodes.jsonl").read_text().splitlines()]
        self.assertEqual(len(root_items), 4)
        partition = json.loads((root / "IID_TRAINS/1_BaseTasks/Map_0_B/collected_episodes.json").read_text())
        self.assertEqual({item["episode_id"] for item in partition}, {"7", "42"})
        episodes, summary = rebuild.rebuild_episodes(root, rebuild.DEFAULT_TASK_DIRECTORIES)
        self.assertEqual(summary["split_counts"], {"train": 2, "test": 2})
        self.assertEqual({item["episode_id"] for item in episodes}, {"7", "42"})

    def test_published_record_capture_and_resume_with_simulated_images(self):
        self.write_partition()
        args = self.args("--episode_ids", "7", "--resume")
        jobs = recorder.load_recording_jobs(args)
        poses = []

        def set_pose(pose, reset_kinematics):
            self.assertTrue(reset_kinematics)
            poses.append(pose)
            return True

        ok, rgb = cv2.imencode(".png", np.zeros((2, 3, 3), dtype=np.uint8))
        self.assertTrue(ok)

        def images(**kwargs):
            self.assertEqual(kwargs["depth_mode"], "float32_m")
            self.assertEqual(kwargs["rgb_mode"], "png")
            return [rgb.tobytes()] * 4, [np.full((2, 3), 1.25, dtype=np.float32)] * 4

        client = SimpleNamespace(
            scene_connections=[[SimpleNamespace(operation_lock=threading.RLock())]],
            drones=[[SimpleNamespace(set_pose=set_pose)]],
            getSceneImageResponses=images,
        )
        recorder.record_scene_job(
            args, client, jobs[0], (dict, dict, dict), threading.Event(),
            progress_callback=lambda: None, text_output_callback=lambda text: None,
        )
        recording = jobs[0]["recordings"][0]
        self.assertEqual(len(poses), 3)
        self.assertEqual(poses[0]["translation"], {"x": 0.0, "y": 0.0, "z": -3.0})
        self.assertEqual(len(list(recording["output_dir"].glob("*.png"))), 12)
        depths = list((recording["output_dir"] / "depth").glob("*.npy"))
        self.assertEqual(len(depths), 12)
        depth = np.load(depths[0])
        self.assertEqual(depth.dtype, np.float32)
        np.testing.assert_allclose(depth, 1.25)
        metadata = json.loads(recording["collected_path"].read_text())
        self.assertEqual(metadata["episode_id"], "7")
        self.assertEqual([step["action"] for step in metadata["steps"]], ["forward", "rotr", "stop"])
        restored_jobs = recorder.load_recording_jobs(args)
        summary = recorder.restore_jobs_from_output(restored_jobs, args.cameras, progress_every=0)
        self.assertEqual(summary["remaining_trajectories"], 0)

    def test_config_paths_and_server_gpu_override(self):
        config = config_from_args(_parse_args(["--config", str(self.config), "--gpus", "2,4"]))
        self.assertEqual(config.root_path, str(self.root / "AerialENVS"))
        self.assertEqual(config.gpus, [2, 4])
        overridden = self.root / "other"
        args = self.args("--dataset_root", str(overridden))
        self.assertEqual(args.output_root, str(overridden / "VideoRECORD"))
        default = resolve_config({})
        self.assertEqual(Path(default["dataset_root"]), ROOT.parent)

    def test_map_discovery_and_legacy_alias(self):
        envs = self.root / "AerialENVS"
        for name, group in (("N_Island_0", "IID_ENVS"), ("U_Factory_1", "OOD_ENVS")):
            script = envs / group / name / (name + ".sh")
            script.parent.mkdir(parents=True)
            script.write_text("#!/bin/sh\n")
            self.assertEqual(resolve_scene_executable(str(envs), name), script)
        self.assertEqual(resolve_scene_executable(str(envs), "FantasyCity").name, "N_Island_0.sh")
        self.assertTrue(discover_scene_scripts(str(envs))["U_Factory_1"]["exists"])
        with self.assertRaises(ValueError):
            resolve_scene_executable(str(envs), "../outside")

    def test_nonphysics_recording_camera_config(self):
        scene = commentjson.loads((ROOT / "config/sim_config/scene_aerialdojo_drone_nonphysics.jsonc").read_text())
        robot = commentjson.loads((ROOT / "config/sim_config" / scene["actors"][0]["robot-config"]).read_text())
        self.assertEqual(robot["physics-type"], "non-physics")
        self.assertFalse(next(link for link in robot["links"] if link["name"] == "Frame")["collision"]["enabled"])
        sensors = {sensor["id"]: sensor for sensor in robot["sensors"]}
        for name in recorder.DEFAULT_CAMERAS:
            captures = {item["image-type"]: item for item in sensors[name]["capture-settings"]}
            for image_type in (0, 2):
                self.assertTrue(captures[image_type]["capture-enabled"])
                self.assertEqual((captures[image_type]["width"], captures[image_type]["height"]), (640, 640))

    def test_same_map_writer_lock_refuses_concurrent_recording(self):
        lock = self.root / "recording.lock"
        with file_lock(lock, blocking=False):
            with self.assertRaisesRegex(RuntimeError, "Another recording"):
                with file_lock(lock, blocking=False):
                    self.fail("second process must not record into the same map")


if __name__ == "__main__":
    unittest.main()
