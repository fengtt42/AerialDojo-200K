"""Current Step3/Step4 trajectory recording integration tests."""

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from AerialDojo.trajectory_recording import multi_scene_astar_record as recorder
from AerialDojo.trajectory_recording import rebuild_collected_index


class CurrentRouteRecordingTest(unittest.TestCase):
    def test_route_index_uses_map_scoped_ids_and_normalizes_v6_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks_root = root / "TASKS_Final_Edition"
            semantic_relative = Path(
                "SemanticOGS/Seen/BaseTasks/Trainset/Test_Map.json"
            )
            semantic_path = tasks_root / semantic_relative
            semantic_path.parent.mkdir(parents=True)
            semantic_path.write_text(
                json.dumps([{
                    "episode_id": "0",
                    "map_name": "Test_Map",
                    "start_object_name": "start_actor",
                    "goal_object_name": "goal_actor",
                    "description": "A goal actor.",
                    "category": "Test",
                    "task": "base_task",
                    "used-in-train": 1,
                }]),
                encoding="utf-8",
            )

            manifests = tasks_root / "manifests"
            manifests.mkdir(parents=True)
            route_index = manifests / "Test_Map.task_route_index.json"
            route_index.write_text(
                json.dumps({
                    "schema_version": "task_route_index_v3",
                    "map_name": "Test_Map",
                    "records": [{
                        "task": "base_task",
                        "split": "Trainset",
                        "episode_id": "0",
                        "semantic_task_file": semantic_relative.as_posix(),
                        "trajectory_task_id": "0_0__1",
                        "trajectory_file": "0_0__1.json",
                    }],
                }),
                encoding="utf-8",
            )

            trajectory_root = root / "Step3_Outputs/Test_Map/results"
            trajectory_root.mkdir(parents=True)
            (trajectory_root / "0_0__1.json").write_text(
                json.dumps({
                    "schema_version": "hybrid_astar_trajectory_v6",
                    "map_name": "Test_Map",
                    "task_id": "0_0__1",
                    "status": "planned",
                    "goal": {"Object_name": "goal_actor"},
                    "trajectory": [
                        {
                            "action": "start",
                            "position_ned_m": [0.0, 0.0, 0.0],
                            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                        {
                            "action": "forward",
                            "position_ned_m": [1.0, 0.0, 0.0],
                            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    ],
                }),
                encoding="utf-8",
            )

            output_root = root / "recordings"
            job_file = root / "record_jobs.yaml"
            job_file.write_text(
                yaml.safe_dump({
                    "server": {"host": "127.0.0.1", "port": 37000},
                    "recording": {"output_root": str(output_root)},
                    "task_source": {
                        "root": str(tasks_root),
                        "map_name": "Unused_Default",
                        "route_index": str(
                            manifests / "{map_name}.task_route_index.json"
                        ),
                        "trajectory_root": str(
                            root / "Step3_Outputs/{map_name}/results"
                        ),
                    },
                    "gpus": {
                        2: {"scenes": [{
                            "scene": "{map_name}", "scene_index": 0,
                        }]},
                    },
                }),
                encoding="utf-8",
            )

            args = recorder.parse_args([
                "--job_file", str(job_file),
                "--map_name", "Test_Map",
                "--dry_run",
            ])
            jobs = recorder.load_recording_jobs(args)

            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["scene"], "Test_Map")
            self.assertEqual(jobs[0]["gpu_id"], 2)
            recording = jobs[0]["recordings"][0]
            self.assertEqual(recording["trajectory_id"], "Test_Map__0_0__1")
            self.assertEqual(recording["trajectory_lookup_id"], "0_0__1")
            self.assertEqual(
                recording["output_dir"],
                output_root
                / "BaseTasks"
                / "Trainset"
                / "Test_Map"
                / "0_0__1",
            )
            self.assertEqual(
                [step["action"] for step in recording["steps"]],
                ["forward", "stop"],
            )
            self.assertEqual(
                recording["steps"][0]["quaternion_wxyz"],
                [1.0, 0.0, 0.0, 0.0],
            )
            episode = recorder.collected_episode_payload(jobs[0], recording)
            self.assertEqual(episode["episode_id"], "0")
            self.assertEqual(episode["recording_id"], "Test_Map__0_0__1")
            self.assertEqual(episode["trajectory_task_id"], "0_0__1")
            self.assertEqual(episode["description"], "A goal actor.")
            self.assertEqual(episode["split"], "train")

            recording["collected_steps"] = [
                {
                    "frame": step["step_index"],
                    "action": step["action"],
                }
                for step in recording["steps"]
            ]
            written = recorder.write_partition_collected_indexes(args, jobs)
            self.assertEqual(
                written,
                [
                    output_root
                    / "BaseTasks"
                    / "Trainset"
                    / "Test_Map"
                    / "collected_episodes.json"
                ],
            )
            partition = json.loads(written[0].read_text(encoding="utf-8"))
            self.assertEqual(partition[0]["episode_id"], "0")
            self.assertEqual(partition[0]["description"], "A goal actor.")
            self.assertEqual(partition[0]["split"], "train")

            slots = recorder.task_scene_slots(
                yaml.safe_load(job_file.read_text(encoding="utf-8")),
                "Test_Map",
                gpu_override=["3", "4"],
            )
            self.assertEqual(
                slots,
                [
                    {"scene_index": 0, "scene": "Test_Map", "gpu_id": 3},
                    {"scene_index": 1, "scene": "Test_Map", "gpu_id": 4},
                ],
            )

    def test_root_index_numbers_train_and_test_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write_partition(task, split, map_name, count):
                path = root / task / split / map_name / "collected_episodes.json"
                path.parent.mkdir(parents=True)
                normalized_split = "train" if split == "Trainset" else "test"
                payload = []
                for index in range(count):
                    payload.append({
                        "episode_id": str(index),
                        "recording_id": "{}__{}__{}__{}".format(
                            map_name, task, split, index
                        ),
                        "trajectory_task_id": "0_0__{}".format(index),
                        "map_name": map_name,
                        "description": "Description {}".format(index),
                        "split": normalized_split,
                        "source_task": {
                            "episode_id": str(index),
                            "map_name": map_name,
                            "task_directory": task,
                            "split": normalized_split,
                            "split_directory": split,
                        },
                    })
                path.write_text(json.dumps(payload), encoding="utf-8")

            write_partition("BaseTasks", "Trainset", "Map_A", 2)
            write_partition("StandardTasks", "Trainset", "Map_B", 1)
            write_partition("BaseTasks", "Testset", "Map_A", 1)
            write_partition("LongTasks", "Testset", "Map_C", 1)

            episodes, summary = rebuild_collected_index.rebuild_episodes(
                root, ["BaseTasks", "StandardTasks", "LongTasks"]
            )

            train = [item for item in episodes if item["split"] == "train"]
            test = [item for item in episodes if item["split"] == "test"]
            self.assertEqual([item["episode_id"] for item in train], ["0", "1", "2"])
            self.assertEqual([item["episode_id"] for item in test], ["0", "1"])
            self.assertEqual(summary["split_counts"], {"train": 3, "test": 2})
            self.assertEqual(summary["map_count"], 3)


if __name__ == "__main__":
    unittest.main()
