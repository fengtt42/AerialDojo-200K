from pathlib import Path
import copy
import tempfile
import unittest
from unittest import mock

import numpy as np


from AerialDojo.trajectory_recording import multi_scene_astar_record as RECORDER


CAMERAS = (
    "FrontCamera",
    "LeftCamera",
    "RightCamera",
    "DownCamera",
)


def planned_steps():
    return [
        {
            "step_index": 0,
            "action": "forward",
            "position_m": [0.0, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "step_index": 1,
            "action": "forward",
            "position_m": [1.0, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "step_index": 2,
            "action": "stop",
            "position_m": [2.0, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    ]


def make_job(output_dir):
    steps = planned_steps()
    trajectory_id = "5001_to_5002"
    recording = {
        "task_order": 0,
        "sequence_index": 0,
        "trajectory_id": trajectory_id,
        "trajectory_file": "test.jsonl",
        "trajectory": {
            "trajectory_id": trajectory_id,
            "goal_object_id": 5002,
            "steps": list(steps),
        },
        "data_index": 0,
        "line_number": 1,
        "start_step": 0,
        "max_steps": 0,
        "source_task": {},
        "estimated_steps": len(steps),
        "steps": steps,
        "output_dir": Path(output_dir),
        "saved_steps": 0,
        "saved_rgb_images": 0,
        "saved_depth_images": 0,
        "collected_steps": [],
        "collected_path": (
            Path(output_dir)
            / "trajectory_5001_to_5002_collected.jsonl"
        ),
    }
    job = {
        "scene_index": 0,
        "scene": "N_Island_0",
        "gpu_id": 0,
        "recordings": [recording],
    }
    return job, recording


def write_frame(recording, step, missing_depth_camera=None):
    output_dir = Path(recording["output_dir"])
    depth_dir = output_dir / "depth"
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    for camera in CAMERAS:
        rgb_name = RECORDER.recording_frame_filename(step, camera, "png")
        (output_dir / rgb_name).write_bytes(b"\x89PNG\r\n\x1a\nresume-test")
        if camera == missing_depth_camera:
            continue
        depth_name = RECORDER.recording_frame_filename(step, camera, "npy")
        np.save(
            depth_dir / depth_name,
            np.ones((2, 3), dtype=np.float32),
            allow_pickle=False,
        )


class RecordingResumeTest(unittest.TestCase):
    def test_async_nas_writer_stages_locally_then_commits_to_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "nas"
            stage_root = Path(directory) / "local_stage"
            output_dir = (
                output_root
                / "LongTasks"
                / "Trainset"
                / "5001_to_5002"
            )
            job, recording = make_job(output_dir)
            writer = RECORDER.AsyncNasFrameWriter(
                stage_root,
                output_root,
                queue_frames=2,
                scene_index=0,
            )
            try:
                result = RECORDER.save_job_png_images(
                    job,
                    recording,
                    recording["steps"][0],
                    (
                        [b"png-data"] * len(CAMERAS),
                        [np.ones((2, 3), dtype=np.float32)] * len(CAMERAS),
                    ),
                    CAMERAS,
                    async_nas_writer=writer,
                )
                RECORDER.append_collected_step(
                    job, recording, recording["steps"][0], result
                )
                writer.flush()
            finally:
                writer.close()

            self.assertEqual(recording["_nas_committed_count"], 1)
            self.assertEqual(len(list(output_dir.glob("*.png"))), 4)
            self.assertEqual(len(list((output_dir / "depth").glob("*.npy"))), 4)
            self.assertTrue(
                all(str(path).startswith(str(output_root)) for path in result["rgb"].values())
            )
            self.assertFalse(any(stage_root.rglob("*.png")))
            self.assertFalse(any(stage_root.rglob("*.npy")))

    def test_async_writer_keeps_active_staging_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "nas"
            stage_root = Path(directory) / "local_stage"
            output_dir = (
                output_root
                / "LongTasks"
                / "Trainset"
                / "5001_to_5002"
            )
            writer = RECORDER.AsyncNasFrameWriter(
                stage_root,
                output_root,
                queue_frames=2,
                scene_index=0,
            )
            stage_output_dir = writer.staging_path(output_dir)
            stage_depth_dir = writer.staging_path(output_dir / "depth")
            stage_depth_dir.mkdir(parents=True)
            rgb_source = stage_output_dir / "previous.png"
            depth_source = stage_depth_dir / "previous.npy"
            rgb_source.write_bytes(b"png-data")
            depth_source.write_bytes(b"npy-data")

            try:
                writer._clean_local_files(
                    (
                        (rgb_source, output_dir / rgb_source.name),
                        (depth_source, output_dir / "depth" / depth_source.name),
                    )
                )
            finally:
                writer.close()

            self.assertFalse(rgb_source.exists())
            self.assertFalse(depth_source.exists())
            self.assertTrue(stage_output_dir.is_dir())
            self.assertTrue(stage_depth_dir.is_dir())

    def test_async_snapshot_only_reports_nas_committed_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            job, recording = make_job(directory)
            shapes = {
                RECORDER.collected_camera_name(camera): [2, 3]
                for camera in CAMERAS
            }
            recording["collected_steps"] = [
                RECORDER.collected_step_payload(
                    step,
                    RECORDER.recording_saved_paths(
                        recording, step, CAMERAS, image_shapes=shapes
                    ),
                )
                for step in recording["steps"][:2]
            ]
            recording["saved_steps"] = 2
            recording["saved_rgb_images"] = 8
            recording["saved_depth_images"] = 8
            recording["_nas_committed_count"] = 1
            recording["_nas_camera_count"] = len(CAMERAS)

            snapshot = RECORDER.scene_job_snapshot(
                job, include_collected=True
            )["recordings"][0]

            self.assertEqual(snapshot["saved_steps"], 1)
            self.assertEqual(snapshot["saved_rgb_images"], 4)
            self.assertEqual(snapshot["saved_depth_images"], 4)
            self.assertEqual(len(snapshot["collected_steps"]), 1)

    def test_partial_frame_resumes_from_first_incomplete_step(self):
        with tempfile.TemporaryDirectory() as directory:
            job, recording = make_job(directory)
            write_frame(recording, recording["steps"][0])
            write_frame(
                recording,
                recording["steps"][1],
                missing_depth_camera="DownCamera",
            )
            write_frame(recording, recording["steps"][2])

            count = RECORDER.restore_recording_from_output(
                job, recording, CAMERAS, write_marker=False
            )

            self.assertEqual(count, 1)
            self.assertEqual(recording["_resume_step_offset"], 1)
            self.assertEqual(len(recording["collected_steps"]), 1)
            self.assertEqual(recording["collected_steps"][0]["frame"], 0)

    def test_partial_metadata_is_more_conservative_than_files(self):
        with tempfile.TemporaryDirectory() as directory:
            job, recording = make_job(directory)
            write_frame(recording, recording["steps"][0])
            write_frame(recording, recording["steps"][1])
            shapes = {
                RECORDER.collected_camera_name(camera): [2, 3]
                for camera in CAMERAS
            }
            saved_paths = RECORDER.recording_saved_paths(
                recording,
                recording["steps"][0],
                CAMERAS,
                image_shapes=shapes,
            )
            recording["collected_steps"] = [
                RECORDER.collected_step_payload(
                    recording["steps"][0], saved_paths
                )
            ]
            RECORDER.write_collected_episode(job, recording)

            new_job, new_recording = make_job(directory)
            count = RECORDER.restore_recording_from_output(
                new_job, new_recording, CAMERAS, write_marker=False
            )

            self.assertEqual(count, 1)
            self.assertEqual(len(new_recording["collected_steps"]), 1)

    def test_completed_folder_gets_marker_and_marker_skips_rescan(self):
        with tempfile.TemporaryDirectory() as directory:
            job, recording = make_job(directory)
            for step in recording["steps"]:
                write_frame(recording, step)

            count = RECORDER.restore_recording_from_output(
                job, recording, CAMERAS, write_marker=True
            )

            self.assertEqual(count, len(recording["steps"]))
            self.assertTrue(RECORDER.completion_marker_path(recording).is_file())

            new_job, new_recording = make_job(directory)
            with mock.patch.object(
                RECORDER,
                "contiguous_recording_file_prefix",
                side_effect=AssertionError("completed marker should skip scan"),
            ):
                count = RECORDER.restore_recording_from_output(
                    new_job, new_recording, CAMERAS, write_marker=False
                )

            self.assertEqual(count, len(new_recording["steps"]))
            self.assertEqual(new_recording["_resume_source"], "marker")

    def test_trusted_partition_index_skips_trajectory_folder_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = (
                Path(directory)
                / "BaseTasks"
                / "Testset"
                / "5001_to_5002"
            )
            job, recording = make_job(output_dir)
            source_task = {
                "file": "TASKS/BaseTasks/Testset/N_Island_0.json",
                "index": 0,
                "task_directory": "BaseTasks",
                "split_directory": "Testset",
            }
            recording["source_task"] = dict(source_task)
            shapes = {
                RECORDER.collected_camera_name(camera): [2, 3]
                for camera in CAMERAS
            }
            recording["collected_steps"] = [
                RECORDER.collected_step_payload(
                    step,
                    RECORDER.recording_saved_paths(
                        recording, step, CAMERAS, image_shapes=shapes
                    ),
                )
                for step in recording["steps"]
            ]
            RECORDER.write_json_atomically(
                output_dir.parent / RECORDER.PARTITION_INDEX_FILENAME,
                [RECORDER.collected_episode_payload(job, recording)],
            )

            new_job, new_recording = make_job(output_dir)
            new_recording["source_task"] = dict(source_task)
            with mock.patch.object(
                RECORDER,
                "restore_recording_from_output",
                side_effect=AssertionError(
                    "trusted partition must not scan trajectory folders"
                ),
            ), mock.patch("builtins.print") as progress_print:
                summary = RECORDER.restore_jobs_from_output(
                    [new_job],
                    CAMERAS,
                    write_markers=False,
                    trusted_complete_task_directories=("BaseTasks",),
                    progress_every=1,
                )

            self.assertEqual(summary["completed_trajectories"], 1)
            self.assertEqual(summary["trusted_completed_trajectories"], 1)
            self.assertTrue(
                any(
                    "resume scan progress: 1/1" in str(call.args[0])
                    for call in progress_print.call_args_list
                    if call.args
                )
            )
            self.assertEqual(
                new_recording["_resume_source"],
                "trusted_partition_index",
            )
            self.assertEqual(
                new_recording["_resume_step_offset"],
                len(new_recording["steps"]),
            )

    def test_available_partition_index_restores_partial_without_folder_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = (
                Path(directory)
                / "LongTasks"
                / "Trainset"
                / "5001_to_5002"
            )
            job, recording = make_job(output_dir)
            source_task = {
                "file": "TASKS/LongTasks/Trainset/N_Island_0.json",
                "index": 0,
                "task_directory": "LongTasks",
                "split_directory": "Trainset",
            }
            recording["source_task"] = dict(source_task)
            shapes = {
                RECORDER.collected_camera_name(camera): [2, 3]
                for camera in CAMERAS
            }
            first_step = recording["steps"][0]
            recording["collected_steps"] = [
                RECORDER.collected_step_payload(
                    first_step,
                    RECORDER.recording_saved_paths(
                        recording,
                        first_step,
                        CAMERAS,
                        image_shapes=shapes,
                    ),
                )
            ]
            RECORDER.write_json_atomically(
                output_dir.parent / RECORDER.PARTITION_INDEX_FILENAME,
                [RECORDER.collected_episode_payload(job, recording)],
            )

            new_job, new_recording = make_job(output_dir)
            new_recording["source_task"] = dict(source_task)
            with mock.patch.object(
                RECORDER,
                "restore_recording_from_output",
                side_effect=AssertionError(
                    "available partition index must skip folder scan"
                ),
            ):
                summary = RECORDER.restore_jobs_from_output(
                    [new_job],
                    CAMERAS,
                    write_markers=False,
                    use_partition_indexes=True,
                )

            self.assertEqual(summary["partial_trajectories"], 1)
            self.assertEqual(summary["indexed_restored_trajectories"], 1)
            self.assertEqual(new_recording["_resume_step_offset"], 1)
            self.assertEqual(
                new_recording["_resume_source"],
                "partition_index_partial",
            )

    def test_missing_folder_is_new_without_per_trajectory_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = (
                Path(directory)
                / "LongTasks"
                / "Testset"
                / "5001_to_5002"
            )
            job, recording = make_job(output_dir)
            recording["source_task"] = {
                "file": "TASKS/LongTasks/Testset/N_Island_0.json",
                "index": 0,
                "task_directory": "LongTasks",
                "split_directory": "Testset",
            }
            with mock.patch.object(
                RECORDER,
                "restore_recording_from_output",
                side_effect=AssertionError(
                    "missing folder should be classified from one directory listing"
                ),
            ):
                summary = RECORDER.restore_jobs_from_output(
                    [job],
                    CAMERAS,
                    write_markers=False,
                    use_partition_indexes=True,
                )

            self.assertEqual(summary["new_trajectories"], 1)
            self.assertEqual(recording["_resume_source"], "new")

    def test_existing_folder_missing_from_index_uses_conservative_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = (
                Path(directory)
                / "StandardTasks"
                / "Trainset"
                / "5001_to_5002"
            )
            job, recording = make_job(output_dir)
            recording["source_task"] = {
                "file": "TASKS/StandardTasks/Trainset/N_Island_0.json",
                "index": 0,
                "task_directory": "StandardTasks",
                "split_directory": "Trainset",
            }
            write_frame(recording, recording["steps"][0])

            with mock.patch.object(
                RECORDER,
                "restore_recording_from_output",
                wraps=RECORDER.restore_recording_from_output,
            ) as folder_scan:
                summary = RECORDER.restore_jobs_from_output(
                    [job],
                    CAMERAS,
                    write_markers=False,
                    use_partition_indexes=True,
                )

            folder_scan.assert_called_once()
            self.assertEqual(summary["partial_trajectories"], 1)
            self.assertEqual(recording["_resume_step_offset"], 1)

    def test_remaining_steps_are_rebalanced_across_current_scenes(self):
        with tempfile.TemporaryDirectory() as directory:
            first_job, template = make_job(Path(directory) / "trajectory_0")
            second_job = {
                "scene_index": 1,
                "scene": "N_Island_0",
                "gpu_id": 1,
                "recordings": [],
            }
            first_job["recordings"] = []
            for task_order, offset in enumerate((0, 1, 2, 3)):
                recording = copy.deepcopy(template)
                recording["task_order"] = task_order
                recording["trajectory_id"] = "trajectory_{}".format(task_order)
                recording["_resume_step_offset"] = offset
                first_job["recordings"].append(recording)

            jobs = [first_job, second_job]
            RECORDER.rebalance_resumed_jobs(jobs)

            loads = [
                sum(
                    RECORDER.recording_remaining_steps(recording)
                    for recording in job["recordings"]
                )
                for job in jobs
            ]
            self.assertEqual(sorted(loads), [3, 3])
            self.assertEqual(
                sum(len(job["recordings"]) for job in jobs),
                4,
            )

    def test_image_timeout_is_retried(self):
        class RetryClient:
            def __init__(self):
                self.calls = 0

            def getSceneImageResponses(self, **kwargs):
                del kwargs
                self.calls += 1
                if self.calls < 3:
                    raise RuntimeError("captured_images timeout")
                return ([b"rgb"] * 4, [np.ones((2, 3))] * 4)

        client = RetryClient()
        with mock.patch.object(RECORDER.time, "sleep"):
            images = RECORDER.capture_scene_images(
                client,
                scene_index=0,
                cameras=CAMERAS,
                retries=3,
                rgb_mode="png",
            )

        self.assertEqual(client.calls, 3)
        self.assertEqual(len(images[0]), 4)


if __name__ == "__main__":
    unittest.main()
