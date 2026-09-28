#!/usr/bin/env python
"""Record multiple A* trajectories across multiple ProjectAirSim scenes."""

import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty, Queue
import re
import shutil
import sys
import threading
import time
import traceback

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .astar_trajectory_record import (  # noqa: E402
    DEFAULT_PLUGIN_CONFIG_DIR,
    DEFAULT_SCENE_CONFIG,
    DEFAULT_TRAJECTORY_FILE,
    make_step_set_pose,
    normalize_recording_trajectory,
    select_steps,
    wxyz_to_xyzw,
)
from .runtime_config import (  # noqa: E402
    DEFAULT_JOB_FILE,
    load_jobs_config,
    recording_settings,
    server_settings,
)


DEFAULT_CAMERAS = (
    "FrontCamera",
    "LeftCamera",
    "RightCamera",
    "DownCamera",
)

TRAJECTORY_ID_PATTERN = re.compile(
    rb'"trajectory_id"\s*:\s*"([^"]+)"'
)
TASK_ID_PATTERN = re.compile(rb'"task_id"\s*:\s*"([^"]+)"')
START_OBJECT_ID_PATTERN = re.compile(
    rb'"start_object_id"\s*:\s*(?:"([^"]+)"|(-?\d+))'
)
TASK_OBJECT_MARKER_PATTERN = re.compile(r"(?<!\d)(\d{4,})(?!\d)")

SEMANTIC_TASK_DIRECTORIES = {
    "base_task": "BaseTasks",
    "standard_task": "StandardTasks",
    "long_task": "LongTasks",
}
SEMANTIC_TASK_TYPES = {
    directory: task_type
    for task_type, directory in SEMANTIC_TASK_DIRECTORIES.items()
}
SEMANTIC_SPLIT_DIRECTORIES = {
    1: "Trainset",
    0: "Testset",
}
SEMANTIC_SPLIT_NAMES = {
    "Trainset": "train",
    "Testset": "test",
}
PARTITION_INDEX_FILENAME = "collected_episodes.json"
COMPLETION_MARKER_FILENAME = ".recording_complete.json"
COMPLETION_MARKER_VERSION = 1

TIMING_OUTPUT_LOCK = threading.Lock()
ANSI_RESET = "\033[0m"
ANSI_DIM = "\033[2m"
ANSI_BOLD = "\033[1m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"
ANSI_CYAN = "\033[36m"
ANSI_MAGENTA = "\033[35m"
ANSI_BLUE = "\033[34m"
SCENE_COLORS = ("\033[96m", "\033[95m", "\033[94m", "\033[93m")


class AsyncNasFrameWriter:
    """把本地暂存帧按顺序异步复制到最终 NAS 目录。"""

    def __init__(self, local_stage_root, output_root, queue_frames, scene_index):
        self.output_root = Path(output_root).expanduser().absolute()
        namespace = hashlib.sha256(
            str(self.output_root).encode("utf-8")
        ).hexdigest()[:12]
        self.stage_output_root = (
            Path(local_stage_root).expanduser().absolute()
            / "output_{}".format(namespace)
        )
        self.stage_output_root.mkdir(parents=True, exist_ok=True)
        self.scene_index = int(scene_index)
        self.queue_frames = max(1, int(queue_frames))
        self._queue = Queue(maxsize=self.queue_frames)
        self._fatal_error = None
        self._fatal_lock = threading.Lock()
        self._prepared_directories = set()
        self._temporary_counter = 0
        self._thread = threading.Thread(
            target=self._run,
            name="nas-writer-scene-{}".format(self.scene_index),
            daemon=False,
        )
        self._thread.start()

    def staging_path(self, destination):
        destination = Path(destination).expanduser().absolute()
        try:
            relative_path = destination.relative_to(self.output_root)
        except ValueError as error:
            raise RuntimeError(
                "NAS 目标路径不在 output_root 中: {}".format(destination)
            ) from error
        return self.stage_output_root / relative_path

    def _current_error(self):
        with self._fatal_lock:
            return self._fatal_error

    def _set_error(self, error):
        with self._fatal_lock:
            if self._fatal_error is None:
                self._fatal_error = error

    def submit(self, file_pairs, recording, committed_count):
        error = self._current_error()
        if error is not None:
            raise RuntimeError(
                "场景 {} 异步写入 NAS 已失败: {}".format(
                    self.scene_index, error
                )
            ) from error
        self._queue.put(
            {
                "file_pairs": tuple(
                    (Path(source), Path(destination))
                    for source, destination in file_pairs
                ),
                "recording": recording,
                "committed_count": int(committed_count),
            }
        )
        error = self._current_error()
        if error is not None:
            raise RuntimeError(
                "场景 {} 异步写入 NAS 已失败: {}".format(
                    self.scene_index, error
                )
            ) from error

    def buffered_frames(self):
        return int(self._queue.qsize())

    def _prepare_destination_directory(self, directory):
        directory_key = str(directory)
        if directory_key in self._prepared_directories:
            return
        directory.mkdir(parents=True, exist_ok=True)
        self._prepared_directories.add(directory_key)

    def _copy_one_file(self, source, destination):
        self._prepare_destination_directory(destination.parent)
        self._temporary_counter += 1
        temporary_path = destination.with_name(
            ".{}.aerialcopy.{}.{}.tmp".format(
                destination.name,
                os.getpid(),
                self._temporary_counter,
            )
        )
        try:
            shutil.copyfile(source, temporary_path)
            os.replace(temporary_path, destination)
        except BaseException:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _clean_local_files(self, file_pairs):
        for source, _destination in file_pairs:
            try:
                source.unlink()
            except FileNotFoundError:
                pass

        # Do not remove staging directories here.  The capture thread can
        # already be writing the next frame in the same trajectory while this
        # writer thread cleans the previous frame.  Removing a momentarily
        # empty ``depth`` directory races with the capture thread's next
        # ``np.save`` and can cause FileNotFoundError.  Empty staging
        # directories are harmless and keeping them preserves the asynchronous
        # producer/consumer separation.

    def _copy_frame(self, task):
        file_pairs = task["file_pairs"]
        for source, destination in file_pairs:
            self._copy_one_file(source, destination)
        self._clean_local_files(file_pairs)
        task["recording"]["_nas_committed_count"] = int(
            task["committed_count"]
        )

    def _run(self):
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                if self._current_error() is None:
                    try:
                        self._copy_frame(task)
                    except BaseException as error:
                        self._set_error(error)
            finally:
                self._queue.task_done()

    def flush(self):
        self._queue.join()
        error = self._current_error()
        if error is not None:
            raise RuntimeError(
                "场景 {} 异步写入 NAS 失败，本地暂存文件已保留: {}".format(
                    self.scene_index, error
                )
            ) from error

    def close(self):
        self._queue.put(None)
        self._thread.join()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Open the configured ProjectAirSim scene instances and record "
            "their assigned trajectories in parallel."
        )
    )
    parser.add_argument("--server_host", default=None)
    parser.add_argument("--server_port", type=int, default=None)
    parser.add_argument(
        "--trajectory_ids",
        nargs="+",
        default=None,
        help="One trajectory id per scene, for example 0_0__10.",
    )
    parser.add_argument(
        "--job_file",
        default=str(DEFAULT_JOB_FILE),
        help=(
            "YAML recording plan. It can either list trajectories explicitly "
            "or load SemanticOGS tasks through a top-level task_source block."
        ),
    )
    parser.add_argument(
        "--map_name",
        default="",
        help=(
            "Record only this map from the YAML task_source. The value also "
            "expands {map_name} placeholders in YAML paths and scene names."
        ),
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=None,
        help=(
            "GPU pool override. Accepts '--gpus 0 1' or '--gpus 0,1'. "
            "In YAML task_source mode, one parallel scene is created per GPU."
        ),
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["FantasyCity"],
        help=(
            "One scene name to reuse for every trajectory, or one scene name "
            "per trajectory."
        ),
    )
    parser.add_argument(
        "--trajectory_file",
        default=str(DEFAULT_TRAJECTORY_FILE),
    )
    parser.add_argument(
        "--sim_config_path",
        default=str(DEFAULT_PLUGIN_CONFIG_DIR),
    )
    parser.add_argument(
        "--scene_config_name",
        default=DEFAULT_SCENE_CONFIG,
    )
    parser.add_argument("--dataset_root", default=None, help="Published dataset root; defaults to the project parent.")
    parser.add_argument("--splits", nargs="+", choices=("IID_TRAINS", "IID_TESTS", "OOD_TRAINS", "OOD_TESTS"))
    parser.add_argument("--tasks", nargs="+", choices=("base", "standard", "long"))
    parser.add_argument("--episode_ids", nargs="+", help="Episode ids within the selected partitions.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum episodes across selected partitions; 0 selects all.")
    parser.add_argument("--connect_timeout", type=int, default=300)
    parser.add_argument(
        "--start_step",
        type=int,
        default=None,
        help="First step used for every trajectory (default: 0).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum steps per trajectory; 0 records all remaining steps.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(DEFAULT_CAMERAS),
    )
    parser.add_argument(
        "--output_root",
        default=None,
    )
    parser.add_argument(
        "--local_stage_root",
        default="",
        help=(
            "Write frames to this local directory first, then copy them "
            "to output_root in a background thread. Empty disables staging."
        ),
    )
    parser.add_argument(
        "--async_nas_queue_frames",
        type=int,
        default=32,
        help=(
            "Maximum locally staged frames waiting for NAS per scene; "
            "the capture worker applies backpressure when the queue is full."
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Seconds to wait between setting poses and requesting images.",
    )
    parser.add_argument(
        "--image_retries",
        type=int,
        default=2,
        help="Retries when a scene fails to return images.",
    )
    parser.add_argument(
        "--manifest_interval",
        type=float,
        default=30.0,
        help=(
            "Minimum seconds between progress manifest writes; the manifest "
            "is always written at startup and shutdown."
        ),
    )
    parser.add_argument(
        "--timing_every",
        type=int,
        default=0,
        help=(
            "Print one timing report every N steps per scene; 0 disables "
            "detailed timing."
        ),
    )
    parser.add_argument(
        "--timing_format",
        choices=("pretty", "json", "both"),
        default="pretty",
        help="Timing output format; pretty is the default terminal view.",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Use ANSI colors for pretty timing output.",
    )
    parser.add_argument(
        "--keep_scenes",
        action="store_true",
        help="Disconnect the client but leave all UE processes running.",
    )
    parser.add_argument(
        "--force_reopen_scenes",
        action="store_true",
        help="Restart UE even when the server has matching live scene slots.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing trajectory images in the output directories.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Scan trajectory folders, skip completed recordings, and "
            "continue partial recordings from their first incomplete frame."
        ),
    )
    parser.add_argument(
        "--resume_trust_complete_task_directory",
        action="append",
        default=[],
        help=(
            "Trust one completed task directory, such as BaseTasks, using "
            "its partition collected_episodes.json indexes instead of "
            "scanning every trajectory folder. Repeat for multiple values."
        ),
    )
    parser.add_argument(
        "--resume_use_partition_indexes",
        action="store_true",
        help=(
            "Use available partition collected_episodes.json indexes to "
            "restore completed and partial recordings before falling back "
            "to trajectory-folder scans."
        ),
    )
    parser.add_argument(
        "--resume_progress_every",
        type=int,
        default=100,
        help=(
            "Print resume scan progress every N trajectories; 0 disables "
            "progress reports."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate trajectories and print assignments without opening UE.",
    )
    args = parser.parse_args(argv)
    args.start_step_override = args.start_step
    args.max_steps_override = args.max_steps
    args.start_step = 0 if args.start_step is None else args.start_step
    args.max_steps = 0 if args.max_steps is None else args.max_steps
    args.published_layout = False
    if args.job_file:
        payload = load_jobs_config(args.job_file)
        args.published_layout = payload.get("task_source", {}).get("layout") == "published"
        if args.published_layout and args.dataset_root:
            from .published_dataset import resolve_config
            # Reload the raw YAML so {dataset_root} templates use the override.
            import yaml
            raw = yaml.safe_load(Path(args.job_file).expanduser().read_text(encoding="utf-8"))
            payload = resolve_config(raw, args.dataset_root)
        args.recording_config = payload
        shared_server = server_settings(payload)
        shared_recording = recording_settings(payload)
        task_source = payload.get("task_source") or {}
        if not isinstance(task_source, dict):
            raise ValueError("record_jobs.yaml 'task_source' must be a mapping")
        if not args.map_name:
            args.map_name = str(task_source.get("map_name") or "")
    else:
        shared_server = {}
        shared_recording = {}
    if args.server_host is None:
        args.server_host = str(shared_server.get("host", "127.0.0.1"))
    if args.server_port is None:
        args.server_port = int(shared_server.get("port", 38000))
    if args.output_root is None:
        args.output_root = str(
            shared_recording.get(
                "output_root", str(REPO_ROOT.parent / "VideoRECORD")
            )
        )
    args.output_root = str(Path(args.output_root).expanduser().absolute())
    return args


def round_robin(values, count, label):
    values = list(values)
    if not values:
        raise ValueError("{} must contain at least one value".format(label))
    return [values[index % len(values)] for index in range(int(count))]


def parse_gpu_ids(values):
    gpu_ids = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            gpu_id = int(token)
            if gpu_id < 0:
                raise ValueError("--gpus must contain non-negative integers")
            gpu_ids.append(gpu_id)
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpu_ids


def expand_scenes(scenes, count):
    scenes = [str(scene) for scene in scenes]
    if len(scenes) == 1:
        return scenes * int(count)
    if len(scenes) != int(count):
        raise ValueError(
            "--scenes must contain one value or {} values, got {}".format(
                int(count),
                len(scenes),
            )
        )
    return scenes


def safe_path_component(value):
    safe = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in str(value)
    )
    return safe or "trajectory"


def split_trajectory_ids(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        ids = []
        for item in value:
            ids.extend(split_trajectory_ids(item))
        return ids
    text = str(value).strip()
    if not text:
        return []
    for delimiter in (";", ",", "\n", "\t"):
        text = text.replace(delimiter, " ")
    return [token for token in text.split(" ") if token]


def resolve_repo_path(path_value):
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def expand_map_template(value, map_name):
    """Expand the one supported YAML placeholder without touching other text."""
    if value is None:
        return value
    return str(value).replace("{map_name}", str(map_name))


def resolve_trajectory_file(path_value):
    path = resolve_repo_path(path_value)
    if path.is_dir():
        candidates = [
            path / "trajectories" / "trajectories.jsonl",
            path / "trajectories.jsonl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return path
    return path


def load_trajectory_catalog(path_value, catalogs=None):
    """Index a legacy JSONL catalog or one current Step3 trajectory JSON."""
    path = resolve_trajectory_file(path_value)
    cache_key = str(path.resolve())
    if catalogs is not None and cache_key in catalogs:
        return catalogs[cache_key]
    if not path.exists():
        raise FileNotFoundError("trajectory file does not exist: {}".format(path))
    if path.is_dir():
        raise ValueError(
            "trajectory directory requires an exact file from task_route_index: "
            "{}".format(path)
        )

    if path.suffix.lower() == ".json":
        trajectory = normalize_recording_trajectory(
            json.loads(path.read_text(encoding="utf-8")), path
        )
        identifiers = {
            str(value)
            for value in (
                trajectory.get("trajectory_id"),
                trajectory.get("task_id"),
            )
            if value not in (None, "")
        }
        if not identifiers:
            raise ValueError(
                "trajectory_id or task_id missing in {}".format(path)
            )
        entry = {
            "trajectory_id": sorted(identifiers)[0],
            "task_id": str(trajectory.get("task_id") or ""),
            "start_object_id": "",
            "data_index": 0,
            "line_number": 1,
            "offset": 0,
            "trajectory": trajectory,
        }
        catalog = {
            "path": path,
            "entries": [entry],
            "by_id": {identifier: entry for identifier in identifiers},
        }
        if catalogs is not None:
            catalogs[cache_key] = catalog
        return catalog

    entries = []
    by_id = {}
    with path.open("rb") as handle:
        data_index = -1
        line_number = 0
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                continue
            data_index += 1
            trajectory_match = TRAJECTORY_ID_PATTERN.search(line)
            task_match = TASK_ID_PATTERN.search(line)
            start_match = START_OBJECT_ID_PATTERN.search(line)
            if not trajectory_match:
                raise ValueError(
                    "trajectory_id missing at {}:{}".format(path, line_number)
                )
            trajectory_id = trajectory_match.group(1).decode("utf-8")
            task_id = (
                task_match.group(1).decode("utf-8") if task_match else ""
            )
            start_object_id = ""
            if start_match:
                start_bytes = start_match.group(1) or start_match.group(2)
                start_object_id = start_bytes.decode("utf-8")
            entry = {
                "trajectory_id": trajectory_id,
                "task_id": task_id,
                "start_object_id": start_object_id,
                "data_index": data_index,
                "line_number": line_number,
                "offset": offset,
            }
            entries.append(entry)
            by_id[trajectory_id] = entry
            if task_id:
                by_id[task_id] = entry

    catalog = {
        "path": path,
        "entries": entries,
        "by_id": by_id,
    }
    if catalogs is not None:
        catalogs[cache_key] = catalog
    return catalog


def trajectory_from_catalog(catalog, trajectory_id):
    trajectory_id = str(trajectory_id)
    try:
        entry = catalog["by_id"][trajectory_id]
    except KeyError as error:
        raise ValueError(
            "trajectory id not found in {}: {}".format(
                catalog["path"], trajectory_id
            )
        ) from error

    trajectory = entry.get("trajectory")
    if trajectory is None:
        with catalog["path"].open("rb") as handle:
            handle.seek(entry["offset"])
            line = handle.readline()
        trajectory = normalize_recording_trajectory(
            json.loads(line.decode("utf-8")),
            "{}:{}".format(catalog["path"], entry["line_number"]),
        )
        entry["trajectory"] = trajectory
    return trajectory, entry["data_index"], entry["line_number"]


def expand_trajectory_selectors(path_value, selectors, catalogs=None):
    """Expand numeric start ids and retain exact ``*_to_*`` trajectory ids."""
    catalog = load_trajectory_catalog(path_value, catalogs=catalogs)
    expanded_ids = []
    seen = set()
    for selector in split_trajectory_ids(selectors):
        selector = str(selector)
        if "_to_" in selector:
            if selector not in catalog["by_id"]:
                raise ValueError(
                    "trajectory id not found in {}: {}".format(
                        catalog["path"], selector
                    )
                )
            matches = [selector]
        else:
            matches = []
            for entry in catalog["entries"]:
                trajectory_id = entry["trajectory_id"]
                start_object_id = entry["start_object_id"]
                if (
                    str(start_object_id) == selector
                    or trajectory_id.startswith("{}_to_".format(selector))
                ):
                    matches.append(trajectory_id)
            if not matches:
                raise ValueError(
                    "no trajectories starting at {} found in {}".format(
                        selector, catalog["path"]
                    )
                )

        for trajectory_id in matches:
            if trajectory_id and trajectory_id not in seen:
                seen.add(trajectory_id)
                expanded_ids.append(trajectory_id)
    return expanded_ids


def cell_value(row, *names, default=""):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def optional_int(value, default):
    if value in (None, ""):
        return default
    return int(value)


def load_yaml_job_file(path):
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "YAML job files require PyYAML; install project requirements with "
            "'python -m pip install -r requirements.txt'"
        ) from error

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("YAML job file must contain a mapping at the top level")
    return payload


def yaml_gpu_blocks(payload):
    gpu_blocks = payload.get("gpus", payload)
    if isinstance(gpu_blocks, dict):
        return list(gpu_blocks.items())
    if isinstance(gpu_blocks, list):
        blocks = []
        for block in gpu_blocks:
            if not isinstance(block, dict):
                raise ValueError("each YAML GPU entry must be a mapping")
            gpu_id = block.get("gpu_id", block.get("gpu"))
            blocks.append((gpu_id, block))
        return blocks
    raise ValueError("YAML 'gpus' must be a mapping or list")


def yaml_scene_entries(gpu_block):
    if isinstance(gpu_block, dict) and "scenes" in gpu_block:
        scenes = gpu_block["scenes"]
    else:
        scenes = gpu_block

    if isinstance(scenes, list):
        return scenes
    if isinstance(scenes, dict):
        entries = []
        for scene_name, scene_config in scenes.items():
            if scene_config is None:
                scene_config = {}
            if not isinstance(scene_config, dict):
                scene_config = {"trajectories": scene_config}
            entry = dict(scene_config)
            entry.setdefault("scene", scene_name)
            entries.append(entry)
        return entries
    raise ValueError("YAML GPU 'scenes' must be a list or mapping")


def task_object_marker(value, field, task_path, task_index):
    """从语义任务对象字段中提取唯一的数字对象标识。"""
    if isinstance(value, bool):
        matches = []
    elif isinstance(value, int):
        matches = [str(value)]
    else:
        text = str(value).strip()
        matches = [text] if text.isdigit() else TASK_OBJECT_MARKER_PATTERN.findall(text)
    if len(matches) != 1:
        raise ValueError(
            (
                "{} task {} field {} must contain exactly one object marker, "
                "got {!r}"
            ).format(
                task_path, task_index, field, value
            )
        )
    marker = int(matches[0])
    if marker < 0:
        raise ValueError(
            "{} task {} field {} has a negative object marker: {}".format(
                task_path, task_index, field, marker
            )
        )
    return marker


def semantic_task_files(task_source, map_name=""):
    """按稳定顺序发现 SemanticOGS 任务 JSON。"""
    if not isinstance(task_source, dict):
        raise ValueError("YAML task_source must be a mapping")
    root_value = expand_map_template(
        cell_value(task_source, "root", "path", "task_root"), map_name
    )
    if not root_value:
        raise ValueError("YAML task_source requires 'root'")
    root = resolve_repo_path(root_value)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        pattern = expand_map_template(
            task_source.get("pattern") or "**/*.json", map_name
        )
        files = sorted(path for path in root.glob(pattern) if path.is_file())
    else:
        raise FileNotFoundError("task_source root does not exist: {}".format(root))
    if not files:
        raise ValueError(
            "task_source did not find any JSON files under {}".format(root)
        )
    non_json = [path for path in files if path.suffix.lower() != ".json"]
    if non_json:
        raise ValueError("task_source matched non-JSON file: {}".format(non_json[0]))
    return root, files


def semantic_task_partition(task_path, task):
    """核对任务类别和训练划分，并返回规范化的输出目录信息。"""
    task_type = str(task.get("task") or "").strip()
    task_directory = SEMANTIC_TASK_DIRECTORIES.get(task_type)
    if task_directory is None:
        raise ValueError(
            "{} has unsupported task type: {!r}".format(task_path, task_type)
        )

    used_in_train = task.get("used-in-train")
    if isinstance(used_in_train, bool):
        train_flag = int(used_in_train)
    else:
        try:
            train_flag = int(used_in_train)
        except (TypeError, ValueError):
            raise ValueError(
                "{} has invalid used-in-train value: {!r}".format(
                    task_path, used_in_train
                )
            )
    split_directory = SEMANTIC_SPLIT_DIRECTORIES.get(train_flag)
    if split_directory is None:
        raise ValueError(
            "{} used-in-train must be 0 or 1, got {!r}".format(
                task_path, used_in_train
            )
        )

    path_parts = set(task_path.parts)
    path_task_directories = path_parts.intersection(
        SEMANTIC_TASK_TYPES
    )
    if path_task_directories and task_directory not in path_task_directories:
        raise ValueError(
            "{} directory disagrees with task={!r}".format(
                task_path, task_type
            )
        )
    path_split_directories = path_parts.intersection(
        SEMANTIC_SPLIT_NAMES
    )
    if path_split_directories and split_directory not in path_split_directories:
        raise ValueError(
            "{} directory disagrees with used-in-train={!r}".format(
                task_path, used_in_train
            )
        )

    return {
        "task_type": task_type,
        "task_directory": task_directory,
        "split": SEMANTIC_SPLIT_NAMES[split_directory],
        "split_directory": split_directory,
        "used_in_train": train_flag,
    }


def recording_id(map_name, trajectory_task_id):
    return "{}__{}".format(str(map_name), str(trajectory_task_id))


def records_from_route_index(
    task_source, map_name, default_trajectory_file, catalogs=None
):
    """Bind current Step4 episodes to exact Step3 result files."""
    if not map_name:
        raise ValueError(
            "current task_route_index mode requires task_source.map_name or "
            "--map_name"
        )
    configured_root = expand_map_template(
        cell_value(task_source, "root", "path", "task_root"), map_name
    )
    if not configured_root:
        raise ValueError("YAML task_source requires 'root'")
    configured_root = resolve_repo_path(configured_root)
    route_index_value = expand_map_template(
        cell_value(task_source, "route_index", "task_route_index"), map_name
    )
    if route_index_value:
        route_index_path = resolve_repo_path(route_index_value)
    else:
        route_index_path = (
            configured_root / "manifests" /
            "{}.task_route_index.json".format(map_name)
        )
    if not route_index_path.is_file():
        raise FileNotFoundError(
            "task route index does not exist: {}".format(route_index_path)
        )
    route_index = json.loads(route_index_path.read_text(encoding="utf-8"))
    if not isinstance(route_index, dict):
        raise ValueError("task route index must be a JSON object")
    if str(route_index.get("map_name") or "") != str(map_name):
        raise ValueError(
            "task route index map_name is {!r}, expected {!r}".format(
                route_index.get("map_name"), map_name
            )
        )
    route_rows = route_index.get("records")
    if not isinstance(route_rows, list) or not route_rows:
        raise ValueError("task route index contains no records")

    trajectory_root_value = expand_map_template(
        cell_value(
            task_source,
            "trajectory_root",
            "trajectory_results",
            "trajectory_dir",
            default=default_trajectory_file,
        ),
        map_name,
    )
    trajectory_root = resolve_repo_path(trajectory_root_value)
    if not trajectory_root.is_dir():
        raise FileNotFoundError(
            "trajectory results directory does not exist: {}".format(
                trajectory_root
            )
        )

    # Step4 paths are relative to TASKS_Final_Edition. An explicit route index
    # under <task-root>/manifests is therefore the most reliable root anchor.
    task_root = (
        route_index_path.parent.parent
        if route_index_path.parent.name == "manifests"
        else configured_root
    )
    start_step = optional_int(task_source.get("start_step"), 0)
    max_steps = optional_int(task_source.get("max_steps"), 0)
    task_documents = {}
    seen_recording_ids = set()
    records = []

    for route_number, route_row in enumerate(route_rows):
        if not isinstance(route_row, dict):
            raise ValueError(
                "task route index record {} must be an object".format(route_number)
            )
        semantic_relative = str(route_row.get("semantic_task_file") or "")
        episode_value = route_row.get("episode_id")
        episode_id = "" if episode_value is None else str(episode_value)
        trajectory_task_id = str(route_row.get("trajectory_task_id") or "")
        trajectory_filename = str(route_row.get("trajectory_file") or "")
        if not all((semantic_relative, episode_id, trajectory_task_id, trajectory_filename)):
            raise ValueError(
                "task route index record {} is missing episode/trajectory fields".format(
                    route_number
                )
            )
        semantic_path = task_root / semantic_relative
        if semantic_path not in task_documents:
            if not semantic_path.is_file():
                raise FileNotFoundError(
                    "semantic task file does not exist: {}".format(semantic_path)
                )
            document = json.loads(semantic_path.read_text(encoding="utf-8"))
            if not isinstance(document, list):
                raise ValueError(
                    "SemanticOGS task file must contain a JSON list: {}".format(
                        semantic_path
                    )
                )
            task_documents[semantic_path] = document
        matches = [
            (index, task)
            for index, task in enumerate(task_documents[semantic_path])
            if isinstance(task, dict)
            and str(task.get("episode_id", index)) == episode_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "{} episode_id {!r} matched {} task rows".format(
                    semantic_path, episode_id, len(matches)
                )
            )
        task_index, task = matches[0]
        if str(task.get("map_name") or "") != str(map_name):
            raise ValueError(
                "{} task {} belongs to map {!r}, expected {!r}".format(
                    semantic_path, task_index, task.get("map_name"), map_name
                )
            )
        partition = semantic_task_partition(semantic_path, task)
        if str(route_row.get("task") or "") != partition["task_type"]:
            raise ValueError(
                "route index task type differs from {} task {}".format(
                    semantic_path, task_index
                )
            )
        if str(route_row.get("split") or "") != partition["split_directory"]:
            raise ValueError(
                "route index split differs from {} task {}".format(
                    semantic_path, task_index
                )
            )

        trajectory_path = trajectory_root / trajectory_filename
        catalog = load_trajectory_catalog(trajectory_path, catalogs=catalogs)
        trajectory, _, _ = trajectory_from_catalog(catalog, trajectory_task_id)
        if str(trajectory.get("map_name") or "") != str(map_name):
            raise ValueError(
                "trajectory {} belongs to map {!r}, expected {!r}".format(
                    trajectory_path, trajectory.get("map_name"), map_name
                )
            )
        unique_id = recording_id(map_name, trajectory_task_id)
        if unique_id in seen_recording_ids:
            raise ValueError(
                "duplicate recording id in task route index: {}".format(unique_id)
            )
        seen_recording_ids.add(unique_id)
        records.append({
            "task_order": len(records),
            "trajectory_file": str(trajectory_path),
            "trajectory_id": unique_id,
            "trajectory_lookup_id": trajectory_task_id,
            "start_step": start_step,
            "max_steps": max_steps,
            "output_parts": [
                partition["task_directory"],
                partition["split_directory"],
                str(map_name),
                trajectory_task_id,
            ],
            "source_task": {
                "file": semantic_relative,
                "index": int(task_index),
                "episode_id": episode_id,
                "map_name": str(map_name),
                "trajectory_task_id": trajectory_task_id,
                "start_object_name": str(task.get("start_object_name") or ""),
                "goal_object_name": str(task.get("goal_object_name") or ""),
                "description": str(task.get("description") or ""),
                "category": str(task.get("category") or ""),
                "task": partition["task_type"],
                "task_directory": partition["task_directory"],
                "split": partition["split"],
                "split_directory": partition["split_directory"],
                "used_in_train": partition["used_in_train"],
            },
        })
    return records, None


def records_from_semantic_tasks(
    task_source, default_trajectory_file, catalogs=None, map_name=""
):
    """把 SemanticOGS 的起终点编号转换为精确轨迹 ID。"""
    effective_map = str(map_name or task_source.get("map_name") or "")
    if task_source.get("route_index") or task_source.get("task_route_index"):
        return records_from_route_index(
            task_source,
            effective_map,
            default_trajectory_file,
            catalogs=catalogs,
        )
    root, task_files = semantic_task_files(task_source, effective_map)
    trajectory_file = cell_value(
        task_source,
        "trajectory_file",
        "trajectory_path",
        "trajectory_jsonl",
        default=default_trajectory_file,
    )
    catalog = load_trajectory_catalog(trajectory_file, catalogs=catalogs)
    expected_map = effective_map
    start_step = optional_int(task_source.get("start_step"), 0)
    max_steps = optional_int(task_source.get("max_steps"), 0)
    records = []
    seen_trajectory_ids = {}

    for task_path in task_files:
        payload = json.loads(task_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(
                "SemanticOGS task file must contain a JSON list: {}".format(
                    task_path
                )
            )
        try:
            source_file = str(task_path.relative_to(REPO_ROOT))
        except ValueError:
            source_file = str(task_path)
        for task_index, task in enumerate(payload):
            if not isinstance(task, dict):
                raise ValueError(
                    "{} task {} must be a JSON object".format(task_path, task_index)
                )
            map_name = str(task.get("map_name") or "")
            if expected_map and map_name != expected_map:
                raise ValueError(
                    "{} task {} map_name is {!r}, expected {!r}".format(
                        task_path, task_index, map_name, expected_map
                    )
                )
            start_id = task_object_marker(
                task.get("start_object_name"),
                "start_object_name",
                task_path,
                task_index,
            )
            goal_id = task_object_marker(
                task.get("goal_object_name"),
                "goal_object_name",
                task_path,
                task_index,
            )
            trajectory_id = "{}_to_{}".format(start_id, goal_id)
            partition = semantic_task_partition(task_path, task)
            if trajectory_id not in catalog["by_id"]:
                raise ValueError(
                    "{} task {} maps to missing trajectory {} in {}".format(
                        task_path, task_index, trajectory_id, catalog["path"]
                    )
                )
            if trajectory_id in seen_trajectory_ids:
                raise ValueError(
                    "SemanticOGS trajectory {} is duplicated by {} and {}".format(
                        trajectory_id,
                        seen_trajectory_ids[trajectory_id],
                        "{}:{}".format(source_file, task_index),
                    )
                )
            seen_trajectory_ids[trajectory_id] = "{}:{}".format(
                source_file, task_index
            )
            records.append(
                {
                    "task_order": len(records),
                    "trajectory_file": str(catalog["path"]),
                    "trajectory_id": trajectory_id,
                    "start_step": start_step,
                    "max_steps": max_steps,
                    "output_parts": [
                        partition["task_directory"],
                        partition["split_directory"],
                        trajectory_id,
                    ],
                    "source_task": {
                        "file": source_file,
                        "index": int(task_index),
                        "episode_id": str(task.get("episode_id", task_index)),
                        "start_object_id": int(start_id),
                        "goal_object_id": int(goal_id),
                        "task": partition["task_type"],
                        "task_directory": partition["task_directory"],
                        "split": partition["split"],
                        "split_directory": partition["split_directory"],
                        "used_in_train": partition["used_in_train"],
                    },
                }
            )
    if not records:
        raise ValueError(
            "SemanticOGS task files do not contain any tasks: {}".format(root)
        )
    return records, catalog


def task_scene_slots(payload, map_name="", gpu_override=None):
    """读取只描述执行资源、不手工列举轨迹的场景槽位。"""
    if "gpus" not in payload:
        raise ValueError("task_source mode requires a top-level 'gpus' mapping")
    slots = []
    next_scene_index = 0
    for gpu_key, gpu_block in yaml_gpu_blocks(payload):
        inherited_gpu_id = gpu_key
        if isinstance(gpu_block, dict):
            inherited_gpu_id = gpu_block.get(
                "gpu_id", gpu_block.get("gpu", inherited_gpu_id)
            )
        if inherited_gpu_id in (None, ""):
            raise ValueError("each YAML GPU block requires a GPU id")
        for scene_entry in yaml_scene_entries(gpu_block):
            if not isinstance(scene_entry, dict):
                raise ValueError("each YAML scene entry must be a mapping")
            scene_index = optional_int(
                cell_value(scene_entry, "scene_index", "scene_slot"),
                next_scene_index,
            )
            next_scene_index += 1
            scene = expand_map_template(
                cell_value(scene_entry, "scene", "scene_id", "name"), map_name
            )
            if not scene:
                raise ValueError(
                    "scene is required for scene_index {}".format(scene_index)
                )
            gpu_id = int(
                cell_value(
                    scene_entry,
                    "gpu_id",
                    "gpu",
                    default=inherited_gpu_id,
                )
            )
            slots.append(
                {
                    "scene_index": int(scene_index),
                    "scene": scene,
                    "gpu_id": gpu_id,
                }
            )
    scene_indices = [slot["scene_index"] for slot in slots]
    if len(scene_indices) != len(set(scene_indices)):
        raise ValueError("task_source scene_index values must be unique")
    validate_scene_indices(scene_indices)
    if gpu_override is not None:
        gpu_ids = parse_gpu_ids(gpu_override)
        slots = [
            {
                "scene_index": scene_index,
                "scene": slots[scene_index % len(slots)]["scene"],
                "gpu_id": gpu_id,
            }
            for scene_index, gpu_id in enumerate(gpu_ids)
        ]
    return sorted(slots, key=lambda slot: slot["scene_index"])


def assign_task_records_to_scenes(records, slots, catalog):
    """按选中帧数做确定性的最长任务优先负载均衡。"""
    if not slots:
        raise ValueError("task_source requires at least one scene slot")
    scheduled = []
    for record in records:
        record_catalog = catalog
        if record.get("trajectory_file"):
            record_path = resolve_trajectory_file(record["trajectory_file"])
            if not (
                isinstance(catalog, dict)
                and "by_id" in catalog
                and Path(catalog["path"]).resolve() == record_path.resolve()
            ):
                record_catalog = load_trajectory_catalog(
                    record_path,
                    catalogs=(
                        catalog
                        if isinstance(catalog, dict) and "by_id" not in catalog
                        else None
                    ),
                )
        trajectory, _, _ = trajectory_from_catalog(
            record_catalog,
            record.get("trajectory_lookup_id", record["trajectory_id"]),
        )
        selected_steps = select_steps(
            trajectory.get("steps", []),
            start_step=record["start_step"],
            max_steps=record["max_steps"],
        )
        if not selected_steps:
            raise ValueError(
                "trajectory {} has no selected steps".format(record["trajectory_id"])
            )
        scheduled.append((len(selected_steps), record))

    loads = {slot["scene_index"]: 0 for slot in slots}
    counts = {slot["scene_index"]: 0 for slot in slots}
    assigned = {slot["scene_index"]: [] for slot in slots}
    for step_count, record in sorted(
        scheduled,
        key=lambda item: (-item[0], item[1]["trajectory_id"]),
    ):
        slot = min(
            slots,
            key=lambda item: (
                loads[item["scene_index"]],
                counts[item["scene_index"]],
                item["scene_index"],
            ),
        )
        scene_index = slot["scene_index"]
        item = dict(record)
        item.update(slot)
        item["estimated_steps"] = int(step_count)
        assigned[scene_index].append(item)
        loads[scene_index] += int(step_count)
        counts[scene_index] += 1

    output = []
    for slot in slots:
        scene_records = sorted(
            assigned[slot["scene_index"]],
            key=lambda item: item["task_order"],
        )
        output.extend(scene_records)
    return output


def records_from_yaml_job_file(
    path,
    default_trajectory_file,
    catalogs=None,
    map_name="",
    gpu_override=None,
):
    """Read GPU -> scenes -> trajectory selectors from a YAML plan."""
    payload = load_yaml_job_file(path)
    task_source = payload.get("task_source")
    if task_source is not None:
        effective_map = str(map_name or task_source.get("map_name") or "")
        records, catalog = records_from_semantic_tasks(
            task_source,
            default_trajectory_file,
            catalogs=catalogs,
            map_name=effective_map,
        )
        return assign_task_records_to_scenes(
            records,
            task_scene_slots(payload, effective_map, gpu_override=gpu_override),
            catalogs if task_source.get("route_index") or task_source.get(
                "task_route_index"
            ) else catalog,
        )
    records = []
    next_scene_index = 0
    for gpu_key, gpu_block in yaml_gpu_blocks(payload):
        inherited_gpu_id = gpu_key
        if isinstance(gpu_block, dict):
            inherited_gpu_id = gpu_block.get(
                "gpu_id", gpu_block.get("gpu", inherited_gpu_id)
            )
        if inherited_gpu_id in (None, ""):
            raise ValueError("each YAML GPU block requires a GPU id")

        for scene_entry in yaml_scene_entries(gpu_block):
            if not isinstance(scene_entry, dict):
                raise ValueError("each YAML scene entry must be a mapping")
            scene_index = optional_int(
                cell_value(scene_entry, "scene_index", "scene_slot"),
                next_scene_index,
            )
            next_scene_index += 1
            scene = cell_value(scene_entry, "scene", "scene_id", "name")
            gpu_id = cell_value(
                scene_entry,
                "gpu_id",
                "gpu",
                default=inherited_gpu_id,
            )
            trajectory_file = cell_value(
                scene_entry,
                "trajectory_file",
                "trajectory_path",
                "trajectory_jsonl",
                "traj_file",
                default=default_trajectory_file,
            )
            selectors = scene_entry.get("trajectories", None)
            if selectors is None:
                selectors = scene_entry.get("trajectory_starts", None)
            trajectory_ids = expand_trajectory_selectors(
                trajectory_file,
                selectors,
                catalogs=catalogs,
            )
            exact_ids = split_trajectory_ids(
                scene_entry.get("trajectory_ids", None)
            )
            if exact_ids:
                trajectory_ids.extend(
                    trajectory_id
                    for trajectory_id in expand_trajectory_selectors(
                        trajectory_file,
                        exact_ids,
                        catalogs=catalogs,
                    )
                    if trajectory_id not in trajectory_ids
                )
            if not trajectory_ids:
                raise ValueError(
                    "YAML scene {} must contain 'trajectories'".format(scene)
                )

            for trajectory_id in trajectory_ids:
                records.append(
                    {
                        "scene_index": scene_index,
                        "gpu_id": gpu_id,
                        "scene": scene,
                        "trajectory_file": trajectory_file,
                        "trajectory_id": trajectory_id,
                        "output_subdir": cell_value(
                            scene_entry, "output_subdir", "output"
                        ),
                        "start_step": cell_value(scene_entry, "start_step"),
                        "max_steps": cell_value(scene_entry, "max_steps"),
                    }
                )
    return records


def records_from_job_file(
    job_file,
    default_trajectory_file,
    catalogs=None,
    map_name="",
    gpu_override=None,
):
    path = resolve_repo_path(job_file)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return records_from_yaml_job_file(
            path,
            default_trajectory_file,
            catalogs=catalogs,
            map_name=map_name,
            gpu_override=gpu_override,
        )
    raise ValueError("--job_file must end with .yaml or .yml: {}".format(path))


def pair_observations_with_next_actions(steps):
    """把每个位姿观测与从该位姿开始执行的下一步动作配对。"""
    paired_steps = []
    for index, step in enumerate(steps):
        paired_step = dict(step)
        if index + 1 < len(steps):
            next_action = str(steps[index + 1].get("action", "")).strip()
            if not next_action or next_action.lower() == "start":
                raise ValueError(
                    "trajectory step {} has no valid next action".format(index)
                )
            paired_step["action"] = next_action
        else:
            paired_step["action"] = "stop"
        paired_steps.append(paired_step)
    return paired_steps


def make_recording(
    args,
    scene_index,
    sequence_index,
    record,
    output_root,
    catalogs=None,
):
    trajectory_id = str(record["trajectory_id"])
    trajectory_lookup_id = str(
        record.get("trajectory_lookup_id", trajectory_id)
    )
    trajectory_file = resolve_trajectory_file(
        record.get("trajectory_file") or args.trajectory_file
    )
    start_step = optional_int(record.get("start_step"), args.start_step)
    max_steps = optional_int(record.get("max_steps"), args.max_steps)
    catalog = load_trajectory_catalog(trajectory_file, catalogs=catalogs)
    trajectory, data_index, line_number = trajectory_from_catalog(
        catalog, trajectory_lookup_id
    )
    paired_steps = pair_observations_with_next_actions(
        trajectory.get("steps", [])
    )
    steps = select_steps(
        paired_steps,
        start_step=start_step,
        max_steps=max_steps,
    )
    if steps:
        # 不论是否由 max_steps 截断，数据集最后一张图都表示停止。
        steps[-1] = dict(steps[-1])
        steps[-1]["action"] = "stop"
    output_parts = record.get("output_parts")
    output_subdir = record.get("output_subdir")
    if output_parts:
        output_dir = output_root.joinpath(
            *(safe_path_component(part) for part in output_parts)
        )
    elif output_subdir:
        output_dir = output_root / safe_path_component(output_subdir)
    else:
        output_dir = output_root / safe_path_component(trajectory_id)
    return {
        "task_order": int(record.get("task_order", sequence_index)),
        "sequence_index": int(sequence_index),
        "trajectory_id": trajectory_id,
        "trajectory_lookup_id": trajectory_lookup_id,
        "trajectory_file": str(trajectory_file),
        "trajectory": trajectory,
        "data_index": int(data_index),
        "line_number": int(line_number),
        "start_step": int(start_step),
        "max_steps": int(max_steps),
        "source_task": dict(record.get("source_task") or {}),
        "estimated_steps": int(record.get("estimated_steps", len(steps))),
        "steps": steps,
        "output_dir": output_dir,
        "saved_steps": 0,
        "saved_rgb_images": 0,
        "saved_depth_images": 0,
        "collected_steps": [],
        "collected_path": output_dir / "trajectory_{}_collected.jsonl".format(
            safe_path_component(trajectory_id)
        ),
    }


def validate_scene_indices(scene_indices):
    sorted_indices = sorted(scene_indices)
    expected_indices = list(range(len(sorted_indices)))
    if sorted_indices != expected_indices:
        raise ValueError(
            "scene_index values must be contiguous from 0, got {}".format(
                sorted_indices
            )
        )


def jobs_from_records(args, records, catalogs=None):
    if not records:
        raise ValueError("--job_file does not contain any trajectory ids")
    output_root = Path(args.output_root)
    scene_groups = {}
    for record in records:
        scene_index = int(record["scene_index"])
        scene = str(record.get("scene") or "")
        gpu_value = record.get("gpu_id")
        trajectory_id = record.get("trajectory_id")
        if not scene:
            raise ValueError("scene is required for scene_index {}".format(scene_index))
        if gpu_value in (None, ""):
            raise ValueError("gpu_id is required for scene_index {}".format(scene_index))
        if not trajectory_id:
            raise ValueError(
                "trajectory_id is required for scene_index {}".format(scene_index)
            )
        gpu_id = int(gpu_value)
        if gpu_id < 0:
            raise ValueError("gpu_id must be non-negative: {}".format(gpu_id))
        if scene_index not in scene_groups:
            scene_groups[scene_index] = {
                "scene_index": scene_index,
                "scene": scene,
                "gpu_id": gpu_id,
                "recordings": [],
            }
        else:
            job = scene_groups[scene_index]
            if job["scene"] != scene or int(job["gpu_id"]) != gpu_id:
                raise ValueError(
                    "scene_index {} has inconsistent scene/gpu values".format(
                        scene_index
                    )
                )
        job = scene_groups[scene_index]
        job["recordings"].append(
            make_recording(
                args,
                scene_index,
                len(job["recordings"]),
                record,
                output_root,
                catalogs=catalogs,
            )
        )
    validate_scene_indices(scene_groups.keys())
    return [scene_groups[index] for index in sorted(scene_groups)]


def records_from_cli_args(args):
    if not args.trajectory_ids:
        raise ValueError("either --job_file or --trajectory_ids is required")
    trajectory_ids = [str(item) for item in args.trajectory_ids]
    scene_names = expand_scenes(args.scenes, len(trajectory_ids))
    gpu_ids = round_robin(
        parse_gpu_ids(args.gpus or ["0", "1"]),
        len(trajectory_ids),
        "--gpus",
    )
    records = []
    for scene_index, trajectory_id in enumerate(trajectory_ids):
        records.append(
            {
                "scene_index": int(scene_index),
                "scene": scene_names[scene_index],
                "gpu_id": int(gpu_ids[scene_index]),
                "trajectory_file": args.trajectory_file,
                "trajectory_id": trajectory_id,
                "start_step": args.start_step,
                "max_steps": args.max_steps,
            }
        )
    return records


def load_recording_jobs(args):
    catalogs = {}
    if getattr(args, "published_layout", False):
        from .published_dataset import records_from_published_dataset
        records = records_from_published_dataset(args.recording_config, args, catalogs)
        slots = task_scene_slots(args.recording_config, args.map_name, gpu_override=args.gpus)
        # No idle slots: scene indices must remain contiguous for endpoint lookup.
        slots = slots[:len(records)]
        for index, slot in enumerate(slots):
            slot["scene_index"] = index
        records = assign_task_records_to_scenes(records, slots, catalogs)
    elif args.job_file:
        records = records_from_job_file(
            args.job_file,
            args.trajectory_file,
            catalogs=catalogs,
            map_name=args.map_name,
            gpu_override=args.gpus,
        )
    else:
        records = records_from_cli_args(args)
    return jobs_from_records(args, records, catalogs=catalogs)


def print_jobs(jobs):
    recording_count = sum(len(job["recordings"]) for job in jobs)
    if recording_count > 100:
        print(
            "scene  gpu  map              trajectories  steps   "
            "first             last"
        )
        print(
            "-----  ---  ---------------  ------------  ------  "
            "----------------  ----------------"
        )
        for job in jobs:
            recordings = job["recordings"]
            print(
                "{:<5}  {:<3}  {:<15}  {:<12}  {:<6}  {:<16}  {}".format(
                    job["scene_index"],
                    job["gpu_id"],
                    job["scene"],
                    len(recordings),
                    sum(len(recording["steps"]) for recording in recordings),
                    recordings[0]["trajectory_id"],
                    recordings[-1]["trajectory_id"],
                )
            )
        print("total trajectories: {}".format(recording_count))
        return
    print("scene  gpu  map              seq  trajectory       steps  output")
    print("-----  ---  ---------------  ---  ---------------  -----  ------")
    for job in jobs:
        for recording in job["recordings"]:
            print(
                "{:<5}  {:<3}  {:<15}  {:<3}  {:<15}  {:<5}  {}".format(
                    job["scene_index"],
                    job["gpu_id"],
                    job["scene"],
                    recording["sequence_index"],
                    recording["trajectory_id"],
                    len(recording["steps"]),
                    recording["output_dir"],
                )
            )


def prepare_output_directories(jobs, overwrite):
    """Reject existing recordings without creating planned trajectory dirs."""
    for job in jobs:
        for recording in job["recordings"]:
            output_dir = recording["output_dir"]
            if not overwrite:
                existing_image = next(
                    output_dir.glob("step_*_camera_*.png"), None
                )
                if existing_image is not None:
                    raise FileExistsError(
                        "output already contains PNG recording: {} "
                        "(use --overwrite)".format(output_dir)
                    )


def build_machines_info(args, jobs):
    return [
        {
            "MACHINE_IP": str(args.server_host),
            "SOCKET_PORT": int(args.server_port),
            "MAX_SCENE_NUM": len(jobs),
            "open_scenes": [job["scene"] for job in jobs],
            "gpus": [job["gpu_id"] for job in jobs],
            "TAKEOFF_ON_CONNECT": False,
            "ENABLE_API_CONTROL_ON_CONNECT": False,
            "ARM_ON_CONNECT": False,
        }
    ]


def capture_images(client, cameras, retries):
    attempts = max(1, int(retries) + 1)
    for attempt in range(1, attempts + 1):
        images = client.getImageResponses(
            cameras=tuple(cameras),
            depth_mode="float32_m",
        )
        if images is not None:
            return images
        if attempt < attempts:
            print("image request failed; retry {}/{}".format(attempt, attempts - 1))
            time.sleep(0.2)
    raise RuntimeError("image request failed for one or more scenes")


def capture_scene_images(
    client,
    scene_index,
    cameras,
    retries,
    return_timings=False,
    rgb_mode="raw",
):
    """Capture one scene so slow/failing scenes do not block other workers."""
    total_started = time.perf_counter()
    retry_sleep_seconds = 0.0
    attempts = max(1, int(retries) + 1)
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            if return_timings:
                images = client.getSceneImageResponses(
                    machine_index=0,
                    scene_index=int(scene_index),
                    cameras=tuple(cameras),
                    depth_mode="float32_m",
                    return_timings=True,
                    rgb_mode=rgb_mode,
                )
            else:
                images = client.getSceneImageResponses(
                    machine_index=0,
                    scene_index=int(scene_index),
                    cameras=tuple(cameras),
                    depth_mode="float32_m",
                    rgb_mode=rgb_mode,
                )
        except Exception as error:
            last_error = error
            images = None
        if images is not None:
            if return_timings:
                rgb_images, depth_images, timings = images
                timings = dict(timings)
                timings.update(
                    {
                        "attempts": int(attempt),
                        "retry_sleep_ms": retry_sleep_seconds * 1000.0,
                        "capture_total_ms": (
                            time.perf_counter() - total_started
                        ) * 1000.0,
                    }
                )
                return (rgb_images, depth_images), timings
            return images
        if attempt < attempts:
            print(
                "scene {} image request failed; retry {}/{}{}".format(
                    scene_index,
                    attempt,
                    attempts - 1,
                    ": {}".format(last_error) if last_error else "",
                )
            )
            retry_sleep_started = time.perf_counter()
            time.sleep(0.2)
            retry_sleep_seconds += time.perf_counter() - retry_sleep_started
    failure = RuntimeError(
        "scene {} image request failed after {} attempts".format(
            scene_index, attempts
        )
    )
    if last_error is not None:
        raise failure from last_error
    raise failure


def set_direct_scene_pose(client, job, recording, round_index, pose_types):
    """Set one nonphysics Drone pose while holding only its scene lock."""
    total_started = time.perf_counter()
    Pose, Quaternion, Vector3 = pose_types
    scene_index = int(job["scene_index"])
    step = recording["steps"][int(round_index)]
    connection = client.scene_connections[0][scene_index]
    drone = client.drones[0][scene_index]
    if connection is None or drone is None:
        raise RuntimeError("scene {} drone is not connected".format(scene_index))
    pose = make_step_set_pose(step, Pose, Quaternion, Vector3)
    lock_started = time.perf_counter()
    with connection.operation_lock:
        lock_wait_ms = (time.perf_counter() - lock_started) * 1000.0
        rpc_started = time.perf_counter()
        if not drone.set_pose(pose, reset_kinematics=True):
            raise RuntimeError(
                "scene {} Drone.set_pose returned False".format(scene_index)
            )
        rpc_ms = (time.perf_counter() - rpc_started) * 1000.0
    return {
        "prepare_ms": (lock_started - total_started) * 1000.0,
        "lock_wait_ms": lock_wait_ms,
        "rpc_ms": rpc_ms,
        "total_ms": (time.perf_counter() - total_started) * 1000.0,
    }


def set_direct_scene_poses(client, active_pairs, round_index, pose_types):
    """Set nonphysics Drone poses sequentially for legacy callers."""
    errors = []
    for job, recording in active_pairs:
        scene_index = int(job["scene_index"])
        try:
            set_direct_scene_pose(
                client,
                job,
                recording,
                round_index,
                pose_types,
            )
        except Exception as error:
            errors.append("scene {}: {}".format(scene_index, error))
    if errors:
        raise RuntimeError("; ".join(errors))


def recording_frame_filename(step, camera, suffix):
    return "step_{:06d}_camera_{}.{}".format(
        int(step["step_index"]),
        safe_path_component(camera),
        str(suffix).lstrip("."),
    )


def save_job_png_images(
    job,
    recording,
    step,
    image_pair,
    cameras,
    return_timings=False,
    async_nas_writer=None,
):
    total_started = time.perf_counter()
    if image_pair is None:
        raise RuntimeError(
            "scene {} returned no images".format(job["scene_index"])
        )
    rgb_images, depth_images = image_pair
    if len(rgb_images) != len(cameras):
        raise RuntimeError(
            "scene {} returned {} RGB images for {} cameras".format(
                job["scene_index"], len(rgb_images), len(cameras)
            )
        )
    if len(depth_images) != len(cameras):
        raise RuntimeError(
            "scene {} returned {} depth images for {} cameras".format(
                job["scene_index"], len(depth_images), len(cameras)
            )
        )

    final_output_dir = Path(recording["output_dir"])
    final_depth_dir = final_output_dir / "depth"
    if async_nas_writer is None:
        write_output_dir = final_output_dir
        write_depth_dir = final_depth_dir
    else:
        write_output_dir = async_nas_writer.staging_path(final_output_dir)
        write_depth_dir = async_nas_writer.staging_path(final_depth_dir)
    write_output_dir.mkdir(parents=True, exist_ok=True)
    write_depth_dir.mkdir(parents=True, exist_ok=True)
    rgb_paths = {}
    depth_paths = {}
    staged_file_pairs = []
    image_shapes = {}
    rgb_bytes = 0
    depth_bytes = 0
    rgb_write_ms = 0.0
    depth_write_ms = 0.0
    for camera, rgb_payload, depth_image in zip(
        cameras, rgb_images, depth_images
    ):
        filename = recording_frame_filename(step, camera, "png")
        camera_key = collected_camera_name(camera)
        rgb_path = final_output_dir / filename
        rgb_write_path = write_output_dir / filename
        encoded_rgb = bytes(rgb_payload)
        rgb_write_started = time.perf_counter()
        rgb_write_path.write_bytes(encoded_rgb)
        rgb_write_ms += (
            time.perf_counter() - rgb_write_started
        ) * 1000.0
        rgb_paths[camera_key] = str(rgb_path)
        if async_nas_writer is not None:
            staged_file_pairs.append((rgb_write_path, rgb_path))
        rgb_bytes += len(encoded_rgb)

        depth_array = np.asarray(depth_image, dtype=np.float32)
        if depth_array.ndim != 2:
            raise RuntimeError(
                "camera {} depth must be a matrix, got shape={}".format(
                    camera, depth_array.shape
                )
            )
        depth_array = np.ascontiguousarray(depth_array)
        depth_path = final_depth_dir / filename.replace(".png", ".npy")
        depth_write_path = (
            write_depth_dir / filename.replace(".png", ".npy")
        )
        depth_write_started = time.perf_counter()
        np.save(depth_write_path, depth_array, allow_pickle=False)
        depth_write_ms += (
            time.perf_counter() - depth_write_started
        ) * 1000.0
        depth_paths[camera_key] = str(depth_path)
        if async_nas_writer is not None:
            staged_file_pairs.append((depth_write_path, depth_path))
        depth_bytes += int(depth_array.nbytes)
        image_shapes[camera_key] = [
            int(depth_array.shape[0]),
            int(depth_array.shape[1]),
        ]

    if async_nas_writer is not None:
        async_nas_writer.submit(
            staged_file_pairs,
            recording,
            committed_count=len(recording.get("collected_steps", [])) + 1,
        )

    timing = {
        "total_ms": (time.perf_counter() - total_started) * 1000.0,
        "rgb_write_ms": rgb_write_ms,
        "depth_write_ms": depth_write_ms,
        "rgb_bytes": int(rgb_bytes),
        "depth_bytes": int(depth_bytes),
        "buffered_frames": (
            async_nas_writer.buffered_frames()
            if async_nas_writer is not None
            else 0
        ),
    }
    result = {
        "rgb": rgb_paths,
        "depth": depth_paths,
        "image_shape": image_shapes,
        "_timing": timing,
    }
    recording["saved_steps"] += 1
    recording["saved_rgb_images"] += len(rgb_paths)
    recording["saved_depth_images"] += len(depth_paths)
    recording["_last_save_timing"] = dict(timing)
    if not return_timings:
        result.pop("_timing", None)
    return result


def collected_camera_name(camera):
    aliases = {
        "frontcamera": "front",
        "leftcamera": "left",
        "rightcamera": "right",
        "downcamera": "down",
    }
    text = str(camera)
    return aliases.get(text.lower(), safe_path_component(text).lower())


def collected_path_value(path_value):
    path = Path(path_value).resolve()
    try:
        return str(path.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def collected_image_reference(reference):
    return collected_path_value(reference)


def action_step_size(action):
    action = str(action).lower()
    if action in ("start", "stop"):
        return 0
    if action in ("rotl", "rotr"):
        return 15.0
    return 1.0


def collected_step_payload(step, saved_paths):
    quaternion_xyzw = wxyz_to_xyzw(step["quaternion_wxyz"])
    return {
        "frame": int(step["step_index"]),
        "rgb": {
            camera: collected_image_reference(reference)
            for camera, reference in saved_paths["rgb"].items()
        },
        "depth": {
            camera: collected_image_reference(reference)
            for camera, reference in saved_paths["depth"].items()
        },
        "action": str(step.get("action", "")),
        "steps_size": action_step_size(step.get("action", "")),
        "position": [float(value) for value in step["position_m"]],
        "quaternion": [float(value) for value in quaternion_xyzw],
        "image_shape": dict(saved_paths["image_shape"]),
    }


def collected_episode_payload(job, recording):
    trajectory = recording["trajectory"]
    collected_steps = recording["collected_steps"]
    planned_steps = trajectory.get("steps", [])
    source_task = dict(recording.get("source_task") or {})
    completed_selection = len(collected_steps) == len(recording["steps"])
    for collected_step, planned_step in zip(
        collected_steps, recording["steps"]
    ):
        if int(collected_step["frame"]) != int(planned_step["step_index"]):
            raise RuntimeError(
                "trajectory {} frame alignment mismatch: {} != {}".format(
                    recording["trajectory_id"],
                    collected_step["frame"],
                    planned_step["step_index"],
                )
            )
        if str(collected_step.get("action", "")) != str(
            planned_step.get("action", "")
        ):
            raise RuntimeError(
                "trajectory {} action alignment mismatch at frame {}".format(
                    recording["trajectory_id"], collected_step["frame"]
                )
            )
    if completed_selection and collected_steps:
        if str(collected_steps[-1].get("action", "")).lower() != "stop":
            raise RuntimeError(
                "trajectory {} final action is not stop".format(
                    recording["trajectory_id"]
                )
            )
        if any(
            str(item.get("action", "")).lower() == "start"
            for item in collected_steps
        ):
            raise RuntimeError(
                "trajectory {} still contains a start action".format(
                    recording["trajectory_id"]
                )
            )
    reached_goal = bool(
        completed_selection
        and collected_steps
        and planned_steps
        and int(collected_steps[-1]["frame"])
        == int(planned_steps[-1]["step_index"])
    )
    goal = trajectory.get("goal") or {}
    goal_object_name = str(
        trajectory.get("goal_object_name")
        or (goal.get("Object_name") if isinstance(goal, dict) else "")
        or source_task.get("goal_object_name")
        or ""
    )
    if not goal_object_name and trajectory.get("goal_object_id", "") != "":
        goal_object_name = "object_{}".format(trajectory["goal_object_id"])
    minimum_clearance_cm = trajectory.get("minimum_clearance_cm") or 0.0
    return {
        "episode_id": str(
            source_task.get("episode_id", recording["trajectory_id"])
        ),
        "recording_id": str(recording["trajectory_id"]),
        "trajectory_task_id": str(
            recording.get("trajectory_lookup_id", recording["trajectory_id"])
        ),
        "map_name": str(source_task.get("map_name") or job["scene"]),
        "object_name": goal_object_name,
        "description": str(source_task.get("description") or ""),
        "task": str(source_task.get("task") or ""),
        "split": str(source_task.get("split") or ""),
        "action_alignment": "observation_then_next_action",
        "source_task": source_task,
        "status": {
            "planned": True,
            "reached_goal": reached_goal,
            "collided": False,
            "n_violations": 0,
            "n_steps": sum(
                1
                for item in collected_steps
                if str(item.get("action", "")).lower()
                not in ("start", "stop")
            ),
            "clearance_m": float(minimum_clearance_cm) / 100.0,
            "planner": (
                "hybrid_astar_v6"
                if str(trajectory.get("schema_version") or "").startswith(
                    "hybrid_astar_trajectory_v"
                )
                else "astar_obstacle_v1"
            ),
        },
        "steps": list(collected_steps),
    }


def write_jsonl_atomically(path, payloads):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name("{}.tmp".format(path.name))
    with temporary_path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")
    temporary_path.replace(path)
    return path


def write_json_atomically(path, payload):
    """原子写入一个标准 JSON 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name("{}.tmp".format(path.name))
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
    temporary_path.replace(path)
    return path


def write_collected_episode(job, recording):
    return write_jsonl_atomically(
        recording["collected_path"],
        [collected_episode_payload(job, recording)],
    )


def recording_saved_paths(recording, step, cameras, image_shapes=None):
    output_dir = Path(recording["output_dir"])
    rgb_paths = {}
    depth_paths = {}
    for camera in cameras:
        camera_key = collected_camera_name(camera)
        rgb_filename = recording_frame_filename(step, camera, "png")
        depth_filename = recording_frame_filename(step, camera, "npy")
        rgb_paths[camera_key] = str(output_dir / rgb_filename)
        depth_paths[camera_key] = str(output_dir / "depth" / depth_filename)
    return {
        "rgb": rgb_paths,
        "depth": depth_paths,
        "image_shape": dict(image_shapes or {}),
    }


def completion_marker_payload(recording, cameras):
    steps = recording["steps"]
    return {
        "version": COMPLETION_MARKER_VERSION,
        "trajectory_id": str(recording["trajectory_id"]),
        "step_count": len(steps),
        "first_frame": int(steps[0]["step_index"]) if steps else None,
        "last_frame": int(steps[-1]["step_index"]) if steps else None,
        "cameras": [str(camera) for camera in cameras],
        "collected_jsonl": Path(recording["collected_path"]).name,
    }


def completion_marker_path(recording):
    return Path(recording["output_dir"]) / COMPLETION_MARKER_FILENAME


def write_completion_marker(recording, cameras):
    return write_json_atomically(
        completion_marker_path(recording),
        completion_marker_payload(recording, cameras),
    )


def load_existing_collected_steps(recording):
    path = Path(recording["collected_path"])
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                steps = payload.get("steps")
                if not isinstance(steps, list):
                    raise ValueError("steps is not a list")
                return steps
    except Exception as error:
        recording.setdefault("_resume_warnings", []).append(
            "cannot read {}: {}".format(path, error)
        )
        return []
    recording.setdefault("_resume_warnings", []).append(
        "collected metadata is empty: {}".format(path)
    )
    return []


def aligned_collected_prefix(existing_steps, planned_steps):
    if existing_steps is None:
        return None
    count = 0
    for existing, planned in zip(existing_steps, planned_steps):
        try:
            frame_matches = int(existing["frame"]) == int(
                planned["step_index"]
            )
        except (KeyError, TypeError, ValueError):
            break
        action_matches = str(existing.get("action", "")) == str(
            planned.get("action", "")
        )
        if not frame_matches or not action_matches:
            break
        count += 1
    return count


def valid_completion_marker(recording, cameras, existing_steps):
    path = completion_marker_path(recording)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        recording.setdefault("_resume_warnings", []).append(
            "cannot read {}: {}".format(path, error)
        )
        return False
    expected = completion_marker_payload(recording, cameras)
    if any(payload.get(key) != value for key, value in expected.items()):
        recording.setdefault("_resume_warnings", []).append(
            "completion marker does not match current plan: {}".format(path)
        )
        return False
    return bool(
        existing_steps is not None
        and len(existing_steps) == len(recording["steps"])
        and aligned_collected_prefix(
            existing_steps, recording["steps"]
        ) == len(recording["steps"])
    )


def scan_recording_file_names(directory, suffix, minimum_size):
    names = set()
    try:
        entries = os.scandir(str(directory))
    except FileNotFoundError:
        return names
    with entries:
        for entry in entries:
            name = entry.name
            if not name.startswith("step_") or not name.endswith(suffix):
                continue
            try:
                if (
                    entry.is_file(follow_symlinks=False)
                    and entry.stat(follow_symlinks=False).st_size >= minimum_size
                ):
                    names.add(name)
            except FileNotFoundError:
                continue
    return names


def contiguous_recording_file_prefix(recording, cameras):
    output_dir = Path(recording["output_dir"])
    if not output_dir.is_dir():
        return 0
    rgb_names = scan_recording_file_names(output_dir, ".png", 8)
    depth_names = scan_recording_file_names(
        output_dir / "depth", ".npy", 129
    )
    count = 0
    for step in recording["steps"]:
        expected_rgb = {
            recording_frame_filename(step, camera, "png")
            for camera in cameras
        }
        expected_depth = {
            recording_frame_filename(step, camera, "npy")
            for camera in cameras
        }
        if not expected_rgb.issubset(rgb_names):
            break
        if not expected_depth.issubset(depth_names):
            break
        count += 1
    return count


def image_shapes_from_recording(recording, step, cameras):
    saved_paths = recording_saved_paths(recording, step, cameras)
    image_shapes = {}
    for camera_key, depth_path in saved_paths["depth"].items():
        depth_array = np.load(
            depth_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        if depth_array.ndim != 2:
            raise ValueError(
                "{} depth shape is {}".format(depth_path, depth_array.shape)
            )
        image_shapes[camera_key] = [
            int(depth_array.shape[0]),
            int(depth_array.shape[1]),
        ]
    return image_shapes


def reconstruct_collected_prefix(recording, cameras, count):
    if count <= 0:
        return []
    image_shapes = image_shapes_from_recording(
        recording, recording["steps"][0], cameras
    )
    collected_steps = []
    for step in recording["steps"][:count]:
        saved_paths = recording_saved_paths(
            recording,
            step,
            cameras,
            image_shapes=image_shapes,
        )
        collected_steps.append(collected_step_payload(step, saved_paths))
    return collected_steps


def restore_recording_from_output(job, recording, cameras, write_marker=True):
    steps = recording["steps"]
    existing_steps = load_existing_collected_steps(recording)
    if valid_completion_marker(recording, cameras, existing_steps):
        complete_count = len(steps)
        collected_steps = list(existing_steps)
        resume_source = "marker"
    else:
        file_count = contiguous_recording_file_prefix(recording, cameras)
        metadata_count = aligned_collected_prefix(existing_steps, steps)
        complete_count = (
            file_count
            if metadata_count is None
            else min(file_count, metadata_count)
        )
        if complete_count and metadata_count is not None:
            collected_steps = list(existing_steps[:complete_count])
            resume_source = "metadata+folder"
        else:
            try:
                collected_steps = reconstruct_collected_prefix(
                    recording, cameras, complete_count
                )
                resume_source = "folder"
            except Exception as error:
                recording.setdefault("_resume_warnings", []).append(
                    "cannot validate existing depth data: {}".format(error)
                )
                complete_count = 0
                collected_steps = []
                resume_source = "restart"

    recording["collected_steps"] = collected_steps
    recording["saved_steps"] = int(complete_count)
    recording["saved_rgb_images"] = int(complete_count * len(cameras))
    recording["saved_depth_images"] = int(complete_count * len(cameras))
    recording["_resume_step_offset"] = int(complete_count)
    recording["_resume_complete"] = bool(complete_count == len(steps))
    recording["_resume_source"] = resume_source
    if recording["_resume_complete"] and write_marker:
        write_collected_episode(job, recording)
        write_completion_marker(recording, cameras)
    return complete_count


def recording_remaining_steps(recording):
    return max(
        0,
        len(recording["steps"])
        - int(recording.get("_resume_step_offset", 0)),
    )


def read_partition_index(index_path):
    """读取并校验一个分区索引，返回按轨迹编号组织的条目。"""
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("分区索引必须是 JSON 数组")
    by_episode = {}
    for item in payload:
        if not isinstance(item, dict) or "episode_id" not in item:
            raise ValueError("分区索引含有无效条目")
        lookup_id = str(item.get("recording_id") or item["episode_id"])
        if lookup_id in by_episode:
            raise ValueError(
                "分区索引存在重复 recording_id={}".format(lookup_id)
            )
        by_episode[lookup_id] = item
    return by_episode


def load_trusted_completion_indexes(jobs, task_directories):
    """读取可信分区汇总索引，不访问分区中的单条轨迹目录。"""
    trusted = {
        str(value).strip()
        for value in (task_directories or [])
        if str(value).strip()
    }
    partition_directories = set()
    for job in jobs:
        for recording in job["recordings"]:
            source_task = recording.get("source_task") or {}
            if str(source_task.get("task_directory") or "") in trusted:
                partition_directories.add(Path(recording["output_dir"]).parent)

    indexes = {}
    for partition_directory in sorted(
        partition_directories, key=lambda item: str(item)
    ):
        index_path = partition_directory / PARTITION_INDEX_FILENAME
        try:
            by_episode = read_partition_index(index_path)
        except Exception as error:
            raise RuntimeError(
                "无法读取可信完成分区索引 {}: {}。如需逐目录检查，请移除 "
                "--resume_trust_complete_task_directory。".format(
                    index_path, error
                )
            ) from error
        indexes[str(partition_directory)] = {
            "path": index_path,
            "episodes": by_episode,
        }
    return trusted, indexes


def load_available_partition_indexes(jobs, existing_indexes=None):
    """尽量读取分区索引，并一次性缓存分区内已有的轨迹目录名。"""
    partition_directories = {
        Path(recording["output_dir"]).parent
        for job in jobs
        for recording in job["recordings"]
    }
    indexes = dict(existing_indexes or {})
    directory_names = {}
    warnings = []
    for partition_directory in sorted(
        partition_directories, key=lambda item: str(item)
    ):
        partition_key = str(partition_directory)
        # 可信完成分区后续只走严格索引校验，不需要再枚举其轨迹目录。
        if partition_key in (existing_indexes or {}):
            directory_names[partition_key] = None
            continue
        if partition_key not in indexes:
            index_path = partition_directory / PARTITION_INDEX_FILENAME
            try:
                by_episode = read_partition_index(index_path)
            except FileNotFoundError:
                pass
            except Exception as error:
                warnings.append(
                    "无法使用分区索引 {}: {}，将检查已有轨迹目录".format(
                        index_path, error
                    )
                )
            else:
                indexes[partition_key] = {
                    "path": index_path,
                    "episodes": by_episode,
                }

        try:
            entries = os.scandir(str(partition_directory))
        except FileNotFoundError:
            directory_names[partition_key] = set()
        except OSError as error:
            directory_names[partition_key] = None
            warnings.append(
                "无法列出分区目录 {}: {}，缺失索引的轨迹将逐条检查".format(
                    partition_directory, error
                )
            )
        else:
            with entries:
                directory_names[partition_key] = {
                    entry.name
                    for entry in entries
                    if entry.is_dir(follow_symlinks=False)
                }
    return indexes, directory_names, warnings


def set_recording_resume_state(recording, cameras, collected_steps, source):
    """统一写入断点恢复状态，避免索引恢复和目录恢复产生差异。"""
    complete_count = len(collected_steps)
    step_count = len(recording["steps"])
    recording["collected_steps"] = list(collected_steps)
    recording["saved_steps"] = complete_count
    recording["saved_rgb_images"] = complete_count * len(cameras)
    recording["saved_depth_images"] = complete_count * len(cameras)
    recording["_resume_step_offset"] = complete_count
    recording["_resume_complete"] = bool(complete_count == step_count)
    recording["_resume_source"] = source
    return complete_count


def restore_recording_from_partition_index(
    recording, cameras, partition_indexes
):
    """从可用分区索引恢复完整或部分轨迹；不可信时返回 None。"""
    partition_directory = Path(recording["output_dir"]).parent
    index = partition_indexes.get(str(partition_directory))
    if index is None:
        return None
    trajectory_id = str(recording["trajectory_id"])
    payload = index["episodes"].get(trajectory_id)
    if payload is None:
        return None

    source_task = payload.get("source_task") or {}
    expected_source_task = recording.get("source_task") or {}
    for key in ("file", "index", "task_directory", "split_directory"):
        if str(source_task.get(key)) != str(expected_source_task.get(key)):
            recording.setdefault("_resume_warnings", []).append(
                "分区索引中的轨迹 {} 来源字段 {} 不匹配，将检查轨迹目录: {}".format(
                    trajectory_id, key, index["path"]
                )
            )
            return None

    existing_steps = payload.get("steps")
    planned_steps = recording["steps"]
    aligned_count = (
        aligned_collected_prefix(existing_steps, planned_steps)
        if isinstance(existing_steps, list)
        else None
    )
    if (
        not isinstance(existing_steps, list)
        or not existing_steps
        or len(existing_steps) > len(planned_steps)
        or aligned_count != len(existing_steps)
    ):
        recording.setdefault("_resume_warnings", []).append(
            "分区索引中的轨迹 {} 帧或动作未连续对齐，将检查轨迹目录: {}".format(
                trajectory_id, index["path"]
            )
        )
        return None

    reached_goal = bool((payload.get("status") or {}).get("reached_goal"))
    is_complete = len(existing_steps) == len(planned_steps)
    if reached_goal != is_complete:
        recording.setdefault("_resume_warnings", []).append(
            "分区索引中的轨迹 {} 完成状态与帧数不一致，将检查轨迹目录: {}".format(
                trajectory_id, index["path"]
            )
        )
        return None

    source = (
        "partition_index_complete"
        if is_complete
        else "partition_index_partial"
    )
    return set_recording_resume_state(
        recording, cameras, existing_steps, source
    )


def initialize_new_recording_resume(recording, cameras):
    """确认轨迹目录不存在时，将其初始化为尚未录制。"""
    return set_recording_resume_state(recording, cameras, [], "new")


def restore_recording_from_trusted_index(
    job, recording, cameras, trusted_indexes
):
    """从单个分区汇总索引恢复完整轨迹，不扫描 PNG/NPY 文件。"""
    partition_directory = Path(recording["output_dir"]).parent
    index = trusted_indexes.get(str(partition_directory))
    if index is None:
        raise RuntimeError(
            "可信完成分区缺少汇总索引: {}".format(partition_directory)
        )
    trajectory_id = str(recording["trajectory_id"])
    payload = index["episodes"].get(trajectory_id)
    if payload is None:
        raise RuntimeError(
            "可信完成分区索引缺少轨迹 {}: {}".format(
                trajectory_id, index["path"]
            )
        )
    if not bool((payload.get("status") or {}).get("reached_goal")):
        raise RuntimeError(
            "可信完成分区中的轨迹未标记完成 {}: {}".format(
                trajectory_id, index["path"]
            )
        )

    source_task = payload.get("source_task") or {}
    expected_source_task = recording.get("source_task") or {}
    for key in ("file", "index", "task_directory", "split_directory"):
        if str(source_task.get(key)) != str(expected_source_task.get(key)):
            raise RuntimeError(
                "可信完成分区中的轨迹 {} 来源字段 {} 不匹配: {}".format(
                    trajectory_id, key, index["path"]
                )
            )

    existing_steps = payload.get("steps")
    planned_steps = recording["steps"]
    if (
        not isinstance(existing_steps, list)
        or len(existing_steps) != len(planned_steps)
        or aligned_collected_prefix(existing_steps, planned_steps)
        != len(planned_steps)
    ):
        raise RuntimeError(
            "可信完成分区中的轨迹 {} 帧或动作未完整对齐: {}".format(
                trajectory_id, index["path"]
            )
        )

    complete_count = len(planned_steps)
    recording["collected_steps"] = list(existing_steps)
    recording["saved_steps"] = complete_count
    recording["saved_rgb_images"] = complete_count * len(cameras)
    recording["saved_depth_images"] = complete_count * len(cameras)
    recording["_resume_step_offset"] = complete_count
    recording["_resume_complete"] = True
    recording["_resume_source"] = "trusted_partition_index"
    return complete_count


def rebalance_resumed_jobs(jobs):
    """按剩余帧重新分配，允许更换服务器上的 GPU/场景数量。"""
    jobs_by_scene = {}
    for job in jobs:
        jobs_by_scene.setdefault(str(job["scene"]), []).append(job)
    for scene_jobs in jobs_by_scene.values():
        recordings = [
            recording
            for job in scene_jobs
            for recording in job["recordings"]
        ]
        assigned = {int(job["scene_index"]): [] for job in scene_jobs}
        loads = {int(job["scene_index"]): 0 for job in scene_jobs}
        counts = {int(job["scene_index"]): 0 for job in scene_jobs}
        pending = [item for item in recordings if recording_remaining_steps(item)]
        completed = [
            item for item in recordings if not recording_remaining_steps(item)
        ]
        for recording in sorted(
            pending,
            key=lambda item: (
                -recording_remaining_steps(item),
                str(item["trajectory_id"]),
            ),
        ):
            target = min(
                scene_jobs,
                key=lambda item: (
                    loads[int(item["scene_index"])],
                    counts[int(item["scene_index"])],
                    int(item["scene_index"]),
                ),
            )
            scene_index = int(target["scene_index"])
            assigned[scene_index].append(recording)
            loads[scene_index] += recording_remaining_steps(recording)
            counts[scene_index] += 1
        for recording in sorted(
            completed, key=lambda item: int(item.get("task_order", 0))
        ):
            target = min(
                scene_jobs,
                key=lambda item: (
                    counts[int(item["scene_index"])],
                    int(item["scene_index"]),
                ),
            )
            scene_index = int(target["scene_index"])
            assigned[scene_index].append(recording)
            counts[scene_index] += 1
        for job in scene_jobs:
            scene_index = int(job["scene_index"])
            job["recordings"] = sorted(
                assigned[scene_index],
                key=lambda item: int(item.get("task_order", 0)),
            )
            for sequence_index, recording in enumerate(job["recordings"]):
                recording["sequence_index"] = int(sequence_index)


def restore_jobs_from_output(
    jobs,
    cameras,
    write_markers=True,
    trusted_complete_task_directories=(),
    use_partition_indexes=False,
    progress_every=0,
):
    scan_started = time.monotonic()
    trusted, trusted_indexes = load_trusted_completion_indexes(
        jobs, trusted_complete_task_directories
    )
    partition_indexes = dict(trusted_indexes)
    partition_directory_names = {}
    partition_warnings = []
    if use_partition_indexes:
        (
            partition_indexes,
            partition_directory_names,
            partition_warnings,
        ) = load_available_partition_indexes(jobs, trusted_indexes)
    summary = {
        "total_trajectories": 0,
        "completed_trajectories": 0,
        "trusted_completed_trajectories": 0,
        "indexed_restored_trajectories": 0,
        "partial_trajectories": 0,
        "new_trajectories": 0,
        "remaining_trajectories": 0,
        "remaining_steps": 0,
        "warnings": list(partition_warnings),
    }
    total_trajectories = sum(len(job["recordings"]) for job in jobs)
    processed_trajectories = 0
    progress_every = max(0, int(progress_every or 0))
    if progress_every:
        print(
            "resume scan progress: 0/{} (0.0%) trusted_indexes={} "
            "resume_indexes={} "
            "elapsed={:.1f}s".format(
                total_trajectories,
                len(trusted_indexes),
                len(partition_indexes),
                time.monotonic() - scan_started,
            ),
            flush=True,
        )
    for job in jobs:
        for recording in job["recordings"]:
            source_task = recording.get("source_task") or {}
            task_directory = str(source_task.get("task_directory") or "")
            if task_directory in trusted:
                complete_count = restore_recording_from_trusted_index(
                    job, recording, cameras, trusted_indexes
                )
                summary["trusted_completed_trajectories"] += 1
            else:
                complete_count = None
                if use_partition_indexes:
                    complete_count = restore_recording_from_partition_index(
                        recording, cameras, partition_indexes
                    )
                    if complete_count is not None:
                        summary["indexed_restored_trajectories"] += 1
                    else:
                        partition_key = str(
                            Path(recording["output_dir"]).parent
                        )
                        existing_names = partition_directory_names.get(
                            partition_key
                        )
                        if (
                            existing_names is not None
                            and Path(recording["output_dir"]).name
                            not in existing_names
                        ):
                            complete_count = initialize_new_recording_resume(
                                recording, cameras
                            )
                if complete_count is None:
                    complete_count = restore_recording_from_output(
                        job,
                        recording,
                        cameras,
                        write_marker=write_markers,
                    )
            step_count = len(recording["steps"])
            summary["total_trajectories"] += 1
            if complete_count == step_count:
                summary["completed_trajectories"] += 1
            elif complete_count:
                summary["partial_trajectories"] += 1
            else:
                summary["new_trajectories"] += 1
            remaining = step_count - complete_count
            if remaining:
                summary["remaining_trajectories"] += 1
                summary["remaining_steps"] += remaining
            summary["warnings"].extend(
                recording.get("_resume_warnings", [])
            )
            processed_trajectories += 1
            if progress_every and (
                processed_trajectories % progress_every == 0
                or processed_trajectories == total_trajectories
            ):
                task_directory = str(
                    source_task.get("task_directory") or "unknown"
                )
                split_directory = str(
                    source_task.get("split_directory") or "unknown"
                )
                percent = (
                    100.0 * processed_trajectories / total_trajectories
                    if total_trajectories
                    else 100.0
                )
                print(
                    "resume scan progress: {}/{} ({:.1f}%) "
                    "completed={} trusted={} indexed={} partial={} new={} "
                    "remaining={} elapsed={:.1f}s last={}/{}:{}".format(
                        processed_trajectories,
                        total_trajectories,
                        percent,
                        summary["completed_trajectories"],
                        summary["trusted_completed_trajectories"],
                        summary["indexed_restored_trajectories"],
                        summary["partial_trajectories"],
                        summary["new_trajectories"],
                        summary["remaining_trajectories"],
                        time.monotonic() - scan_started,
                        task_directory,
                        split_directory,
                        recording["trajectory_id"],
                    ),
                    flush=True,
                )
    rebalance_resumed_jobs(jobs)
    return summary


def print_resume_summary(summary, jobs):
    print(
        "resume scan: completed={} partial={} new={} remaining={} "
        "remaining_steps={} trusted_complete={} indexed_restore={}".format(
            summary["completed_trajectories"],
            summary["partial_trajectories"],
            summary["new_trajectories"],
            summary["remaining_trajectories"],
            summary["remaining_steps"],
            summary.get("trusted_completed_trajectories", 0),
            summary.get("indexed_restored_trajectories", 0),
        )
    )
    for job in jobs:
        pending = [
            item for item in job["recordings"] if recording_remaining_steps(item)
        ]
        print(
            "resume scene={} gpu={} pending_trajectories={} "
            "pending_steps={}".format(
                job["scene_index"],
                job["gpu_id"],
                len(pending),
                sum(recording_remaining_steps(item) for item in pending),
            )
        )
    for warning in summary.get("warnings", [])[:20]:
        print("resume warning: {}".format(warning))
    if len(summary.get("warnings", [])) > 20:
        print(
            "resume warning: {} additional warnings omitted".format(
                len(summary["warnings"]) - 20
            )
        )


def append_collected_step(job, recording, step, saved_paths):
    del job  # Kept in the signature for compatibility with existing callers.
    payload = collected_step_payload(step, saved_paths)
    recording["collected_steps"].append(payload)
    return payload


def initialize_collected_episodes(jobs):
    for job in jobs:
        for recording in job["recordings"]:
            recording["collected_steps"] = []


def write_collected_index(args, jobs):
    if getattr(args, "published_layout", False):
        from .published_dataset import merge_episode_index
        return merge_episode_index(
            Path(args.output_root) / "collected_episodes.jsonl",
            [collected_episode_payload(job, recording) for job in jobs
             for recording in job["recordings"] if recording.get("collected_steps")],
            jsonl=True,
        )
    return write_jsonl_atomically(
        Path(args.output_root) / "collected_episodes.jsonl",
        [
            collected_episode_payload(job, recording)
            for job in jobs
            for recording in job["recordings"]
        ],
    )


def write_partition_collected_indexes(args, jobs):
    """Write one index per task/split/map recording partition."""
    grouped_recordings = {}
    for job in jobs:
        for recording in job["recordings"]:
            source_task = recording.get("source_task") or {}
            task_directory = source_task.get("task_directory")
            split_directory = source_task.get("split_directory")
            # 没有成功写入任何图像时，不创建任务轨迹目录或划分目录。
            if (
                not task_directory
                or not split_directory
                or not recording.get("collected_steps")
            ):
                continue
            partition_directory = (
                Path(args.output_root)
                / safe_path_component(task_directory)
                / safe_path_component(split_directory)
            )
            map_name = str(source_task.get("map_name") or "")
            if map_name:
                partition_directory = (
                    partition_directory / safe_path_component(map_name)
                )
            if source_task.get("output_partition_parts"):
                partition_directory = Path(args.output_root).joinpath(
                    *(safe_path_component(part) for part in source_task["output_partition_parts"])
                )
            expected_parent = Path(recording["output_dir"]).parent
            if expected_parent != partition_directory:
                raise RuntimeError(
                    "trajectory {} output is outside its task partition: {}".format(
                        recording["trajectory_id"], expected_parent
                    )
                )
            grouped_recordings.setdefault(partition_directory, []).append(
                (job, recording)
            )

    written_paths = []
    for partition_directory, items in sorted(
        grouped_recordings.items(), key=lambda item: str(item[0])
    ):
        items.sort(
            key=lambda item: (
                str(item[1].get("source_task", {}).get("file", "")),
                int(item[1].get("source_task", {}).get("index", 0)),
            )
        )
        writer = write_json_atomically
        if getattr(args, "published_layout", False):
            from .published_dataset import merge_episode_index
            writer = merge_episode_index
        written_paths.append(
            writer(
                partition_directory / PARTITION_INDEX_FILENAME,
                [
                    collected_episode_payload(job, recording)
                    for job, recording in items
                ],
            )
        )
    return written_paths


def update_recording_timing(recording, durations_ms):
    timing_lock = recording.setdefault("_timing_lock", threading.Lock())
    with timing_lock:
        stats = recording.setdefault(
            "_timing_stats",
            {"samples": 0, "total_ms": {}, "max_ms": {}},
        )
        stats["samples"] += 1
        for name, value in durations_ms.items():
            value = float(value)
            stats["total_ms"][name] = (
                stats["total_ms"].get(name, 0.0) + value
            )
            stats["max_ms"][name] = max(
                stats["max_ms"].get(name, 0.0),
                value,
            )


def recording_timing_summary(recording):
    timing_lock = recording.get("_timing_lock")
    if timing_lock is None:
        stats = dict(recording.get("_timing_stats") or {})
    else:
        with timing_lock:
            stats = copy_timing_stats(recording.get("_timing_stats") or {})
    samples = int(stats.get("samples", 0))
    if not samples:
        return {"samples": 0, "average_ms": {}, "max_ms": {}}
    return {
        "samples": samples,
        "average_ms": {
            name: round(float(value) / samples, 3)
            for name, value in sorted(stats.get("total_ms", {}).items())
        },
        "max_ms": {
            name: round(float(value), 3)
            for name, value in sorted(stats.get("max_ms", {}).items())
        },
    }


def copy_timing_stats(stats):
    return {
        "samples": int(stats.get("samples", 0)),
        "total_ms": dict(stats.get("total_ms", {})),
        "max_ms": dict(stats.get("max_ms", {})),
    }


def timing_color_enabled(mode="auto", stream=None):
    mode = str(mode or "auto").lower()
    if mode == "always":
        return True
    if mode == "never" or "NO_COLOR" in os.environ:
        return False
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def colorize(text, color, enabled, bold=False):
    text = str(text)
    if not enabled:
        return text
    prefix = (ANSI_BOLD if bold else "") + color
    return "{}{}{}".format(prefix, text, ANSI_RESET)


def human_duration_ms(value):
    value = float(value or 0.0)
    if value >= 1000.0:
        return "{:.3f} s".format(value / 1000.0)
    if value >= 100.0:
        return "{:.1f} ms".format(value)
    if value >= 10.0:
        return "{:.2f} ms".format(value)
    return "{:.3f} ms".format(value)


def colored_duration(value, enabled):
    value = float(value or 0.0)
    text = "{:>10}".format(human_duration_ms(value))
    if value >= 750.0:
        color = ANSI_RED
    elif value >= 250.0:
        color = ANSI_YELLOW
    else:
        color = ANSI_GREEN
    return colorize(text, color, enabled, bold=value >= 750.0)


def human_bytes(value):
    value = float(value or 0.0)
    if value >= 1024.0 * 1024.0:
        return "{:.2f} MiB".format(value / (1024.0 * 1024.0))
    if value >= 1024.0:
        return "{:.1f} KiB".format(value / 1024.0)
    return "{} B".format(int(value))


def timing_metric(label, value, enabled, label_color=ANSI_CYAN):
    return "{} {}".format(
        colorize(label, label_color, enabled, bold=True),
        colored_duration(value, enabled),
    )


def format_frame_timing(payload, color=False):
    ms = payload.get("ms", {})
    scene_index = int(payload.get("scene", 0))
    scene_color = SCENE_COLORS[scene_index % len(SCENE_COLORS)]
    divider = colorize("-" * 88, ANSI_DIM, color)
    frame_total = float(ms.get("frame_total", 0.0))
    stage_values = {
        "pose": float(ms.get("pose_total", 0.0)),
        "settle": float(ms.get("settle", 0.0)),
        "capture": float(ms.get("capture_total", 0.0)),
        "save": float(ms.get("save_total", 0.0)),
        "metadata": float(ms.get("metadata", 0.0)),
        "manifest": float(ms.get("progress_manifest", 0.0)),
    }
    accounted_ms = sum(stage_values.values())
    other_ms = max(0.0, frame_total - accounted_ms)
    stage_values["other"] = other_ms
    bottleneck_name, bottleneck_ms = max(
        stage_values.items(), key=lambda item: item[1]
    )
    bottleneck_percent = (
        bottleneck_ms * 100.0 / frame_total if frame_total > 0.0 else 0.0
    )
    header = colorize(
        "[FRAME] scene={}  gpu={}  seq={}  trajectory={}  step={}".format(
            scene_index,
            payload.get("gpu", "?"),
            payload.get("sequence", "?"),
            payload.get("trajectory", "?"),
            payload.get("step", "?"),
        ),
        scene_color,
        color,
        bold=True,
    )
    lines = [header, divider]
    lines.append(
        "  {} | {} | {} | {} | {}".format(
            timing_metric("TOTAL", frame_total, color, scene_color),
            timing_metric("pose", ms.get("pose_total", 0.0), color, ANSI_BLUE),
            timing_metric("settle", ms.get("settle", 0.0), color, ANSI_BLUE),
            timing_metric(
                "capture", ms.get("capture_total", 0.0), color, ANSI_YELLOW
            ),
            timing_metric("save", ms.get("save_total", 0.0), color, ANSI_GREEN),
        )
    )
    lines.append(
        "  {} {} ({:.1f}% of frame)".format(
            colorize("BOTTLENECK", ANSI_RED, color, bold=True),
            timing_metric(bottleneck_name, bottleneck_ms, color, ANSI_RED),
            bottleneck_percent,
        )
    )
    lines.append(
        "  {} | {} | {}".format(
            timing_metric(
                "metadata", ms.get("metadata", 0.0), color, ANSI_BLUE
            ),
            timing_metric(
                "manifest",
                ms.get("progress_manifest", 0.0),
                color,
                ANSI_BLUE,
            ),
            timing_metric("other", other_ms, color, ANSI_BLUE),
        )
    )
    lines.append(
        "  {} | {} | {} | attempts {}".format(
            timing_metric(
                "camera wall", ms.get("camera_batch", 0.0), color, ANSI_YELLOW
            ),
            timing_metric(
                "max RPC", ms.get("camera_rpc_max", 0.0), color, ANSI_MAGENTA
            ),
            timing_metric(
                "lock wait", ms.get("capture_lock_wait", 0.0), color, ANSI_BLUE
            ),
            payload.get("capture_attempts", 1),
        )
    )
    lines.append(
        "  {} | {} | {}".format(
            timing_metric(
                "response wait max",
                ms.get("camera_response_wait_max", 0.0),
                color,
                ANSI_RED,
            ),
            timing_metric(
                "response decode sum",
                ms.get("camera_response_decode_sum", 0.0),
                color,
                ANSI_MAGENTA,
            ),
            timing_metric(
                "image process sum",
                ms.get("camera_processing_sum", 0.0),
                color,
                ANSI_BLUE,
            ),
        )
    )
    lines.append(
        "  {} | {} | {}".format(
            timing_metric(
                "request pack sum",
                ms.get("camera_request_pack_sum", 0.0),
                color,
                ANSI_BLUE,
            ),
            timing_metric(
                "request send sum",
                ms.get("camera_request_send_sum", 0.0),
                color,
                ANSI_BLUE,
            ),
            timing_metric(
                "camera start skew",
                ms.get("camera_start_skew", 0.0),
                color,
                ANSI_BLUE,
            ),
        )
    )
    lines.append(colorize("  CAMERAS", ANSI_MAGENTA, color, bold=True))
    for item in payload.get("camera_ms", []):
        camera_name = str(item.get("camera", "camera"))
        lines.append(
            "    {} {} | {} | {} | start {} | end {}".format(
                colorize(
                    "{:<12}".format(camera_name),
                    scene_color,
                    color,
                    bold=True,
                ),
                timing_metric("RPC", item.get("rpc_ms", 0.0), color),
                timing_metric(
                    "process", item.get("processing_ms", 0.0), color
                ),
                timing_metric("total", item.get("total_ms", 0.0), color),
                colored_duration(item.get("start_offset_ms", 0.0), color),
                colored_duration(item.get("end_offset_ms", 0.0), color),
            )
        )
        lines.append(
            "      NNG pack {} | context {} | send {}".format(
                colored_duration(item.get("request_pack_ms", 0.0), color),
                colored_duration(item.get("context_ms", 0.0), color),
                colored_duration(item.get("request_send_ms", 0.0), color),
            )
        )
        lines.append(
            "      RESPONSE wait {} | decode {} | payload {}".format(
                colored_duration(item.get("response_wait_ms", 0.0), color),
                colored_duration(
                    item.get("response_decode_ms", 0.0), color
                ),
                human_bytes(item.get("response_bytes", 0)),
            )
        )
        lines.append(
            "      IMAGE RGB unpack {} | prepare {} | depth unpack {} | convert {}".format(
                colored_duration(item.get("rgb_unpack_ms", 0.0), color),
                colored_duration(item.get("rgb_prepare_ms", 0.0), color),
                colored_duration(item.get("depth_unpack_ms", 0.0), color),
                colored_duration(item.get("depth_convert_ms", 0.0), color),
            )
        )
    image_counts = payload.get("images", {})
    output_bytes = payload.get("bytes", {})
    lines.append(
        "  {} PNG RGB {} ({}) | depth NPY {} ({})".format(
            colorize("OUTPUT", ANSI_GREEN, color, bold=True),
            image_counts.get("rgb", "?"),
            human_bytes(output_bytes.get("rgb", 0)),
            image_counts.get("depth", "?"),
            human_bytes(output_bytes.get("depth", 0)),
        )
    )
    lines.append(
        "  {} | {}".format(
            timing_metric(
                "PNG write",
                ms.get("png_rgb_write", 0.0),
                color,
                ANSI_GREEN,
            ),
            timing_metric(
                "depth NPY write",
                ms.get("png_depth_write", 0.0),
                color,
                ANSI_GREEN,
            ),
        )
    )
    lines.append(divider)
    return "\n".join(lines)


def format_timing_summary(payload, color=False):
    timing = payload.get("timing", {})
    average_ms = timing.get("average_ms", {})
    maximum_ms = timing.get("max_ms", {})
    scene_index = int(payload.get("scene", 0))
    scene_color = SCENE_COLORS[scene_index % len(SCENE_COLORS)]
    divider = colorize("=" * 88, ANSI_DIM, color)
    lines = [
        colorize(
            "[SUMMARY] scene={}  gpu={}  trajectory={}  samples={}".format(
                scene_index,
                payload.get("gpu", "?"),
                payload.get("trajectory", "?"),
                timing.get("samples", 0),
            ),
            scene_color,
            color,
            bold=True,
        ),
        divider,
    ]
    for label, key in (
        ("frame", "frame_total"),
        ("capture", "capture_total"),
        ("camera wall", "camera_batch"),
        ("response wait", "camera_response_wait_max"),
        ("response decode sum", "camera_response_decode_sum"),
        ("image process sum", "camera_processing_sum"),
        ("save", "save_total"),
        ("manifest", "progress_manifest"),
    ):
        lines.append(
            "  {:<12} average {} | maximum {}".format(
                label,
                colored_duration(average_ms.get(key, 0.0), color),
                colored_duration(maximum_ms.get(key, 0.0), color),
            )
        )
    lines.append(
        "  episode write {}".format(
            colored_duration(payload.get("episode_write_ms", 0.0), color)
        )
    )
    lines.append(divider)
    return "\n".join(lines)


def format_round_timing(payload, color=False):
    ms = payload.get("ms", {})
    divider = colorize("=" * 88, ANSI_DIM, color)
    lines = [
        colorize(
            "[ROUND] frame={}  scenes={}  wall={}".format(
                payload.get("round", "?"),
                len(payload.get("scenes", [])),
                human_duration_ms(ms.get("round_wall", 0.0)),
            ),
            ANSI_YELLOW,
            color,
            bold=True,
        ),
        divider,
        "  {} | {}".format(
            timing_metric(
                "start skew", ms.get("start_skew", 0.0), color, ANSI_BLUE
            ),
            timing_metric("end skew", ms.get("end_skew", 0.0), color, ANSI_BLUE),
        ),
    ]
    for item in payload.get("scenes", []):
        scene_index = int(item.get("scene", 0))
        scene_color = SCENE_COLORS[scene_index % len(SCENE_COLORS)]
        lines.append(
            "  {} trajectory={} step={} total={}".format(
                colorize(
                    "scene {:<2}".format(scene_index),
                    scene_color,
                    color,
                    bold=True,
                ),
                item.get("trajectory", "?"),
                item.get("step", "?"),
                colored_duration(item.get("frame_total_ms", 0.0), color),
            )
        )
    lines.append(divider)
    return "\n".join(lines)


class RoundTimingAggregator:
    def __init__(self, scene_indices):
        self.scene_indices = tuple(sorted(int(index) for index in scene_indices))
        self.pending = {}
        self.lock = threading.Lock()

    def add(self, payload):
        round_index = int(payload["scene_frame"])
        scene_index = int(payload["scene"])
        compact = {
            "scene": scene_index,
            "trajectory": str(payload.get("trajectory", "")),
            "step": int(payload.get("step", 0)),
            "frame_total_ms": float(
                payload.get("ms", {}).get("frame_total", 0.0)
            ),
            "wall_started_s": float(payload.get("wall_started_s", 0.0)),
            "wall_ended_s": float(payload.get("wall_ended_s", 0.0)),
        }
        with self.lock:
            bucket = self.pending.setdefault(round_index, {})
            bucket[scene_index] = compact
            if any(index not in bucket for index in self.scene_indices):
                return None
            scene_items = [bucket[index] for index in self.scene_indices]
            del self.pending[round_index]

        starts = [item["wall_started_s"] for item in scene_items]
        ends = [item["wall_ended_s"] for item in scene_items]
        return {
            "event": "round",
            "round": round_index,
            "scenes": scene_items,
            "wall_started_s": min(starts),
            "wall_ended_s": max(ends),
            "ms": {
                "round_wall": (max(ends) - min(starts)) * 1000.0,
                "start_skew": (max(starts) - min(starts)) * 1000.0,
                "end_skew": (max(ends) - min(ends)) * 1000.0,
            },
        }


def print_timing_payload(payload, args):
    timing_format = str(getattr(args, "timing_format", "pretty"))
    use_color = timing_color_enabled(getattr(args, "color", "auto"))
    outputs = []
    if timing_format in ("pretty", "both"):
        if payload.get("event") == "trajectory_summary":
            outputs.append(format_timing_summary(payload, color=use_color))
        elif payload.get("event") == "round":
            outputs.append(format_round_timing(payload, color=use_color))
        else:
            outputs.append(format_frame_timing(payload, color=use_color))
    if timing_format in ("json", "both"):
        prefixes = {
            "trajectory_summary": "TIMING_SUMMARY",
            "round": "TIMING_ROUND",
        }
        prefix = prefixes.get(payload.get("event"), "TIMING")
        outputs.append(
            "{} {}".format(
                prefix,
                json.dumps(payload, ensure_ascii=True, sort_keys=True),
            )
        )
    with TIMING_OUTPUT_LOCK:
        print("\n".join(outputs), flush=True)


def manifest_payload(args, jobs, status, endpoints=None):
    local_stage_root = str(
        getattr(args, "local_stage_root", "") or ""
    ).strip()
    storage = {
        "format": "png_npy",
        "rgb_encoding": "png",
        "depth_encoding": "float32_npy_metres",
        "depth_unit": "metre",
        "depth_semantics": "perspective_range",
        "writer": (
            "local_stage_async_nas"
            if local_stage_root
            else "scene_capture_process"
        ),
        "output_root": str(Path(args.output_root)),
        "local_stage_root": local_stage_root,
        "async_nas_queue_frames": int(
            getattr(args, "async_nas_queue_frames", 0)
        ),
    }
    return {
        "status": str(status),
        "map_name": str(getattr(args, "map_name", "") or ""),
        "server": {
            "host": str(args.server_host),
            "port": int(args.server_port),
        },
        "trajectory_file": str(Path(args.trajectory_file).resolve()),
        "start_step": int(args.start_step),
        "max_steps": int(args.max_steps),
        "cameras": [str(camera) for camera in args.cameras],
        "timing_every": int(getattr(args, "timing_every", 0)),
        "timing_format": str(getattr(args, "timing_format", "pretty")),
        "resume": {
            "enabled": bool(getattr(args, "resume", False)),
            "overwrite": bool(getattr(args, "overwrite", False)),
        },
        "capture": {
            "worker": "one_process_per_scene",
        },
        "storage": storage,
        "collected_index": str(
            Path(args.output_root) / "collected_episodes.jsonl"
        ),
        "partition_index_filename": PARTITION_INDEX_FILENAME,
        "endpoints": endpoints or [],
        "jobs": [
            {
                "scene_index": job["scene_index"],
                "scene": job["scene"],
                "gpu_id": job["gpu_id"],
                "capture_process_pid": job.get("_capture_process_pid"),
                "recordings": [
                    {
                        "sequence_index": recording["sequence_index"],
                        "trajectory_id": recording["trajectory_id"],
                        "trajectory_lookup_id": recording[
                            "trajectory_lookup_id"
                        ],
                        "trajectory_file": recording["trajectory_file"],
                        "source_task": dict(
                            recording.get("source_task") or {}
                        ),
                        "step_count": len(recording["steps"]),
                        "saved_steps": recording["saved_steps"],
                        "saved_rgb_images": recording["saved_rgb_images"],
                        "saved_depth_images": recording["saved_depth_images"],
                        "resume_step_offset": int(
                            recording.get("_resume_step_offset", 0)
                        ),
                        "remaining_steps": recording_remaining_steps(
                            recording
                        ),
                        "resume_source": recording.get("_resume_source"),
                        "resume_complete": bool(
                            recording.get("_resume_complete", False)
                        ),
                        "output_dir": str(recording["output_dir"]),
                        "collected_jsonl": str(recording["collected_path"]),
                        "timing": recording_timing_summary(recording),
                    }
                    for recording in job["recordings"]
                ],
            }
            for job in jobs
        ],
    }


def write_manifest(args, jobs, status, endpoints=None):
    manifest_path = Path(args.output_root) / "run_manifest.json"
    if getattr(args, "published_layout", False):
        from .published_dataset import run_directory
        manifest_path = run_directory(args) / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_name("{}.tmp".format(manifest_path.name))
    temporary_path.write_text(
        json.dumps(
            manifest_payload(args, jobs, status, endpoints=endpoints),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    return manifest_path


def record_scene_job(
    args,
    client,
    job,
    pose_types,
    stop_event,
    progress_callback,
    frame_timing_callback=None,
    timing_output_callback=None,
    text_output_callback=None,
    async_nas_writer=None,
):
    """Record one scene's trajectories sequentially and independently."""
    timing_every = max(0, int(getattr(args, "timing_every", 0)))
    scene_frame_index = 0
    for recording in job["recordings"]:
        if stop_event.is_set():
            return
        if async_nas_writer is not None:
            recording["_nas_committed_count"] = len(
                recording.get("collected_steps", [])
            )
            recording["_nas_camera_count"] = len(args.cameras)
        resume_offset = int(recording.get("_resume_step_offset", 0))
        if resume_offset >= len(recording["steps"]):
            continue
        episode_write_ms = 0.0
        try:
            for round_index in range(resume_offset, len(recording["steps"])):
                step = recording["steps"][round_index]
                if stop_event.is_set():
                    return
                collect_timing = bool(
                    timing_every and round_index % timing_every == 0
                )
                frame_started = time.perf_counter()
                wall_started = time.time()
                pose_timing = set_direct_scene_pose(
                    client,
                    job,
                    recording,
                    round_index,
                    pose_types,
                )
                settle_started = time.perf_counter()
                if float(args.interval) > 0:
                    time.sleep(float(args.interval))
                settle_ms = (time.perf_counter() - settle_started) * 1000.0

                if collect_timing:
                    image_pair, capture_timing = capture_scene_images(
                        client,
                        job["scene_index"],
                        args.cameras,
                        args.image_retries,
                        return_timings=True,
                        rgb_mode="png",
                    )
                else:
                    image_pair = capture_scene_images(
                        client,
                        job["scene_index"],
                        args.cameras,
                        args.image_retries,
                        rgb_mode="png",
                    )
                    capture_timing = {}
                saved_paths = save_job_png_images(
                    job,
                    recording,
                    step,
                    image_pair,
                    args.cameras,
                    return_timings=collect_timing,
                    async_nas_writer=async_nas_writer,
                )
                save_timing = saved_paths.get(
                    "_timing", recording.get("_last_save_timing", {})
                )
                metadata_started = time.perf_counter()
                append_collected_step(job, recording, step, saved_paths)
                metadata_ms = (
                    time.perf_counter() - metadata_started
                ) * 1000.0
                progress_started = time.perf_counter()
                progress_callback()
                progress_ms = (
                    time.perf_counter() - progress_started
                ) * 1000.0
                if collect_timing:
                    camera_timings = capture_timing.get("cameras", [])
                    camera_rpc_values = [
                        float(item.get("rpc_ms", 0.0))
                        for item in camera_timings
                    ]
                    camera_processing_values = [
                        float(item.get("processing_ms", 0.0))
                        for item in camera_timings
                    ]
                    camera_response_wait_values = [
                        float(item.get("response_wait_ms", 0.0))
                        for item in camera_timings
                    ]
                    camera_start_offsets = [
                        float(item.get("start_offset_ms", 0.0))
                        for item in camera_timings
                    ]
                    durations_ms = {
                        "pose_prepare": pose_timing["prepare_ms"],
                        "pose_lock_wait": pose_timing["lock_wait_ms"],
                        "pose_rpc": pose_timing["rpc_ms"],
                        "pose_total": pose_timing["total_ms"],
                        "settle": settle_ms,
                        "capture_total": capture_timing.get(
                            "capture_total_ms", 0.0
                        ),
                        "capture_client_total": capture_timing.get(
                            "total_ms", 0.0
                        ),
                        "capture_retry_sleep": capture_timing.get(
                            "retry_sleep_ms", 0.0
                        ),
                        "capture_lock_wait": capture_timing.get(
                            "lock_wait_ms", 0.0
                        ),
                        "camera_batch": capture_timing.get(
                            "camera_batch_ms", 0.0
                        ),
                        "camera_rpc_max": max(camera_rpc_values, default=0.0),
                        "camera_rpc_sum": sum(camera_rpc_values),
                        "camera_processing_sum": sum(
                            camera_processing_values
                        ),
                        "camera_response_wait_max": max(
                            camera_response_wait_values, default=0.0
                        ),
                        "camera_response_wait_sum": sum(
                            camera_response_wait_values
                        ),
                        "camera_response_decode_sum": sum(
                            float(item.get("response_decode_ms", 0.0))
                            for item in camera_timings
                        ),
                        "camera_request_pack_sum": sum(
                            float(item.get("request_pack_ms", 0.0))
                            for item in camera_timings
                        ),
                        "camera_request_send_sum": sum(
                            float(item.get("request_send_ms", 0.0))
                            for item in camera_timings
                        ),
                        "camera_rgb_unpack_sum": sum(
                            float(item.get("rgb_unpack_ms", 0.0))
                            for item in camera_timings
                        ),
                        "camera_depth_unpack_sum": sum(
                            float(item.get("depth_unpack_ms", 0.0))
                            for item in camera_timings
                        ),
                        "camera_depth_convert_sum": sum(
                            float(item.get("depth_convert_ms", 0.0))
                            for item in camera_timings
                        ),
                        "camera_start_skew": (
                            max(camera_start_offsets)
                            - min(camera_start_offsets)
                            if camera_start_offsets
                            else 0.0
                        ),
                        "save_total": save_timing.get("total_ms", 0.0),
                        "png_rgb_write": save_timing.get(
                            "rgb_write_ms", 0.0
                        ),
                        "png_depth_write": save_timing.get(
                            "depth_write_ms", 0.0
                        ),
                        "metadata": metadata_ms,
                        "progress_manifest": progress_ms,
                        "frame_total": (
                            time.perf_counter() - frame_started
                        ) * 1000.0,
                    }
                    update_recording_timing(recording, durations_ms)
                    timing_payload = {
                        "event": "frame",
                        "scene": int(job["scene_index"]),
                        "gpu": int(job["gpu_id"]),
                        "sequence": int(recording["sequence_index"]),
                        "scene_frame": int(scene_frame_index),
                        "trajectory": str(recording["trajectory_id"]),
                        "step": int(step["step_index"]),
                        "wall_started_s": round(wall_started, 6),
                        "wall_ended_s": round(time.time(), 6),
                        "capture_attempts": int(
                            capture_timing.get("attempts", 1)
                        ),
                        "ms": {
                            name: round(float(value), 3)
                            for name, value in durations_ms.items()
                        },
                        "camera_ms": [
                            {
                                key: round(float(value), 3)
                                if isinstance(value, float)
                                else value
                                for key, value in item.items()
                                if key not in ("rgb_bytes", "depth_bytes")
                            }
                            for item in camera_timings
                        ],
                        "bytes": {
                            "rgb": int(save_timing.get("rgb_bytes", 0)),
                            "depth": int(save_timing.get("depth_bytes", 0)),
                        },
                        "images": {
                            "rgb": len(saved_paths["rgb"]),
                            "depth": len(saved_paths["depth"]),
                        },
                    }
                    if timing_output_callback is None:
                        print_timing_payload(timing_payload, args)
                    else:
                        timing_output_callback(timing_payload)
                    if frame_timing_callback is not None:
                        frame_timing_callback(timing_payload)
                else:
                    message = (
                        "scene={} gpu={} seq={} trajectory={} step={} "
                        "rgb={} depth={}"
                    ).format(
                        job["scene_index"],
                        job["gpu_id"],
                        recording["sequence_index"],
                        recording["trajectory_id"],
                        step["step_index"],
                        len(saved_paths["rgb"]),
                        len(saved_paths["depth"]),
                    )
                    if text_output_callback is None:
                        print(message, flush=True)
                    else:
                        text_output_callback(message)
                scene_frame_index += 1
        finally:
            sync_error = None
            if async_nas_writer is not None:
                try:
                    async_nas_writer.flush()
                except BaseException as error:
                    sync_error = error
                committed_count = int(
                    recording.get("_nas_committed_count", 0)
                )
                if committed_count < len(recording["collected_steps"]):
                    recording["collected_steps"] = list(
                        recording["collected_steps"][:committed_count]
                    )
                    recording["saved_steps"] = committed_count
                    recording["saved_rgb_images"] = (
                        committed_count * len(args.cameras)
                    )
                    recording["saved_depth_images"] = (
                        committed_count * len(args.cameras)
                    )
            if recording["collected_steps"]:
                episode_write_started = time.perf_counter()
                write_collected_episode(job, recording)
                episode_write_ms = (
                    time.perf_counter() - episode_write_started
                ) * 1000.0
            if sync_error is not None:
                raise sync_error
        if len(recording["collected_steps"]) != len(recording["steps"]):
            raise RuntimeError(
                "trajectory {} stopped with {}/{} collected steps".format(
                    recording["trajectory_id"],
                    len(recording["collected_steps"]),
                    len(recording["steps"]),
                )
            )
        write_completion_marker(recording, args.cameras)
        recording["_resume_complete"] = True
        recording["_resume_step_offset"] = len(recording["steps"])
        progress_callback()
        if timing_every:
            summary_payload = {
                "event": "trajectory_summary",
                "scene": int(job["scene_index"]),
                "gpu": int(job["gpu_id"]),
                "trajectory": str(recording["trajectory_id"]),
                "episode_write_ms": round(episode_write_ms, 3),
                "timing": recording_timing_summary(recording),
            }
            if timing_output_callback is None:
                print_timing_payload(summary_payload, args)
            else:
                timing_output_callback(summary_payload)
        else:
            message = "scene={} trajectory={} completed steps={}".format(
                job["scene_index"],
                recording["trajectory_id"],
                recording["saved_steps"],
            )
            if text_output_callback is None:
                print(message, flush=True)
            else:
                text_output_callback(message)


def scene_job_snapshot(job, include_collected=False):
    recordings = []
    for recording in job["recordings"]:
        timing_lock = recording.get("_timing_lock")
        if timing_lock is None:
            timing_stats = copy_timing_stats(
                recording.get("_timing_stats") or {}
            )
        else:
            with timing_lock:
                timing_stats = copy_timing_stats(
                    recording.get("_timing_stats") or {}
                )
        committed_count = recording.get("_nas_committed_count")
        if committed_count is None:
            snapshot_saved_steps = int(recording.get("saved_steps", 0))
            snapshot_rgb_images = int(
                recording.get("saved_rgb_images", 0)
            )
            snapshot_depth_images = int(
                recording.get("saved_depth_images", 0)
            )
            snapshot_collected_steps = list(
                recording.get("collected_steps", [])
            )
        else:
            snapshot_saved_steps = int(committed_count)
            camera_count = int(
                recording.get("_nas_camera_count", len(DEFAULT_CAMERAS))
            )
            snapshot_rgb_images = snapshot_saved_steps * camera_count
            snapshot_depth_images = snapshot_saved_steps * camera_count
            snapshot_collected_steps = list(
                recording.get("collected_steps", [])[:snapshot_saved_steps]
            )
        item = {
            "sequence_index": int(recording["sequence_index"]),
            "saved_steps": snapshot_saved_steps,
            "saved_rgb_images": snapshot_rgb_images,
            "saved_depth_images": snapshot_depth_images,
            "timing_stats": timing_stats,
        }
        if include_collected:
            item["collected_steps"] = snapshot_collected_steps
        recordings.append(item)
    return {
        "scene_index": int(job["scene_index"]),
        "recordings": recordings,
    }


def apply_scene_job_snapshot(job, snapshot):
    if int(snapshot["scene_index"]) != int(job["scene_index"]):
        raise RuntimeError("scene progress snapshot was routed incorrectly")
    by_sequence = {
        int(recording["sequence_index"]): recording
        for recording in job["recordings"]
    }
    for item in snapshot.get("recordings", []):
        recording = by_sequence[int(item["sequence_index"])]
        for name in (
            "saved_steps",
            "saved_rgb_images",
            "saved_depth_images",
        ):
            recording[name] = int(item.get(name, 0))
        recording["_timing_stats"] = copy_timing_stats(
            item.get("timing_stats") or {}
        )
        if "collected_steps" in item:
            recording["collected_steps"] = list(item["collected_steps"])


def _run_scene_recording_process(
    args,
    machines_info,
    job,
    endpoint,
    event_queue,
    stop_event,
):
    """Own one scene's ProjectAirSim connection and capture pipeline."""
    from AerialDojo.projectairsim_plugin import ProjectAirSimSimulatorClientTool
    from projectairsim.types import Pose, Quaternion, Vector3

    scene_index = int(job["scene_index"])
    client = None
    async_nas_writer = None
    failure = None
    try:
        client = ProjectAirSimSimulatorClientTool(
            machines_info,
            sim_config_path=args.sim_config_path,
            scene_config_name=args.scene_config_name,
        )
        client.connect_existing_scene(
            0,
            scene_index,
            endpoint,
            connection_timeout=float(args.connect_timeout),
        )
        if str(getattr(args, "local_stage_root", "")).strip():
            async_nas_writer = AsyncNasFrameWriter(
                args.local_stage_root,
                args.output_root,
                args.async_nas_queue_frames,
                scene_index,
            )
        event_queue.put(
            {
                "type": "started",
                "scene": scene_index,
                "capture_pid": os.getpid(),
                "async_nas": bool(async_nas_writer is not None),
            }
        )

        def send_progress():
            event_queue.put(
                {
                    "type": "progress",
                    "scene": scene_index,
                    "snapshot": scene_job_snapshot(job),
                }
            )

        def send_timing(payload):
            event_queue.put(
                {
                    "type": "timing",
                    "scene": scene_index,
                    "payload": payload,
                }
            )

        def send_text(message):
            event_queue.put(
                {
                    "type": "text",
                    "scene": scene_index,
                    "message": str(message),
                }
            )

        record_scene_job(
            args,
            client,
            job,
            (Pose, Quaternion, Vector3),
            stop_event,
            send_progress,
            timing_output_callback=send_timing,
            text_output_callback=send_text,
            async_nas_writer=async_nas_writer,
        )
    except BaseException as error:
        failure = {
            "error": "scene {} capture process failed: {}".format(
                scene_index, error
            ),
            "traceback": traceback.format_exc(),
        }
        stop_event.set()
    finally:
        cleanup_errors = []
        if async_nas_writer is not None:
            try:
                async_nas_writer.close()
            except Exception as error:
                cleanup_errors.append(
                    "异步 NAS 写入线程: {}".format(error)
                )
        if client is not None:
            try:
                client.disconnectClients()
            except Exception as error:
                cleanup_errors.append("ProjectAirSim disconnect: {}".format(error))
        if cleanup_errors:
            if failure is None:
                failure = {
                    "error": "scene {} cleanup failed: {}".format(
                        scene_index, "; ".join(cleanup_errors)
                    ),
                    "traceback": "",
                }
            else:
                failure["error"] += "; " + "; ".join(cleanup_errors)

    event_queue.put(
        {
            "type": "error" if failure is not None else "complete",
            "scene": scene_index,
            "snapshot": scene_job_snapshot(job, include_collected=True),
            "error": failure,
        }
    )


def run_recording(args, jobs):
    from AerialDojo.projectairsim_plugin import ProjectAirSimSimulatorClientTool

    if not (args.resume and not args.overwrite):
        prepare_output_directories(jobs, overwrite=args.overwrite)
        initialize_collected_episodes(jobs)
    machines_info = build_machines_info(args, jobs)
    client = ProjectAirSimSimulatorClientTool(
        machines_info,
        sim_config_path=args.sim_config_path,
        scene_config_name=args.scene_config_name,
    )
    connected = False
    status = "failed"
    endpoints = []
    manifest_lock = threading.Lock()
    last_manifest_write = [0.0]
    round_timings = RoundTimingAggregator(
        job["scene_index"] for job in jobs
    )
    scene_processes = {}
    process_context = multiprocessing.get_context("spawn")
    process_events = process_context.Queue()
    stop_event = process_context.Event()

    def write_progress_manifest(force=False):
        now = time.monotonic()
        interval = max(0.0, float(args.manifest_interval))
        with manifest_lock:
            if (
                not force
                and last_manifest_write[0]
                and now - last_manifest_write[0] < interval
            ):
                return None
            path = write_manifest(args, jobs, "running", endpoints=endpoints)
            last_manifest_write[0] = now
            return path

    try:
        print("opening {} scenes...".format(len(jobs)))
        client.run_call(
            airsim_timeout=int(args.connect_timeout),
            reuse_existing=not bool(args.force_reopen_scenes),
        )
        connected = True
        endpoints = client.endpoints_by_machine[0]
        client.disconnectClients()
        manifest_path = write_progress_manifest(force=True)
        print("manifest: {}".format(manifest_path))

        for job in jobs:
            scene_index = int(job["scene_index"])
            process = process_context.Process(
                target=_run_scene_recording_process,
                args=(
                    args,
                    machines_info,
                    job,
                    endpoints[scene_index],
                    process_events,
                    stop_event,
                ),
                name="capture-scene-{}".format(scene_index),
            )
            process.start()
            scene_processes[scene_index] = process
            job["_capture_process_pid"] = process.pid
        print(
            "started {} independent scene capture processes".format(
                len(scene_processes)
            )
        )

        jobs_by_scene = {
            int(job["scene_index"]): job for job in jobs
        }
        finished_scenes = set()
        worker_errors = []
        dead_since = {}
        while len(finished_scenes) < len(scene_processes):
            try:
                event = process_events.get(timeout=0.5)
            except Empty:
                now = time.monotonic()
                for scene_index, process in scene_processes.items():
                    if scene_index in finished_scenes or process.exitcode is None:
                        continue
                    first_seen = dead_since.setdefault(scene_index, now)
                    if now - first_seen < 1.0:
                        continue
                    finished_scenes.add(scene_index)
                    stop_event.set()
                    worker_errors.append(
                        "scene {} capture process exited with code {} without a final event".format(
                            scene_index, process.exitcode
                        )
                    )
                continue

            event_type = event.get("type")
            scene_index = int(event["scene"])
            job = jobs_by_scene[scene_index]
            if event_type == "started":
                job["_capture_process_pid"] = int(event["capture_pid"])
                print(
                    "scene {} process: capture pid={} output={}".format(
                        scene_index,
                        event["capture_pid"],
                        (
                            "local-stage->nas"
                            if event.get("async_nas")
                            else "png+npy"
                        ),
                    )
                )
                write_progress_manifest()
            elif event_type == "progress":
                apply_scene_job_snapshot(job, event["snapshot"])
                write_progress_manifest()
            elif event_type == "timing":
                payload = event["payload"]
                print_timing_payload(payload, args)
                if payload.get("event") == "frame":
                    round_payload = round_timings.add(payload)
                    if round_payload is not None:
                        print_timing_payload(round_payload, args)
            elif event_type == "text":
                print(event["message"], flush=True)
            elif event_type in ("complete", "error"):
                apply_scene_job_snapshot(job, event["snapshot"])
                finished_scenes.add(scene_index)
                if event_type == "error":
                    stop_event.set()
                    failure = event.get("error") or {}
                    worker_errors.append(
                        "{}\n{}".format(
                            failure.get(
                                "error",
                                "scene {} capture process failed".format(
                                    scene_index
                                ),
                            ),
                            failure.get("traceback", ""),
                        ).rstrip()
                    )
                write_progress_manifest(force=True)
            else:
                stop_event.set()
                worker_errors.append(
                    "scene {} sent unknown process event: {}".format(
                        scene_index, event_type
                    )
                )
                finished_scenes.add(scene_index)

        for scene_index, process in scene_processes.items():
            process.join(timeout=10.0)
            if process.is_alive():
                stop_event.set()
                process.terminate()
                process.join(timeout=5.0)
                worker_errors.append(
                    "scene {} capture process did not stop cleanly".format(
                        scene_index
                    )
                )
            elif process.exitcode not in (0, None):
                worker_errors.append(
                    "scene {} capture process exited with code {}".format(
                        scene_index, process.exitcode
                    )
                )
        if worker_errors:
            raise RuntimeError("; ".join(worker_errors))

        status = "completed"
        print("all trajectories completed")
    finally:
        stop_event.set()
        for scene_index, process in scene_processes.items():
            if process.is_alive():
                process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
                print(
                    "warning: terminated scene {} capture process".format(
                        scene_index
                    )
                )
        process_events.close()
        try:
            if connected and args.keep_scenes:
                client.disconnectClients()
                print("disconnected client; all UE scenes are kept open")
            else:
                client.closeScenes()
                print("closed all UE scenes")
        except Exception as error:
            print("warning: failed to clean up scenes: {}".format(error))
        write_manifest(args, jobs, status, endpoints=endpoints)
        collected_index_path = write_collected_index(args, jobs)
        print("collected metadata: {}".format(collected_index_path))
        for partition_index_path in write_partition_collected_indexes(args, jobs):
            print("partition metadata: {}".format(partition_index_path))


def main(argv=None):
    args = parse_args(argv)
    if getattr(args, "published_layout", False) and not args.dry_run:
        from .published_dataset import file_lock, run_directory
        with file_lock(run_directory(args) / "recording.lock", blocking=False):
            return _main(args)
    return _main(args)


def _main(args):
    jobs = load_recording_jobs(args)
    resume_summary = None
    if args.resume and not args.overwrite:
        resume_summary = restore_jobs_from_output(
            jobs,
            args.cameras,
            write_markers=not args.dry_run,
            trusted_complete_task_directories=(
                args.resume_trust_complete_task_directory
            ),
            use_partition_indexes=args.resume_use_partition_indexes,
            progress_every=args.resume_progress_every,
        )
    print_jobs(jobs)
    if resume_summary is not None:
        print_resume_summary(resume_summary, jobs)
    if args.dry_run:
        print("dry-run complete; no UE scene was opened")
        return 0
    if (
        resume_summary is not None
        and resume_summary["remaining_trajectories"] == 0
    ):
        manifest_path = write_manifest(args, jobs, "completed", endpoints=[])
        print("manifest: {}".format(manifest_path))
        collected_index_path = write_collected_index(args, jobs)
        print("collected metadata: {}".format(collected_index_path))
        for partition_index_path in write_partition_collected_indexes(args, jobs):
            print("partition metadata: {}".format(partition_index_path))
        print("all trajectories already completed; no UE scene was opened")
        return 0
    run_recording(args, jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
