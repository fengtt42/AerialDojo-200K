"""Bind published SemanticOGS episodes to TrajectoryDATA without replanning."""

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("IID_TRAINS", "IID_TESTS", "OOD_TRAINS", "OOD_TESTS")
TASKS = {
    "base": ("1_BaseTasks", "B", "base_task"),
    "standard": ("2_StandardTasks", "S", "standard_task"),
    "long": ("3_LongHorizonTasks", "L", "long_task"),
}


def component(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Invalid dataset path component: {!r}".format(value))
    return value


def resolve_config(payload, dataset_root=None):
    """Resolve dataset paths against the installed project, independent of cwd."""
    payload = copy.deepcopy(payload)
    root = Path(dataset_root or payload.get("dataset_root") or "..").expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    payload["dataset_root"] = str(root)
    for block, key, default in (
        ("server", "root_path", "{dataset_root}/AerialENVS"),
        ("recording", "output_root", "{dataset_root}/VideoRECORD"),
    ):
        settings = payload.setdefault(block, {})
        path = Path(str(settings.get(key) or default).replace("{dataset_root}", str(root))).expanduser()
        settings[key] = str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())
    return payload


def records_from_published_dataset(payload, args, catalogs):
    # Lazy import avoids a module cycle and shares the existing v6 pose conversion.
    from . import multi_scene_astar_record as recorder

    root = Path(payload["dataset_root"])
    source = payload["task_source"]
    map_name = component(args.map_name or source.get("map_name") or "")
    splits = args.splits or source.get("splits") or list(SPLITS)
    tasks = args.tasks or source.get("tasks") or list(TASKS)
    if any(split not in SPLITS for split in splits) or any(task not in TASKS for task in tasks):
        raise ValueError("Unknown published dataset split or task category")
    episode_ids = args.episode_ids or source.get("episode_ids")
    selected_ids = {component(value) for value in episode_ids} if episode_ids else None
    limit = args.limit if args.limit is not None else int(source.get("limit", 0))
    if limit < 0:
        raise ValueError("--limit must be non-negative")
    start_step = args.start_step_override
    max_steps = args.max_steps_override
    start_step = int(source.get("start_step", 0) if start_step is None else start_step)
    max_steps = int(source.get("max_steps", 0) if max_steps is None else max_steps)
    if start_step < 0 or max_steps < 0:
        raise ValueError("start_step and max_steps must be non-negative")
    records = []
    found_ids = set()
    for split in splits:
        is_train = split.endswith("TRAINS")
        for task in tasks:
            category, suffix, task_type = TASKS[task]
            partition = "{}_{}".format(map_name, suffix)
            task_folder = partition + ("_Train" if is_train else "_Test")
            relative = Path(split) / category / task_folder
            task_path = root / "SemanticOGS" / relative / "Task.json"
            if not task_path.is_file():
                continue  # A map normally belongs to only IID or OOD.
            document = json.loads(task_path.read_text(encoding="utf-8"))
            if not isinstance(document, list):
                raise ValueError("Task.json must contain a list: {}".format(task_path))
            seen_ids = set()
            for index, task_row in enumerate(document):
                episode_id = component(task_row["episode_id"])
                if episode_id in seen_ids:
                    raise ValueError("Duplicate episode_id {} in {}".format(episode_id, task_path))
                seen_ids.add(episode_id)
                if selected_ids is not None and episode_id not in selected_ids:
                    continue
                found_ids.add(episode_id)
                if limit and len(records) >= limit:
                    continue
                if task_row.get("map_name") != map_name or task_row.get("task") != task_type:
                    raise ValueError("Task map/category differs from its directory: {} episode {}".format(task_path, episode_id))
                if int(task_row["used-in-train"]) != int(is_train):
                    raise ValueError("Task split differs from its directory: {} episode {}".format(task_path, episode_id))
                trajectory_path = root / "TrajectoryDATA" / relative / (episode_id + ".json")
                catalog = recorder.load_trajectory_catalog(trajectory_path, catalogs=catalogs)
                entry = catalog["entries"][0]
                trajectory = entry["trajectory"]
                if trajectory.get("map_name") != map_name:
                    raise ValueError("Trajectory map differs from task: {}".format(trajectory_path))
                trajectory_id = str(trajectory.get("task_id") or entry["trajectory_id"])
                # Episode numbers are local to a partition; planner ids may repeat.
                unique_id = "__".join((split, partition, episode_id))
                partition_parts = [split, category, partition]
                records.append({
                    "task_order": len(records),
                    "trajectory_file": str(trajectory_path),
                    "trajectory_id": unique_id,
                    "trajectory_lookup_id": trajectory_id,
                    "start_step": start_step,
                    "max_steps": max_steps,
                    "output_parts": partition_parts + [episode_id],
                    "source_task": {
                        "file": str(Path("SemanticOGS") / relative / "Task.json"),
                        "index": index,
                        "episode_id": episode_id,
                        "map_name": map_name,
                        "trajectory_task_id": trajectory_id,
                        "trajectory_file": str(Path("TrajectoryDATA") / relative / (episode_id + ".json")),
                        "start_object_name": task_row.get("start_object_name", ""),
                        "goal_object_name": task_row.get("goal_object_name", ""),
                        "description": task_row.get("description", ""),
                        "category": task_row.get("category", ""),
                        "Landmark": task_row.get("Landmark", ""),
                        "Direction": task_row.get("Direction", ""),
                        "task": task_type,
                        "task_directory": category,
                        "split": "train" if is_train else "test",
                        "split_directory": split,
                        "used_in_train": int(is_train),
                        "dataset_partition": split,
                        "output_partition_parts": partition_parts,
                        "layout": "published",
                        "task_payload": dict(task_row),
                    },
                })
    if selected_ids is not None and selected_ids - found_ids:
        raise ValueError("Episode ids absent from selected partitions: {}".format(sorted(selected_ids - found_ids)))
    if not records:
        raise ValueError("No tasks matched {} under {}".format(map_name, root / "SemanticOGS"))
    return records


@contextmanager
def file_lock(path, blocking=True):
    """Coordinate independent Linux recording processes using the same NAS."""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError("Another recording process holds {}".format(path)) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def merge_episode_index(path, episodes, jsonl=False):
    """Update selected episodes without deleting earlier recordings."""
    from . import multi_scene_astar_record as recorder

    path = Path(path)
    with file_lock(path.with_name(path.name + ".lock")):
        existing = []
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            existing = [json.loads(line) for line in text.splitlines() if line.strip()] if jsonl else json.loads(text)
        merged = {str(item["recording_id"]): item for item in existing}
        merged.update({str(item["recording_id"]): item for item in episodes})
        writer = recorder.write_jsonl_atomically if jsonl else recorder.write_json_atomically
        return writer(path, list(merged.values()))


def run_directory(args):
    return Path(args.output_root) / "_runs" / component(args.map_name)
