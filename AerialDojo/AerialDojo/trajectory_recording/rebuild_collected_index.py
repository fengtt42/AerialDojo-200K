#!/usr/bin/env python
"""Rebuild the root recording index from task/split/map partitions."""

import argparse
import json
import os
from pathlib import Path


DEFAULT_RECORD_ROOT = Path(__file__).resolve().parents[3] / "VideoRECORD"
DEFAULT_TASK_DIRECTORIES = ("1_BaseTasks", "2_StandardTasks", "3_LongHorizonTasks",
                            "BaseTasks", "StandardTasks", "LongTasks")
SPLIT_DIRECTORIES = (("Trainset", "train"), ("Testset", "test"))
PARTITION_INDEX_FILENAME = "collected_episodes.json"
ROOT_INDEX_FILENAME = "collected_episodes.jsonl"
SUMMARY_FILENAME = "collected_episodes.summary.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Scan task/split/map recording indexes and rebuild one root JSONL; "
            "Published episode_id values are preserved within each partition."
        )
    )
    parser.add_argument(
        "--record-root", type=Path, default=DEFAULT_RECORD_ROOT
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <record-root>/collected_episodes.jsonl.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASK_DIRECTORIES),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print counts without replacing the root indexes.",
    )
    return parser.parse_args(argv)


def normalized_split(value):
    aliases = {
        "train": "train",
        "trainset": "train",
        "test": "test",
        "testset": "test",
    }
    return aliases.get(str(value or "").strip().lower())


def discover_partition_indexes(record_root, task_directories):
    record_root = Path(record_root)
    discovered = []
    for split_directory, split in SPLIT_DIRECTORIES:
        for task_directory in task_directories:
            split_root = record_root / task_directory / split_directory
            if not split_root.is_dir():
                continue
            for map_directory in sorted(
                (path for path in split_root.iterdir() if path.is_dir()),
                key=lambda path: path.name,
            ):
                index_path = map_directory / PARTITION_INDEX_FILENAME
                if index_path.is_file():
                    discovered.append({
                        "task_directory": str(task_directory),
                        "split_directory": split_directory,
                        "split": split,
                        "map_name": map_directory.name,
                        "path": index_path,
                    })
    from .published_dataset import SPLITS, TASKS
    for split_directory in SPLITS:
        split = "train" if split_directory.endswith("TRAINS") else "test"
        for task_directory, suffix, _ in TASKS.values():
            if task_directory not in task_directories:
                continue
            split_root = record_root / split_directory / task_directory
            if not split_root.is_dir():
                continue
            for map_directory in sorted(split_root.iterdir()):
                index_path = map_directory / PARTITION_INDEX_FILENAME
                ending = "_" + suffix
                if map_directory.name.endswith(ending) and index_path.is_file():
                    discovered.append({
                        "task_directory": task_directory,
                        "split_directory": split_directory,
                        "split": split,
                        "map_name": map_directory.name[:-len(ending)],
                        "layout": "published",
                        "output_partition_parts": [split_directory, task_directory, map_directory.name],
                        "path": index_path,
                    })
    if not discovered:
        raise FileNotFoundError(
            "no task/split/map partition indexes found under {}".format(
                record_root
            )
        )
    return discovered


def load_partition(partition):
    path = partition["path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("partition index must be a JSON list: {}".format(path))
    episodes = []
    for position, episode in enumerate(payload, start=1):
        if not isinstance(episode, dict):
            raise ValueError(
                "{} item {} must be a JSON object".format(path, position)
            )
        source_task = episode.get("source_task")
        if not isinstance(source_task, dict):
            raise ValueError(
                "{} item {} has no source_task".format(path, position)
            )
        actual_map = str(
            episode.get("map_name") or source_task.get("map_name") or ""
        )
        if actual_map != partition["map_name"]:
            raise ValueError(
                "{} item {} map_name is {!r}, expected {!r}".format(
                    path, position, actual_map, partition["map_name"]
                )
            )
        actual_split = normalized_split(
            episode.get("split")
            or source_task.get("split")
            or source_task.get("split_directory")
        )
        if actual_split != partition["split"]:
            raise ValueError(
                "{} item {} split is inconsistent with its directory".format(
                    path, position
                )
            )
        if str(source_task.get("task_directory") or "") != partition[
            "task_directory"
        ]:
            raise ValueError(
                "{} item {} task directory is inconsistent".format(path, position)
            )
        if not str(episode.get("description") or "").strip():
            raise ValueError(
                "{} item {} has an empty description".format(path, position)
            )
        if not str(episode.get("recording_id") or ""):
            raise ValueError(
                "{} item {} has no recording_id".format(path, position)
            )
        if partition.get("layout") == "published":
            if source_task.get("output_partition_parts") != partition["output_partition_parts"]:
                raise ValueError("Published partition metadata mismatch: {}".format(path))
            if str(episode.get("episode_id")) != str(source_task.get("episode_id")):
                raise ValueError("Published episode_id changed: {}".format(path))
        episodes.append(dict(episode))
    return episodes


def rebuild_episodes(record_root, task_directories):
    partitions = discover_partition_indexes(record_root, task_directories)
    counters = {"train": 0, "test": 0}
    seen_recording_ids = set()
    episodes = []
    maps = set()
    for partition in partitions:
        maps.add(partition["map_name"])
        for episode in load_partition(partition):
            recording_id = str(episode["recording_id"])
            if recording_id in seen_recording_ids:
                raise ValueError(
                    "duplicate recording_id across partitions: {}".format(
                        recording_id
                    )
                )
            seen_recording_ids.add(recording_id)
            split = partition["split"]
            if partition.get("layout") != "published":
                episode["episode_id"] = str(counters[split])
            episode["split"] = split
            counters[split] += 1
            episodes.append(episode)
    summary = {
        "schema_version": "recording_collection_summary_v1",
        "record_root": str(Path(record_root).resolve()),
        "partition_count": len(partitions),
        "map_count": len(maps),
        "maps": sorted(maps),
        "episode_count": len(episodes),
        "split_counts": counters,
    }
    return episodes, summary


def write_jsonl_atomically(path, episodes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for episode in episodes:
                json.dump(episode, handle, ensure_ascii=False)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def write_json_atomically(path, payload):
    path = Path(path)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def main(argv=None):
    args = parse_args(argv)
    record_root = args.record_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else record_root / ROOT_INDEX_FILENAME
    )
    episodes, summary = rebuild_episodes(record_root, args.tasks)
    summary["output"] = str(output)
    if not args.dry_run:
        write_jsonl_atomically(output, episodes)
        write_json_atomically(record_root / SUMMARY_FILENAME, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    main()
