"""读取 config/record_jobs.yaml 中的共享运行配置。"""

from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "config"
DEFAULT_JOB_FILE = CONFIG_ROOT / "record_jobs.yaml"


def load_jobs_config(path: Any = DEFAULT_JOB_FILE) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("record_jobs.yaml must contain a mapping at the top level")
    if payload.get("task_source", {}).get("layout") == "published":
        from .published_dataset import resolve_config
        return resolve_config(payload)
    return payload


def _mapping(payload: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise ValueError("record_jobs.yaml '{}' must be a mapping".format(key))
    return dict(value)


def configured_gpu_ids(payload: Mapping[str, Any]) -> List[int]:
    gpu_blocks = payload.get("gpus")
    if not isinstance(gpu_blocks, dict) or not gpu_blocks:
        raise ValueError("record_jobs.yaml 'gpus' must be a non-empty mapping")
    gpu_ids = []
    for gpu_key, gpu_block in gpu_blocks.items():
        gpu_value = gpu_key
        if isinstance(gpu_block, dict):
            gpu_value = gpu_block.get("gpu_id", gpu_block.get("gpu", gpu_key))
        gpu_id = int(gpu_value)
        if gpu_id < 0:
            raise ValueError("GPU ids must be non-negative")
        if gpu_id not in gpu_ids:
            gpu_ids.append(gpu_id)
    return gpu_ids


def server_settings(payload: Mapping[str, Any]) -> Dict[str, Any]:
    settings = _mapping(payload, "server")
    settings["gpus"] = configured_gpu_ids(payload)
    return settings


def recording_settings(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(payload, "recording")
