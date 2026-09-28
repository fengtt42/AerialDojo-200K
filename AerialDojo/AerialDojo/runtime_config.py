"""Load the simulator server configuration used by online evaluation."""

from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_CONFIG_FILE = REPO_ROOT / "config" / "server_config.yaml"


def load_runtime_config(
    path: Any = DEFAULT_SERVER_CONFIG_FILE,
) -> Dict[str, Any]:
    """Read a server YAML file and return its top-level mapping."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("server_config.yaml must contain a mapping")
    if payload.get("task_source", {}).get("layout") == "published":
        from .trajectory_recording.published_dataset import resolve_config
        return resolve_config(payload)
    return dict(payload)


def _mapping(payload: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise ValueError("server_config.yaml '{}' must be a mapping".format(key))
    return dict(value)


def configured_gpu_ids(payload: Mapping[str, Any]) -> List[int]:
    """Return the non-negative, unique GPU ids listed in the YAML file."""
    configured = payload.get("gpus")
    if isinstance(configured, dict):
        configured = [
            block.get("gpu_id", block.get("gpu", key)) if isinstance(block, dict) else key
            for key, block in configured.items()
        ]
    if not isinstance(configured, list) or not configured:
        raise ValueError("config 'gpus' must be a non-empty list or mapping")
    gpu_ids = []
    for value in configured:
        gpu_id = int(value)
        if gpu_id < 0:
            raise ValueError("GPU ids must be non-negative")
        if gpu_id not in gpu_ids:
            gpu_ids.append(gpu_id)
    return gpu_ids


def server_settings(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return server settings with the configured GPU pool attached."""
    settings = _mapping(payload, "server")
    settings["gpus"] = configured_gpu_ids(payload)
    return settings
