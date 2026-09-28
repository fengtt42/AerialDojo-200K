"""导航策略注册表与工厂。"""

import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Type

import yaml

from .base import NavigationPolicy


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_CONFIG_FILE = REPO_ROOT / "config" / "policy_config.yaml"


class PolicyFactory:
    """按短名称或 ``module:Class`` 创建导航策略。"""

    _registry: Dict[str, Type[NavigationPolicy]] = {}

    @staticmethod
    def load_config(path: Any = DEFAULT_POLICY_CONFIG_FILE) -> Dict[str, Any]:
        """读取在线策略评测 YAML，并返回可直接使用的普通字典。"""
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        if not isinstance(payload, dict):
            raise ValueError("policy_config.yaml must contain a mapping")
        policy_config = payload.get("policy_config", {})
        if not isinstance(policy_config, dict):
            raise ValueError("policy_config.yaml 'policy_config' must be a mapping")
        return dict(payload)

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized = str(name).strip().lower()
        if not normalized:
            raise ValueError("policy name must not be empty")
        return normalized

    @classmethod
    def register(
        cls,
        name: str,
        aliases: Iterable[str] = (),
        replace: bool = False,
    ):
        """装饰器：注册一个策略类及其可选别名。"""
        names = (name,) + tuple(aliases)

        def decorator(policy_class: Type[NavigationPolicy]):
            if not isinstance(policy_class, type) or not issubclass(
                policy_class, NavigationPolicy
            ):
                raise TypeError("registered policy must subclass NavigationPolicy")
            for item in names:
                normalized = cls._normalize_name(item)
                existing = cls._registry.get(normalized)
                if (
                    existing is not None
                    and existing is not policy_class
                    and not replace
                ):
                    raise ValueError("policy is already registered: {}".format(item))
                cls._registry[normalized] = policy_class
            return policy_class

        return decorator

    @classmethod
    def import_modules(cls, module_names: Iterable[str]) -> None:
        """导入会自行调用 ``register`` 的外部策略模块。"""
        for module_name in module_names:
            importlib.import_module(str(module_name).strip())

    @classmethod
    def resolve(cls, name: str) -> Type[NavigationPolicy]:
        """解析注册名，或动态解析 ``python.module:Class``。"""
        raw_name = str(name).strip()
        registered = cls._registry.get(cls._normalize_name(raw_name))
        if registered is not None:
            return registered

        if ":" not in raw_name:
            raise KeyError(
                "unknown policy {!r}; available: {}".format(
                    raw_name,
                    ", ".join(cls.available()) or "<none>",
                )
            )
        module_name, attribute_path = raw_name.split(":", 1)
        if not module_name or not attribute_path:
            raise ValueError("dynamic policy must use 'module:Class' format")
        target = importlib.import_module(module_name)
        for attribute in attribute_path.split("."):
            target = getattr(target, attribute)
        if not isinstance(target, type) or not issubclass(target, NavigationPolicy):
            raise TypeError(
                "dynamic policy must be a NavigationPolicy subclass: {}".format(
                    raw_name
                )
            )
        return target

    @classmethod
    def create(cls, name: str, **kwargs) -> NavigationPolicy:
        """实例化指定策略。"""
        return cls.resolve(name)(**kwargs)

    @classmethod
    def available(cls) -> Tuple[str, ...]:
        """返回所有已注册短名称。"""
        return tuple(sorted(cls._registry))
