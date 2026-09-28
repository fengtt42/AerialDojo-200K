"""在线导航策略的公共接口与动作枚举。"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Mapping


class PolicyAction(str, Enum):
    """策略可以返回的动作。

    当前所有平移动作固定为 1 米，转向固定为 15 度。
    """

    MoveForward = "forward"
    MoveLeft = "left"
    MoveRight = "right"
    MoveUp = "ascend"
    MoveDown = "descend"
    TurnLeft = "rotl"
    TurnRight = "rotr"
    Stop = "stop"


class NavigationPolicy(ABC):
    """每轮根据完整感知观测产生一个动作。"""

    def reset(self, observation: Mapping[str, Any]) -> None:
        """重置回合级内部状态；无状态策略可以不实现。"""

    @abstractmethod
    def forward(self, observation: Mapping[str, Any]) -> PolicyAction:
        """接收 RGB、深度、位姿等观测并返回下一步动作。"""
        raise NotImplementedError
