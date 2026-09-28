"""CLIP-H 在线导航策略实现。"""

import io
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..env_utils_uav import yaw_from_quaternion
from .base import NavigationPolicy, PolicyAction
from .factory import PolicyFactory


CAMERA_ACTIONS = (
    PolicyAction.MoveForward,
    PolicyAction.MoveLeft,
    PolicyAction.MoveRight,
    PolicyAction.MoveDown,
)
EXPECTED_CAMERAS = ("0", "1", "2", "3")


@PolicyFactory.register("cliph", aliases=("clip", "clip_heuristic"))
class ClipHeuristicPolicy(NavigationPolicy):
    """使用 CLIP 图文相似度选方向，并应用 UAV-ON 的安全修正规则。"""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        device: str = "auto",
        stop_threshold: float = 0.30,
        min_descend_clearance_m: float = 5.0,
        boundary_radius_m: float = 50.0,
        horizontal_step_m: float = 1.0,
    ) -> None:
        self.torch, clip_model, clip_processor = self._load_clip_types()
        if device == "auto":
            device = "cuda:0" if self.torch.cuda.is_available() else "cpu"
        self.device = str(device)
        self.model = clip_model.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = clip_processor.from_pretrained(model_name)
        self.stop_threshold = float(stop_threshold)
        self.min_descend_clearance_m = float(min_descend_clearance_m)
        self.boundary_radius_m = float(boundary_radius_m)
        self.horizontal_step_m = float(horizontal_step_m)
        self.start_position: Optional[Tuple[float, float, float]] = None
        self.previous_action: Optional[PolicyAction] = None

    @staticmethod
    def _load_clip_types():
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise RuntimeError(
                "CLIP-H 需要 torch、transformers 和 Pillow；"
                "请先在当前环境中安装这些依赖。"
            ) from error
        return torch, CLIPModel, CLIPProcessor

    def reset(self, observation: Mapping[str, Any]) -> None:
        """根据初始观察重置单个回合的策略状态。"""
        task = observation.get("task", {})
        start_position = task.get("start_position", observation["pose"][:3])
        self.start_position = tuple(float(value) for value in start_position[:3])
        self.previous_action = None

    def forward(self, observation: Mapping[str, Any]) -> PolicyAction:
        """读取本轮 RGB-D 感知数据并返回下一步动作。"""
        if self.start_position is None:
            self.reset(observation)

        rgb_images = observation.get("rgb")
        depth_images = observation.get("depth")
        if rgb_images is None or len(rgb_images) != len(EXPECTED_CAMERAS):
            # 碰撞后可能无法获取图像，返回 Stop 避免崩溃
            return PolicyAction.Stop
        if depth_images is None or len(depth_images) != len(EXPECTED_CAMERAS):
            return PolicyAction.Stop

        description = str(observation.get("task", {}).get("description", "")).strip()
        if not description:
            raise RuntimeError("CLIP-H 需要 task.description 目标描述")

        pil_images = [self._to_pil_image(payload) for payload in rgb_images]
        model_inputs = self.processor(
            text=[description],
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {
            key: value.to(self.device) for key, value in model_inputs.items()
        }
        with self.torch.no_grad():
            outputs = self.model(**model_inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            image_features = image_features / image_features.norm(
                dim=1, keepdim=True
            )
            text_features = text_features / text_features.norm(
                dim=1, keepdim=True
            )
            scores = (image_features @ text_features.T)[:, 0]

        score_values = [float(value) for value in scores.detach().cpu().tolist()]
        ranked_indices = sorted(
            range(len(score_values)),
            key=lambda index: score_values[index],
            reverse=True,
        )
        top_index = ranked_indices[0]
        selected_rank = 0
        action = CAMERA_ACTIONS[top_index]
        selected_score = score_values[top_index]

        if selected_score >= self.stop_threshold:
            action = PolicyAction.Stop
        else:
            if self._is_lateral_reversal(self.previous_action, action):
                selected_rank = min(1, len(ranked_indices) - 1)
                action = CAMERA_ACTIONS[ranked_indices[selected_rank]]
                selected_score = score_values[ranked_indices[selected_rank]]

            nearest_depth_m = self._nearest_depth_metres(depth_images)
            if (
                action is PolicyAction.MoveDown
                and nearest_depth_m <= self.min_descend_clearance_m
            ):
                selected_rank = self._next_non_descend_rank(
                    ranked_indices, selected_rank + 1
                )
                action = CAMERA_ACTIONS[ranked_indices[selected_rank]]
                selected_score = score_values[ranked_indices[selected_rank]]

            if selected_score >= self.stop_threshold:
                action = PolicyAction.Stop

        redirected_action = self._redirect_out_of_bounds(action, observation)
        self.previous_action = redirected_action
        return redirected_action

    @staticmethod
    def _to_pil_image(payload):
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("CLIP-H 需要 Pillow") from error
        if isinstance(payload, (bytes, bytearray, memoryview)):
            with Image.open(io.BytesIO(bytes(payload))) as image:
                return image.convert("RGB")
        array = np.asarray(payload, dtype=np.uint8)
        return Image.fromarray(array).convert("RGB")

    @staticmethod
    def _nearest_depth_metres(depth_images: Sequence[Any]) -> float:
        nearest_values = []
        for depth_image in depth_images:
            depth = np.asarray(depth_image)
            if depth.size == 0:
                continue
            if depth.dtype == np.uint8:
                nearest_values.append(float(np.min(depth)) / 255.0 * 100.0)
            else:
                nearest_values.append(float(np.min(depth.astype(np.float32))))
        return min(nearest_values, default=float("inf"))

    @staticmethod
    def _is_lateral_reversal(
        previous: Optional[PolicyAction], current: PolicyAction
    ) -> bool:
        return (previous, current) in (
            (PolicyAction.MoveLeft, PolicyAction.MoveRight),
            (PolicyAction.MoveRight, PolicyAction.MoveLeft),
        )

    @staticmethod
    def _next_non_descend_rank(
        ranked_indices: Sequence[int], start_rank: int
    ) -> int:
        for rank in range(start_rank, len(ranked_indices)):
            if CAMERA_ACTIONS[ranked_indices[rank]] is not PolicyAction.MoveDown:
                return rank
        return 0

    def _redirect_out_of_bounds(
        self, action: PolicyAction, observation: Mapping[str, Any]
    ) -> PolicyAction:
        if action not in (
            PolicyAction.MoveForward,
            PolicyAction.MoveLeft,
            PolicyAction.MoveRight,
        ):
            return action
        x, y = [float(value) for value in observation["pose"][:2]]
        yaw = yaw_from_quaternion(observation["pose"][3:])
        if action is PolicyAction.MoveForward:
            dx, dy = math.cos(yaw), math.sin(yaw)
        elif action is PolicyAction.MoveLeft:
            dx, dy = -math.cos(yaw + math.pi / 2.0), -math.sin(
                yaw + math.pi / 2.0
            )
        else:
            dx, dy = math.cos(yaw + math.pi / 2.0), math.sin(
                yaw + math.pi / 2.0
            )
        next_x = x + dx * self.horizontal_step_m
        next_y = y + dy * self.horizontal_step_m
        start_x, start_y, _ = self.start_position
        if (
            abs(next_x - start_x) > self.boundary_radius_m
            or abs(next_y - start_y) > self.boundary_radius_m
        ):
            return PolicyAction.TurnLeft
        return action
