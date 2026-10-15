"""Qwen3-VL example policy using front RGB and native uint8 depth."""

import io
import json
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .base import NavigationPolicy, PolicyAction
from .factory import PolicyFactory


# Preserve the recovered base_rgbd prompt byte for byte. Changing the prompt
# changes the navigation policy even when the checkpoint and simulator match.
SYSTEM_PROMPT = """You control a UAV using its current front-facing RGB image and depth image.
Choose exactly one action that best moves toward the described target while avoiding visible obstacles. The first image is the front-facing RGB image. The second image is the aligned front-facing depth image. Each depth pixel is an unsigned 8-bit value from 0 to 255, with distance_m = pixel_value / 255 * 100; 0 is the nearest encoded distance and 255 is 100 metres or farther. The action must be one of: moveforward, moveleft, moveright, moveup, movedown, turnleft, turnright, stop. Return exactly one valid JSON object in the form {"action":"moveforward"}. Do not include Markdown, explanations, or any other keys. Output stop only when the UAV is within 3 metres of the target; stopping farther away is not successful."""

_ACTION_ALIASES = {
    "moveforward": "forward",
    "moveleft": "left", "moveright": "right",
    "moveup": "ascend", "movedown": "descend",
    "turnleft": "rotl", "turnright": "rotr",
}


@PolicyFactory.register("qwen3vl")
class Qwen3VLPolicy(NavigationPolicy):
    """One local Qwen3-VL checkpoint behind the public NavigationPolicy API."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 32,
        max_pixels_per_image: int = 200704,
        history_length: int = 10,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoProcessor

        model_class = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
        if model_class is None:
            raise RuntimeError(
                "This Transformers version does not support Qwen3-VL; "
                "install a version compatible with the checkpoint."
            )
        self.torch = torch
        self.device = str(device)
        self.max_new_tokens = int(max_new_tokens)
        self.history_length = int(history_length)
        if self.history_length < 0:
            raise ValueError("history_length must be non-negative")
        if self.max_new_tokens < 1 or int(max_pixels_per_image) < 1:
            raise ValueError("token and pixel limits must be positive")
        model_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, max_pixels=int(max_pixels_per_image)
        )
        image_processor = self.processor.image_processor
        processor_size = image_processor.size
        if isinstance(processor_size, Mapping):
            if not {"height", "width"}.issubset(processor_size):
                image_processor.size = {
                    **processor_size, "longest_edge": int(max_pixels_per_image)
                }
        elif hasattr(processor_size, "longest_edge"):
            processor_size.longest_edge = int(max_pixels_per_image)
        image_processor.max_pixels = int(max_pixels_per_image)
        self.model = model_class.from_pretrained(
            model_name_or_path, dtype=model_dtype, low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()
        self.last_raw_response = ""
        self.last_valid = True

    def reset(self, observation: Mapping[str, Any]) -> None:
        self.last_raw_response = ""
        self.last_valid = True

    def _history(self, observation: Mapping[str, Any]):
        history = list(observation.get("trajectory", []))[-self.history_length :]
        compact = []
        for item in history:
            state = item.get("sensors", {}).get("state", {})
            compact.append({
                "action": item.get("action", ""),
                "position": [round(float(v), 2) for v in state.get("position", [])],
            })
        return compact

    @staticmethod
    def _images(observation: Mapping[str, Any]):
        rgb, depth = observation.get("rgb"), observation.get("depth")
        if not rgb or not depth or len(rgb) != 1 or len(depth) != 1:
            raise ValueError("Qwen3-VL example requires one front RGB and depth image")
        with Image.open(io.BytesIO(bytes(rgb[0]))) as source:
            front_rgb = source.convert("RGB")
        depth_array = np.asarray(depth[0])
        if depth_array.dtype != np.uint8 or depth_array.ndim != 2:
            raise TypeError("Qwen3-VL example requires native uint8 depth")
        front_depth = Image.fromarray(depth_array, mode="L").convert("RGB")
        if front_depth.size != front_rgb.size:
            front_depth = front_depth.resize(front_rgb.size, Image.Resampling.NEAREST)
        return front_rgb, front_depth

    @staticmethod
    def _parse_action(response: str) -> PolicyAction:
        try:
            payload = json.loads(response)
            action = str(payload["action"]).strip().lower()
            return PolicyAction(_ACTION_ALIASES.get(action, action))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def forward(self, observation: Mapping[str, Any]) -> PolicyAction:
        front_rgb, front_depth = self._images(observation)
        task = observation.get("task", {})
        goal = {
            "target_name": str(task.get("goal_true_name", task.get("category", ""))),
            "category": str(task.get("category", "")),
            "landmark": str(task.get("Landmark", "")),
            "description": str(task.get("description", "")),
        }
        state = {
            "current_pose_xyzw": [round(float(v), 3) for v in observation["pose"]],
            "step": int(observation.get("step", 0)),
            "recent_history": self._history(observation),
        }
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "text", "text":
                    "Task input (target object, landmark reference, and description):\n"
                    + json.dumps(goal, ensure_ascii=False)
                    + "\nCurrent navigation state and recent actions:\n"
                    + json.dumps(state, ensure_ascii=False)
                    + "\nTask memory from previous steps (empty at task start):\n(empty)"
                    + "\nObservation images follow in the documented order."},
                {"type": "image", "image": front_rgb},
                {"type": "image", "image": front_depth},
                {"type": "text", "text": "Choose the next action. Output JSON only."},
            ]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, use_cache=True,
            )
        trimmed = output_ids[:, inputs.input_ids.shape[1]:]
        self.last_raw_response = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        action = self._parse_action(self.last_raw_response)
        self.last_valid = action is not None
        return action or PolicyAction.Stop
