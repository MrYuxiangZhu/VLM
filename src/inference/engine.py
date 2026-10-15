from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from ..data.datasets import SYSTEM_PROMPT
from ..evaluation.metrics import parse_json


class VisionPredictor:
    """Load a model once and reuse it for batch evaluation or repeated inference."""

    def __init__(self, model_path: str, max_new_tokens: int = 512, device: str = "auto") -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        ).eval()

    def predict(self, media: str, instruction: str, media_type: str = "video", fps: float = 2.0) -> dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        media_content: dict[str, Any] = {"type": media_type, media_type: str(Path(media))}
        if media_type == "video":
            media_content["fps"] = fps
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [media_content, {"type": "text", "text": instruction}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt", "padding": True}
        if image_inputs:
            kwargs["images"] = image_inputs
        if video_inputs:
            kwargs["videos"] = video_inputs
        inputs = self.processor(**kwargs)
        inputs = {key: value.to(self.model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1]:]
        text_output = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return parse_json(text_output)
