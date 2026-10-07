from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from ..data.datasets import SYSTEM_PROMPT
from ..evaluation.metrics import parse_json


def predict(model_path: str, media: str, instruction: str, media_type: str = "video", fps: float = 2.0) -> dict:
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    media_content = {"type": media_type, media_type: str(Path(media))}
    if media_type == "video":
        media_content["fps"] = fps
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [media_content, {"type": "text", "text": instruction}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated = output[:, inputs["input_ids"].shape[1]:]
    return parse_json(processor.batch_decode(generated, skip_special_tokens=True)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--media", required=True)
    parser.add_argument("--media-type", choices=["image", "video"], default="video")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--instruction", default="检测所有目标，并输出视频级类别和归一化边界框。")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = predict(args.model, args.media, args.instruction, args.media_type, args.fps)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
