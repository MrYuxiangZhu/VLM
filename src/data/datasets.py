from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset, Dataset

from ..common.schema import Record, load_records

SYSTEM_PROMPT = """你是视觉目标检测与分类模型。只输出合法 JSON，不要输出 Markdown。格式必须是：
{"video_class": "类别", "detections": [{"label": "类别", "score": 0.0, "bbox": [x1, y1, x2, y2]}]}
其中 bbox 是相对于图像或视频宽高归一化的 [左上x, 左上y, 右下x, 右下y]。"""


class VisionDataset(ABC, Dataset):
    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]: ...


class JsonlVisionDataset(VisionDataset):
    def __init__(self, annotation_file: str, media_root: str = "", media_type: str = "video", fps: float = 2.0) -> None:
        self.records = load_records(annotation_file)
        self.media_root = Path(media_root) if media_root else Path(annotation_file).parent
        self.media_type = media_type
        self.fps = fps

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        media_path = Path(record.video)
        if not media_path.is_absolute():
            media_path = self.media_root / media_path
        media = {"type": self.media_type, self.media_type: str(media_path)}
        if self.media_type == "video":
            media["fps"] = self.fps
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [media, {"type": "text", "text": record.instruction}]},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(record.output, ensure_ascii=False)}]},
        ]
        return {"messages": messages, "media": str(media_path), "media_type": self.media_type}


class CocoDataset(JsonlVisionDataset):
    """COCO is converted to the common JSONL contract before training."""

    pass


def build_dataset(spec: dict[str, Any] | list[dict[str, Any]]) -> VisionDataset:
    if isinstance(spec, list):
        if not spec:
            raise ValueError("dataset list must not be empty")
        return ConcatDataset([build_dataset(item) for item in spec])
    dataset_type = spec.get("type", "jsonl")
    if dataset_type == "jsonl":
        return JsonlVisionDataset(spec["annotations"], spec.get("media_root", ""), spec.get("media_type", "video"), spec.get("fps", 2.0))
    if dataset_type == "coco":
        return CocoDataset(spec["annotations"], spec["media_root"], "image", 0.0)
    raise ValueError(f"Unknown dataset type: {dataset_type}")
