from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Detection(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)
    bbox: list[float]

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[float]) -> list[float]:
        if len(value) != 4 or any(not 0 <= x <= 1 for x in value):
            raise ValueError("bbox must be [x1, y1, x2, y2] normalized to [0, 1]")
        if value[2] < value[0] or value[3] < value[1]:
            raise ValueError("bbox coordinates must be ordered")
        return value


class Annotation(BaseModel):
    video_class: str
    detections: list[Detection] = Field(default_factory=list)


class Record(BaseModel):
    video: str
    instruction: str
    output: Annotation | dict[str, Any]

    @field_validator("output", mode="before")
    @classmethod
    def normalize_output(cls, value: Any) -> Any:
        return value if isinstance(value, dict) else json.loads(value)


def load_records(path: str | Path) -> list[Record]:
    records: list[Record] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if line.strip():
                try:
                    records.append(Record.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"Invalid annotation at {path}:{line_number}: {exc}") from exc
    return records
