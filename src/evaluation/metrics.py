from __future__ import annotations

import json
from typing import Any


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(area_left + area_right - intersection, 1e-8)


def detection_f1(pred: dict[str, Any], target: dict[str, Any], threshold: float = 0.5) -> float:
    unmatched = list(target.get("detections", []))
    true_positive = 0
    for item in pred.get("detections", []):
        match = next((x for x in unmatched if x.get("label") == item.get("label") and iou(x["bbox"], item["bbox"]) >= threshold), None)
        if match is not None:
            unmatched.remove(match)
            true_positive += 1
    precision = true_positive / max(len(pred.get("detections", [])), 1)
    recall = true_positive / max(len(target.get("detections", [])), 1)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def parse_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model output does not contain a JSON object")
    return json.loads(text[start : end + 1])
