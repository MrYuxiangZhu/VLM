from __future__ import annotations

import json
from typing import Any


def _valid_bbox(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(isinstance(x, (int, float)) for x in value) and value[0] <= value[2] and value[1] <= value[3]


def iou(left: list[float], right: list[float]) -> float:
    if not _valid_bbox(left) or not _valid_bbox(right):
        return 0.0
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(pred: dict[str, Any], target: dict[str, Any], threshold: float = 0.5) -> dict[str, float]:
    predictions = [item for item in pred.get("detections", []) if isinstance(item, dict)]
    unmatched = [item for item in target.get("detections", []) if isinstance(item, dict)]
    true_positive, ious = 0, []
    for item in sorted(predictions, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        candidates = [(index, iou(item.get("bbox"), target_item.get("bbox"))) for index, target_item in enumerate(unmatched) if target_item.get("label") == item.get("label")]
        if not candidates:
            continue
        index, overlap = max(candidates, key=lambda pair: pair[1])
        if overlap >= threshold:
            unmatched.pop(index)
            true_positive += 1
            ious.append(overlap)
    false_positive = len(predictions) - true_positive
    false_negative = len(unmatched)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0 if not unmatched else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "mean_iou": sum(ious) / len(ious) if ious else 0.0}


def detection_f1(pred: dict[str, Any], target: dict[str, Any], threshold: float = 0.5) -> float:
    return match_detections(pred, target, threshold)["f1"]


def parse_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model output does not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model output JSON must be an object")
    return value
