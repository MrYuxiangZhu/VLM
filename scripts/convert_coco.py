from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def convert(annotation: str, image_root: str, output: str, category_names: str = "") -> None:
    source = json.loads(Path(annotation).read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in source["categories"]}
    selected = set(category_names.split(",")) if category_names else set(categories.values())
    image_map = {item["id"]: item for item in source["images"]}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for item in source["annotations"]:
        label = categories[item["category_id"]]
        if label not in selected or item.get("iscrowd", 0):
            continue
        image = image_map[item["image_id"]]
        x, y, width, height = item["bbox"]
        grouped[item["image_id"]].append({
            "label": label,
            "score": 1.0,
            "bbox": [x / image["width"], y / image["height"], (x + width) / image["width"], (y + height) / image["height"]],
        })
    with Path(output).open("w", encoding="utf-8") as file:
        for image_id, image in image_map.items():
            detections = grouped.get(image_id, [])
            labels = [item["label"] for item in detections]
            record = {
                "video": image["file_name"],
                "instruction": f"检测图像中的目标，从 [{', '.join(sorted(selected))}] 中选择图像级类别。",
                "output": {"video_class": labels[0] if labels else "background", "detections": detections},
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COCO detection JSON to the common VLM JSONL format")
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--image-root", required=True, help="COCO train2017 or val2017 directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--categories", default="", help="Optional comma-separated category whitelist")
    args = parser.parse_args()
    convert(args.annotation, args.image_root, args.output, args.categories)


if __name__ == "__main__":
    main()
