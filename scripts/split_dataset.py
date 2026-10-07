from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = [line for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    random.Random(args.seed).shuffle(rows)
    split = max(1, int(len(rows) * args.val_ratio))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "val.jsonl").write_text("\n".join(rows[:split]) + "\n", encoding="utf-8")
    (output / "train.jsonl").write_text("\n".join(rows[split:]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
