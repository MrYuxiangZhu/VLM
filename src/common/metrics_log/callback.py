"""原生 TrainerCallback：把训练/评估指标实时持久化到 output_dir/metrics/。

产物：
  metrics.jsonl      每条 log 一行 JSON，含 wall_time / step / epoch / type(train|eval)
  train_metrics.csv  训练日志长表（step, epoch, loss, grad_norm, learning_rate）
  eval_metrics.csv   评估日志长表（step, epoch, eval_loss, eval_runtime, ...）
  run_summary.json   训练结束时生成的全程汇总

对接 HF Trainer：在 Trainer(...) 里加 callbacks=[MetricsCallback(output_dir, logger)] 即可。
任何支持 `Trainer.log()` 的 transformers 版本都兼容；不引入第三方依赖。
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


class MetricsCallback(TrainerCallback):
    """把 on_log 收到的 dict 追加写入 jsonl，并按 train/eval 拆成两张 csv。"""

    def __init__(self, output_dir: str, logger: Any | None = None) -> None:
        self.metrics_dir = Path(output_dir) / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.metrics_dir / "metrics.jsonl"
        self.train_csv = self.metrics_dir / "train_metrics.csv"
        self.eval_csv = self.metrics_dir / "eval_metrics.csv"
        self.summary_path = self.metrics_dir / "run_summary.json"
        self.logger = logger
        self._t0 = time.time()
        # 断点续训时保留旧记录，只在文件头不存在时写 csv header
        self._train_header = not self.train_csv.exists()
        self._eval_header = not self.eval_csv.exists()

    # ------------------------------------------------------------------ #
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        if not state.is_world_process_zero or not logs:
            return
        record = {
            "wall_time": round(time.time() - self._t0, 3),
            "log_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step": state.global_step,
            "epoch": logs.get("epoch"),
            "type": "eval" if "eval_loss" in logs else "train",
        }
        record.update({k: _to_jsonable(v) for k, v in logs.items() if k != "epoch"})

        # JSONL 全量追加
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # CSV 分表
        if record["type"] == "eval":
            self._append_csv(self.eval_csv, record, "_eval_header",
                             ["step", "epoch", "eval_loss", "eval_runtime",
                              "eval_samples_per_second", "eval_steps_per_second"])
            self._eval_header = False
        else:
            self._append_csv(self.train_csv, record, "_train_header",
                             ["step", "epoch", "loss", "grad_norm", "learning_rate"])
            self._train_header = False

    @staticmethod
    def _append_csv(path: Path, record: dict[str, Any], header_flag: str, columns: list[str]) -> None:
        write_header = not path.exists() or (header_flag == "_train_header" and path.stat().st_size == 0)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(record)

    # ------------------------------------------------------------------ #
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **_: Any,
    ) -> None:
        if not state.is_world_process_zero:
            return
        try:
            from .parser import load_metrics, split_train_eval
            from .analyzer import analyze
            df = load_metrics(self.jsonl_path)
            train_df, eval_df = split_train_eval(df)
            summary = analyze(train_df, eval_df)
            summary["total_wall_time_sec"] = round(time.time() - self._t0, 3)
            summary["total_steps"] = state.global_step
            self.summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if self.logger:
                for line in summary["suggestions"]:
                    self.logger.info("[metrics] %s", line)
        except Exception as exc:  # 汇总失败不影响训练主流程
            if self.logger:
                self.logger.warning("metrics summary failed: %s", exc)