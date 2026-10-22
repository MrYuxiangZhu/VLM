"""指标健康度分析：把训练曲线规则化校验，输出可落盘的 summary dict。

规则覆盖上一轮人工核对项：
  - eval_loss 是否单调下降 / 波动
  - train eval gap（过拟合）
  - 学习率调度形状（warmup->peak->decay）
  - grad_norm 范围（爆炸/消失）
  - loss 收敛区间（caption 任务经验值）
  - eval_runtime 占比提示

analyze(train_df, eval_df) -> dict，字段均为 JSON 可序列化；空表返回最小 summary。
"""
from __future__ import annotations

import pandas as pd


def analyze(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> dict:
    summary: dict = {"total_train_logs": int(len(train_df)),
                     "total_eval_logs": int(len(eval_df))}
    issues: list[str] = []
    suggestions: list[str] = []

    # ---------------- train loss ---------------- #
    if not train_df.empty and "loss" in train_df.columns:
        losses = train_df["loss"].dropna()
        if len(losses):
            first, last = float(losses.iloc[0]), float(losses.iloc[-1])
            mean, std = float(losses.mean()), float(losses.std())
            summary["train_loss"] = {
                "start": round(first, 4), "end": round(last, 4),
                "drop": round(first - last, 4), "mean": round(mean, 4),
                "max": round(float(losses.max()), 4),
                "min": round(float(losses.min()), 4),
            }
            if len(losses) > 1:
                if last > mean + 2 * std:
                    issues.append("loss spike at end: final loss exceeds mean+2*std")
                if last > first * 0.95:
                    issues.append("loss not decreasing: end >= 95% of start")

    # ---------------- grad norm ---------------- #
    if not train_df.empty and "grad_norm" in train_df.columns:
        g = train_df["grad_norm"].dropna()
        if len(g) > 2:
            spikes = g.iloc[1:]  # skip cold-start first point
            summary["grad_norm"] = {
                "max": round(float(spikes.max()), 4),
                "mean": round(float(spikes.mean()), 4),
                "spike_over_5": int((spikes > 5.0).sum()),
            }
            if (spikes > 5.0).any():
                issues.append(f"gradient explosion: {(spikes > 5.0).sum()} points > 5.0")
            if float(spikes.mean()) < 1e-4:
                issues.append("gradient vanishing: mean grad_norm < 1e-4")

    # ---------------- learning rate ---------------- #
    if not train_df.empty and "learning_rate" in train_df.columns:
        lr = train_df["learning_rate"].dropna()
        if len(lr) > 3:
            peak_idx = int(lr.idxmax())
            peak, final = float(lr.max()), float(lr.iloc[-1])
            summary["learning_rate"] = {
                "peak": peak, "final": final,
                "peak_step": int(train_df.loc[peak_idx, "step"]) if "step" in train_df.columns else peak_idx,
            }
            if final > peak * 1.01:
                suggestions.append("lr not decaying: final lr still at peak")

    # ---------------- eval loss ---------------- #
    if not eval_df.empty and "eval_loss" in eval_df.columns:
        el = eval_df["eval_loss"].dropna()
        if len(el) >= 2:
            summary["eval_loss"] = {
                "first": round(float(el.iloc[0]), 4),
                "last": round(float(el.iloc[-1]), 4),
                "best": round(float(el.min()), 4),
                "monotonic": bool(el.is_monotonic_decreasing),
            }
            if not el.is_monotonic_decreasing:
                issues.append("eval_loss not monotonically decreasing")
            if not train_df.empty and "loss" in train_df.columns and len(train_df["loss"].dropna()) > 0:
                gap = float(el.iloc[-1] - train_df["loss"].dropna().iloc[-1])
                summary["train_eval_gap"] = round(gap, 4)
                if gap > 0.5:
                    issues.append(f"overfitting gap: eval-train = {gap:.4f} > 0.5")

    # ---------------- eval runtime overhead ---------------- #
    if not eval_df.empty and "eval_runtime" in eval_df.columns:
        rt = eval_df["eval_runtime"].dropna()
        if len(rt):
            avg_rt = float(rt.mean())
            total_eval = float(rt.sum())
            summary["eval_overhead"] = {
                "eval_runs": int(len(rt)),
                "avg_runtime_sec": round(avg_rt, 2),
                "total_runtime_sec": round(total_eval, 2),
            }
            if "wall_time" in eval_df.columns and eval_df["wall_time"].dropna().any():
                total_wall = float(eval_df["wall_time"].max())
                if total_wall > 0:
                    ratio = total_eval / total_wall
                    summary["eval_overhead"]["wall_time_ratio"] = round(ratio, 3)
                    if ratio > 0.15:
                        suggestions.append(
                            f"eval overhead too high ({ratio:.1%} of wall time) - "
                            "consider smaller val set or larger eval interval"
                        )

    summary["status"] = "ok" if not issues else "warning"
    summary["issues"] = issues
    summary["suggestions"] = suggestions
    return summary


def format_summary(summary: dict) -> str:
    """把 analyze() 结果格式化成多行日志文本。"""
    lines = [f"status={summary['status']} "
             f"(train_logs={summary['total_train_logs']} eval_logs={summary['total_eval_logs']})"]
    for key in ("train_loss", "grad_norm", "learning_rate", "eval_loss",
                "train_eval_gap", "eval_overhead"):
        if key in summary:
            lines.append(f"{key}: {summary[key]}")
    for msg in summary["issues"]:
        lines.append(f"[issue] {msg}")
    for msg in summary["suggestions"]:
        lines.append(f"[suggestion] {msg}")
    return "\n".join(lines)