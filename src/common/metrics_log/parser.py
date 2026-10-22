"""离线解析器：把 Trainer stdout 里的 dict 行 / 在线 metrics.jsonl 还原成结构化表。

用法：
  df = load_log("outputs/logs/train_lora.log")        # 从纯文本日志解析
  df = load_metrics("outputs/x/metrics/metrics.jsonl") # 从在线 jsonl 读取
  train_df, eval_df = split_train_eval(df)

依赖 pandas；解析失败/无指标时返回空 DataFrame，不抛异常。
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd

# 标准 python-dict 行：{'loss': 2.25, ...}
_DICT_RE = re.compile(r"\{[^{}]*\}")
# tqdm 进度条: {'loss': 2.25}%|...   -> 前缀数字
_TQDM_PREFIX_RE = re.compile(r"^\s*\d+(?:\.\d+)?%\|")


def _extract_dicts(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = _TQDM_PREFIX_RE.sub("", line).strip()
        for m in _DICT_RE.finditer(line):
            try:
                d = ast.literal_eval(m.group())
            except (SyntaxError, ValueError):
                continue
            if isinstance(d, dict) and d:
                out.append(d)
    return out


def _classify(df: pd.DataFrame) -> pd.DataFrame:
    has_loss = df.get("loss", pd.Series([None] * len(df), index=df.index)).notna()
    has_eval = df.get("eval_loss", pd.Series([None] * len(df), index=df.index)).notna()
    df["type"] = "other"
    df.loc[has_eval, "type"] = "eval"
    df.loc[has_loss & ~has_eval, "type"] = "train"
    return df


def load_log(path: str | Path) -> pd.DataFrame:
    """从纯文本训练日志解析指标行（tqdm 行/裸 dict 行均可）。"""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    rows = _extract_dicts(p.read_text(encoding="utf-8", errors="ignore"))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df = df.sort_values("epoch").reset_index(drop=True)
    df["step"] = range(len(df))
    return _classify(df)


def load_metrics(path: str | Path) -> pd.DataFrame:
    """读取在线 MetricsCallback 写出的 metrics.jsonl。"""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    rows = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df = df.sort_values("epoch").reset_index(drop=True)
    return _classify(df)


def split_train_eval(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把混合表拆成 train / eval 两张，并按已知列排序。"""
    if df.empty or "type" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    train_cols = ["step", "epoch", "loss", "grad_norm", "learning_rate", "wall_time"]
    eval_cols = ["step", "epoch", "eval_loss", "eval_runtime",
                 "eval_samples_per_second", "eval_steps_per_second", "wall_time"]
    train_df = df[df["type"] == "train"]
    eval_df = df[df["type"] == "eval"]
    train_df = train_df[[c for c in train_cols if c in train_df.columns]]
    eval_df = eval_df[[c for c in eval_cols if c in eval_df.columns]]
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)