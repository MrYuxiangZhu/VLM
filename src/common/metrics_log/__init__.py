"""训练指标持久化与分析模块（独立于业务代码，可拖到任意 HF Trainer 项目复用）。

- MetricsCallback : 训练/评估指标实时落盘（jsonl + csv + summary）
- load_metrics    : 读取在线 metrics.jsonl
- load_log        : 解析离线 stdout 日志中的 dict 行
- split_train_eval: 拆分 train/eval 长表
- analyze         : 训练健康度规则校验，返回 summary dict
- format_summary  : summary dict -> 多行可读文本
"""
from .analyzer import analyze, format_summary
from .callback import MetricsCallback
from .parser import load_log, load_metrics, split_train_eval

__all__ = [
    "MetricsCallback",
    "load_log",
    "load_metrics",
    "split_train_eval",
    "analyze",
    "format_summary",
]