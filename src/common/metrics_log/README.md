# metrics_log — 训练指标持久化与自动分析

独立于业务代码，拖进任何 HF Trainer 项目即可用，无第三方新增依赖（仅 `pandas`）。

## 在线（训练时自动落盘）

`src/training/train.py` 已接入：

```python
from src.common.metrics_log import MetricsCallback
Trainer(..., callbacks=[MetricsCallback(cfg["output_dir"], logger)])
```

产物在 `output_dir/metrics/`：

| 文件 | 内容 |
|---|---|
| `metrics.jsonl` | 每条 log 一行 JSON，含 wall_time / step / epoch / type(train\|eval) |
| `train_metrics.csv` | 训练长表：step, epoch, loss, grad_norm, learning_rate |
| `eval_metrics.csv` | 评估长表：step, epoch, eval_loss, eval_runtime, 吞吐 |
| `run_summary.json` | 训练结束的全程汇总（自动跑 analyzer） |

## 离线（已有 stdout 日志直接分析）

```python
from src.common.metrics_log import load_log, split_train_eval, analyze, format_summary

df = load_log("/home/users/yuxiang.zhu/vlm/logs/train_lora.log")
train_df, eval_df = split_train_eval(df)
print(format_summary(analyze(train_df, eval_df)))
```

能把 tqdm 进度条里嵌入的 `{'loss': ...}` 行一并还原。

## analyzer 内置健康规则

- train loss 收敛/末端 spike
- grad_norm 爆炸(>5)/消失(<1e-4) 检测（自动剔除冷启动首点）
- learning_rate 调度形状（peak / final / 是否未衰减）
- eval_loss 单调性与 train/eval gap（过拟合）
- eval_runtime 占 wall-time 比，>15% 给出优化建议