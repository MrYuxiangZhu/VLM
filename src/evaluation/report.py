from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METRICS = [
    ("detection_f1", "Detection F1", True),
    ("classification_accuracy", "Classification Accuracy", True),
    ("json_valid_rate", "Valid JSON Rate", True),
    ("mean_iou", "Mean IoU", True),
]


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"metrics file must contain a JSON object: {path}")
    return data


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def _delta(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "N/A"
    return f"{(after - before) * 100:+.2f} pp"


def _metric_row(before: dict[str, Any], after: dict[str, Any], key: str, label: str) -> str:
    old, new = _number(before.get(key)), _number(after.get(key))
    direction = "up" if old is not None and new is not None and new >= old else "down"
    return f"""<tr><td>{html.escape(label)}</td><td>{_pct(old)}</td><td>{_pct(new)}</td>
    <td class=\"{direction}\">{html.escape(_delta(old, new))}</td></tr>"""


def _bar(label: str, before: float | None, after: float | None) -> str:
    old_width = max(0.0, min(100.0, (before or 0.0) * 100))
    new_width = max(0.0, min(100.0, (after or 0.0) * 100))
    return f"""<div class=\"bar-row\"><div class=\"bar-label\">{html.escape(label)}</div>
    <div class=\"bars\"><div class=\"bar before\" style=\"width:{old_width:.2f}%\"><span>{_pct(before)}</span></div>
    <div class=\"bar after\" style=\"width:{new_width:.2f}%\"><span>{_pct(after)}</span></div></div></div>"""


def build_report(before: dict[str, Any], after: dict[str, Any], before_name: str, after_name: str) -> str:
    rows = "".join(_metric_row(before, after, key, label) for key, label, _ in METRICS)
    bars = "".join(_bar(label, _number(before.get(key)), _number(after.get(key))) for key, label, _ in METRICS)
    sample_before = before.get("samples", "N/A")
    sample_after = after.get("samples", "N/A")
    generated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    return f"""<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>模型微调前后评估报告</title><style>
:root{{--bg:#f4f7fb;--card:#fff;--ink:#172033;--muted:#667085;--line:#e5eaf2;--blue:#3976f6;--green:#18a36b;--red:#e05252}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:42px 22px 64px}}h1{{font-size:32px;margin:0 0 8px}}h2{{font-size:20px;margin:0 0 18px}}h3{{font-size:16px;margin:0 0 6px}}.sub{{color:var(--muted);margin-bottom:28px}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 4px 16px #162b4d0a}}.value{{font-size:27px;font-weight:700;margin-top:5px}}.muted{{color:var(--muted);font-size:13px}}.section{{margin-top:18px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:13px 10px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted);font-weight:600;font-size:13px}}.up{{color:var(--green);font-weight:700}}.down{{color:var(--red);font-weight:700}}.legend{{display:flex;gap:18px;color:var(--muted);font-size:13px;margin-bottom:12px}}.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}}.before-dot{{background:#aab5c7}}.after-dot{{background:var(--blue)}}.bar-row{{display:grid;grid-template-columns:210px 1fr;gap:14px;align-items:center;margin:16px 0}}.bar-label{{font-weight:600}}.bars{{display:flex;flex-direction:column;gap:5px}}.bar{{height:23px;border-radius:5px;min-width:2px;color:#fff;font-size:12px;padding:1px 8px;text-align:right;white-space:nowrap}}.before{{background:#aab5c7}}.after{{background:var(--blue)}}.note{{background:#eef4ff;border-left:4px solid var(--blue);padding:13px 16px;color:#33415c;border-radius:6px}}@media(max-width:760px){{.grid{{grid-template-columns:repeat(2,1fr)}}.bar-row{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>模型微调前后评估报告</h1><div class=\"sub\">生成时间：{html.escape(generated)} · 同一验证集对比</div>
<div class=\"grid\"><div class=\"card\"><div class=\"muted\">微调前模型</div><div class=\"value\">{html.escape(before_name)}</div><div class=\"muted\">样本数：{html.escape(str(sample_before))}</div></div>
<div class=\"card\"><div class=\"muted\">微调后模型</div><div class=\"value\">{html.escape(after_name)}</div><div class=\"muted\">样本数：{html.escape(str(sample_after))}</div></div>
<div class=\"card\"><div class=\"muted\">Detection F1 变化</div><div class=\"value\">{html.escape(_delta(_number(before.get('detection_f1')), _number(after.get('detection_f1'))))}</div></div>
<div class=\"card\"><div class=\"muted\">分类准确率变化</div><div class=\"value\">{html.escape(_delta(_number(before.get('classification_accuracy')), _number(after.get('classification_accuracy'))))}</div></div></div>
<section class=\"card section\"><h2>核心指标对比</h2><table><thead><tr><th>指标</th><th>微调前</th><th>微调后</th><th>变化</th></tr></thead><tbody>{rows}</tbody></table></section>
<section class=\"card section\"><h2>指标可视化</h2><div class=\"legend\"><span><i class=\"dot before-dot\"></i>微调前</span><span><i class=\"dot after-dot\"></i>微调后</span></div>{bars}</section>
<section class=\"card section\"><h2>评估解读</h2><div class=\"note\">正值表示微调后指标提升，负值表示下降。请确保两次评估使用相同的验证集、类别集合、IoU 阈值、采样 FPS 和推理参数，否则指标变化不能直接归因于微调。</div></section>
<section class=\"card section\"><h2>原始评估结果</h2><pre>{html.escape(json.dumps({'before': before, 'after': after}, ensure_ascii=False, indent=2))}</pre></section>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate before/after fine-tuning HTML report")
    parser.add_argument("--before", required=True, help="Baseline metrics JSON")
    parser.add_argument("--after", required=True, help="Fine-tuned metrics JSON")
    parser.add_argument("--output", required=True, help="HTML output path")
    parser.add_argument("--before-name", default="Baseline")
    parser.add_argument("--after-name", default="Fine-tuned")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(_load(args.before), _load(args.after), args.before_name, args.after_name), encoding="utf-8")
    print(f"HTML report written to: {output}")


if __name__ == "__main__":
    main()
