from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a concise Chinese latent-z power report.")
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _fmt(value, unit: str = "", digits: int = 2) -> str:
    if value is None:
        return "无法计算"
    return f"{float(value):.{digits}f}{unit}"


def _value(row: dict, key: str, fallback_key: str | None = None):
    value = row.get(key)
    if value is not None:
        return value
    if fallback_key is None:
        return None
    return row.get(fallback_key)


def _change(before_value, after_value):
    if before_value is None or after_value is None:
        return None
    if abs(float(before_value)) < 1.0e-12:
        return None
    return 100.0 * (float(after_value) - float(before_value)) / float(before_value)


def main() -> None:
    args = parse_args()
    data = json.loads(args.analysis_json.read_text())
    before = data["before"]
    after = data["after"]
    comparison = data["comparison"]

    verdict = "有效"
    saved = comparison["human_abs_power_saved_percent"]
    roi = comparison["assist_efficiency_human_saved_per_exo_abs_power"]
    if saved is None or saved <= 0:
        verdict = "未观察到人体功率降低"
    elif roi is not None and roi < 1.0:
        verdict = "人体功率降低，但机械助力效率低于 1"

    before_metabolic = _value(
        before,
        "human_lower_limb_metabolic_proxy_mean_w",
        "human_knee_abs_power_mean_w",
    )
    after_metabolic = _value(
        after,
        "human_lower_limb_metabolic_proxy_mean_w",
        "human_knee_abs_power_mean_w",
    )
    before_mechanical = _value(before, "human_lower_limb_mechanical_abs_power_mean_w")
    after_mechanical = _value(after, "human_lower_limb_mechanical_abs_power_mean_w")
    mechanical_change = _change(before_mechanical, after_mechanical)

    text = f"""# Latent-Z 助力功率对比

结论：**{verdict}**

| 指标 | Baseline | Assisted | 变化 |
|---|---:|---:|---:|
| 人体下肢代谢功率 proxy（髋+膝+踝） | {_fmt(before_metabolic, ' W')} | {_fmt(after_metabolic, ' W')} | {_fmt(comparison['human_abs_power_saved_w'], ' W')} saved / {_fmt(saved, '%')} |
| 人体下肢机械绝对功率（髋+膝+踝） | {_fmt(before_mechanical, ' W')} | {_fmt(after_mechanical, ' W')} | {_fmt(mechanical_change, '%')} |
| 外骨骼助力关节绝对功率 | {_fmt(before['exo_knee_abs_power_mean_w'], ' W')} | {_fmt(after['exo_knee_abs_power_mean_w'], ' W')} | - |
| 人机分开计总功率（代谢 proxy + 外骨骼机械） | {_fmt(before_metabolic, ' W')} | {_fmt(float(after_metabolic) + float(after['exo_knee_abs_power_mean_w']) if after_metabolic is not None else None, ' W')} | {_fmt(comparison['net_system_input_power_change_percent'], '%')} |
| 人机合计下肢绝对功率 | {_fmt(before['combined_knee_abs_power_mean_w'], ' W')} | {_fmt(after['combined_knee_abs_power_mean_w'], ' W')} | {_fmt(comparison['combined_abs_power_change_percent'], '%')} |

助力效率：**{_fmt(roi, '', 3)} W/W**  
含义：每 1 W 外骨骼助力关节绝对机械输出，节省多少 W 人体下肢（髋+膝+踝）代谢功率 proxy。

代谢 proxy：先抵消 XML passive joint forces（仿真稳定用关节刚度/阻尼），再计算正机械功率 / 0.25 + 负机械功率幅值 / 1.20。

数据：
- baseline: `{before['source']}`
- assisted: `{after['source']}`
- report json: `{args.analysis_json}`
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text)
    print(f"[INFO] Wrote concise report: {args.output}")


if __name__ == "__main__":
    main()
