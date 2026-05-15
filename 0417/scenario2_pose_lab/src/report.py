from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd


def write_markdown_report(title: str, summary: dict, single_df: pd.DataFrame | None, out_path: str | Path, chart_paths: Iterable[str] | None = None) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# {title}\n")
    lines.append("## Summary\n")
    for k, v in summary.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    if chart_paths:
        lines.append("## Charts\n")
        for p in chart_paths:
            lines.append(f"- {p}")
        lines.append("")

    if single_df is not None and not single_df.empty:
        lines.append("## Window Results (head)\n")
        try:
            lines.append(single_df.head(20).to_markdown(index=False))
        except Exception:
            lines.append(single_df.head(20).to_string(index=False))
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def write_batch_report(batch_df: pd.DataFrame, out_dir: str | Path) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "batch_report.md"

    lines = ["# Batch Experiment Report\n"]
    lines.append(f"- total_runs: {len(batch_df)}")

    key_metrics = [
        "first_result_latency_ms",
        "ai_avg_latency_ms",
        "ai_p95_latency_ms",
        "payload_avg_kb",
        "tcp_est_send_ms_avg",
        "pose_detect_rate_avg",
        "hand_detect_rate_avg",
        "face_detect_rate_avg",
        "keypoint_completeness_avg",
        "stability_score_avg",
        "user_smoothness_score_avg",
        "user_visual_quality_score_avg",
    ]
    for c in key_metrics:
        if c in batch_df.columns:
            lines.append(f"- mean_{c}: {batch_df[c].mean():.4f}")
    lines.append("")

    try:
        lines.append("## Results\n")
        lines.append(batch_df.to_markdown(index=False))
    except Exception:
        lines.append(batch_df.to_string(index=False))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)
