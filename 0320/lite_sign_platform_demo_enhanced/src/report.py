from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd


def write_markdown_report(
    title: str,
    summary: dict,
    single_df: pd.DataFrame | None,
    out_path: str | Path,
    chart_paths: Iterable[str] | None = None,
) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# {title}\n")
    lines.append("## Executive Summary\n")

    for k, v in summary.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    if chart_paths:
        lines.append("## Charts\n")
        for p in chart_paths:
            lines.append(f"- {p}")
        lines.append("")

    if single_df is not None and not single_df.empty:
        lines.append("## Key Findings\n")
        try:
            if "ai_latency_ms" in single_df.columns:
                worst = single_df.sort_values("ai_latency_ms", ascending=False).iloc[0]
                lines.append(
                    f"- Worst latency window: #{int(worst['window_id'])}, latency={worst['ai_latency_ms']:.2f} ms"
                )
            if "baseline_similarity" in single_df.columns:
                best = single_df.sort_values("baseline_similarity", ascending=False).iloc[0]
                lines.append(
                    f"- Best quality window: #{int(best['window_id'])}, similarity={best['baseline_similarity']:.3f}"
                )
        except Exception:
            pass

        lines.append("")
        lines.append("## Window Results (head)\n")
        try:
            lines.append(single_df.head(20).to_markdown(index=False))
        except Exception:
            lines.append(single_df.head(20).to_string(index=False))
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def write_batch_report(batch_df: pd.DataFrame, out_dir: str | Path) -> str:
    """
    为批量实验结果生成 Markdown 报告。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "batch_report.md"

    lines: list[str] = []
    lines.append("# Batch Experiment Report\n")

    if batch_df is None or batch_df.empty:
        lines.append("No batch results available.\n")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return str(out_path)

    lines.append("## Overview\n")
    lines.append(f"- Total runs: {len(batch_df)}")

    metric_candidates = [
        "ai_avg_latency_ms",
        "ai_p95_latency_ms",
        "overlay_skew_avg_ms",
        "infer_avg_ms",
        "pose_detect_rate_avg",
        "hand_detect_rate_avg",
        "keypoint_completeness_avg",
        "baseline_similarity_avg",
    ]

    for col in metric_candidates:
        if col in batch_df.columns:
            try:
                lines.append(f"- Mean {col}: {batch_df[col].mean():.4f}")
            except Exception:
                pass
    lines.append("")

    lines.append("## Best / Worst Cases\n")

    try:
        if "baseline_similarity_avg" in batch_df.columns:
            best_quality = batch_df.sort_values("baseline_similarity_avg", ascending=False).iloc[0]
            lines.append(
                f"- Best quality: {best_quality.get('run_name', 'N/A')} "
                f"(baseline_similarity_avg={best_quality['baseline_similarity_avg']:.4f})"
            )
        if "ai_avg_latency_ms" in batch_df.columns:
            worst_latency = batch_df.sort_values("ai_avg_latency_ms", ascending=False).iloc[0]
            lines.append(
                f"- Worst latency: {worst_latency.get('run_name', 'N/A')} "
                f"(ai_avg_latency_ms={worst_latency['ai_avg_latency_ms']:.2f})"
            )
    except Exception:
        pass

    lines.append("")
    lines.append("## Batch Results\n")
    try:
        lines.append(batch_df.to_markdown(index=False))
    except Exception:
        lines.append(batch_df.to_string(index=False))
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)