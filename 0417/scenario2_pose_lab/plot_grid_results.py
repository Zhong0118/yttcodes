from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def factor_plot(df: pd.DataFrame, factor: str, metric: str, out_path: Path):
    if factor not in df.columns or metric not in df.columns:
        return

    sub = df[[factor, metric]].dropna()
    if sub.empty:
        return

    agg = sub.groupby(factor, as_index=False)[metric].mean().sort_values(factor)

    plt.figure(figsize=(8, 5))
    plt.plot(agg[factor], agg[metric], marker="o")
    plt.xlabel(factor)
    plt.ylabel(metric)
    plt.title(f"{metric} vs {factor}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def draw_heatmap(df: pd.DataFrame, row: str, col: str, val: str, out_path: Path, fmt: str = ".3f"):
    if row not in df.columns or col not in df.columns or val not in df.columns:
        return

    sub = df[[row, col, val]].dropna()
    if sub.empty:
        return

    pivot = sub.pivot_table(index=row, columns=col, values=val, aggfunc="mean")
    if pivot.empty:
        return

    pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
    arr = pivot.values.astype(float)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(y) for y in pivot.index])
    ax.set_title(f"{val} | rows={row}, cols={col}")
    plt.colorbar(im, ax=ax)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isnan(arr[i, j]):
                ax.text(j, i, format(arr[i, j], fmt), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, annotate: bool = True):
    if x_col not in df.columns or y_col not in df.columns:
        return

    cols = [x_col, y_col]
    if "name" in df.columns:
        cols = ["name"] + cols

    sub = df[cols].dropna()
    if sub.empty:
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(sub[x_col], sub[y_col])

    if annotate and "name" in sub.columns:
        for _, row in sub.iterrows():
            plt.annotate(str(row["name"]), (row[x_col], row[y_col]), fontsize=7)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def bar_plot(df: pd.DataFrame, metric: str, out_path: Path):
    if metric not in df.columns:
        return

    cols = [metric]
    if "name" in df.columns:
        cols = ["name", metric]

    sub = df[cols].dropna().copy()
    if sub.empty:
        return

    if "name" not in sub.columns:
        sub["name"] = [f"run_{i:03d}" for i in range(len(sub))]

    sub = sub.sort_values(metric, ascending=False)

    plt.figure(figsize=(14, 6))
    plt.bar(sub["name"], sub[metric])
    plt.xticks(rotation=75, ha="right")
    plt.title(metric)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot charts for X3D grid experiment results.")
    parser.add_argument("--runs-summary-csv", required=True)
    args = parser.parse_args()

    csv_path = Path(args.runs_summary_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)
    out_dir = csv_path.parent / "charts"
    ensure_dir(out_dir)

    # 单因素折线图
    factor_metric_pairs = [
        ("effective_clip_frames", "video_top1_accuracy"),
        ("effective_clip_frames", "chunk_top1_accuracy"),
        ("effective_clip_frames", "infer_avg_ms"),
        ("effective_clip_frames", "end_to_end_avg_ms"),

        ("input_resize", "video_top1_accuracy"),
        ("input_resize", "chunk_top1_accuracy"),
        ("input_resize", "infer_avg_ms"),
        ("input_resize", "end_to_end_avg_ms"),

        ("effective_sampling_rate", "video_top1_accuracy"),
        ("effective_sampling_rate", "infer_avg_ms"),

        ("stride_frames", "video_top1_accuracy"),
        ("stride_frames", "prediction_switch_rate_avg"),

        ("bandwidth_mbps", "tx_avg_ms"),
        ("bandwidth_mbps", "end_to_end_avg_ms"),
        ("bandwidth_mbps", "video_top1_accuracy"),

        ("network_delay_ms", "end_to_end_avg_ms"),
        ("network_delay_ms", "first_result_ms"),
        ("network_delay_ms", "video_top1_accuracy"),

        ("packet_loss", "n_dropped_chunks"),
        ("packet_loss", "video_top1_accuracy"),
        ("packet_loss", "chunk_top1_accuracy"),

        ("jpeg_quality", "tx_avg_ms"),
        ("jpeg_quality", "video_top1_accuracy")
    ]

    for factor, metric in factor_metric_pairs:
        factor_plot(df, factor, metric, out_dir / f"factor_{factor}_vs_{metric}.png")

    # 热力图
    heatmaps = [
        ("effective_clip_frames", "bandwidth_mbps", "video_top1_accuracy"),
        ("effective_clip_frames", "bandwidth_mbps", "end_to_end_avg_ms"),
        ("effective_clip_frames", "network_delay_ms", "video_top1_accuracy"),
        ("effective_clip_frames", "network_delay_ms", "end_to_end_avg_ms"),
        ("input_resize", "bandwidth_mbps", "video_top1_accuracy"),
        ("input_resize", "packet_loss", "video_top1_accuracy"),
        ("network_delay_ms", "packet_loss", "video_top1_accuracy"),
        ("network_delay_ms", "packet_loss", "end_to_end_avg_ms")
    ]

    for row, col, val in heatmaps:
        draw_heatmap(df, row, col, val, out_dir / f"heatmap_{row}_{col}_{val}.png")

    # 散点图
    scatter_plot(df, "infer_avg_ms", "video_top1_accuracy", out_dir / "scatter_infer_vs_acc.png")
    scatter_plot(df, "end_to_end_avg_ms", "video_top1_accuracy", out_dir / "scatter_e2e_vs_acc.png")
    scatter_plot(df, "tx_avg_ms", "video_top1_accuracy", out_dir / "scatter_tx_vs_acc.png")

    # 柱状图
    metrics = [
        "video_top1_accuracy",
        "chunk_top1_accuracy",
        "infer_avg_ms",
        "infer_p95_ms",
        "tx_avg_ms",
        "first_result_ms",
        "end_to_end_avg_ms",
        "n_dropped_chunks",
        "prediction_switch_rate_avg"
    ]
    for metric in metrics:
        bar_plot(df, metric, out_dir / f"bar_{metric}.png")

    index_md = out_dir / "charts_index.md"
    pngs = sorted(out_dir.glob("*.png"))
    lines = ["# Charts Index", f"- total_pngs: {len(pngs)}", ""]
    for p in pngs:
        lines.append(f"- {p.name}")
    index_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[DONE] charts written to {out_dir}")


if __name__ == "__main__":
    main()