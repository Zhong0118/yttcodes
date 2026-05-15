import os
import json
import math
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] failed to read csv: {path} -> {e}")
    return None


def safe_read_json(path: Path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] failed to read json: {path} -> {e}")
    return None


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    兼容不同版本的字段名。
    """
    rename_map = {}

    if "ai_latency_ms" in df.columns and "ai_avg_latency_ms" not in df.columns:
        rename_map["ai_latency_ms"] = "ai_avg_latency_ms"
    if "overlay_skew_ms" in df.columns and "overlay_skew_avg_ms" not in df.columns:
        rename_map["overlay_skew_ms"] = "overlay_skew_avg_ms"
    if "infer_ms" in df.columns and "infer_avg_ms" not in df.columns:
        rename_map["infer_ms"] = "infer_avg_ms"
    if "baseline_similarity" in df.columns and "baseline_similarity_avg" not in df.columns:
        rename_map["baseline_similarity"] = "baseline_similarity_avg"
    if "pose_detect_rate" in df.columns and "pose_detect_rate_avg" not in df.columns:
        rename_map["pose_detect_rate"] = "pose_detect_rate_avg"
    if "hand_detect_rate" in df.columns and "hand_detect_rate_avg" not in df.columns:
        rename_map["hand_detect_rate"] = "hand_detect_rate_avg"
    if "keypoint_completeness" in df.columns and "keypoint_completeness_avg" not in df.columns:
        rename_map["keypoint_completeness"] = "keypoint_completeness_avg"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def make_run_timeline_plot(run_dir: Path, out_dir: Path):
    windows_csv = run_dir / "windows.csv"
    df = safe_read_csv(windows_csv)
    if df is None or df.empty:
        return

    x_col = None
    for c in ["window_id", "window_index", "start_sec", "start_ms"]:
        if c in df.columns:
            x_col = c
            break
    if x_col is None:
        x_col = df.index

    fig = plt.figure(figsize=(12, 8))

    plot_count = 0
    metrics = [
        ("ai_latency_ms", "AI Latency per Window"),
        ("baseline_similarity", "Baseline Similarity per Window"),
        ("pose_detect_rate", "Pose Detect Rate per Window"),
        ("hand_detect_rate", "Hand Detect Rate per Window"),
    ]

    for idx, (col, title) in enumerate(metrics, start=1):
        if col in df.columns:
            plot_count += 1
            ax = fig.add_subplot(2, 2, plot_count)
            ax.plot(df[x_col], df[col], marker="o")
            ax.set_title(title)
            ax.set_xlabel(x_col)
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)

    if plot_count == 0:
        plt.close(fig)
        return

    plt.tight_layout()
    out_path = out_dir / f"{run_dir.name}_timeline.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {out_path}")


def make_batch_bar_plots(batch_df: pd.DataFrame, out_dir: Path):
    batch_df = batch_df.copy()
    if "run_name" not in batch_df.columns:
        batch_df["run_name"] = [f"run_{i:03d}" for i in range(len(batch_df))]

    metrics = [
        "ai_avg_latency_ms",
        "overlay_skew_avg_ms",
        "infer_avg_ms",
        "baseline_similarity_avg",
        "pose_detect_rate_avg",
        "hand_detect_rate_avg",
        "keypoint_completeness_avg",
    ]

    for metric in metrics:
        if metric not in batch_df.columns:
            continue

        df = batch_df[["run_name", metric]].dropna().sort_values(metric, ascending=False)
        if df.empty:
            continue

        plt.figure(figsize=(12, 6))
        plt.bar(df["run_name"], df[metric])
        plt.xticks(rotation=60, ha="right")
        plt.ylabel(metric)
        plt.title(f"{metric} across runs")
        plt.grid(True, axis="y", alpha=0.3)
        out_path = out_dir / f"bar_{metric}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved {out_path}")


def make_scatter_and_pareto(batch_df: pd.DataFrame, out_dir: Path):
    batch_df = batch_df.copy()

    if "run_name" not in batch_df.columns:
        batch_df["run_name"] = [f"run_{i:03d}" for i in range(len(batch_df))]

    if "ai_avg_latency_ms" not in batch_df.columns or "baseline_similarity_avg" not in batch_df.columns:
        return

    df = batch_df[["run_name", "ai_avg_latency_ms", "baseline_similarity_avg"]].dropna()
    if df.empty:
        return

    # 散点图
    plt.figure(figsize=(9, 7))
    plt.scatter(df["ai_avg_latency_ms"], df["baseline_similarity_avg"])
    for _, row in df.iterrows():
        plt.annotate(
            row["run_name"],
            (row["ai_avg_latency_ms"], row["baseline_similarity_avg"]),
            fontsize=8,
            alpha=0.8,
        )
    plt.xlabel("ai_avg_latency_ms")
    plt.ylabel("baseline_similarity_avg")
    plt.title("Latency vs Quality")
    plt.grid(True, alpha=0.3)
    out_path = out_dir / "scatter_latency_vs_quality.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {out_path}")

    # Pareto 前沿
    pts = df.sort_values(["ai_avg_latency_ms", "baseline_similarity_avg"], ascending=[True, False]).reset_index(drop=True)
    pareto_idx = []
    best_quality = -float("inf")
    for i, row in pts.iterrows():
        q = row["baseline_similarity_avg"]
        if q > best_quality:
            pareto_idx.append(i)
            best_quality = q
    pareto = pts.iloc[pareto_idx].copy()

    plt.figure(figsize=(9, 7))
    plt.scatter(df["ai_avg_latency_ms"], df["baseline_similarity_avg"])
    plt.plot(pareto["ai_avg_latency_ms"], pareto["baseline_similarity_avg"], marker="o")
    for _, row in pareto.iterrows():
        plt.annotate(
            row["run_name"],
            (row["ai_avg_latency_ms"], row["baseline_similarity_avg"]),
            fontsize=8,
            alpha=0.8,
        )
    plt.xlabel("ai_avg_latency_ms")
    plt.ylabel("baseline_similarity_avg")
    plt.title("Pareto Frontier: Low Latency vs High Quality")
    plt.grid(True, alpha=0.3)
    out_path = out_dir / "pareto_latency_quality.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {out_path}")


def draw_heatmap(pivot_df: pd.DataFrame, title: str, out_path: Path):
    if pivot_df.empty:
        return

    values = pivot_df.values.astype(float)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(values, aspect="auto")

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels([str(x) for x in pivot_df.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels([str(y) for y in pivot_df.index])

    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {out_path}")


def make_heatmaps(batch_df: pd.DataFrame, out_dir: Path):
    specs = [
        ("bandwidth_mbps", "ai_resolution", "baseline_similarity_avg", "heatmap_bandwidth_resolution_similarity.png"),
        ("bandwidth_mbps", "ai_fps", "ai_avg_latency_ms", "heatmap_bandwidth_fps_latency.png"),
        ("net_delay_ms", "h264_batch_size", "ai_avg_latency_ms", "heatmap_delay_batch_latency.png"),
        ("window_seconds", "ai_fps", "baseline_similarity_avg", "heatmap_window_fps_similarity.png"),
    ]

    for row_key, col_key, val_key, filename in specs:
        if row_key not in batch_df.columns or col_key not in batch_df.columns or val_key not in batch_df.columns:
            continue

        sub = batch_df[[row_key, col_key, val_key]].dropna()
        if sub.empty:
            continue

        pivot = sub.pivot_table(index=row_key, columns=col_key, values=val_key, aggfunc="mean")
        if pivot.empty:
            continue

        title = f"{val_key} | rows={row_key}, cols={col_key}"
        draw_heatmap(pivot, title, out_dir / filename)


def make_single_factor_plots(batch_df: pd.DataFrame, out_dir: Path):
    """
    用来做老师比较喜欢的“单因素敏感性”图。
    """
    factor_metric_pairs = [
        ("bandwidth_mbps", "baseline_similarity_avg"),
        ("bandwidth_mbps", "ai_avg_latency_ms"),
        ("net_delay_ms", "ai_avg_latency_ms"),
        ("ai_resolution", "baseline_similarity_avg"),
        ("ai_resolution", "infer_avg_ms"),
        ("ai_fps", "baseline_similarity_avg"),
        ("ai_fps", "ai_avg_latency_ms"),
        ("window_seconds", "baseline_similarity_avg"),
        ("window_seconds", "ai_avg_latency_ms"),
        ("h264_batch_size", "ai_avg_latency_ms"),
        ("jpeg_quality", "baseline_similarity_avg"),
    ]

    for factor, metric in factor_metric_pairs:
        if factor not in batch_df.columns or metric not in batch_df.columns:
            continue

        sub = batch_df[[factor, metric]].dropna()
        if sub.empty:
            continue

        agg = sub.groupby(factor, as_index=False)[metric].mean().sort_values(factor)

        plt.figure(figsize=(8, 5))
        plt.plot(agg[factor], agg[metric], marker="o")
        plt.xlabel(factor)
        plt.ylabel(metric)
        plt.title(f"{metric} vs {factor}")
        plt.grid(True, alpha=0.3)
        out_path = out_dir / f"factor_{factor}_vs_{metric}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved {out_path}")


def make_summary_markdown(batch_dir: Path, batch_df: pd.DataFrame, out_dir: Path):
    md_path = out_dir / "charts_index.md"
    lines = []
    lines.append("# Charts Index\n")
    lines.append(f"- batch_dir: {batch_dir}\n")
    lines.append(f"- total_runs: {len(batch_df)}\n")

    key_cols = [
        "ai_avg_latency_ms",
        "overlay_skew_avg_ms",
        "infer_avg_ms",
        "baseline_similarity_avg",
        "pose_detect_rate_avg",
        "hand_detect_rate_avg",
        "keypoint_completeness_avg",
    ]
    for c in key_cols:
        if c in batch_df.columns:
            lines.append(f"- mean_{c}: {batch_df[c].mean():.4f}")

    lines.append("\n## Files\n")
    for p in sorted(out_dir.glob("*.png")):
        lines.append(f"- {p.name}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] saved {md_path}")


def collect_batch_summary(batch_dir: Path) -> pd.DataFrame:
    batch_csv = batch_dir / "batch_summary.csv"
    if batch_csv.exists():
        df = pd.read_csv(batch_csv)
        return normalize_metric_columns(df)

    # 如果没有 batch_summary.csv，就从 run_*/summary.json 重建
    rows = []
    for run_dir in sorted(batch_dir.glob("run_*")):
        summary_json = run_dir / "summary.json"
        row = safe_read_json(summary_json)
        if row:
            rows.append(row)

    if not rows:
        raise FileNotFoundError(f"batch_summary.csv and run_*/summary.json both missing in {batch_dir}")

    df = pd.DataFrame(rows)
    return normalize_metric_columns(df)


def generate_all_run_timeline_plots(batch_dir: Path, out_dir: Path):
    run_chart_dir = out_dir / "per_run"
    ensure_dir(run_chart_dir)

    for run_dir in sorted(batch_dir.glob("run_*")):
        make_run_timeline_plot(run_dir, run_chart_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=str, required=True, help="path to results/batch_xxx")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    if not batch_dir.exists():
        raise FileNotFoundError(f"batch dir not found: {batch_dir}")

    out_dir = batch_dir / "charts"
    ensure_dir(out_dir)

    batch_df = collect_batch_summary(batch_dir)

    print(f"[INFO] loaded batch rows: {len(batch_df)}")
    print(f"[INFO] output dir: {out_dir}")

    generate_all_run_timeline_plots(batch_dir, out_dir)
    make_batch_bar_plots(batch_df, out_dir)
    make_scatter_and_pareto(batch_df, out_dir)
    make_heatmaps(batch_df, out_dir)
    make_single_factor_plots(batch_df, out_dir)
    make_summary_markdown(batch_dir, batch_df, out_dir)

    print("[DONE] all charts generated.")


if __name__ == "__main__":
    main()