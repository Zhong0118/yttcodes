import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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
    rename_map = {}

    aliases = {
        "ai_latency_ms": "ai_avg_latency_ms",
        "overlay_skew_ms": "overlay_skew_avg_ms",
        "infer_ms": "infer_avg_ms",
        "baseline_similarity": "baseline_similarity_avg",
        "pose_detect_rate": "pose_detect_rate_avg",
        "hand_detect_rate": "hand_detect_rate_avg",
        "keypoint_completeness": "keypoint_completeness_avg",
        "stability_score": "stability_score_avg",
        "motion_energy": "motion_energy_avg",
    }

    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            rename_map[old] = new

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def collect_batch_summary(batch_dir: Path) -> pd.DataFrame:
    batch_csv = batch_dir / "batch_summary.csv"
    if batch_csv.exists():
        df = pd.read_csv(batch_csv)
        return normalize_metric_columns(df)

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


def sort_numeric_axis(values):
    try:
        return sorted(values, key=lambda x: float(x))
    except Exception:
        return sorted(values)


def draw_heatmap(pivot_df: pd.DataFrame, title: str, out_path: Path, fmt: str = ".3f"):
    if pivot_df.empty:
        return

    pivot_df = pivot_df.copy()
    pivot_df = pivot_df.loc[sort_numeric_axis(list(pivot_df.index)), sort_numeric_axis(list(pivot_df.columns))]
    values = pivot_df.values.astype(float)

    fig, ax = plt.subplots(figsize=(9, 7))
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
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {out_path}")


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
        df = df.reset_index().rename(columns={"index": "row_index"})
        x_col = "row_index"

    fig = plt.figure(figsize=(12, 8))
    plot_count = 0
    metrics = [
        ("ai_latency_ms", "AI Latency per Window"),
        ("baseline_similarity", "Baseline Similarity per Window"),
        ("pose_detect_rate", "Pose Detect Rate per Window"),
        ("hand_detect_rate", "Hand Detect Rate per Window"),
        ("stability_score", "Stability Score per Window"),
        ("keypoint_completeness", "Keypoint Completeness per Window"),
    ]

    for col, title in metrics:
        if col in df.columns:
            plot_count += 1
            ax = fig.add_subplot(3, 2, plot_count)
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
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {out_path}")


def generate_all_run_timeline_plots(batch_dir: Path, out_dir: Path):
    run_chart_dir = out_dir / "per_run"
    ensure_dir(run_chart_dir)
    for run_dir in sorted(batch_dir.glob("run_*")):
        make_run_timeline_plot(run_dir, run_chart_dir)


def make_batch_bar_plots(batch_df: pd.DataFrame, out_dir: Path):
    batch_df = batch_df.copy()
    if "run_name" not in batch_df.columns:
        batch_df["run_name"] = [f"run_{i:03d}" for i in range(len(batch_df))]

    metrics = [
        "ai_avg_latency_ms",
        "ai_p95_latency_ms",
        "overlay_skew_avg_ms",
        "infer_avg_ms",
        "pose_detect_rate_avg",
        "hand_detect_rate_avg",
        "keypoint_completeness_avg",
        "stability_score_avg",
        "motion_energy_avg",
        "baseline_similarity_avg",
    ]

    for metric in metrics:
        if metric not in batch_df.columns:
            continue

        df = batch_df[["run_name", metric]].dropna().sort_values(metric, ascending=False)
        if df.empty:
            continue

        plt.figure(figsize=(13, 6))
        plt.bar(df["run_name"], df[metric])
        plt.xticks(rotation=70, ha="right")
        plt.ylabel(metric)
        plt.title(f"{metric} across runs")
        plt.grid(True, axis="y", alpha=0.3)
        out_path = out_dir / f"bar_{metric}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved {out_path}")


def make_single_factor_plots(batch_df: pd.DataFrame, out_dir: Path):
    factor_metric_pairs = [
        ("bandwidth_mbps", "ai_avg_latency_ms"),
        ("bandwidth_mbps", "ai_p95_latency_ms"),
        ("bandwidth_mbps", "pose_detect_rate_avg"),
        ("bandwidth_mbps", "hand_detect_rate_avg"),
        ("bandwidth_mbps", "keypoint_completeness_avg"),
        ("bandwidth_mbps", "stability_score_avg"),

        ("window_seconds", "ai_avg_latency_ms"),
        ("window_seconds", "ai_p95_latency_ms"),
        ("window_seconds", "pose_detect_rate_avg"),
        ("window_seconds", "hand_detect_rate_avg"),
        ("window_seconds", "keypoint_completeness_avg"),
        ("window_seconds", "stability_score_avg"),

        ("ai_fps", "ai_avg_latency_ms"),
        ("ai_fps", "stability_score_avg"),
        ("ai_resolution", "hand_detect_rate_avg"),
        ("net_delay_ms", "ai_avg_latency_ms"),
        ("h264_batch_size", "ai_avg_latency_ms"),
        ("jpeg_quality", "hand_detect_rate_avg"),
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
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved {out_path}")


def make_bandwidth_window_heatmaps(batch_df: pd.DataFrame, out_dir: Path):
    row_key = "bandwidth_mbps"
    col_key = "window_seconds"

    metrics = [
        ("ai_avg_latency_ms", "bw_window_heatmap_latency.png", ".1f"),
        ("ai_p95_latency_ms", "bw_window_heatmap_p95_latency.png", ".1f"),
        ("pose_detect_rate_avg", "bw_window_heatmap_pose_detect.png", ".3f"),
        ("hand_detect_rate_avg", "bw_window_heatmap_hand_detect.png", ".3f"),
        ("keypoint_completeness_avg", "bw_window_heatmap_completeness.png", ".3f"),
        ("stability_score_avg", "bw_window_heatmap_stability.png", ".3f"),
    ]

    if row_key not in batch_df.columns or col_key not in batch_df.columns:
        return

    for metric, filename, fmt in metrics:
        if metric not in batch_df.columns:
            continue

        sub = batch_df[[row_key, col_key, metric]].dropna()
        if sub.empty:
            continue

        pivot = sub.pivot_table(index=row_key, columns=col_key, values=metric, aggfunc="mean")
        if pivot.empty:
            continue

        draw_heatmap(
            pivot_df=pivot,
            title=f"{metric} | rows=bandwidth_mbps, cols=window_seconds",
            out_path=out_dir / filename,
            fmt=fmt,
        )


def make_general_heatmaps(batch_df: pd.DataFrame, out_dir: Path):
    specs = [
        ("bandwidth_mbps", "ai_resolution", "hand_detect_rate_avg", "heatmap_bandwidth_resolution_hand_detect.png", ".3f"),
        ("bandwidth_mbps", "ai_fps", "ai_avg_latency_ms", "heatmap_bandwidth_fps_latency.png", ".1f"),
        ("net_delay_ms", "h264_batch_size", "ai_avg_latency_ms", "heatmap_delay_batch_latency.png", ".1f"),
        ("window_seconds", "ai_fps", "stability_score_avg", "heatmap_window_fps_stability.png", ".3f"),
        ("window_seconds", "ai_fps", "ai_avg_latency_ms", "heatmap_window_fps_latency.png", ".1f"),
    ]

    for row_key, col_key, val_key, filename, fmt in specs:
        if row_key not in batch_df.columns or col_key not in batch_df.columns or val_key not in batch_df.columns:
            continue

        sub = batch_df[[row_key, col_key, val_key]].dropna()
        if sub.empty:
            continue

        pivot = sub.pivot_table(index=row_key, columns=col_key, values=val_key, aggfunc="mean")
        if pivot.empty:
            continue

        draw_heatmap(
            pivot_df=pivot,
            title=f"{val_key} | rows={row_key}, cols={col_key}",
            out_path=out_dir / filename,
            fmt=fmt,
        )


def make_scatter_and_pareto(batch_df: pd.DataFrame, out_dir: Path):
    batch_df = batch_df.copy()

    if "run_name" not in batch_df.columns:
        batch_df["run_name"] = [f"run_{i:03d}" for i in range(len(batch_df))]

    quality_metric = None
    for c in ["baseline_similarity_avg", "hand_detect_rate_avg", "stability_score_avg", "keypoint_completeness_avg"]:
        if c in batch_df.columns and batch_df[c].notna().sum() > 0:
            quality_metric = c
            break

    if "ai_avg_latency_ms" not in batch_df.columns or quality_metric is None:
        return

    df = batch_df[["run_name", "ai_avg_latency_ms", quality_metric]].dropna()
    if df.empty:
        return

    plt.figure(figsize=(9, 7))
    plt.scatter(df["ai_avg_latency_ms"], df[quality_metric])
    for _, row in df.iterrows():
        plt.annotate(
            row["run_name"],
            (row["ai_avg_latency_ms"], row[quality_metric]),
            fontsize=8,
            alpha=0.8,
        )
    plt.xlabel("ai_avg_latency_ms")
    plt.ylabel(quality_metric)
    plt.title(f"Latency vs Quality ({quality_metric})")
    plt.grid(True, alpha=0.3)
    out_path = out_dir / "scatter_latency_vs_quality.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {out_path}")

    pts = df.sort_values(["ai_avg_latency_ms", quality_metric], ascending=[True, False]).reset_index(drop=True)
    pareto_idx = []
    best_quality = -float("inf")
    for i, row in pts.iterrows():
        q = row[quality_metric]
        if q > best_quality:
            pareto_idx.append(i)
            best_quality = q
    pareto = pts.iloc[pareto_idx].copy()

    plt.figure(figsize=(9, 7))
    plt.scatter(df["ai_avg_latency_ms"], df[quality_metric], alpha=0.7)
    plt.plot(pareto["ai_avg_latency_ms"], pareto[quality_metric], marker="o")
    for _, row in pareto.iterrows():
        plt.annotate(
            row["run_name"],
            (row["ai_avg_latency_ms"], row[quality_metric]),
            fontsize=8,
            alpha=0.8,
        )
    plt.xlabel("ai_avg_latency_ms")
    plt.ylabel(quality_metric)
    plt.title(f"Pareto Frontier: Low Latency vs High Quality ({quality_metric})")
    plt.grid(True, alpha=0.3)
    out_path = out_dir / "pareto_latency_quality.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {out_path}")


def make_window_1s_focus_plots(batch_df: pd.DataFrame, out_dir: Path):
    if "window_seconds" not in batch_df.columns:
        return

    sub = batch_df.copy()
    if "run_name" not in sub.columns:
        sub["run_name"] = [f"run_{i:03d}" for i in range(len(sub))]

    focus = sub[sub["window_seconds"].isin([1.0, 2.0, 3.0, 4.0, 6.0])].copy()
    if focus.empty:
        return

    metrics = [
        "ai_avg_latency_ms",
        "ai_p95_latency_ms",
        "hand_detect_rate_avg",
        "stability_score_avg",
        "keypoint_completeness_avg",
    ]

    for metric in metrics:
        if metric not in focus.columns:
            continue

        agg = focus.groupby("window_seconds", as_index=False)[metric].mean().sort_values("window_seconds")

        plt.figure(figsize=(8, 5))
        plt.plot(agg["window_seconds"], agg[metric], marker="o")
        plt.axvline(1.0, linestyle="--", alpha=0.5)
        plt.xlabel("window_seconds")
        plt.ylabel(metric)
        plt.title(f"{metric} vs window_seconds (1s focus)")
        plt.grid(True, alpha=0.3)
        out_path = out_dir / f"window_focus_{metric}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved {out_path}")


def make_bandwidth_focus_plots(batch_df: pd.DataFrame, out_dir: Path):
    if "bandwidth_mbps" not in batch_df.columns:
        return

    sub = batch_df.copy()
    metrics = [
        "ai_avg_latency_ms",
        "ai_p95_latency_ms",
        "hand_detect_rate_avg",
        "stability_score_avg",
        "keypoint_completeness_avg",
    ]

    for metric in metrics:
        if metric not in sub.columns:
            continue

        agg = sub.groupby("bandwidth_mbps", as_index=False)[metric].mean().sort_values("bandwidth_mbps")

        plt.figure(figsize=(8, 5))
        plt.plot(agg["bandwidth_mbps"], agg[metric], marker="o")
        plt.xlabel("bandwidth_mbps")
        plt.ylabel(metric)
        plt.title(f"{metric} vs bandwidth_mbps")
        plt.grid(True, alpha=0.3)
        out_path = out_dir / f"bandwidth_focus_{metric}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved {out_path}")


def make_summary_markdown(batch_dir: Path, batch_df: pd.DataFrame, out_dir: Path):
    md_path = out_dir / "charts_index.md"
    lines = []
    lines.append("# Charts Index\n")
    lines.append(f"- batch_dir: {batch_dir}")
    lines.append(f"- total_runs: {len(batch_df)}\n")

    key_cols = [
        "ai_avg_latency_ms",
        "ai_p95_latency_ms",
        "overlay_skew_avg_ms",
        "infer_avg_ms",
        "pose_detect_rate_avg",
        "hand_detect_rate_avg",
        "keypoint_completeness_avg",
        "stability_score_avg",
        "motion_energy_avg",
        "baseline_similarity_avg",
    ]
    for c in key_cols:
        if c in batch_df.columns and batch_df[c].notna().sum() > 0:
            lines.append(f"- mean_{c}: {batch_df[c].mean():.4f}")
    lines.append("")

    lines.append("## Generated PNG Files\n")
    for p in sorted(out_dir.rglob("*.png")):
        rel = p.relative_to(out_dir)
        lines.append(f"- {rel.as_posix()}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] saved {md_path}")


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
    make_single_factor_plots(batch_df, out_dir)
    make_bandwidth_window_heatmaps(batch_df, out_dir)
    make_general_heatmaps(batch_df, out_dir)
    make_scatter_and_pareto(batch_df, out_dir)
    make_window_1s_focus_plots(batch_df, out_dir)
    make_bandwidth_focus_plots(batch_df, out_dir)
    make_summary_markdown(batch_dir, batch_df, out_dir)

    print("[DONE] all charts generated.")


if __name__ == "__main__":
    main()