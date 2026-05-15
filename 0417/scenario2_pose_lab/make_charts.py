from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def collect_batch_summary(batch_dir: Path) -> pd.DataFrame:
    csv_path = batch_dir / "batch_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    return pd.read_csv(csv_path)


def make_per_run_timeline(batch_dir: Path, out_dir: Path):
    """
    单轮图：
    1) 质量/原有指标图
    2) 新增传输图（上下线）
    """
    per_run_dir = out_dir / "per_run"
    ensure_dir(per_run_dir)

    for run_dir in sorted(batch_dir.glob("run_*")):
        csv_path = run_dir / "windows.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        x = df["window_id"] if "window_id" in df.columns else np.arange(len(df))

        # -------------------------
        # 图 1：质量/效果
        # -------------------------
        fig1 = plt.figure(figsize=(12, 8))
        metrics1 = [
            ("ai_latency_ms", "AI Latency"),
            ("pose_detect_rate", "Pose Detect Rate"),
            ("hand_detect_rate", "Hand Detect Rate"),
            ("stability_score", "Stability Score"),
            ("keypoint_completeness", "Keypoint Completeness"),
            ("user_smoothness_score", "User Smoothness"),
        ]
        count = 0
        for col, title in metrics1:
            if col in df.columns:
                count += 1
                ax = fig1.add_subplot(3, 2, count)
                ax.plot(x, df[col], marker="o")
                ax.set_title(title)
                ax.set_xlabel("window_id")
                ax.grid(True, alpha=0.3)
        if count > 0:
            plt.tight_layout()
            fig1.savefig(per_run_dir / f"{run_dir.name}_quality_timeline.png", dpi=180, bbox_inches="tight")
        plt.close(fig1)

        # -------------------------
        # 图 2：上下线传输
        # -------------------------
        fig2 = plt.figure(figsize=(12, 8))
        metrics2 = [
            ("uplink_queue_ms", "Uplink Queue"),
            ("uplink_tx_ms", "Uplink TX"),
            ("server_recv_gap_ms", "Server Recv Gap"),
            ("first_result_latency_ms", "First Result Latency"),
            ("downlink_render_delay_ms", "Downlink Render Delay"),
            ("user_perceived_latency_ms", "User Perceived Latency"),
        ]
        count = 0
        for col, title in metrics2:
            if col in df.columns:
                count += 1
                ax = fig2.add_subplot(3, 2, count)
                ax.plot(x, df[col], marker="o")
                ax.set_title(title)
                ax.set_xlabel("window_id")
                ax.grid(True, alpha=0.3)
        if count > 0:
            plt.tight_layout()
            fig2.savefig(per_run_dir / f"{run_dir.name}_transport_timeline.png", dpi=180, bbox_inches="tight")
        plt.close(fig2)


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


def draw_heatmap(df: pd.DataFrame, row: str, col: str, val: str, out_path: Path, fmt: str = ".2f"):
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


def scatter_pareto(df: pd.DataFrame, quality: str, out_dir: Path):
    if "ai_avg_latency_ms" not in df.columns or quality not in df.columns:
        return
    cols = ["ai_avg_latency_ms", quality]
    if "name" in df.columns:
        cols = ["name"] + cols
    sub = df[cols].dropna().copy()
    if sub.empty:
        return

    plt.figure(figsize=(9, 7))
    plt.scatter(sub["ai_avg_latency_ms"], sub[quality])
    if "name" in sub.columns:
        for _, row in sub.iterrows():
            plt.annotate(str(row["name"]), (row["ai_avg_latency_ms"], row[quality]), fontsize=7)
    plt.xlabel("ai_avg_latency_ms")
    plt.ylabel(quality)
    plt.title(f"Latency vs Quality ({quality})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_latency_vs_quality.png", dpi=180, bbox_inches="tight")
    plt.close()

    pts = sub.sort_values(["ai_avg_latency_ms", quality], ascending=[True, False]).reset_index(drop=True)
    keep = []
    best_q = -1e18
    for i, row in pts.iterrows():
        if row[quality] > best_q:
            keep.append(i)
            best_q = row[quality]
    pareto = pts.iloc[keep]

    plt.figure(figsize=(9, 7))
    plt.scatter(sub["ai_avg_latency_ms"], sub[quality], alpha=0.6)
    plt.plot(pareto["ai_avg_latency_ms"], pareto[quality], marker="o")
    if "name" in pareto.columns:
        for _, row in pareto.iterrows():
            plt.annotate(str(row["name"]), (row["ai_avg_latency_ms"], row[quality]), fontsize=7)
    plt.xlabel("ai_avg_latency_ms")
    plt.ylabel(quality)
    plt.title(f"Pareto Frontier ({quality})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pareto_latency_quality.png", dpi=180, bbox_inches="tight")
    plt.close()


def make_bar(df: pd.DataFrame, metrics: list[str], out_dir: Path):
    for metric in metrics:
        if metric not in df.columns:
            continue
        cols = [metric]
        if "name" in df.columns:
            cols = ["name", metric]
        sub = df[cols].dropna().copy()
        if sub.empty:
            continue
        if "name" not in sub.columns:
            sub["name"] = [f"run_{i:03d}" for i in range(len(sub))]
        sub = sub.sort_values(metric, ascending=False)

        plt.figure(figsize=(14, 6))
        plt.bar(sub["name"], sub[metric])
        plt.xticks(rotation=75, ha="right")
        plt.title(f"{metric} across runs")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"bar_{metric}.png", dpi=180, bbox_inches="tight")
        plt.close()


def make_uplink_downlink_bars(df: pd.DataFrame, out_dir: Path):
    if "name" not in df.columns:
        df = df.copy()
        df["name"] = [f"run_{i:03d}" for i in range(len(df))]

    # uplink-only
    uplink_metrics = ["uplink_queue_avg_ms", "uplink_tx_avg_ms", "first_result_latency_avg_ms"]
    for metric in uplink_metrics:
        if metric not in df.columns:
            continue
        sub = df[["name", metric]].dropna().sort_values(metric, ascending=False)
        plt.figure(figsize=(14, 6))
        plt.bar(sub["name"], sub[metric])
        plt.xticks(rotation=75, ha="right")
        plt.title(f"Uplink metric: {metric}")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"uplink_bar_{metric}.png", dpi=180, bbox_inches="tight")
        plt.close()

    # downlink-only
    downlink_metrics = ["downlink_render_delay_avg_ms", "user_perceived_latency_avg_ms", "display_stutter_score_avg"]
    for metric in downlink_metrics:
        if metric not in df.columns:
            continue
        sub = df[["name", metric]].dropna().sort_values(metric, ascending=False)
        plt.figure(figsize=(14, 6))
        plt.bar(sub["name"], sub[metric])
        plt.xticks(rotation=75, ha="right")
        plt.title(f"Downlink metric: {metric}")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"downlink_bar_{metric}.png", dpi=180, bbox_inches="tight")
        plt.close()


def make_stacked_latency_bar(df: pd.DataFrame, out_dir: Path):
    required = ["uplink_queue_avg_ms", "uplink_tx_avg_ms", "infer_avg_ms", "downlink_render_delay_avg_ms"]
    if not all(c in df.columns for c in required):
        return

    plot_df = df.copy()
    if "name" not in plot_df.columns:
        plot_df["name"] = [f"run_{i:03d}" for i in range(len(plot_df))]

    plot_df = plot_df[["name"] + required].fillna(0.0)

    x = np.arange(len(plot_df))
    plt.figure(figsize=(16, 7))
    plt.bar(x, plot_df["uplink_queue_avg_ms"], label="uplink_queue")
    plt.bar(x, plot_df["uplink_tx_avg_ms"], bottom=plot_df["uplink_queue_avg_ms"], label="uplink_tx")
    plt.bar(
        x,
        plot_df["infer_avg_ms"],
        bottom=plot_df["uplink_queue_avg_ms"] + plot_df["uplink_tx_avg_ms"],
        label="infer",
    )
    plt.bar(
        x,
        plot_df["downlink_render_delay_avg_ms"],
        bottom=plot_df["uplink_queue_avg_ms"] + plot_df["uplink_tx_avg_ms"] + plot_df["infer_avg_ms"],
        label="downlink_render",
    )
    plt.xticks(x, plot_df["name"], rotation=75, ha="right")
    plt.ylabel("ms")
    plt.title("End-to-End Latency Composition (Uplink/Infer/Downlink)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "stacked_uplink_downlink_latency.png", dpi=180, bbox_inches="tight")
    plt.close()


def make_latency_pies(df: pd.DataFrame, out_dir: Path):
    required = ["uplink_queue_avg_ms", "uplink_tx_avg_ms", "infer_avg_ms", "downlink_render_delay_avg_ms"]
    if not all(c in df.columns for c in required):
        return

    plot_df = df.copy()
    if "name" not in plot_df.columns:
        plot_df["name"] = [f"run_{i:03d}" for i in range(len(plot_df))]

    # base on first, best latency, worst latency
    candidates = []

    if len(plot_df) > 0:
        candidates.append(("first_run", plot_df.iloc[0]))

    if "ai_avg_latency_ms" in plot_df.columns:
        best = plot_df.sort_values("ai_avg_latency_ms", ascending=True).iloc[0]
        worst = plot_df.sort_values("ai_avg_latency_ms", ascending=False).iloc[0]
        candidates.append(("best_latency_run", best))
        candidates.append(("worst_latency_run", worst))

    used_names = set()
    for tag, row in candidates:
        row_name = str(row.get("name", tag))
        if row_name in used_names:
            continue
        used_names.add(row_name)

        vals = [
            float(row.get("uplink_queue_avg_ms", 0.0) or 0.0),
            float(row.get("uplink_tx_avg_ms", 0.0) or 0.0),
            float(row.get("infer_avg_ms", 0.0) or 0.0),
            float(row.get("downlink_render_delay_avg_ms", 0.0) or 0.0),
        ]
        labels = ["uplink_queue", "uplink_tx", "infer", "downlink_render"]

        plt.figure(figsize=(7, 7))
        plt.pie(vals, labels=labels, autopct="%1.1f%%")
        plt.title(f"Latency Composition Pie - {row_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"pie_latency_composition_{row_name}.png", dpi=180, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    df = collect_batch_summary(batch_dir)
    out_dir = batch_dir / "charts"
    ensure_dir(out_dir)

    make_per_run_timeline(batch_dir, out_dir)

    # old/general factor plots
    factor_metric_pairs = [
        ("uplink_resolution", "hand_detect_rate_avg"),
        ("uplink_resolution", "ai_avg_latency_ms"),
        ("uplink_fps", "ai_avg_latency_ms"),
        ("uplink_fps", "stability_score_avg"),
        ("window_seconds", "ai_avg_latency_ms"),
        ("window_seconds", "hand_detect_rate_avg"),
        ("window_seconds", "stability_score_avg"),
        ("chunk_size_frames", "ai_avg_latency_ms"),
        ("chunk_size_frames", "hand_detect_rate_avg"),
        ("tcp_bandwidth_mbps", "ai_avg_latency_ms"),
        ("tcp_bandwidth_mbps", "payload_avg_kb"),
        ("tcp_fixed_rtt_ms", "ai_avg_latency_ms"),
        ("downlink_resolution", "user_visual_quality_score_avg"),
        ("downlink_fps", "user_smoothness_score_avg"),
        ("downlink_fps", "downlink_perceived_latency_ms_avg"),

        # new uplink/downlink factor plots
        ("tcp_bandwidth_mbps", "uplink_tx_avg_ms"),
        ("window_seconds", "uplink_tx_avg_ms"),
        ("window_seconds", "uplink_queue_avg_ms"),
        ("window_seconds", "first_result_latency_avg_ms"),
        ("downlink_fps", "downlink_render_delay_avg_ms"),
        ("downlink_fps", "display_fps_actual_avg"),
        ("downlink_fps", "display_stutter_score_avg"),
        ("downlink_resolution", "user_perceived_latency_avg_ms"),
    ]
    for factor, metric in factor_metric_pairs:
        factor_plot(df, factor, metric, out_dir / f"factor_{factor}_vs_{metric}.png")

    # heatmaps
    draw_heatmap(df, "tcp_bandwidth_mbps", "window_seconds", "ai_avg_latency_ms", out_dir / "heatmap_bw_window_latency.png", ".1f")
    draw_heatmap(df, "tcp_bandwidth_mbps", "window_seconds", "hand_detect_rate_avg", out_dir / "heatmap_bw_window_hand_detect.png", ".3f")
    draw_heatmap(df, "tcp_bandwidth_mbps", "window_seconds", "stability_score_avg", out_dir / "heatmap_bw_window_stability.png", ".3f")
    draw_heatmap(df, "uplink_fps", "window_seconds", "ai_avg_latency_ms", out_dir / "heatmap_fps_window_latency.png", ".1f")
    draw_heatmap(df, "uplink_fps", "window_seconds", "stability_score_avg", out_dir / "heatmap_fps_window_stability.png", ".3f")
    draw_heatmap(df, "uplink_resolution", "uplink_fps", "hand_detect_rate_avg", out_dir / "heatmap_res_fps_hand_detect.png", ".3f")
    draw_heatmap(df, "downlink_resolution", "downlink_fps", "user_visual_quality_score_avg", out_dir / "heatmap_downlink_quality.png", ".3f")

    # new uplink/downlink heatmaps
    draw_heatmap(df, "tcp_bandwidth_mbps", "window_seconds", "uplink_tx_avg_ms", out_dir / "heatmap_uplink_bw_window_tx.png", ".1f")
    draw_heatmap(df, "tcp_bandwidth_mbps", "window_seconds", "first_result_latency_avg_ms", out_dir / "heatmap_uplink_bw_window_first_result.png", ".1f")
    draw_heatmap(df, "downlink_resolution", "downlink_fps", "downlink_render_delay_avg_ms", out_dir / "heatmap_downlink_render_delay.png", ".1f")
    draw_heatmap(df, "downlink_resolution", "downlink_fps", "user_perceived_latency_avg_ms", out_dir / "heatmap_uplink_vs_downlink_perceived_latency.png", ".1f")

    # bars
    metrics = [
        "first_result_latency_ms",
        "first_result_latency_avg_ms",
        "ai_avg_latency_ms",
        "ai_p95_latency_ms",
        "uplink_queue_avg_ms",
        "uplink_tx_avg_ms",
        "server_recv_gap_avg_ms",
        "downlink_render_delay_avg_ms",
        "display_fps_actual_avg",
        "display_stutter_score_avg",
        "user_perceived_latency_avg_ms",
        "payload_avg_kb",
        "pose_detect_rate_avg",
        "hand_detect_rate_avg",
        "face_detect_rate_avg",
        "keypoint_completeness_avg",
        "stability_score_avg",
        "user_smoothness_score_avg",
        "user_visual_quality_score_avg",
    ]
    make_bar(df, metrics, out_dir)

    # dedicated uplink/downlink bars
    make_uplink_downlink_bars(df, out_dir)

    # stacked + pie
    make_stacked_latency_bar(df, out_dir)
    make_latency_pies(df, out_dir)

    # scatter / pareto
    quality_col = "hand_detect_rate_avg" if "hand_detect_rate_avg" in df.columns else "stability_score_avg"
    scatter_pareto(df, quality_col, out_dir)

    md = out_dir / "charts_index.md"
    pngs = sorted(out_dir.rglob("*.png"))
    lines = ["# Scenario2 Charts Index\n", f"- total_pngs: {len(pngs)}"]
    for p in pngs:
        lines.append(f"- {p.relative_to(out_dir).as_posix()}")
    md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[DONE] charts written to {out_dir}")

if __name__ == "__main__":
    main()