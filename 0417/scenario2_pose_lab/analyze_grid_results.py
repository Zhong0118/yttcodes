from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# utils
# =========================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_runs_summary(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    elif path.suffix.lower() == ".xlsx":
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Unsupported save format: {path}")


def normalize_for_score(series: pd.Series, larger_is_better: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return pd.Series([np.nan] * len(series), index=series.index)

    s_min = s.min()
    s_max = s.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return pd.Series([1.0] * len(series), index=series.index)

    if larger_is_better:
        return (s - s_min) / (s_max - s_min)
    else:
        return (s_max - s) / (s_max - s_min)


def get_default_fixed_values(df: pd.DataFrame, fixed_cols: List[str]) -> Dict[str, object]:
    """
    从 full_grid 结果里自动选一组“最常见配置”作为固定值。
    """
    fixed_values = {}
    for c in fixed_cols:
        if c not in df.columns:
            continue
        mode_vals = df[c].mode(dropna=True)
        if len(mode_vals) > 0:
            fixed_values[c] = mode_vals.iloc[0]
    return fixed_values


def filter_by_fixed_values(df: pd.DataFrame, fixed_values: Dict[str, object], ignore_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    固定其他变量，只让目标变量变化。
    """
    ignore_cols = ignore_cols or []
    sub = df.copy()
    for c, v in fixed_values.items():
        if c in ignore_cols:
            continue
        if c not in sub.columns:
            continue
        sub = sub[sub[c] == v]
    return sub


# =========================================================
# ranking / preprocessing
# =========================================================

def build_tradeoff_score(
    df: pd.DataFrame,
    acc_col: str = "video_top1_accuracy",
    latency_col: str = "end_to_end_avg_ms",
    stability_col: str = "prediction_switch_rate_avg",
    dropped_col: str = "n_dropped_chunks",
    w_acc: float = 0.5,
    w_latency: float = 0.3,
    w_stability: float = 0.1,
    w_drop: float = 0.1,
) -> pd.DataFrame:
    out = df.copy()

    out["_score_acc"] = normalize_for_score(out[acc_col], larger_is_better=True) if acc_col in out.columns else np.nan
    out["_score_latency"] = normalize_for_score(out[latency_col], larger_is_better=False) if latency_col in out.columns else np.nan
    out["_score_stability"] = normalize_for_score(out[stability_col], larger_is_better=False) if stability_col in out.columns else np.nan
    out["_score_drop"] = normalize_for_score(out[dropped_col], larger_is_better=False) if dropped_col in out.columns else np.nan

    out["tradeoff_score"] = (
        w_acc * out["_score_acc"].fillna(0)
        + w_latency * out["_score_latency"].fillna(0)
        + w_stability * out["_score_stability"].fillna(0)
        + w_drop * out["_score_drop"].fillna(0)
    )
    return out


def export_rankings(df: pd.DataFrame, out_dir: Path, topk: int = 20):
    ensure_dir(out_dir)

    # 1) accuracy top
    if "video_top1_accuracy" in df.columns:
        acc_top = df.sort_values(["video_top1_accuracy", "chunk_top1_accuracy"], ascending=[False, False]).head(topk)
        save_df(acc_top, out_dir / "top_video_accuracy.csv")

    # 2) latency best
    if "end_to_end_avg_ms" in df.columns:
        latency_top = df.sort_values(["end_to_end_avg_ms", "video_top1_accuracy"], ascending=[True, False]).head(topk)
        save_df(latency_top, out_dir / "lowest_end_to_end_latency.csv")

    # 3) tradeoff
    trade = build_tradeoff_score(df)
    trade_top = trade.sort_values("tradeoff_score", ascending=False).head(topk)
    save_df(trade_top, out_dir / "best_tradeoff.csv")

    save_df(trade.sort_values("tradeoff_score", ascending=False), out_dir / "runs_summary_with_tradeoff.csv")


# =========================================================
# plotting helpers
# =========================================================

def bar_plot_topk(df: pd.DataFrame, value_col: str, out_path: Path, title: str, ascending: bool, topk: int = 15):
    if value_col not in df.columns:
        return
    sub = df.dropna(subset=[value_col]).copy()
    if sub.empty:
        return

    if "name" not in sub.columns:
        sub["name"] = [f"run_{i:04d}" for i in range(len(sub))]

    sub = sub.sort_values(value_col, ascending=ascending).head(topk)

    plt.figure(figsize=(14, 6))
    plt.bar(sub["name"], sub[value_col])
    plt.xticks(rotation=75, ha="right")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str, annotate_topk: int = 20):
    if x_col not in df.columns or y_col not in df.columns:
        return
    cols = [x_col, y_col]
    if "name" in df.columns:
        cols = ["name"] + cols

    sub = df[cols].dropna().copy()
    if sub.empty:
        return

    plt.figure(figsize=(9, 7))
    plt.scatter(sub[x_col], sub[y_col], alpha=0.8)

    if "name" in sub.columns:
        # 标注一部分：按 y_col 高低优先
        anno = sub.sort_values(y_col, ascending=False).head(annotate_topk)
        for _, row in anno.iterrows():
            plt.annotate(str(row["name"]), (row[x_col], row[y_col]), fontsize=7)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def factor_plot(df: pd.DataFrame, factor: str, metric: str, out_path: Path, title: Optional[str] = None):
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
    plt.title(title or f"{metric} vs {factor}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def draw_heatmap(df: pd.DataFrame, row: str, col: str, val: str, out_path: Path, fmt: str = ".3f", title: Optional[str] = None):
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
    ax.set_title(title or f"{val} | rows={row}, cols={col}")
    plt.colorbar(im, ax=ax)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isnan(arr[i, j]):
                ax.text(j, i, format(arr[i, j], fmt), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Layer 1: overall ranking
# =========================================================

def plot_overall_rankings(df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)

    bar_plot_topk(
        df, "video_top1_accuracy",
        out_dir / "bar_top_video_accuracy.png",
        title="Top Configs by Video Top-1 Accuracy",
        ascending=False,
        topk=15,
    )

    bar_plot_topk(
        df, "end_to_end_avg_ms",
        out_dir / "bar_lowest_end_to_end_latency.png",
        title="Top Configs by Lowest End-to-End Latency",
        ascending=True,
        topk=15,
    )

    trade = build_tradeoff_score(df)
    bar_plot_topk(
        trade, "tradeoff_score",
        out_dir / "bar_best_tradeoff.png",
        title="Top Configs by Tradeoff Score",
        ascending=False,
        topk=15,
    )

    scatter_plot(
        df,
        x_col="end_to_end_avg_ms",
        y_col="video_top1_accuracy",
        out_path=out_dir / "scatter_accuracy_vs_e2e.png",
        title="Video Accuracy vs End-to-End Latency",
    )

    scatter_plot(
        df,
        x_col="infer_avg_ms",
        y_col="video_top1_accuracy",
        out_path=out_dir / "scatter_accuracy_vs_infer.png",
        title="Video Accuracy vs Inference Latency",
    )

    trade = build_tradeoff_score(df)
    scatter_plot(
        trade,
        x_col="end_to_end_avg_ms",
        y_col="tradeoff_score",
        out_path=out_dir / "scatter_tradeoff_vs_e2e.png",
        title="Tradeoff Score vs End-to-End Latency",
    )


# =========================================================
# Layer 2: single-factor trends
# =========================================================

def plot_single_factor_slices(df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)

    # 这些列视为“其他变量”，用于固定成基准切片
    control_cols = [
        "effective_clip_frames",
        "effective_sampling_rate",
        "stride_frames",
        "bandwidth_mbps",
        "network_delay_ms",
        "packet_loss",
        "input_resize",
        "jpeg_quality",
    ]
    fixed_values = get_default_fixed_values(df, control_cols)

    # 定义需要画的单因素趋势
    plans = [
        ("effective_clip_frames", ["video_top1_accuracy", "chunk_top1_accuracy", "end_to_end_avg_ms"]),
        ("bandwidth_mbps", ["tx_avg_ms", "end_to_end_avg_ms", "video_top1_accuracy"]),
        ("packet_loss", ["video_top1_accuracy", "chunk_top1_accuracy", "n_dropped_chunks"]),
        ("effective_sampling_rate", ["video_top1_accuracy", "infer_avg_ms"]),
        ("network_delay_ms", ["end_to_end_avg_ms", "first_result_ms", "video_top1_accuracy"]),
    ]

    slice_meta = []

    for factor, metrics in plans:
        if factor not in df.columns:
            continue

        use_fixed = {k: v for k, v in fixed_values.items() if k != factor}
        sub = filter_by_fixed_values(df, use_fixed, ignore_cols=[factor])

        if sub.empty or sub[factor].nunique() <= 1:
            # 退化方案：不固定其他变量，直接全量聚合
            sub = df.copy()
            slice_type = "global_aggregate"
        else:
            slice_type = "fixed_slice"

        for metric in metrics:
            if metric not in sub.columns:
                continue
            factor_plot(
                sub,
                factor=factor,
                metric=metric,
                out_path=out_dir / f"factor_{factor}_vs_{metric}.png",
                title=f"{metric} vs {factor} ({slice_type})",
            )

        slice_meta.append(
            {
                "factor": factor,
                "slice_type": slice_type,
                "n_rows": len(sub),
                **{f"fixed_{k}": v for k, v in use_fixed.items()}
            }
        )

    if slice_meta:
        pd.DataFrame(slice_meta).to_csv(out_dir / "single_factor_slice_meta.csv", index=False, encoding="utf-8-sig")


# =========================================================
# Layer 3: two-factor heatmaps
# =========================================================

def plot_two_factor_heatmaps(df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)

    plans = [
        ("effective_clip_frames", "bandwidth_mbps", "video_top1_accuracy"),
        ("effective_clip_frames", "bandwidth_mbps", "end_to_end_avg_ms"),
        ("network_delay_ms", "packet_loss", "end_to_end_avg_ms"),
        ("network_delay_ms", "packet_loss", "video_top1_accuracy"),
    ]

    for row, col, val in plans:
        draw_heatmap(
            df,
            row=row,
            col=col,
            val=val,
            out_path=out_dir / f"heatmap_{row}_{col}_{val}.png",
            title=f"{val} | {row} x {col}",
        )


# =========================================================
# report helpers
# =========================================================

def export_preprocessed_summaries(df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)

    # 每个主要因素按均值聚合
    factors = [
        "effective_clip_frames",
        "bandwidth_mbps",
        "packet_loss",
        "effective_sampling_rate",
        "network_delay_ms",
        "input_resize",
        "stride_frames",
        "jpeg_quality",
    ]

    metrics = [
        "video_top1_accuracy",
        "chunk_top1_accuracy",
        "prediction_switch_rate_avg",
        "infer_avg_ms",
        "tx_avg_ms",
        "end_to_end_avg_ms",
        "n_dropped_chunks",
    ]

    for factor in factors:
        if factor not in df.columns:
            continue
        use_cols = [factor] + [m for m in metrics if m in df.columns]
        sub = df[use_cols].dropna(subset=[factor]).copy()
        if sub.empty:
            continue
        agg = sub.groupby(factor, as_index=False).mean(numeric_only=True).sort_values(factor)
        save_df(agg, out_dir / f"agg_by_{factor}.csv")

    # 关键双因素聚合
    pairs = [
        ("effective_clip_frames", "bandwidth_mbps"),
        ("network_delay_ms", "packet_loss"),
    ]
    for a, b in pairs:
        if a not in df.columns or b not in df.columns:
            continue
        use_cols = [a, b] + [m for m in metrics if m in df.columns]
        sub = df[use_cols].dropna(subset=[a, b]).copy()
        if sub.empty:
            continue
        agg = sub.groupby([a, b], as_index=False).mean(numeric_only=True).sort_values([a, b])
        save_df(agg, out_dir / f"agg_by_{a}_{b}.csv")


def write_analysis_index(out_dir: Path):
    pngs = sorted(out_dir.rglob("*.png"))
    csvs = sorted(out_dir.rglob("*.csv"))
    lines = [
        "# Analysis Outputs",
        "",
        f"- total_pngs: {len(pngs)}",
        f"- total_csvs: {len(csvs)}",
        "",
        "## PNG files",
    ]
    for p in pngs:
        lines.append(f"- {p.relative_to(out_dir).as_posix()}")

    lines.append("")
    lines.append("## CSV files")
    for p in csvs:
        lines.append(f"- {p.relative_to(out_dir).as_posix()}")

    (out_dir / "analysis_index.md").write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze runs_summary.csv for X3D grid experiments.")
    parser.add_argument("--runs-summary-csv", required=True)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    csv_path = Path(args.runs_summary_csv)
    df = load_runs_summary(csv_path)

    numeric_cols = [
        "video_top1_accuracy",
        "chunk_top1_accuracy",
        "prediction_switch_rate_avg",
        "infer_avg_ms",
        "infer_p95_ms",
        "tx_avg_ms",
        "first_result_ms",
        "end_to_end_avg_ms",
        "n_dropped_chunks",
        "effective_clip_frames",
        "effective_sampling_rate",
        "input_resize",
        "input_crop",
        "stride_frames",
        "bandwidth_mbps",
        "network_delay_ms",
        "packet_loss",
        "jpeg_quality",
    ]
    df = safe_numeric(df, numeric_cols)

    out_dir = csv_path.parent / "analysis_outputs"
    ensure_dir(out_dir)

    # 1) 预处理汇总
    preprocess_dir = out_dir / "01_preprocessed"
    export_preprocessed_summaries(df, preprocess_dir)

    # 2) 总体排行
    ranking_dir = out_dir / "02_overall_rankings"
    export_rankings(df, ranking_dir, topk=args.topk)
    plot_overall_rankings(df, ranking_dir)

    # 3) 单因素趋势
    single_factor_dir = out_dir / "03_single_factor_trends"
    plot_single_factor_slices(df, single_factor_dir)

    # 4) 双因素热力图
    heatmap_dir = out_dir / "04_two_factor_heatmaps"
    plot_two_factor_heatmaps(df, heatmap_dir)

    # 5) 保存增强后的总表
    trade = build_tradeoff_score(df)
    save_df(trade, out_dir / "runs_summary_enhanced.csv")
    save_df(trade, out_dir / "runs_summary_enhanced.xlsx")

    write_analysis_index(out_dir)

    print(f"[DONE] analysis outputs written to: {out_dir}")


if __name__ == "__main__":
    main()