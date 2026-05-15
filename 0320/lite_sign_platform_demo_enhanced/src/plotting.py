from __future__ import annotations

from pathlib import Path
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path, paths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    paths.append(str(path))


def plot_single_run(df: pd.DataFrame, out_dir: str | Path) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    # Timeline dashboard
    fig = plt.figure(figsize=(12, 8))
    axes = [fig.add_subplot(4, 1, i + 1) for i in range(4)]
    x = df['window_id']
    axes[0].plot(x, df['ai_latency_ms'], marker='o')
    axes[0].set_ylabel('AI Latency (ms)')
    axes[0].set_title('Window Timeline Dashboard')
    axes[1].plot(x, df['overlay_skew_ms'], marker='o')
    axes[1].set_ylabel('Skew (ms)')
    if 'baseline_similarity' in df.columns:
        axes[2].plot(x, df['baseline_similarity'], marker='o')
        axes[2].set_ylabel('Similarity')
    else:
        axes[2].plot(x, df['hand_detect_rate'], marker='o')
        axes[2].set_ylabel('Hand Rate')
    axes[3].plot(x, df['pose_detect_rate'], marker='o', label='Pose')
    axes[3].plot(x, df['hand_detect_rate'], marker='s', label='Hand')
    axes[3].set_ylabel('Detect Rate')
    axes[3].set_xlabel('Window ID')
    axes[3].legend()
    fig.tight_layout()
    _save(fig, out / 'timeline_dashboard.png', paths)

    # Simple line plots
    plots = [
        ('ai_latency_ms', 'AI Latency (ms)', 'single_latency.png'),
        ('overlay_skew_ms', 'Overlay Skew (ms)', 'single_skew.png'),
        ('baseline_similarity', 'Baseline Similarity', 'single_similarity.png'),
        ('hand_detect_rate', 'Hand Detect Rate', 'single_hand_rate.png'),
    ]
    for col, ylabel, name in plots:
        if col not in df.columns:
            continue
        fig = plt.figure(figsize=(8, 4.5))
        plt.plot(df['window_id'], df[col], marker='o')
        plt.xlabel('Window ID')
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.tight_layout()
        _save(fig, out / name, paths)

    # Latency vs quality scatter
    if 'baseline_similarity' in df.columns:
        fig = plt.figure(figsize=(7, 5))
        plt.scatter(df['ai_latency_ms'], df['baseline_similarity'])
        plt.xlabel('AI Latency (ms)')
        plt.ylabel('Baseline Similarity')
        plt.title('Latency vs Similarity')
        plt.tight_layout()
        _save(fig, out / 'latency_vs_similarity.png', paths)

    # Confidence/category distribution
    if 'label' in df.columns:
        counts = df['label'].value_counts()
        fig = plt.figure(figsize=(8, 4.5))
        plt.bar(counts.index.astype(str), counts.values)
        plt.xticks(rotation=20, ha='right')
        plt.ylabel('Count')
        plt.title('Window Label Distribution')
        plt.tight_layout()
        _save(fig, out / 'label_distribution.png', paths)

    return paths


def plot_batch_summary(df: pd.DataFrame, out_dir: str | Path) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    candidate_cols = [
        ('bandwidth_mbps', 'baseline_similarity_avg', 'batch_bandwidth_similarity.png'),
        ('bandwidth_mbps', 'ai_avg_latency_ms', 'batch_bandwidth_latency.png'),
        ('ai_resolution', 'baseline_similarity_avg', 'batch_resolution_similarity.png'),
        ('ai_fps', 'ai_avg_latency_ms', 'batch_fps_latency.png'),
    ]
    for x, y, name in candidate_cols:
        if x not in df.columns or y not in df.columns:
            continue
        fig = plt.figure(figsize=(8, 4.5))
        plt.scatter(df[x], df[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'{y} vs {x}')
        plt.tight_layout()
        _save(fig, out / name, paths)

    # Pareto chart: latency vs quality
    if 'ai_avg_latency_ms' in df.columns and 'baseline_similarity_avg' in df.columns:
        fig = plt.figure(figsize=(8, 5.5))
        plt.scatter(df['ai_avg_latency_ms'], df['baseline_similarity_avg'])
        pareto = compute_pareto_front(df, 'ai_avg_latency_ms', 'baseline_similarity_avg', minimize_x=True, maximize_y=True)
        if not pareto.empty:
            pareto = pareto.sort_values('ai_avg_latency_ms')
            plt.plot(pareto['ai_avg_latency_ms'], pareto['baseline_similarity_avg'])
        plt.xlabel('Average AI Latency (ms)')
        plt.ylabel('Average Baseline Similarity')
        plt.title('Pareto: Latency vs Quality')
        plt.tight_layout()
        _save(fig, out / 'pareto_latency_quality.png', paths)

    # Heatmaps for sensitive factors
    heatmap_specs = [
        ('bandwidth_mbps', 'ai_resolution', 'baseline_similarity_avg', 'heatmap_bandwidth_resolution_similarity.png'),
        ('bandwidth_mbps', 'ai_fps', 'ai_avg_latency_ms', 'heatmap_bandwidth_fps_latency.png'),
        ('net_delay_ms', 'h264_batch_size', 'ai_avg_latency_ms', 'heatmap_delay_batch_latency.png'),
        ('window_seconds', 'ai_fps', 'baseline_similarity_avg', 'heatmap_window_fps_similarity.png'),
    ]
    for x, y, z, name in heatmap_specs:
        if all(c in df.columns for c in [x, y, z]):
            path = plot_heatmap(df, x, y, z, out / name)
            if path:
                paths.append(str(path))
    return paths


def plot_heatmap(df: pd.DataFrame, x: str, y: str, z: str, path: str | Path) -> Path | None:
    pivot = df.pivot_table(index=y, columns=x, values=z, aggfunc='mean')
    if pivot.empty:
        return None
    fig = plt.figure(figsize=(8, 5.5))
    plt.imshow(pivot.values, aspect='auto')
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{z} heatmap')
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                plt.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8)
    plt.colorbar()
    plt.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def compute_pareto_front(df: pd.DataFrame, x_col: str, y_col: str, minimize_x: bool = True, maximize_y: bool = True) -> pd.DataFrame:
    work = df[[x_col, y_col]].copy()
    mask = work[x_col].notna() & work[y_col].notna()
    work = work[mask]
    if work.empty:
        return df.iloc[0:0].copy()
    selected = []
    records = df.loc[work.index]
    for idx, row in records.iterrows():
        dominated = False
        for jdx, other in records.iterrows():
            if idx == jdx:
                continue
            x_better = other[x_col] <= row[x_col] if minimize_x else other[x_col] >= row[x_col]
            y_better = other[y_col] >= row[y_col] if maximize_y else other[y_col] <= row[y_col]
            strictly = other[x_col] != row[x_col] or other[y_col] != row[y_col]
            if x_better and y_better and strictly:
                dominated = True
                break
        if not dominated:
            selected.append(idx)
    return df.loc[selected].copy()
