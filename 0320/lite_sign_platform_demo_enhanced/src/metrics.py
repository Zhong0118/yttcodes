from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def summarize_window_df(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    summary = {
        "n_windows": int(len(df)),
        "ai_avg_latency_ms": float(df["ai_latency_ms"].mean()),
        "ai_p95_latency_ms": float(df["ai_latency_ms"].quantile(0.95)),
        "overlay_skew_avg_ms": float(df["overlay_skew_ms"].mean()),
        "infer_avg_ms": float(df["infer_ms"].mean()),
        "pose_detect_rate_avg": float(df["pose_detect_rate"].mean()),
        "hand_detect_rate_avg": float(df["hand_detect_rate"].mean()),
        "keypoint_completeness_avg": float(df["keypoint_completeness"].mean()),
        "motion_energy_avg": float(df["motion_energy"].mean()),
        "stability_score_avg": float(df["stability_score"].mean()),
        "effective_fps": float(df["n_frames"].sum() / max((df["end_ms"].max() - df["start_ms"].min()) / 1000.0, 1e-6)),
    }
    if "baseline_similarity" in df.columns:
        summary["baseline_similarity_avg"] = float(df["baseline_similarity"].mean())
    return summary
