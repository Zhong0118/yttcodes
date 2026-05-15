from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


def p95(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, 95))


def _safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return float("nan")
    return float(s.mean())


def _safe_p95(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return float("nan")
    return p95(s)


def summarize_window_df(df: pd.DataFrame) -> Dict[str, float | int | str]:
    if df is None or df.empty:
        return {"n_windows": 0}

    summary: Dict[str, float | int | str] = {
        "n_windows": int(len(df)),
    }

    # -----------------------------
    # core existing stats
    # -----------------------------
    if "first_result_latency_ms" in df.columns:
        summary["first_result_latency_ms"] = _safe_mean(df["first_result_latency_ms"])
        summary["first_result_latency_avg_ms"] = _safe_mean(df["first_result_latency_ms"])
        summary["first_result_latency_p95_ms"] = _safe_p95(df["first_result_latency_ms"])

    if "ai_latency_ms" in df.columns:
        summary["ai_avg_latency_ms"] = _safe_mean(df["ai_latency_ms"])
        summary["ai_p95_latency_ms"] = _safe_p95(df["ai_latency_ms"])

    if "payload_kb" in df.columns:
        summary["payload_avg_kb"] = _safe_mean(df["payload_kb"])
        summary["payload_p95_kb"] = _safe_p95(df["payload_kb"])

    if "infer_ms" in df.columns:
        summary["infer_avg_ms"] = _safe_mean(df["infer_ms"])
        summary["infer_p95_ms"] = _safe_p95(df["infer_ms"])

    # -----------------------------
    # new uplink stats
    # -----------------------------
    if "uplink_queue_ms" in df.columns:
        summary["uplink_queue_avg_ms"] = _safe_mean(df["uplink_queue_ms"])
        summary["uplink_queue_p95_ms"] = _safe_p95(df["uplink_queue_ms"])

    if "uplink_tx_ms" in df.columns:
        summary["uplink_tx_avg_ms"] = _safe_mean(df["uplink_tx_ms"])
        summary["uplink_tx_p95_ms"] = _safe_p95(df["uplink_tx_ms"])

    if "server_recv_gap_ms" in df.columns:
        summary["server_recv_gap_avg_ms"] = _safe_mean(df["server_recv_gap_ms"])
        summary["server_recv_gap_p95_ms"] = _safe_p95(df["server_recv_gap_ms"])

    # -----------------------------
    # new downlink stats
    # -----------------------------
    if "downlink_render_delay_ms" in df.columns:
        summary["downlink_render_delay_avg_ms"] = _safe_mean(df["downlink_render_delay_ms"])
        summary["downlink_render_delay_p95_ms"] = _safe_p95(df["downlink_render_delay_ms"])

    if "display_fps_actual" in df.columns:
        summary["display_fps_actual_avg"] = _safe_mean(df["display_fps_actual"])

    if "display_stutter_score" in df.columns:
        summary["display_stutter_score_avg"] = _safe_mean(df["display_stutter_score"])
        summary["display_stutter_score_p95"] = _safe_p95(df["display_stutter_score"])

    if "user_perceived_latency_ms" in df.columns:
        summary["user_perceived_latency_avg_ms"] = _safe_mean(df["user_perceived_latency_ms"])
        summary["user_perceived_latency_p95_ms"] = _safe_p95(df["user_perceived_latency_ms"])

    # -----------------------------
    # quality / model stats
    # -----------------------------
    for src, dst in [
        ("pose_detect_rate", "pose_detect_rate_avg"),
        ("hand_detect_rate", "hand_detect_rate_avg"),
        ("face_detect_rate", "face_detect_rate_avg"),
        ("keypoint_completeness", "keypoint_completeness_avg"),
        ("motion_energy", "motion_energy_avg"),
        ("stability_score", "stability_score_avg"),
        ("downlink_frame_interval_ms", "downlink_frame_interval_ms_avg"),
        ("downlink_perceived_latency_ms", "downlink_perceived_latency_ms_avg"),
        ("user_smoothness_score", "user_smoothness_score_avg"),
        ("user_visual_quality_score", "user_visual_quality_score_avg"),
    ]:
        if src in df.columns:
            summary[dst] = _safe_mean(df[src])

    # -----------------------------
    # stacked-bar helpers
    # -----------------------------
    summary["uplink_total_avg_ms"] = (
        float(summary.get("uplink_queue_avg_ms", 0.0) or 0.0)
        + float(summary.get("uplink_tx_avg_ms", 0.0) or 0.0)
    )
    summary["downlink_total_avg_ms"] = float(summary.get("downlink_render_delay_avg_ms", 0.0) or 0.0)

    return summary