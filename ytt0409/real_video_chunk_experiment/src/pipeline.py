from __future__ import annotations
from typing import Dict, Any, List, Tuple
import pandas as pd

from src.config import RunConfig
from src.video_utils import (
    read_video,
    sample_to_target_fps,
    resize_frames,
    split_into_chunks,
    estimate_payload_kb,
)
from src.network_sim import simulate_transfer
from src.models import build_model


def analyze_single_video(cfg: RunConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frames, src_fps = read_video(cfg.video_path)
    sampled_frames, sampled_indices = sample_to_target_fps(frames, src_fps=src_fps, target_fps=cfg.ai_fps)
    resized_frames = resize_frames(sampled_frames, size=cfg.ai_input_size)
    chunks = split_into_chunks(
        resized_frames,
        frame_indices=sampled_indices,
        src_fps=src_fps,
        clip_len=cfg.clip_len,
        stride_len=cfg.stride_len,
    )

    model = build_model(cfg.model_name)

    rows: List[Dict[str, Any]] = []
    first_result_ms = None

    for chunk in chunks:
        payload_kb = estimate_payload_kb(chunk["frames"], jpeg_quality=cfg.jpeg_quality)
        net = simulate_transfer(
            payload_kb=payload_kb,
            bandwidth_kbps=cfg.bandwidth_kbps,
            network_delay_ms=cfg.network_delay_ms,
        )
        pred = model.predict_chunk(chunk["frames"])

        infer_ms = float(pred["infer_ms"])
        chunk_first_result_ms = float(net["transfer_ms"] + infer_ms)
        end_to_end_ms = float(chunk_first_result_ms + 8.0)

        if first_result_ms is None:
            first_result_ms = chunk_first_result_ms

        rows.append({
            "chunk_id": int(chunk["chunk_id"]),
            "start_sec": float(chunk["start_sec"]),
            "end_sec": float(chunk["end_sec"]),
            "duration_sec": float(chunk["end_sec"] - chunk["start_sec"]),
            "n_frames": int(chunk["n_frames"]),
            "pred_label": pred["pred_label"],
            "confidence": float(pred["confidence"]),
            "infer_ms": infer_ms,
            "payload_kb": float(net["payload_kb"]),
            "tx_ms": float(net["tx_ms"]),
            "network_delay_ms": float(net["network_delay_ms"]),
            "first_result_ms": chunk_first_result_ms,
            "end_to_end_ms": end_to_end_ms,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No chunks were produced. Try reducing clip_len or checking the video.")

    summary = {
        "video_path": cfg.video_path,
        "src_fps": float(src_fps),
        "sampled_fps": int(cfg.ai_fps),
        "n_source_frames": int(len(frames)),
        "n_sampled_frames": int(len(sampled_frames)),
        "n_chunks": int(len(df)),
        "infer_avg_ms": float(df["infer_ms"].mean()),
        "infer_p95_ms": float(df["infer_ms"].quantile(0.95)),
        "first_result_ms": float(first_result_ms),
        "end_to_end_avg_ms": float(df["end_to_end_ms"].mean()),
        "payload_avg_kb": float(df["payload_kb"].mean()),
        "tx_avg_ms": float(df["tx_ms"].mean()),
    }
    return df, summary
