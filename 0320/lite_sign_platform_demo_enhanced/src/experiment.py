from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable
import itertools
import math
import pandas as pd
import numpy as np

from .analyzer import KeypointAnalyzer
from .config import ExperimentConfig
from .metrics import cosine_similarity, summarize_window_df
from .video_io import VideoSource, packetize_video_frame
from .utils import ensure_dir, now_ts, save_json


class LiteSignExperiment:
    def __init__(self, analyzer: KeypointAnalyzer | None = None):
        self.analyzer = analyzer or KeypointAnalyzer()

    

    def run_single(self, video_path: str, config: ExperimentConfig, baseline_embeddings: list[np.ndarray] | None = None) -> tuple[pd.DataFrame, dict]:
        source = VideoSource(video_path)
        packets = []
        keep_every = max(1, int(round(source.fps / max(config.ai_fps, 1e-6))))
        for idx, src_ts_ms, frame in source.iter_frames(max_duration_sec=config.max_duration_sec):
            if idx % keep_every != 0:
                continue
            pkt = packetize_video_frame(
                frame_idx=idx,
                src_ts_ms=src_ts_ms,
                frame_bgr=frame,
                bandwidth_mbps=config.bandwidth_mbps,
                net_delay_ms=config.net_delay_ms,
                ai_resolution=config.ai_resolution,
                jpeg_quality=config.jpeg_quality,
                extra_ai_delay_ms=config.extra_ai_delay_ms,
            )
            packets.append(pkt)
        rows = self._window_infer(packets, source.fps, config, baseline_embeddings)
        df = pd.DataFrame(rows)
        summary = summarize_window_df(df)
        summary.update({f"cfg_{k}": v for k, v in asdict(config).items()})
        return df, summary

    def build_baseline_embeddings(self, video_path: str, max_duration_sec: float | None = None, ai_fps: float = 8.0) -> list[np.ndarray]:
        cfg = ExperimentConfig(
            bandwidth_mbps=1e9,
            net_delay_ms=0.0,
            ai_resolution=480,
            ai_fps=ai_fps,
            window_seconds=4.0,
            stride_seconds=4.0,
            extra_ai_delay_ms=0.0,
            jpeg_quality=95,
            h264_batch_size=1,
            max_duration_sec=max_duration_sec,
        )
        df, _ = self.run_single(video_path, cfg, baseline_embeddings=None)
        embeddings = [np.fromstring(s, sep=' ') for s in df['embedding_str'].tolist()]
        return embeddings

    def _window_infer(self, packets, src_fps: float, config: ExperimentConfig, baseline_embeddings: list[np.ndarray] | None):
        if not packets:
            return []
        rows = []
        win_ms = config.window_seconds * 1000.0
        stride_ms = config.stride_seconds * 1000.0
        t0 = packets[0].src_ts_ms
        tend = packets[-1].src_ts_ms
        window_id = 0
        start = t0
        while start <= tend:
            end = start + win_ms
            sel = [p for p in packets if start <= p.src_ts_ms < end]
            if not sel:
                start += stride_ms
                window_id += 1
                continue
            res = self.analyzer.analyze_window([p.ai_frame_bgr for p in sel], start, end, window_id)
            last_ai_available = max(p.ai_available_ts_ms for p in sel)
            ai_latency_ms = max(0.0, last_ai_available - end) + res.infer_ms + (config.h264_batch_size - 1) * 3.0
            overlay_skew_ms = ai_latency_ms
            baseline_similarity = None
            if baseline_embeddings is not None and window_id < len(baseline_embeddings):
                baseline_similarity = cosine_similarity(res.embedding, baseline_embeddings[window_id])
            rows.append({
                "window_id": window_id,
                "start_ms": start,
                "end_ms": end,
                "label": res.label,
                "confidence": res.confidence,
                "infer_ms": res.infer_ms,
                "pose_detect_rate": res.pose_detect_rate,
                "hand_detect_rate": res.hand_detect_rate,
                "keypoint_completeness": res.keypoint_completeness,
                "motion_energy": res.motion_energy,
                "stability_score": res.stability_score,
                "n_frames": res.n_frames,
                "ai_latency_ms": ai_latency_ms,
                "overlay_skew_ms": overlay_skew_ms,
                "avg_encoded_kb": float(np.mean([p.encoded_size_bytes for p in sel]) / 1024.0),
                "avg_encode_ms": float(np.mean([p.simulated_encode_ms for p in sel])),
                "baseline_similarity": baseline_similarity,
                "embedding_str": ' '.join(f'{x:.6f}' for x in res.embedding.tolist()),
            })
            start += stride_ms
            window_id += 1
        return rows


def batch_configs(grid: dict, fast: bool = False):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    all_cfgs = []
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        all_cfgs.append(cfg)
    if fast:
        return all_cfgs[: min(12, len(all_cfgs))]
    return all_cfgs
