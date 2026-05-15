from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from .classifier import ShortPoseUnderstandingModel
from .config import Scenario2Config
from .extractor import PoseStickExtractor, resize_keep_aspect
from .metrics import summarize_window_df
from .windowing import SlidingWindowBuffer


def estimate_tcp_send_ms(payload_kb: float, bandwidth_mbps: float, fixed_rtt_ms: float) -> float:
    bandwidth_kBps = max(1e-6, bandwidth_mbps * 125.0)  # 1 Mbps = 125 kB/s
    serialization_ms = (payload_kb / bandwidth_kBps) * 1000.0
    return float(serialization_ms + fixed_rtt_ms)


def estimate_payload_kb(window: List[Dict], cfg: Scenario2Config) -> float:
    serialized = json.dumps(window)
    return len(serialized.encode("utf-8")) / 1024.0


def downlink_quality_score(resolution: int) -> float:
    mapping = {360: 0.55, 480: 0.72, 720: 0.9, 1080: 1.0}
    if resolution in mapping:
        return mapping[resolution]
    if resolution < 360:
        return 0.4
    return min(1.0, 0.55 + (resolution - 360) / 900.0)


def downlink_smoothness_score(fps: float) -> float:
    if fps <= 10:
        return 0.45
    if fps <= 15:
        return 0.68
    if fps <= 24:
        return 0.88
    return 0.95


class Scenario2Experiment:
    def __init__(self, extractor: PoseStickExtractor | None = None, model: ShortPoseUnderstandingModel | None = None):
        self.extractor = extractor or PoseStickExtractor()
        self.model = model or ShortPoseUnderstandingModel()

    def run_single(self, video_path: str, cfg: Scenario2Config) -> Tuple[pd.DataFrame, Dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 1e-3:
            src_fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            source_video_duration_sec = total_frames / src_fps
        else:
            source_video_duration_sec = float("nan")

        if cfg.max_duration_sec is not None:
            max_frames = int(round(cfg.max_duration_sec * src_fps))
        else:
            max_frames = None

        if np.isnan(source_video_duration_sec):
            effective_video_duration_sec = float(cfg.max_duration_sec) if cfg.max_duration_sec is not None else float("nan")
        else:
            if cfg.max_duration_sec is None:
                effective_video_duration_sec = float(source_video_duration_sec)
            else:
                effective_video_duration_sec = float(min(source_video_duration_sec, cfg.max_duration_sec))

        # sample frames to approximate uplink fps
        sample_period = max(1, int(round(src_fps / max(cfg.uplink_fps, 1e-6))))
        buffer = SlidingWindowBuffer(cfg.uplink_fps, cfg.window_seconds, cfg.stride_seconds)

        rows: List[Dict] = []
        first_result_latency_ms = None
        prev_server_recv_ts = None
        prev_client_render_ts = None

        frame_idx = 0
        sampled_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            if frame_idx % sample_period != 0:
                frame_idx += 1
                continue

            sampled_idx += 1
            frame_idx += 1

            t_frame_capture = time.time()
            frame_up = resize_keep_aspect(frame, cfg.uplink_resolution)

            feat = self.extractor.extract(frame_up)
            feat["frame_idx"] = int(frame_idx)
            feat["sampled_idx"] = int(sampled_idx)
            feat["ts"] = float(t_frame_capture)

            window = buffer.push(feat)
            if window is None:
                continue

            # -----------------------------
            # window/chunk lifecycle timestamps
            # -----------------------------
            t_chunk_ready = time.time()

            # queue before actual tx
            simulated_queue_ms = float(getattr(cfg, "extra_uplink_delay_ms", 0.0))
            t_send_start = t_chunk_ready + simulated_queue_ms / 1000.0

            payload_kb = estimate_payload_kb(window, cfg)
            uplink_tx_ms = estimate_tcp_send_ms(payload_kb, cfg.tcp_bandwidth_mbps, cfg.tcp_fixed_rtt_ms)
            t_send_end = t_send_start + uplink_tx_ms / 1000.0

            # approximate server receive time
            server_recv_ts = t_send_end

            if prev_server_recv_ts is None:
                server_recv_gap_ms = float("nan")
            else:
                server_recv_gap_ms = (server_recv_ts - prev_server_recv_ts) * 1000.0
            prev_server_recv_ts = server_recv_ts

            # server inference
            infer_t0 = time.time()
            result = self.model.infer(window)
            infer_ms = (time.time() - infer_t0) * 1000.0
            server_result_ready_ts = server_recv_ts + infer_ms / 1000.0

            # client receives result
            client_result_recv_ts = server_result_ready_ts

            # -----------------------------
            # downlink/render simulation
            # -----------------------------
            ideal_display_interval_ms = 1000.0 / max(cfg.downlink_fps, 1e-6)
            downlink_render_delay_ms = float(max(0.0, ideal_display_interval_ms / 2.0 + cfg.extra_downlink_delay_ms))
            client_render_ts = client_result_recv_ts + downlink_render_delay_ms / 1000.0

            # actual display fps
            if prev_client_render_ts is None:
                display_fps_actual = float(cfg.downlink_fps)
                display_stutter_score = 0.0
            else:
                actual_dt_ms = (client_render_ts - prev_client_render_ts) * 1000.0
                display_fps_actual = 1000.0 / max(actual_dt_ms, 1e-6)
                display_stutter_score = abs(actual_dt_ms - ideal_display_interval_ms) / max(ideal_display_interval_ms, 1e-6)
            prev_client_render_ts = client_render_ts

            # total perceived latency
            user_perceived_latency_ms = (client_render_ts - t_chunk_ready) * 1000.0

            # AI latency keeps your original "window wait + tx + infer" interpretation
            window_wait_ms = cfg.window_seconds * 1000.0
            ai_latency_ms = window_wait_ms + uplink_tx_ms + infer_ms + downlink_render_delay_ms

            if first_result_latency_ms is None:
                first_result_latency_ms = (client_result_recv_ts - t_chunk_ready) * 1000.0

            user_visual_quality_score = downlink_quality_score(cfg.downlink_resolution)
            user_smoothness_score = downlink_smoothness_score(display_fps_actual)

            row = {
                "window_id": len(rows),

                # -----------------------------
                # 上线统计
                # -----------------------------
                "uplink_queue_ms": float(simulated_queue_ms),
                "uplink_tx_ms": float(uplink_tx_ms),
                "server_recv_gap_ms": float(server_recv_gap_ms),
                "first_result_latency_ms": float((client_result_recv_ts - t_chunk_ready) * 1000.0),

                # -----------------------------
                # 下线统计
                # -----------------------------
                "downlink_render_delay_ms": float(downlink_render_delay_ms),
                "display_fps_actual": float(display_fps_actual),
                "display_stutter_score": float(display_stutter_score),
                "user_perceived_latency_ms": float(user_perceived_latency_ms),

                # -----------------------------
                # 核心时延与负载
                # -----------------------------
                "ai_latency_ms": float(ai_latency_ms),
                "tcp_est_send_ms": float(uplink_tx_ms),
                "payload_kb": float(payload_kb),
                "infer_ms": float(infer_ms),

                # -----------------------------
                # 配置镜像
                # -----------------------------
                "window_seconds": cfg.window_seconds,
                "chunk_size_frames": cfg.chunk_size_frames,
                "uplink_resolution": cfg.uplink_resolution,
                "uplink_fps": cfg.uplink_fps,
                "downlink_resolution": cfg.downlink_resolution,
                "downlink_fps": cfg.downlink_fps,
                "tcp_bandwidth_mbps": cfg.tcp_bandwidth_mbps,
                "tcp_fixed_rtt_ms": cfg.tcp_fixed_rtt_ms,

                # -----------------------------
                # 用户体验代理指标
                # -----------------------------
                "downlink_frame_interval_ms": float(ideal_display_interval_ms),
                "downlink_perceived_latency_ms": float(downlink_render_delay_ms),
                "user_visual_quality_score": float(user_visual_quality_score),
                "user_smoothness_score": float(user_smoothness_score),
            }
            row.update(result)
            rows.append(row)

        cap.release()
        df = pd.DataFrame(rows)
        summary = summarize_window_df(df)
        summary.update(cfg.to_dict())

        summary["source_video_duration_sec"] = float(source_video_duration_sec)
        summary["effective_video_duration_sec"] = float(effective_video_duration_sec)
        summary["source_video_total_frames"] = int(total_frames) if total_frames > 0 else -1
        summary["source_video_fps"] = float(src_fps)

        return df, summary


def save_single_outputs(df: pd.DataFrame, summary: Dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "windows.csv", index=False, encoding="utf-8-sig")
    df.to_excel(out_dir / "windows.xlsx", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)