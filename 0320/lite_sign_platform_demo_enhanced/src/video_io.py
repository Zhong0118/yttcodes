from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator
import cv2
import numpy as np
import time


@dataclass
class FramePacket:
    frame_idx: int
    src_ts_ms: float
    user_ts_ms: float
    ai_available_ts_ms: float
    frame_bgr: np.ndarray
    ai_frame_bgr: np.ndarray
    encoded_size_bytes: int
    simulated_encode_ms: float


class VideoSource:
    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open video: {self.video_path}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.duration_sec = self.frame_count / self.fps if self.fps > 0 else 0.0

    def iter_frames(self, max_duration_sec: float | None = None) -> Generator[tuple[int, float, np.ndarray], None, None]:
        idx = 0
        max_idx = None
        if max_duration_sec is not None:
            max_idx = int(max_duration_sec * self.fps)
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if max_idx is not None and idx >= max_idx:
                break
            src_ts_ms = idx * 1000.0 / self.fps
            yield idx, src_ts_ms, frame
            idx += 1
        self.cap.release()


def resize_keep_aspect(frame: np.ndarray, short_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if min(h, w) == short_side:
        return frame
    if h < w:
        new_h = short_side
        new_w = int(w * short_side / h)
    else:
        new_w = short_side
        new_h = int(h * short_side / w)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def simulate_compression(frame_bgr: np.ndarray, jpeg_quality: int) -> tuple[np.ndarray, int, float]:
    t0 = time.perf_counter()
    ok, enc = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError('JPEG encode failed')
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    encode_ms = (time.perf_counter() - t0) * 1000.0
    return dec, int(enc.size), encode_ms


def packetize_video_frame(
    frame_idx: int,
    src_ts_ms: float,
    frame_bgr: np.ndarray,
    bandwidth_mbps: float,
    net_delay_ms: float,
    ai_resolution: int,
    jpeg_quality: int,
    extra_ai_delay_ms: float,
) -> FramePacket:
    user_ts_ms = src_ts_ms
    resized = resize_keep_aspect(frame_bgr, ai_resolution)
    ai_frame, size_bytes, encode_ms = simulate_compression(resized, jpeg_quality)
    tx_ms = size_bytes * 8.0 / max(bandwidth_mbps, 1e-6) / 1000.0
    ai_available_ts_ms = src_ts_ms + tx_ms + net_delay_ms + encode_ms + extra_ai_delay_ms
    return FramePacket(
        frame_idx=frame_idx,
        src_ts_ms=src_ts_ms,
        user_ts_ms=user_ts_ms,
        ai_available_ts_ms=ai_available_ts_ms,
        frame_bgr=frame_bgr,
        ai_frame_bgr=ai_frame,
        encoded_size_bytes=size_bytes,
        simulated_encode_ms=encode_ms,
    )


def write_side_by_side_video(
    original_video_path: str,
    out_path: str | Path,
    ai_resolution: int,
    jpeg_quality: int,
    max_duration_sec: float | None = None,
    overlay_text: str | None = None,
) -> str:
    src = VideoSource(original_video_path)
    out_path = str(out_path)
    cap = cv2.VideoCapture(original_video_path)
    fps = src.fps or 25.0
    max_idx = int(max_duration_sec * fps) if max_duration_sec else None
    idx = 0
    writer = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_idx is not None and idx >= max_idx:
            break

        ai_frame = resize_keep_aspect(frame, ai_resolution)
        ai_frame, _, _ = simulate_compression(ai_frame, jpeg_quality)
        h1, w1 = frame.shape[:2]
        target_h = max(h1, ai_frame.shape[0])

        def pad_h(img, h):
            if img.shape[0] == h:
                return img
            scale = h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            return cv2.resize(img, (new_w, h))

        left = pad_h(frame, target_h)
        right = pad_h(ai_frame, target_h)
        combo = np.hstack([left, right])

        cv2.putText(combo, 'User View', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(combo, 'AI View', (left.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        if overlay_text:
            cv2.putText(combo, overlay_text, (20, combo.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (combo.shape[1], combo.shape[0]))

        writer.write(combo)
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    return out_path