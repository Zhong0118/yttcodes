from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np


def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 25.0

    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, float(fps)


def sample_to_target_fps(frames: List[np.ndarray], src_fps: float, target_fps: int) -> Tuple[List[np.ndarray], List[int]]:
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")
    if not frames:
        return [], []

    if src_fps <= target_fps:
        return frames, list(range(len(frames)))

    step = src_fps / target_fps
    sampled = []
    indices = []
    idx = 0.0
    while int(idx) < len(frames):
        i = int(idx)
        sampled.append(frames[i])
        indices.append(i)
        idx += step
    return sampled, indices


def resize_frames(frames: List[np.ndarray], size: int) -> List[np.ndarray]:
    return [cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR) for f in frames]


def split_into_chunks(frames: List[np.ndarray], frame_indices: List[int], src_fps: float, clip_len: int, stride_len: int):
    if clip_len <= 0 or stride_len <= 0:
        raise ValueError("clip_len and stride_len must be positive")

    chunks = []
    n = len(frames)
    start = 0
    chunk_id = 0
    while start + clip_len <= n:
        end = start + clip_len
        chunk_frames = frames[start:end]
        raw_indices = frame_indices[start:end]
        start_sec = raw_indices[0] / src_fps
        end_sec = (raw_indices[-1] + 1) / src_fps
        chunks.append({
            "chunk_id": chunk_id,
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "frames": chunk_frames,
            "n_frames": len(chunk_frames),
        })
        start += stride_len
        chunk_id += 1
    return chunks


def estimate_payload_kb(chunk_frames: List[np.ndarray], jpeg_quality: int) -> float:
    total_bytes = 0
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    for frame in chunk_frames:
        ok, enc = cv2.imencode(".jpg", frame, encode_param)
        if ok:
            total_bytes += int(enc.size)
    return total_bytes / 1024.0
