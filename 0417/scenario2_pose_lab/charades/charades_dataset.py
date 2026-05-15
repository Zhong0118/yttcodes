from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_video_clip_frames(
    video_path: str,
    clip_start_sec: float,
    clip_end_sec: float,
    num_frames: int,
    sampling_rate: int,
    side_size: int,
    crop_size: int,
) -> np.ndarray:
    """
    读取指定时间段的 clip，并采样成固定 num_frames。
    返回 T,H,W,C
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 25.0

    start_idx = max(0, int(round(clip_start_sec * fps)))
    end_idx = max(start_idx + 1, int(round(clip_end_sec * fps)))

    frames: List[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx > end_idx:
            break
        if frame_idx >= start_idx:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames in clip range: {video_path} [{clip_start_sec}, {clip_end_sec}]")

    # 先按 sampling_rate 尝试取样；不够再线性补
    needed_span = num_frames * sampling_rate
    if len(frames) >= needed_span:
        start = max(0, (len(frames) - needed_span) // 2)
        idxs = [start + i * sampling_rate for i in range(num_frames)]
    else:
        idxs = np.linspace(0, len(frames) - 1, num_frames).astype(int).tolist()

    sampled = [frames[i] for i in idxs]
    arr = np.stack(sampled, axis=0)  # T,H,W,C

    # resize + center crop
    out = []
    for img in arr:
        h, w = img.shape[:2]
        short_side = min(h, w)
        scale = side_size / float(max(1, short_side))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        h2, w2 = img.shape[:2]
        y0 = max(0, (h2 - crop_size) // 2)
        x0 = max(0, (w2 - crop_size) // 2)
        img = img[y0:y0 + crop_size, x0:x0 + crop_size]

        if img.shape[0] != crop_size or img.shape[1] != crop_size:
            pad_h = crop_size - img.shape[0]
            pad_w = crop_size - img.shape[1]
            img = np.pad(
                img,
                ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            img = img[:crop_size, :crop_size]

        out.append(img)

    return np.stack(out, axis=0)


def to_tensor(frames_thwc: np.ndarray) -> torch.Tensor:
    """
    T,H,W,C -> C,T,H,W
    """
    x = torch.from_numpy(frames_thwc).float() / 255.0
    x = x.permute(3, 0, 1, 2)  # C,T,H,W

    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    x = (x - mean) / std
    return x


class CharadesClipDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        num_classes: int,
        num_frames: int = 8,
        sampling_rate: int = 8,
        side_size: int = 256,
        crop_size: int = 224,
    ):
        self.df = pd.read_csv(manifest_csv)
        self.num_classes = int(num_classes)
        self.num_frames = int(num_frames)
        self.sampling_rate = int(sampling_rate)
        self.side_size = int(side_size)
        self.crop_size = int(crop_size)

        required = {"video_path", "start_sec", "end_sec", "label_indices"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in manifest: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        video_path = str(row["video_path"])
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])

        frames = load_video_clip_frames(
            video_path=video_path,
            clip_start_sec=start_sec,
            clip_end_sec=end_sec,
            num_frames=self.num_frames,
            sampling_rate=self.sampling_rate,
            side_size=self.side_size,
            crop_size=self.crop_size,
        )
        video = to_tensor(frames)

        target = torch.zeros(self.num_classes, dtype=torch.float32)
        label_indices_str = str(row["label_indices"]).strip()
        if label_indices_str and label_indices_str.lower() != "nan":
            for x in label_indices_str.split(";"):
                x = x.strip()
                if x != "":
                    target[int(x)] = 1.0

        sample = {
            "video": video,
            "target": target,
            "video_id": str(row.get("video_id", "")),
            "clip_id": int(row.get("clip_id", idx)),
            "video_path": video_path,
            "start_sec": start_sec,
            "end_sec": end_sec,
        }
        return sample