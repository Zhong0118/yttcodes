from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2


@dataclass
class ActionSegment:
    action_id: str
    start_sec: float
    end_sec: float


def load_charades_classes(classes_txt: str | Path) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, str]]:
    """
    读取 Charades_v1_classes.txt
    返回：
      action_id_to_index: {"c000": 0, ...}
      index_to_action_id: {0: "c000", ...}
      action_id_to_text:  {"c000": "Holding some clothes", ...}
    """
    classes_txt = Path(classes_txt)
    action_id_to_index: Dict[str, int] = {}
    index_to_action_id: Dict[int, str] = {}
    action_id_to_text: Dict[str, str] = {}

    with open(classes_txt, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f if x.strip()]

    for idx, line in enumerate(lines):
        parts = line.split(" ", 1)
        action_id = parts[0].strip()
        action_text = parts[1].strip() if len(parts) > 1 else action_id

        action_id_to_index[action_id] = idx
        index_to_action_id[idx] = action_id
        action_id_to_text[action_id] = action_text

    return action_id_to_index, index_to_action_id, action_id_to_text


def parse_actions_field(actions_str: str | None) -> List[ActionSegment]:
    """
    解析类似:
      c092 11.90 21.20;c147 0.00 12.60
    """
    if actions_str is None:
        return []

    s = str(actions_str).strip()
    if s == "" or s.lower() == "nan":
        return []

    out: List[ActionSegment] = []
    items = [x.strip() for x in s.split(";") if x.strip()]
    for item in items:
        parts = item.split()
        if len(parts) != 3:
            continue
        action_id = parts[0]
        start_sec = float(parts[1])
        end_sec = float(parts[2])
        out.append(ActionSegment(action_id=action_id, start_sec=start_sec, end_sec=end_sec))
    return out


def probe_video_duration(video_path: str | Path) -> float:
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps is None or fps <= 1e-6 or n_frames is None or n_frames <= 0:
        return 0.0
    return float(n_frames / fps)


def find_video_path(videos_root: str | Path, video_id: str) -> Optional[Path]:
    """
    按 Charades id 找视频。
    优先找 .mp4，其次其他常见格式。
    """
    videos_root = Path(videos_root)
    exts = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"]
    for ext in exts:
        p = videos_root / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


def overlap_seconds(a0: float, a1: float, b0: float, b1: float) -> float:
    left = max(a0, b0)
    right = min(a1, b1)
    return max(0.0, right - left)


def build_multihot_for_clip(
    clip_start: float,
    clip_end: float,
    segments: List[ActionSegment],
    action_id_to_index: Dict[str, int],
    min_overlap_ratio: float = 0.0,
) -> Tuple[List[int], List[str]]:
    """
    对一个 clip 构造多标签。
    若某 action 与 clip 的重叠 / clip_len >= min_overlap_ratio，则认为该 action 出现。
    """
    clip_len = max(1e-6, clip_end - clip_start)

    indices: List[int] = []
    ids: List[str] = []

    for seg in segments:
        ov = overlap_seconds(clip_start, clip_end, seg.start_sec, seg.end_sec)
        ratio = ov / clip_len
        if ratio >= min_overlap_ratio and seg.action_id in action_id_to_index:
            indices.append(action_id_to_index[seg.action_id])
            ids.append(seg.action_id)

    indices = sorted(set(indices))
    ids = sorted(set(ids))
    return indices, ids


def save_csv(rows: List[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            pass
        return

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)