from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from charades_utils import (
    build_multihot_for_clip,
    find_video_path,
    load_charades_classes,
    parse_actions_field,
    probe_video_duration,
    save_csv,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare clip-level manifest for Charades.")
    parser.add_argument("--annotation-csv", required=True, help="例如 Charades_v1_train.csv 或你整理后的 csv")
    parser.add_argument("--classes-txt", required=True, help="Charades_v1_classes.txt")
    parser.add_argument("--videos-root", required=True, help="视频目录，视频名是 id.mp4")
    parser.add_argument("--out-dir", default="charades_manifest_outputs")

    parser.add_argument("--clip-seconds", type=float, default=2.56, help="clip 时长，默认约对应 8x8@25fps")
    parser.add_argument("--stride-seconds", type=float, default=1.28, help="clip 步长")
    parser.add_argument("--min-overlap-ratio", type=float, default=0.1, help="动作和 clip 重叠比例阈值")
    parser.add_argument("--drop-empty-clips", action="store_true", help="是否丢弃没有任何动作标签的 clip")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    action_id_to_index, index_to_action_id, action_id_to_text = load_charades_classes(args.classes_txt)
    df = pd.read_csv(args.annotation_csv)

    required_cols = {"id", "actions"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in annotation csv: {missing}")

    rows: List[dict] = []
    skipped_rows: List[dict] = []

    for _, row in df.iterrows():
        video_id = str(row["id"]).strip()
        actions_str = row["actions"]

        video_path = find_video_path(args.videos_root, video_id)
        if video_path is None:
            skipped_rows.append(
                {
                    "video_id": video_id,
                    "reason": "video_not_found",
                }
            )
            continue

        try:
            duration_sec = probe_video_duration(video_path)
        except Exception as e:
            skipped_rows.append(
                {
                    "video_id": video_id,
                    "reason": f"probe_failed: {e}",
                }
            )
            continue

        segments = parse_actions_field(actions_str)
        if duration_sec <= 0:
            skipped_rows.append(
                {
                    "video_id": video_id,
                    "reason": "invalid_duration",
                }
            )
            continue

        clip_len = float(args.clip_seconds)
        stride = float(args.stride_seconds)

        clip_id = 0
        start_sec = 0.0

        while start_sec < duration_sec:
            end_sec = min(duration_sec, start_sec + clip_len)
            label_indices, label_ids = build_multihot_for_clip(
                clip_start=start_sec,
                clip_end=end_sec,
                segments=segments,
                action_id_to_index=action_id_to_index,
                min_overlap_ratio=float(args.min_overlap_ratio),
            )

            if args.drop_empty_clips and len(label_indices) == 0:
                start_sec += stride
                clip_id += 1
                continue

            label_texts = [action_id_to_text[x] for x in label_ids]

            rows.append(
                {
                    "video_id": video_id,
                    "video_path": str(video_path.resolve()),
                    "clip_id": clip_id,
                    "start_sec": round(start_sec, 4),
                    "end_sec": round(end_sec, 4),
                    "duration_sec": round(duration_sec, 4),
                    "label_ids": ";".join(label_ids),
                    "label_indices": ";".join(str(x) for x in label_indices),
                    "label_texts": " | ".join(label_texts),
                }
            )

            if end_sec >= duration_sec:
                break

            start_sec += stride
            clip_id += 1

    manifest_csv = out_dir / "charades_clip_manifest.csv"
    save_csv(rows, manifest_csv)

    skipped_csv = out_dir / "skipped_videos.csv"
    save_csv(skipped_rows, skipped_csv)

    meta = {
        "annotation_csv": str(Path(args.annotation_csv).resolve()),
        "classes_txt": str(Path(args.classes_txt).resolve()),
        "videos_root": str(Path(args.videos_root).resolve()),
        "clip_seconds": args.clip_seconds,
        "stride_seconds": args.stride_seconds,
        "min_overlap_ratio": args.min_overlap_ratio,
        "drop_empty_clips": bool(args.drop_empty_clips),
        "n_classes": len(action_id_to_index),
        "n_clips": len(rows),
        "n_skipped_videos": len(skipped_rows),
    }

    with open(out_dir / "manifest_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[OUT] {manifest_csv}")


if __name__ == "__main__":
    main()