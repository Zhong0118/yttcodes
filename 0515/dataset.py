from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.charades_pipeline.common import overlap_seconds, parse_actions, write_csv_dicts


MANIFEST_FIELDS = [
    "video_id",
    "clip_id",
    "clip_mode",
    "start_sec",
    "end_sec",
    "duration_sec",
    "action_id",
    "action_start_sec",
    "action_end_sec",
    "action_duration_sec",
    "delay_sec",
    "context_before_sec",
    "overlap_sec",
    "video_length_sec",
    "video_path",
]


def build_action_rows(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    with Path(args.annotations).open("r", encoding="utf-8", newline="") as f:
        for record in csv.DictReader(f):
            video_id = record["id"]
            video_path = Path(args.video_dir) / f"{video_id}.mp4"
            if args.require_video and not video_path.exists():
                continue
            video_length = float(record.get("length") or 0)
            actions = parse_actions(record.get("actions"))
            if not actions:
                continue
            for action_index, action in enumerate(actions):
                clip_start = max(0.0, action.start_sec - args.context_before_sec)
                clip_end = min(video_length, action.end_sec + args.delay_sec) if video_length else action.end_sec + args.delay_sec
                if clip_end <= clip_start:
                    continue
                rows.append(
                    {
                        "video_id": video_id,
                        "clip_id": f"{video_id}_a{action_index:03d}",
                        "clip_mode": "action",
                        "start_sec": f"{clip_start:.3f}",
                        "end_sec": f"{clip_end:.3f}",
                        "duration_sec": f"{clip_end - clip_start:.3f}",
                        "action_id": action.action_id,
                        "action_start_sec": f"{action.start_sec:.3f}",
                        "action_end_sec": f"{action.end_sec:.3f}",
                        "action_duration_sec": f"{action.duration_sec:.3f}",
                        "delay_sec": f"{args.delay_sec:.3f}",
                        "context_before_sec": f"{args.context_before_sec:.3f}",
                        "overlap_sec": f"{action.duration_sec:.3f}",
                        "video_length_sec": f"{video_length:.3f}",
                        "video_path": str(video_path),
                    }
                )
            if args.max_videos and len({row["video_id"] for row in rows}) >= args.max_videos:
                break
    return rows


def build_fixed_rows(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    seen_videos = 0
    with Path(args.annotations).open("r", encoding="utf-8", newline="") as f:
        for record in csv.DictReader(f):
            video_id = record["id"]
            video_path = Path(args.video_dir) / f"{video_id}.mp4"
            if args.require_video and not video_path.exists():
                continue
            video_length = float(record.get("length") or 0)
            if video_length <= 0:
                continue
            actions = parse_actions(record.get("actions"))
            start = 0.0
            clip_index = 0
            while start < video_length:
                end = min(video_length, start + args.clip_seconds)
                if end - start < args.min_clip_seconds:
                    break
                overlaps = [
                    (action.action_id, overlap_seconds(start, end, action.start_sec, action.end_sec), action)
                    for action in actions
                ]
                overlaps = [item for item in overlaps if item[1] >= args.min_overlap_sec]
                overlaps.sort(key=lambda x: x[1], reverse=True)
                labels = ";".join(item[0] for item in overlaps)
                primary = overlaps[0][2] if overlaps else None
                overlap = overlaps[0][1] if overlaps else 0.0
                rows.append(
                    {
                        "video_id": video_id,
                        "clip_id": f"{video_id}_f{clip_index:03d}",
                        "clip_mode": "fixed",
                        "start_sec": f"{start:.3f}",
                        "end_sec": f"{end:.3f}",
                        "duration_sec": f"{end - start:.3f}",
                        "action_id": labels,
                        "action_start_sec": f"{primary.start_sec:.3f}" if primary else "",
                        "action_end_sec": f"{primary.end_sec:.3f}" if primary else "",
                        "action_duration_sec": f"{primary.duration_sec:.3f}" if primary else "",
                        "delay_sec": "",
                        "context_before_sec": "",
                        "overlap_sec": f"{overlap:.3f}",
                        "video_length_sec": f"{video_length:.3f}",
                        "video_path": str(video_path),
                    }
                )
                clip_index += 1
                start += args.stride_seconds
            seen_videos += 1
            if args.max_videos and seen_videos >= args.max_videos:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Charades clip manifest.")
    parser.add_argument("--annotations", default="Charades/Charades_v1_train.csv")
    parser.add_argument("--video-dir", default="Charades_v1_480")
    parser.add_argument("--out-manifest", default="outputs/manifest.csv")
    parser.add_argument("--mode", choices=["action", "fixed"], default="action")
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--require-video", action="store_true")
    parser.add_argument("--clip-seconds", type=float, default=2.56)
    parser.add_argument("--stride-seconds", type=float, default=1.28)
    parser.add_argument("--min-clip-seconds", type=float, default=0.25)
    parser.add_argument("--min-overlap-sec", type=float, default=0.10)
    parser.add_argument("--delay-sec", type=float, default=0.0, help="Cut action clips at action_end + delay_sec.")
    parser.add_argument("--context-before-sec", type=float, default=0.0, help="Include look-back context before action_start.")
    args = parser.parse_args()

    rows = build_action_rows(args) if args.mode == "action" else build_fixed_rows(args)
    write_csv_dicts(args.out_manifest, rows, MANIFEST_FIELDS)
    print(f"Wrote {len(rows)} clips to {args.out_manifest}")


if __name__ == "__main__":
    main()
