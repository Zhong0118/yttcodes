from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from src.charades_pipeline.common import read_csv_dicts, write_csv_dicts


def main() -> None:
    parser = argparse.ArgumentParser(description="Physically cut clips from a manifest using ffmpeg.")
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--out-dir", default="outputs/clips")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv_dicts(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    output_rows = []
    for index, row in enumerate(rows, 1):
        source = Path(row["video_path"])
        clip_path = out_dir / row["video_id"] / f'{row["clip_id"]}.mp4'
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        row["clip_path"] = str(clip_path)
        output_rows.append(row)
        if args.dry_run:
            continue
        if clip_path.exists() and not args.overwrite:
            continue
        cmd = [
            "ffmpeg",
            "-y" if args.overwrite else "-n",
            "-ss",
            row["start_sec"],
            "-to",
            row["end_sec"],
            "-i",
            str(source),
            "-c",
            "copy",
            str(clip_path),
        ]
        subprocess.run(cmd, check=True)
        print(f"[{index}/{len(rows)}] {clip_path}")

    fields = list(output_rows[0].keys()) if output_rows else []
    if fields:
        write_csv_dicts(Path(args.out_dir) / "clip_manifest.csv", output_rows, fields)
    print(f"Prepared {len(output_rows)} clip paths under {args.out_dir}")


if __name__ == "__main__":
    main()
