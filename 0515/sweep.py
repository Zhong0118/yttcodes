from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
from pathlib import Path


def parse_list(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Small grid sweep over segmentation delay and network latency.")
    parser.add_argument("--annotations", default="Charades/Charades_v1_train.csv")
    parser.add_argument("--video-dir", default="Charades_v1_480")
    parser.add_argument("--out-dir", default="outputs/sweep")
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--delay-sec", default="0,0.5,1.0")
    parser.add_argument("--network-delay-ms", default="20,50,100")
    parser.add_argument("--mode", choices=["oracle", "random", "empty"], default="oracle")
    args = parser.parse_args()

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for delay_sec, network_delay_ms in itertools.product(
        parse_list(args.delay_sec, float),
        parse_list(args.network_delay_ms, float),
    ):
        run_name = f"delay_{delay_sec:g}_net_{network_delay_ms:g}".replace(".", "p")
        run_dir = root / run_name
        manifest = run_dir / "manifest.csv"
        results = run_dir / "clip_results.csv"
        metrics = run_dir / "metrics"
        subprocess.run(
            [
                "python",
                "dataset.py",
                "--annotations",
                args.annotations,
                "--video-dir",
                args.video_dir,
                "--out-manifest",
                str(manifest),
                "--mode",
                "action",
                "--max-videos",
                str(args.max_videos),
                "--require-video",
                "--delay-sec",
                str(delay_sec),
            ],
            check=True,
        )
        subprocess.run(
            [
                "python",
                "simulate_results.py",
                "--manifest",
                str(manifest),
                "--out",
                str(results),
                "--mode",
                args.mode,
                "--network-delay-ms",
                str(network_delay_ms),
            ],
            check=True,
        )
        subprocess.run(
            [
                "python",
                "aggregate_metrics.py",
                "--clip-results",
                str(results),
                "--out-dir",
                str(metrics),
            ],
            check=True,
        )
        with (metrics / "run_summary.json").open("r", encoding="utf-8") as f:
            summary = json.load(f)
        rows.append(
            {
                "run": run_name,
                "delay_sec": delay_sec,
                "network_delay_ms": network_delay_ms,
                "clip_count": summary["clip_count"],
                "hit@1": summary["hit@1"],
                "hit@3": summary["hit@3"],
                "hit@5": summary["hit@5"],
                "mAP": summary["mAP"],
                "micro_f1": summary["micro_f1"],
                "macro_f1": summary["macro_f1"],
                "end_to_end_avg_ms": summary["end_to_end_avg_ms"],
                "detection_delay_p90_ms": summary["detection_delay_ms"]["p90"],
            }
        )
    if rows:
        with (root / "runs_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote sweep summary to {root / 'runs_summary.csv'}")


if __name__ == "__main__":
    main()
