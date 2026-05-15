from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import time


import pandas as pd

from src.config import get_preset
from src.report import write_batch_report, write_markdown_report
from src.simulator import Scenario2Experiment, save_single_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="input video path")
    parser.add_argument("--preset", default="scenario2_full", help="smoke or scenario2_full")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--max-duration-sec", type=float, default=None)
    args = parser.parse_args()

    configs = get_preset(args.preset)
    exp = Scenario2Experiment()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    batch_dir = output_root / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for i, cfg in enumerate(configs):
        if args.max_duration_sec is not None:
            cfg.max_duration_sec = args.max_duration_sec

        run_dir = batch_dir / f"run_{i:03d}_{cfg.name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        t_run_start = time.time()
        print(f"[{i+1}/{len(configs)}] Running {cfg.name} ...", flush=True)

        df, summary = exp.run_single(args.video, cfg)

        t_run_end = time.time()
        run_wall_time_sec = t_run_end - t_run_start
        run_wall_time_min = run_wall_time_sec / 60.0

        # 写入这一轮真实运行耗时
        summary["name"] = cfg.name
        summary["run_wall_time_sec"] = float(run_wall_time_sec)
        summary["run_wall_time_min"] = float(run_wall_time_min)

        save_single_outputs(df, summary, run_dir)

        write_markdown_report(
            title=f"Scenario2 Single Report - {cfg.name}",
            summary=summary,
            single_df=df,
            out_path=run_dir / "report.md",
        )

        rows.append(summary)

        print(
            f"[{i+1}/{len(configs)}] Finished {cfg.name} | "
            f"wall_time={run_wall_time_sec:.2f}s ({run_wall_time_min:.2f} min) | "
            f"video_used={summary.get('effective_video_duration_sec', float('nan')):.2f}s / "
            f"video_total={summary.get('source_video_duration_sec', float('nan')):.2f}s",
            flush=True,
        )

    batch_df = pd.DataFrame(rows)
    batch_df.to_csv(batch_dir / "batch_summary.csv", index=False, encoding="utf-8-sig")
    batch_df.to_excel(batch_dir / "batch_summary.xlsx", index=False)
    write_batch_report(batch_df, batch_dir)

    # also export exact used configs
    with open(batch_dir / "used_configs.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Done. Results saved to: {batch_dir}")


if __name__ == "__main__":
    main()
