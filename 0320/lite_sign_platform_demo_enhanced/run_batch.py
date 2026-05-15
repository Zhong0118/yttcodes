import os
import json
import argparse
from datetime import datetime

import pandas as pd

from src.config import build_experiment_configs_from_list
from src.experiment import LiteSignExperiment
from src.report import write_batch_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="input video path")
    parser.add_argument("--output-root", type=str, default="results", help="output root dir")
    parser.add_argument("--max-duration-sec", type=float, default=None, help="override max duration")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(args.output_root, f"batch_{ts}")
    os.makedirs(batch_dir, exist_ok=True)

    experiment = LiteSignExperiment()
    configs = build_experiment_configs_from_list()

    all_rows = []

    for idx, cfg in enumerate(configs):
        if args.max_duration_sec is not None:
            cfg.max_duration_sec = args.max_duration_sec

        run_name = f"run_{idx:03d}"
        run_dir = os.path.join(batch_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        print(f"[{idx + 1}/{len(configs)}] Running {run_name} ...")

        # 直接调用真实存在的方法
        df, summary = experiment.run_single(
            video_path=args.video,
            config=cfg,
            baseline_embeddings=None,
        )

        # 保存单次窗口结果
        windows_csv = os.path.join(run_dir, "windows.csv")
        windows_xlsx = os.path.join(run_dir, "windows.xlsx")
        df.to_csv(windows_csv, index=False, encoding="utf-8-sig")
        df.to_excel(windows_xlsx, index=False)

        summary_row = {
            "run_name": run_name,
            "bandwidth_mbps": cfg.bandwidth_mbps,
            "net_delay_ms": cfg.net_delay_ms,
            "ai_resolution": cfg.ai_resolution,
            "ai_fps": cfg.ai_fps,
            "window_seconds": cfg.window_seconds,
            "stride_seconds": cfg.stride_seconds,
            "extra_ai_delay_ms": cfg.extra_ai_delay_ms,
            "jpeg_quality": cfg.jpeg_quality,
            "h264_batch_size": cfg.h264_batch_size,
            "max_duration_sec": cfg.max_duration_sec,
        }
        summary_row.update(summary)
        all_rows.append(summary_row)

        with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_row, f, ensure_ascii=False, indent=2)

    df_batch = pd.DataFrame(all_rows)
    csv_path = os.path.join(batch_dir, "batch_summary.csv")
    xlsx_path = os.path.join(batch_dir, "batch_summary.xlsx")
    df_batch.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df_batch.to_excel(xlsx_path, index=False)

    write_batch_report(df_batch, batch_dir)

    print(f"Done. Results saved to: {batch_dir}")
    print(f"- {csv_path}")
    print(f"- {xlsx_path}")


if __name__ == "__main__":
    main()