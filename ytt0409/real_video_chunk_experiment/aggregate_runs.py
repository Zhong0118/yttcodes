from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs")
    args = parser.parse_args()

    root = Path(args.input)
    rows = []

    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "run_meta.json"
        summary_path = run_dir / "run_summary.json"
        if meta_path.exists() and summary_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            rows.append({**meta, **summary})

    if not rows:
        raise RuntimeError("No run results found. Please run analyze_video.py or sweep_video.py first.")

    df = pd.DataFrame(rows)
    out_csv = root / "runs_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"saved: {out_csv}")
    print(df.groupby('exp_group', as_index=False)[['infer_avg_ms', 'first_result_ms', 'end_to_end_avg_ms']].mean())


if __name__ == "__main__":
    main()
