from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing plotting dependency: {exc.name}. Install requirements.txt if you need figures; "
        "the CSV/JSON metric pipeline does not require plotting."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot action duration and latency distributions.")
    parser.add_argument("--clip-results", default="outputs/clip_results.csv")
    parser.add_argument("--summary", default="outputs/metrics/action_summary.csv")
    parser.add_argument("--run-summary", default="outputs/metrics/run_summary.json")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = pd.read_csv(args.clip_results).fillna("")
    actions = pd.read_csv(args.summary) if Path(args.summary).exists() else pd.DataFrame()

    duration = pd.to_numeric(clips.get("action_duration_sec", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(duration):
        plt.figure(figsize=(8, 4.5))
        plt.hist(duration, bins=30, color="#2f6f73", edgecolor="white")
        plt.xlabel("Action duration (sec)")
        plt.ylabel("Count")
        plt.title("Action Time Distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "action_duration_hist.png", dpi=160)
        plt.close()

    delay = pd.to_numeric(clips.get("detection_delay_ms", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(delay):
        plt.figure(figsize=(8, 4.5))
        plt.hist(delay, bins=30, color="#8a5a2b", edgecolor="white")
        plt.xlabel("Detection delay (ms)")
        plt.ylabel("Count")
        plt.title("Delay Distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "delay_hist.png", dpi=160)
        plt.close()

    if len(actions):
        top = actions.sort_values("clip_count", ascending=False).head(20)
        plt.figure(figsize=(10, 5))
        plt.bar(top["action_id"], top["hit@5"], color="#586f9c")
        plt.xlabel("Action")
        plt.ylabel("hit@5")
        plt.title("Top Actions by Clip Count")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "top_action_hit5.png", dpi=160)
        plt.close()

    if Path(args.run_summary).exists():
        with Path(args.run_summary).open("r", encoding="utf-8") as f:
            summary = json.load(f)

        metric_labels = ["hit@1", "hit@3", "hit@5", "mAP", "micro_f1", "macro_f1"]
        metric_values = [summary.get(label, 0) for label in metric_labels]
        plt.figure(figsize=(8, 4.5))
        plt.bar(metric_labels, metric_values, color=["#516b91", "#5f8a68", "#b08a3c", "#8b5e83", "#4d7f8f", "#9b5a4a"])
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("Overall Recognition Performance")
        for i, value in enumerate(metric_values):
            plt.text(i, min(1.0, value + 0.02), f"{value:.3f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "overall_performance.png", dpi=160)
        plt.close()

        labels = ["infer", "tx", "end_to_end"]
        values = [summary.get("infer_avg_ms", 0), summary.get("tx_avg_ms", 0), summary.get("end_to_end_avg_ms", 0)]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, values, color=["#3f7d58", "#b55f4c", "#4f5d75"])
        plt.ylabel("Average ms")
        plt.title("Clip Latency Metrics")
        plt.tight_layout()
        plt.savefig(out_dir / "latency_bar.png", dpi=160)
        plt.close()

    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
