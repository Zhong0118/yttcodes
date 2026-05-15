from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def import_plotting():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing plotting dependency: {exc.name}") from exc
    return plt


def load_summary(run_dir: str, name: str) -> dict:
    path = Path(run_dir) / "metrics" / "run_summary.json"
    if not path.exists():
        raise SystemExit(f"Missing run summary: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["name"] = name
    data["run_dir"] = run_dir
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple experiment run_summary.json files.")
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("NAME", "RUN_DIR"),
        required=True,
        help="Experiment name and directory containing metrics/run_summary.json.",
    )
    parser.add_argument("--out-dir", default="outputs/experiments/comparison_train2000_test500")
    args = parser.parse_args()

    plt = import_plotting()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [load_summary(run_dir, name) for name, run_dir in args.run]

    table_rows = []
    for item in summaries:
        table_rows.append(
            {
                "name": item["name"],
                "clip_count": item.get("clip_count", 0),
                "hit@1": item.get("hit@1", 0),
                "hit@3": item.get("hit@3", 0),
                "hit@5": item.get("hit@5", 0),
                "mAP": item.get("mAP", 0),
                "micro_f1": item.get("micro_f1", 0),
                "macro_f1": item.get("macro_f1", 0),
                "infer_avg_ms": item.get("infer_avg_ms", 0),
                "tx_avg_ms": item.get("tx_avg_ms", 0),
                "end_to_end_avg_ms": item.get("end_to_end_avg_ms", 0),
                "delay_p50_ms": item.get("detection_delay_ms", {}).get("p50", 0),
                "delay_p90_ms": item.get("detection_delay_ms", {}).get("p90", 0),
            }
        )

    with (out_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    names = [row["name"] for row in table_rows]
    metric_names = ["hit@1", "hit@3", "hit@5", "mAP", "micro_f1", "macro_f1"]
    width = 0.12
    x = list(range(len(names)))
    plt.figure(figsize=(10, 5))
    for offset, metric in enumerate(metric_names):
        values = [row[metric] for row in table_rows]
        positions = [pos + (offset - (len(metric_names) - 1) / 2) * width for pos in x]
        plt.bar(positions, values, width=width, label=metric)
    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Recognition Performance Comparison")
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "performance_comparison.png", dpi=160)
    plt.close()

    latency_metrics = ["infer_avg_ms", "tx_avg_ms", "end_to_end_avg_ms"]
    width = 0.22
    plt.figure(figsize=(9, 5))
    for offset, metric in enumerate(latency_metrics):
        values = [row[metric] for row in table_rows]
        positions = [pos + (offset - 1) * width for pos in x]
        plt.bar(positions, values, width=width, label=metric)
    plt.xticks(x, names)
    plt.ylabel("Average ms")
    plt.title("Latency Comparison")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "latency_comparison.png", dpi=160)
    plt.close()

    delay_metrics = ["delay_p50_ms", "delay_p90_ms"]
    width = 0.28
    plt.figure(figsize=(8, 4.5))
    for offset, metric in enumerate(delay_metrics):
        values = [row[metric] for row in table_rows]
        positions = [pos + (offset - 0.5) * width for pos in x]
        plt.bar(positions, values, width=width, label=metric)
    plt.xticks(x, names)
    plt.ylabel("Delay ms")
    plt.title("Detection Delay Comparison")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "delay_comparison.png", dpi=160)
    plt.close()

    print(f"Wrote comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
