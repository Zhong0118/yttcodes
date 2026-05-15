from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from src.charades_pipeline.common import (
    parse_top_labels,
    parse_top_scores,
    percentile,
    read_csv_dicts,
    split_labels,
    write_csv_dicts,
    write_json,
)


def to_float(value: object) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def numeric(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = to_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def hit_at(top_labels: list[str], gt_labels: list[str], k: int) -> float:
    if not gt_labels:
        return 0.0
    gt = set(gt_labels)
    return float(any(label in gt for label in top_labels[:k]))


def average_precision(y_true: list[int], y_score: list[float]) -> float:
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    positives = sum(y_true)
    if positives == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, (_, truth) in enumerate(pairs, 1):
        if truth:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def enrich_rows(rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        gt_labels = split_labels(row.get("action_id"))
        pred_labels = parse_top_labels(row.get("topk_labels"))
        pred_scores = parse_top_scores(row.get("topk_labels"))
        item = dict(row)
        item["_gt_labels"] = gt_labels
        item["_pred_labels"] = pred_labels
        item["_pred_scores"] = pred_scores
        for k in [1, 3, 5]:
            item[f"hit@{k}"] = hit_at(pred_labels, gt_labels, k)
        enriched.append(item)
    return enriched


def macro_micro_f1(rows: list[dict], labels: list[str]) -> tuple[float, float]:
    global_tp = global_fp = global_fn = 0
    label_f1 = []
    for label in labels:
        tp = fp = fn = 0
        for row in rows:
            gt = set(row["_gt_labels"])
            pred = set(row["_pred_labels"])
            tp += int(label in pred and label in gt)
            fp += int(label in pred and label not in gt)
            fn += int(label not in pred and label in gt)
        global_tp += tp
        global_fp += fp
        global_fn += fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        label_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    precision = global_tp / (global_tp + global_fp) if global_tp + global_fp else 0.0
    recall = global_tp / (global_tp + global_fn) if global_tp + global_fn else 0.0
    micro = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return micro, mean(label_f1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate clip-level results into action/run summaries.")
    parser.add_argument("--clip-results", default="outputs/clip_results.csv")
    parser.add_argument("--out-dir", default="outputs/metrics")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = enrich_rows(read_csv_dicts(args.clip_results))

    by_action: dict[str, list[dict]] = defaultdict(list)
    all_labels = set()
    for row in rows:
        all_labels.update(row["_gt_labels"])
        all_labels.update(row["_pred_scores"].keys())
        for action_id in row["_gt_labels"]:
            by_action[action_id].append(row)

    action_rows = []
    for action_id, group in by_action.items():
        durations = numeric(group, "action_duration_sec")
        action_rows.append(
            {
                "action_id": action_id,
                "clip_count": len(group),
                "hit@1": f"{mean([row['hit@1'] for row in group]):.6f}",
                "hit@3": f"{mean([row['hit@3'] for row in group]):.6f}",
                "hit@5": f"{mean([row['hit@5'] for row in group]):.6f}",
                "avg_confidence": f"{mean(numeric(group, 'confidence')):.6f}",
                "action_duration_avg_sec": f"{mean(durations):.6f}",
                "action_duration_p50_sec": f"{percentile(durations, 0.50):.6f}",
                "action_duration_p90_sec": f"{percentile(durations, 0.90):.6f}",
                "infer_avg_ms": f"{mean(numeric(group, 'infer_ms')):.6f}",
                "tx_avg_ms": f"{mean(numeric(group, 'tx_ms')):.6f}",
                "end_to_end_avg_ms": f"{mean(numeric(group, 'end_to_end_ms')):.6f}",
                "detection_delay_avg_ms": f"{mean(numeric(group, 'detection_delay_ms')):.6f}",
            }
        )
    action_rows.sort(key=lambda row: (-int(row["clip_count"]), row["action_id"]))
    write_csv_dicts(
        out_dir / "action_summary.csv",
        action_rows,
        [
            "action_id",
            "clip_count",
            "hit@1",
            "hit@3",
            "hit@5",
            "avg_confidence",
            "action_duration_avg_sec",
            "action_duration_p50_sec",
            "action_duration_p90_sec",
            "infer_avg_ms",
            "tx_avg_ms",
            "end_to_end_avg_ms",
            "detection_delay_avg_ms",
        ],
    )

    sorted_labels = sorted(all_labels)
    ap_by_label = {}
    for label in sorted_labels:
        y_true = [int(label in row["_gt_labels"]) for row in rows]
        y_score = [row["_pred_scores"].get(label, 0.0) for row in rows]
        ap_by_label[label] = average_precision(y_true, y_score)
    micro_f1, macro_f1 = macro_micro_f1(rows, sorted_labels)

    delays = numeric(rows, "detection_delay_ms")
    durations = numeric(rows, "action_duration_sec")
    run_summary = {
        "clip_count": len(rows),
        "labeled_clip_count": sum(bool(row["_gt_labels"]) for row in rows),
        "hit@1": mean([row["hit@1"] for row in rows]),
        "hit@3": mean([row["hit@3"] for row in rows]),
        "hit@5": mean([row["hit@5"] for row in rows]),
        "mAP": mean(list(ap_by_label.values())),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "infer_avg_ms": mean(numeric(rows, "infer_ms")),
        "tx_avg_ms": mean(numeric(rows, "tx_ms")),
        "end_to_end_avg_ms": mean(numeric(rows, "end_to_end_ms")),
        "detection_delay_ms": {
            "avg": mean(delays),
            "p50": percentile(delays, 0.50),
            "p90": percentile(delays, 0.90),
            "p95": percentile(delays, 0.95),
        },
        "action_duration_sec": {
            "avg": mean(durations),
            "p50": percentile(durations, 0.50),
            "p90": percentile(durations, 0.90),
        },
    }
    write_json(out_dir / "run_summary.json", run_summary)

    visible_rows = []
    for row in rows:
        visible = {k: v for k, v in row.items() if not k.startswith("_")}
        visible["hit@1"] = f"{row['hit@1']:.6f}"
        visible["hit@3"] = f"{row['hit@3']:.6f}"
        visible["hit@5"] = f"{row['hit@5']:.6f}"
        visible_rows.append(visible)
    fields = list(visible_rows[0].keys()) if visible_rows else []
    if fields:
        write_csv_dicts(out_dir / "clip_metrics.csv", visible_rows, fields)
    print(f"Wrote summaries to {out_dir}")


if __name__ == "__main__":
    main()
