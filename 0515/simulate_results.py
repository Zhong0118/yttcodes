from __future__ import annotations

import argparse
import time

from src.charades_pipeline.common import read_class_names, read_csv_dicts, seeded_rng, split_labels, write_csv_dicts


RESULT_FIELDS = [
    "video_id",
    "clip_id",
    "clip_mode",
    "start_sec",
    "end_sec",
    "action_id",
    "action_start_sec",
    "action_end_sec",
    "action_duration_sec",
    "top1_label",
    "confidence",
    "topk_labels",
    "infer_ms",
    "tx_ms",
    "end_to_end_ms",
    "detection_delay_ms",
]


def choose_labels(mode: str, gt_labels: list[str], all_labels: list[str], topk: int, rng) -> list[tuple[str, float]]:
    if mode == "empty":
        return []
    if mode == "oracle" and gt_labels:
        ordered = gt_labels[:topk]
        candidates = [label for label in all_labels if label not in ordered]
        rng.shuffle(candidates)
        ordered.extend(candidates[: max(0, topk - len(ordered))])
    else:
        ordered = rng.sample(all_labels, min(topk, len(all_labels)))

    scores = []
    for rank, label in enumerate(ordered[:topk]):
        base = 0.95 if mode == "oracle" and label in gt_labels else 0.55
        score = max(0.01, base - rank * 0.08 + rng.uniform(-0.03, 0.03))
        scores.append((label, round(score, 4)))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Create clip_results.csv without a trained model.")
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--classes", default="Charades/Charades_v1_classes.txt")
    parser.add_argument("--out", default="outputs/clip_results.csv")
    parser.add_argument("--mode", choices=["oracle", "random", "empty"], default="oracle")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--network-delay-ms", type=float, default=50.0)
    parser.add_argument("--bandwidth-mbps", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    classes = read_class_names(args.classes)
    all_labels = sorted(classes)
    rng = seeded_rng(args.seed)
    results = []

    for row in read_csv_dicts(args.manifest):
        tic = time.perf_counter()
        gt_labels = split_labels(row.get("action_id"))
        preds = choose_labels(args.mode, gt_labels, all_labels, args.topk, rng)
        infer_ms = (time.perf_counter() - tic) * 1000.0 + rng.uniform(8.0, 35.0)
        duration_sec = float(row.get("duration_sec") or 0.0)
        pseudo_payload_mb = max(0.01, duration_sec * 0.22)
        tx_ms = args.network_delay_ms + pseudo_payload_mb * 8.0 / max(args.bandwidth_mbps, 0.001) * 1000.0
        action_end = row.get("action_end_sec")
        detection_delay_ms = ""
        if action_end:
            detection_delay_ms = f"{max(0.0, float(row['end_sec']) - float(action_end)) * 1000.0:.3f}"
        topk_labels = ";".join(f"{label}:{score:.4f}" for label, score in preds)
        results.append(
            {
                "video_id": row["video_id"],
                "clip_id": row["clip_id"],
                "clip_mode": row["clip_mode"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "action_id": row.get("action_id", ""),
                "action_start_sec": row.get("action_start_sec", ""),
                "action_end_sec": row.get("action_end_sec", ""),
                "action_duration_sec": row.get("action_duration_sec", ""),
                "top1_label": preds[0][0] if preds else "",
                "confidence": f"{preds[0][1]:.4f}" if preds else "0.0000",
                "topk_labels": topk_labels,
                "infer_ms": f"{infer_ms:.3f}",
                "tx_ms": f"{tx_ms:.3f}",
                "end_to_end_ms": f"{infer_ms + tx_ms:.3f}",
                "detection_delay_ms": detection_delay_ms,
            }
        )

    write_csv_dicts(args.out, results, RESULT_FIELDS)
    print(f"Wrote {len(results)} rows to {args.out} using mode={args.mode}")


if __name__ == "__main__":
    main()
