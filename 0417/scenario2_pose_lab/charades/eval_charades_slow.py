from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from charades_dataset import CharadesClipDataset
from charades_utils import load_charades_classes
from slow_charades_model import load_slow_r50_charades


def compute_multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)

    aps = []
    for c in range(y_true.shape[1]):
        if np.sum(y_true[:, c]) == 0:
            continue
        ap = average_precision_score(y_true[:, c], y_prob[:, c])
        if not np.isnan(ap):
            aps.append(float(ap))
    mAP = float(np.mean(aps)) if aps else float("nan")

    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_precision = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    micro_recall = float(recall_score(y_true, y_pred, average="micro", zero_division=0))

    top1_hit = 0
    top3_hit = 0
    n = len(y_true)
    for i in range(n):
        gt_indices = np.where(y_true[i] > 0)[0].tolist()
        if len(gt_indices) == 0:
            continue

        ranking = np.argsort(-y_prob[i])
        top1 = ranking[:1].tolist()
        top3 = ranking[:3].tolist()

        if any(x in gt_indices for x in top1):
            top1_hit += 1
        if any(x in gt_indices for x in top3):
            top3_hit += 1

    hit1 = float(top1_hit / max(1, n))
    hit3 = float(top3_hit / max(1, n))

    return {
        "mAP": mAP,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "hit@1": hit1,
        "hit@3": hit3,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SLOW_R50 on Charades multi-label clips.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--classes-txt", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="charades_slow_eval")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--sampling-rate", type=int, default=8)
    parser.add_argument("--side-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    action_id_to_index, index_to_action_id, action_id_to_text = load_charades_classes(args.classes_txt)
    num_classes = len(action_id_to_index)

    ds = CharadesClipDataset(
        manifest_csv=args.manifest,
        num_classes=num_classes,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        side_size=args.side_size,
        crop_size=args.crop_size,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = load_slow_r50_charades(num_classes=num_classes, device=device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    all_probs = []
    all_targets = []
    rows: List[Dict] = []

    with torch.no_grad():
        for batch in tqdm(loader):
            videos = batch["video"].to(device)
            targets = batch["target"].numpy()

            logits = model(videos)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(targets)

            for i in range(len(probs)):
                ranking = np.argsort(-probs[i]).tolist()
                top1 = ranking[:1]
                top3 = ranking[:3]

                gt_indices = np.where(targets[i] > 0)[0].tolist()
                gt_ids = [index_to_action_id[x] for x in gt_indices]
                top1_ids = [index_to_action_id[x] for x in top1]
                top3_ids = [index_to_action_id[x] for x in top3]

                rows.append(
                    {
                        "video_id": batch["video_id"][i],
                        "clip_id": int(batch["clip_id"][i]),
                        "start_sec": float(batch["start_sec"][i]),
                        "end_sec": float(batch["end_sec"][i]),
                        "gt_action_ids": ";".join(gt_ids),
                        "pred_top1_ids": ";".join(top1_ids),
                        "pred_top3_ids": ";".join(top3_ids),
                        "hit@1": int(any(x in gt_indices for x in top1)),
                        "hit@3": int(any(x in gt_indices for x in top3)),
                    }
                )

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = compute_multilabel_metrics(all_targets, all_probs, threshold=float(args.threshold))

    pd.DataFrame(rows).to_csv(out_dir / "clip_predictions.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()