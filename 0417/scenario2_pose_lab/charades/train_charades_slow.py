from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from charades_dataset import CharadesClipDataset
from charades_utils import load_charades_classes
from slow_charades_model import freeze_backbone_except_head, load_slow_r50_charades


def compute_multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)

    # mAP
    aps = []
    for c in range(y_true.shape[1]):
        if np.sum(y_true[:, c]) == 0:
            continue
        ap = average_precision_score(y_true[:, c], y_prob[:, c])
        if not math.isnan(ap):
            aps.append(float(ap))
    mAP = float(np.mean(aps)) if aps else float("nan")

    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_precision = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    micro_recall = float(recall_score(y_true, y_pred, average="micro", zero_division=0))

    # hit@1 / hit@3
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


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    all_probs = []
    all_targets = []

    for batch in tqdm(loader, leave=False):
        videos = batch["video"].to(device)
        targets = batch["target"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(videos)
            loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        tgts = targets.detach().cpu().numpy()

        losses.append(float(loss.item()))
        all_probs.append(probs)
        all_targets.append(tgts)

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = compute_multilabel_metrics(all_targets, all_probs, threshold=0.5)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train SLOW_R50 on Charades multi-label clips.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--classes-txt", required=True)
    parser.add_argument("--out-dir", default="charades_slow_runs")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--sampling-rate", type=int, default=8)
    parser.add_argument("--side-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)

    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    from datetime import datetime
    run_dir = out_dir / f"slow_charades_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    action_id_to_index, index_to_action_id, action_id_to_text = load_charades_classes(args.classes_txt)
    num_classes = len(action_id_to_index)

    train_ds = CharadesClipDataset(
        manifest_csv=args.train_manifest,
        num_classes=num_classes,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        side_size=args.side_size,
        crop_size=args.crop_size,
    )
    val_ds = CharadesClipDataset(
        manifest_csv=args.val_manifest,
        num_classes=num_classes,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        side_size=args.side_size,
        crop_size=args.crop_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = load_slow_r50_charades(num_classes=num_classes, device=device)

    if args.freeze_backbone:
        freeze_backbone_except_head(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_map = -1.0
    history: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.epochs} ==========")

        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        print(json.dumps(row, ensure_ascii=False, indent=2))

        # 保存 last
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_classes": num_classes,
                "args": vars(args),
            },
            run_dir / "last.pt",
        )

        cur_map = float(val_metrics.get("mAP", float("nan")))
        if not math.isnan(cur_map) and cur_map > best_map:
            best_map = cur_map
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "num_classes": num_classes,
                    "args": vars(args),
                    "best_val_mAP": best_map,
                },
                run_dir / "best.pt",
            )

    import pandas as pd
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(run_dir / "train_history.csv", index=False, encoding="utf-8-sig")

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"[DONE] run_dir = {run_dir}")


if __name__ == "__main__":
    main()