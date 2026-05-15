from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path


def import_runtime():
    try:
        import cv2
        import torch
        from torch.utils.data import DataLoader, Dataset
        from torchvision.models.video import mc3_18, r2plus1d_18, r3d_18
        from tqdm import tqdm
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency in current Python environment: {exc.name}") from exc
    return cv2, torch, DataLoader, Dataset, {"r3d_18": r3d_18, "mc3_18": mc3_18, "r2plus1d_18": r2plus1d_18}, tqdm


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_class_ids(path: str | Path) -> list[str]:
    ids = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line.split(maxsplit=1)[0])
    return sorted(ids)


def build_model(torch, builders, arch: str, num_classes: int, device: str):
    if arch not in builders:
        raise SystemExit(f"Unsupported arch={arch}. Choose one of: {', '.join(builders)}")
    model = builders[arch](weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    return model


def sample_clip(cv2, torch, row: dict[str, str], num_frames: int, resize: int, train: bool):
    video_path = row["video_path"]
    start_sec = float(row["start_sec"])
    end_sec = float(row["end_sec"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(start_sec * fps))
    end_frame = max(start_frame + 1, int(end_sec * fps))
    if total_frames > 0:
        end_frame = min(end_frame, total_frames - 1)

    available = max(1, end_frame - start_frame + 1)
    if available >= num_frames:
        if train:
            max_offset = max(0, available - num_frames)
            offset = random.randint(0, max_offset)
            positions = [start_frame + offset + i * max(1, available // num_frames) for i in range(num_frames)]
            positions = [min(pos, end_frame) for pos in positions]
        else:
            step = (end_frame - start_frame) / max(1, num_frames - 1)
            positions = [int(round(start_frame + i * step)) for i in range(num_frames)]
    else:
        positions = [start_frame + min(i, available - 1) for i in range(num_frames)]

    frames = []
    last = None
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            if last is None:
                tensor = torch.zeros(3, resize, resize)
            else:
                tensor = last.clone()
            frames.append(tensor)
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_LINEAR)
        if train and random.random() < 0.5:
            frame = cv2.flip(frame, 1)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        last = tensor
        frames.append(tensor)
    cap.release()
    clip = torch.stack(frames, dim=1)
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    return (clip - mean) / std


def main() -> None:
    cv2, torch, DataLoader, Dataset, builders, tqdm = import_runtime()

    class CharadesClipDataset(Dataset):
        def __init__(self, rows, class_to_idx, num_frames, resize, train):
            self.rows = rows
            self.class_to_idx = class_to_idx
            self.num_frames = num_frames
            self.resize = resize
            self.train = train

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            clip = sample_clip(cv2, torch, row, self.num_frames, self.resize, self.train)
            label = torch.zeros(len(self.class_to_idx), dtype=torch.float32)
            for action_id in str(row.get("action_id", "")).split(";"):
                action_id = action_id.strip()
                if action_id in self.class_to_idx:
                    label[self.class_to_idx[action_id]] = 1.0
            return clip, label

    parser = argparse.ArgumentParser(description="Train a Charades multi-label video classifier from a clip manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--classes", default="Charades/Charades_v1_classes.txt")
    parser.add_argument("--out-dir", default="checkpoints/r3d18_train2000_e3")
    parser.add_argument("--arch", choices=["r3d_18", "mc3_18", "r2plus1d_18"], default="r3d_18")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--resize", type=int, default=112)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        args.device = "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_ids = read_class_ids(args.classes)
    class_to_idx = {action_id: idx for idx, action_id in enumerate(class_ids)}
    rows = [row for row in read_manifest(args.manifest) if row.get("action_id") and Path(row["video_path"]).exists()]
    random.shuffle(rows)
    val_size = max(1, int(len(rows) * args.val_ratio)) if len(rows) > 10 else 0
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]
    print(f"Training clips: {len(train_rows)} | validation clips: {len(val_rows)} | classes: {len(class_ids)}")

    train_set = CharadesClipDataset(train_rows, class_to_idx, args.num_frames, args.resize, True)
    val_set = CharadesClipDataset(val_rows, class_to_idx, args.num_frames, args.resize, False) if val_rows else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    val_loader = (
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if val_set
        else None
    )

    model = build_model(torch, builders, args.arch, len(class_ids), args.device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.device == "cuda")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        tic = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs} train", dynamic_ncols=True)
        for clips, labels in pbar:
            clips = clips.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and args.device == "cuda"):
                logits = model(clips)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * clips.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= max(1, len(train_set))

        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            seen = 0
            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"epoch {epoch}/{args.epochs} val", dynamic_ncols=True)
                for clips, labels in pbar:
                    clips = clips.to(args.device, non_blocking=True)
                    labels = labels.to(args.device, non_blocking=True)
                    logits = model(clips)
                    loss = criterion(logits, labels)
                    val_loss += float(loss.item()) * clips.size(0)
                    seen += clips.size(0)
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
            val_loss /= max(1, seen)
        else:
            val_loss = train_loss

        epoch_state = {
            "epoch": epoch,
            "arch": args.arch,
            "num_classes": len(class_ids),
            "class_ids": class_ids,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
        }
        torch.save(epoch_state, out_dir / f"epoch_{epoch:03d}.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(epoch_state, out_dir / "best.pth")
        elapsed_min = (time.time() - tic) / 60.0
        print(
            f"epoch {epoch}/{args.epochs} done | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | elapsed={elapsed_min:.1f} min"
        )

    print(f"Done. Best checkpoint: {out_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
