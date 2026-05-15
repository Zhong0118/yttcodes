from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.charades_pipeline.common import read_class_names, read_csv_dicts, write_csv_dicts


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


def build_model(torch, builders, arch: str, num_classes: int, checkpoint: str, device: str):
    if arch not in builders:
        raise SystemExit(f"Unsupported arch={arch}. Choose one of: {', '.join(builders)}")
    model = builders[arch](weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    if not checkpoint:
        raise SystemExit(
            "Real inference needs --checkpoint for a Charades-trained multi-label model. "
            "Use simulate_results.py for oracle/random placeholder runs."
        )
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    cleaned = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        key = key.removeprefix("model.")
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"Warning: missing checkpoint keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"Warning: unexpected checkpoint keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    model.to(device)
    model.eval()
    return model


def sample_clip_tensor(cv2, torch, video_path: str, start_sec: float, end_sec: float, num_frames: int, resize: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(start_sec * fps))
    end_frame = max(start_frame + 1, int(end_sec * fps))
    if total_frames > 0:
        end_frame = min(end_frame, total_frames - 1)

    if num_frames == 1:
        positions = [start_frame]
    else:
        step = (end_frame - start_frame) / max(1, num_frames - 1)
        positions = [int(round(start_frame + i * step)) for i in range(num_frames)]

    frames = []
    last = None
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            if last is None:
                frame = torch.zeros(3, resize, resize)
                frames.append(frame)
                continue
            frames.append(last.clone())
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        last = tensor
        frames.append(tensor)
    cap.release()

    clip = torch.stack(frames, dim=1)
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    return (clip - mean) / std


def make_dataset_class(cv2, torch, Dataset):
    class CharadesInferenceDataset(Dataset):
        def __init__(self, rows, num_frames: int, resize: int):
            self.rows = rows
            self.num_frames = num_frames
            self.resize = resize

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            clip = sample_clip_tensor(
                cv2,
                torch,
                row["video_path"],
                float(row["start_sec"]),
                float(row["end_sec"]),
                self.num_frames,
                self.resize,
            )
            return index, clip

    return CharadesInferenceDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real clip inference from a Charades manifest.")
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--classes", default="Charades/Charades_v1_classes.txt")
    parser.add_argument("--checkpoint", required=True, help="Charades-trained checkpoint with 157 multi-label outputs.")
    parser.add_argument("--arch", choices=["r3d_18", "mc3_18", "r2plus1d_18"], default="r3d_18")
    parser.add_argument("--out", default="outputs/real_clip_results.csv")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--network-delay-ms", type=float, default=50.0)
    parser.add_argument("--bandwidth-mbps", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-topk", type=int, default=0, help="Print top-k labels for a few clips while running.")
    parser.add_argument("--print-every", type=int, default=50, help="Print one prediction every N clips when --print-topk is set.")
    args = parser.parse_args()

    cv2, torch, DataLoader, Dataset, builders, tqdm = import_runtime()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        args.device = "cpu"

    classes = read_class_names(args.classes)
    class_ids = sorted(classes)
    model = build_model(torch, builders, args.arch, len(class_ids), args.checkpoint, args.device)
    rows = read_csv_dicts(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    results_by_index: dict[int, dict] = {}
    dataset_class = make_dataset_class(cv2, torch, Dataset)
    dataset = dataset_class(rows, args.num_frames, args.resize)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    with torch.no_grad():
        for batch_indices, clips in tqdm(loader, desc="inference", dynamic_ncols=True):
            clips = clips.to(args.device, non_blocking=True)
            tic = time.perf_counter()
            logits = model(clips)
            infer_ms = (time.perf_counter() - tic) * 1000.0
            probs = torch.sigmoid(logits)
            values, indices = torch.topk(probs, k=min(args.topk, len(class_ids)), dim=1)
            per_clip_infer_ms = infer_ms / max(1, clips.size(0))
            for batch_pos, original_index in enumerate(batch_indices.tolist()):
                row = rows[original_index]
                preds = [
                    (class_ids[int(label_index)], float(score))
                    for score, label_index in zip(values[batch_pos].cpu(), indices[batch_pos].cpu())
                ]
                duration_sec = float(row.get("duration_sec") or 0.0)
                pseudo_payload_mb = max(0.01, duration_sec * 0.22)
                tx_ms = args.network_delay_ms + pseudo_payload_mb * 8.0 / max(args.bandwidth_mbps, 0.001) * 1000.0
                detection_delay_ms = ""
                if row.get("action_end_sec"):
                    detection_delay_ms = f"{max(0.0, float(row['end_sec']) - float(row['action_end_sec'])) * 1000.0:.3f}"
                results_by_index[original_index] = {
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
                    "confidence": f"{preds[0][1]:.6f}" if preds else "0.000000",
                    "topk_labels": ";".join(f"{label}:{score:.6f}" for label, score in preds),
                    "infer_ms": f"{per_clip_infer_ms:.3f}",
                    "tx_ms": f"{tx_ms:.3f}",
                    "end_to_end_ms": f"{per_clip_infer_ms + tx_ms:.3f}",
                    "detection_delay_ms": detection_delay_ms,
                }
                if args.print_topk and (original_index + 1 == len(rows) or (original_index + 1) % args.print_every == 0):
                    visible = preds[: min(args.print_topk, len(preds))]
                    text = ", ".join(f"{label}:{score:.3f}" for label, score in visible)
                    print(f"[{original_index + 1}/{len(rows)}] {row['clip_id']} -> {text}")

    results = [results_by_index[index] for index in range(len(rows))]
    write_csv_dicts(args.out, results, RESULT_FIELDS)
    print(f"Wrote real inference results to {args.out}")


if __name__ == "__main__":
    main()
