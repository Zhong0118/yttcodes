from __future__ import annotations

import argparse
from datetime import datetime
import csv
import json
import re
import urllib.request
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch


def normalize_label(label: str) -> str:
    s = str(label).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


DEFAULT_ALIASES = {
    "run": "running",
    "jog": "running",
    "drink": "drinking",
    "eat": "eating",
    "walk": "walking",
    "sit": "sitting",
    "jumping_jacks": "jumping jacks",
}


def canonicalize_label(label: str, alias_map: Dict[str, str]) -> str:
    x = normalize_label(label)
    return alias_map.get(x, x)


def load_aliases(alias_json: str | None) -> Dict[str, str]:
    alias_map = {}
    for k, v in DEFAULT_ALIASES.items():
        alias_map[normalize_label(k)] = normalize_label(v)

    if alias_json:
        with open(alias_json, "r", encoding="utf-8") as f:
            user_aliases = json.load(f)
        for k, v in user_aliases.items():
            alias_map[normalize_label(k)] = normalize_label(v)

    return alias_map


def load_kinetics_categories(local_json: str | None = None) -> List[str]:
    """
    读取 Kinetics-400 的类别名，返回按 class_id 排序后的类别列表。
    """
    json_path = Path(local_json) if local_json else Path("kinetics_classnames.json")

    if not json_path.exists():
        json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
        urllib.request.urlretrieve(json_url, str(json_path))

    with open(json_path, "r", encoding="utf-8") as f:
        kinetics_classnames = json.load(f)

    id_to_name = {}
    for k, v in kinetics_classnames.items():
        id_to_name[int(v)] = str(k).replace('"', "")

    return [id_to_name[i] for i in sorted(id_to_name.keys())]


def get_model_params(model_name: str) -> Dict[str, int]:
    """
    按 PyTorch Hub X3D 官方示例设置。x3d_xs: 4x12, 182; x3d_s: 13x6, 182; x3d_m: 16x5, 256
    """
    params = {
        "x3d_xs": {"num_frames": 4, "sampling_rate": 12, "side_size": 182, "crop_size": 182},
        "x3d_s": {"num_frames": 13, "sampling_rate": 6, "side_size": 182, "crop_size": 182},
        "x3d_m": {"num_frames": 16, "sampling_rate": 5, "side_size": 256, "crop_size": 256},
    }
    if model_name not in params:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return params[model_name]


def load_x3d_model(model_name: str, device: torch.device):
    model = torch.hub.load("facebookresearch/pytorchvideo", model_name, pretrained=True)
    model = model.eval().to(device)
    return model


def load_video_frames_cv2(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded: {video_path}")

    return frames


def temporal_sample_with_rate(frames: List[np.ndarray], num_frames: int, sampling_rate: int) -> np.ndarray:
    """
    尽量模拟官方 clip_duration = num_frames * sampling_rate / fps 的采样思路。
    对于任意长度视频，这里从全视频中均匀选一个覆盖范围，再按 sampling_rate 取样。
    """
    total = len(frames)

    need = num_frames * sampling_rate
    if total >= need:
        start = max(0, (total - need) // 2)
        idxs = [start + i * sampling_rate for i in range(num_frames)]
    else:
        idxs = np.linspace(0, total - 1, num_frames).astype(int).tolist()

    sampled = [frames[i] for i in idxs]
    return np.stack(sampled, axis=0)  # T,H,W,C


def resize_short_and_center_crop(frames_thwc: np.ndarray, side_size: int, crop_size: int) -> np.ndarray:
    out = []
    for img in frames_thwc:
        h, w = img.shape[:2]
        short_side = min(h, w)
        scale = side_size / float(max(1, short_side))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        h2, w2 = img.shape[:2]
        y0 = max(0, (h2 - crop_size) // 2)
        x0 = max(0, (w2 - crop_size) // 2)
        img = img[y0:y0 + crop_size, x0:x0 + crop_size]

        if img.shape[0] != crop_size or img.shape[1] != crop_size:
            pad_h = crop_size - img.shape[0]
            pad_w = crop_size - img.shape[1]
            img = np.pad(
                img,
                ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            img = img[:crop_size, :crop_size]

        out.append(img)

    return np.stack(out, axis=0)


def to_model_input(frames_thwc: np.ndarray) -> torch.Tensor:
    """
    PyTorch Hub X3D 官方示例使用 mean=[0.45]*3, std=[0.225]*3
    """
    x = torch.from_numpy(frames_thwc).float() / 255.0  # T,H,W,C
    x = x.permute(3, 0, 1, 2)  # C,T,H,W

    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    x = (x - mean) / std
    x = x.unsqueeze(0)  # 1,C,T,H,W
    return x


def top1_predict(
    model,
    model_name: str,
    categories: List[str],
    video_path: str,
    device: torch.device,
    alias_map: Dict[str, str],
) -> Dict:
    params = get_model_params(model_name)

    frames = load_video_frames_cv2(video_path)
    frames = temporal_sample_with_rate(
        frames,
        num_frames=params["num_frames"],
        sampling_rate=params["sampling_rate"],
    )
    frames = resize_short_and_center_crop(
        frames,
        side_size=params["side_size"],
        crop_size=params["crop_size"],
    )
    x = to_model_input(frames).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    pred_idx = int(pred_idx.item())
    conf = float(conf.item())

    raw_label = categories[pred_idx]
    canonical_label = canonicalize_label(raw_label, alias_map)

    return {
        "pred_idx": pred_idx,
        "pred_label_raw": raw_label,
        "pred_label_canonical": canonical_label,
        "confidence": conf,
    }


def save_csv(rows: List[Dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            pass
        return

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--out-dir", default="outputs_eval")
    parser.add_argument("--alias-json", default=None)
    parser.add_argument("--kinetics-json", default=None, help="本地 kinetics_classnames.json 路径；不填则自动下载")
    parser.add_argument("--model-name", default="x3d_xs", choices=["x3d_xs", "x3d_s", "x3d_m"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv)

    base_out_dir = Path(args.out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_out_dir / f"eval_{args.model_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    alias_map = load_aliases(args.alias_json)

    df = pd.read_csv(manifest_csv)
    if "video_path" not in df.columns or "canonical_label" not in df.columns:
        raise ValueError("manifest csv must contain columns: video_path, canonical_label")

    if args.limit is not None:
        df = df.head(args.limit).copy()

    device = torch.device(args.device)
    model = load_x3d_model(args.model_name, device)
    categories = load_kinetics_categories(args.kinetics_json)

    results = []
    skipped = []

    for idx, row in df.iterrows():
        video_path = str(row["video_path"])
        gt_label = canonicalize_label(row["canonical_label"], alias_map)

        try:
            pred = top1_predict(
                model=model,
                model_name=args.model_name,
                categories=categories,
                video_path=video_path,
                device=device,
                alias_map=alias_map,
            )

            correct = int(pred["pred_label_canonical"] == gt_label)

            results.append(
                {
                    "video_path": video_path,
                    "gt_label": gt_label,
                    "pred_label_raw": pred["pred_label_raw"],
                    "pred_label_canonical": pred["pred_label_canonical"],
                    "confidence": pred["confidence"],
                    "correct": correct,
                }
            )

            print(
                f"[{len(results)+len(skipped)}] "
                f"{Path(video_path).name} | gt={gt_label} | pred={pred['pred_label_canonical']} | ok={correct}"
            )

        except Exception as e:
            skipped.append(
                {
                    "video_path": video_path,
                    "gt_label": gt_label,
                    "error": str(e),
                }
            )
            print(f"[SKIP] {video_path} | {e}")

    save_csv(results, out_dir / "sample_results.csv")
    save_csv(skipped, out_dir / "skipped_videos.csv")

    model_params = get_model_params(args.model_name)


    if results:
        res_df = pd.DataFrame(results)
        accuracy = float(res_df["correct"].mean())

        class_stats = []
        for lb, sub in res_df.groupby("gt_label"):
            class_stats.append(
                {
                    "label": lb,
                    "n_samples": int(len(sub)),
                    "accuracy": float(sub["correct"].mean()),
                }
            )
        save_csv(class_stats, out_dir / "per_class_accuracy.csv")

        labels = sorted(set(res_df["gt_label"].tolist()) | set(res_df["pred_label_canonical"].tolist()))
        cm = pd.crosstab(
            res_df["gt_label"],
            res_df["pred_label_canonical"],
            rownames=["gt"],
            colnames=["pred"],
            dropna=False,
        )
        cm = cm.reindex(index=labels, columns=labels, fill_value=0)
        cm.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")

        

        summary = {
            "model_name": args.model_name,
            "backend_used": "pytorchvideo_hub",

            "effective_clip_frames": int(model_params["num_frames"]),  # 实际送进模型的帧数
            "effective_sampling_rate": int(model_params["sampling_rate"]),  # 时间维采样间隔
            "input_resize": int(model_params["side_size"]),
            "input_crop": int(model_params["crop_size"]),

            "n_total_manifest_rows": int(len(df)),
            "n_evaluated": int(len(res_df)),
            "n_skipped": int(len(skipped)),
            "top1_accuracy": accuracy,
            "n_gt_classes_in_eval": int(res_df["gt_label"].nunique()),
            "n_pred_classes_in_eval": int(res_df["pred_label_canonical"].nunique()),
            "avg_confidence": float(res_df["confidence"].mean()),
        }
    else:
        summary = {
            "model_name": args.model_name,
            "backend_used": "pytorchvideo_hub",

            "effective_clip_frames": int(model_params["num_frames"]),
            "effective_sampling_rate": int(model_params["sampling_rate"]),
            "input_resize": int(model_params["side_size"]),
            "input_crop": int(model_params["crop_size"]),

            "n_total_manifest_rows": int(len(df)),
            "n_evaluated": 0,
            "n_skipped": int(len(skipped)),
            "top1_accuracy": None,
        }

    run_config = {
        "manifest_csv": str(manifest_csv),
        "resolved_out_dir": str(out_dir),
        "model_name": args.model_name,
        "device": str(device),
        "limit": args.limit,
        "alias_json": args.alias_json,
        "kinetics_json": args.kinetics_json,
        "num_frames": model_params["num_frames"],
        "sampling_rate": model_params["sampling_rate"],
        "side_size": model_params["side_size"],
        "crop_size": model_params["crop_size"],
    }

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2)

    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()