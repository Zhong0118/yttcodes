from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import re
import time
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch


# =========================================================
# label helpers
# =========================================================

DEFAULT_ALIASES = {
    "run": "running",
    "jog": "running",
    "drink": "drinking",
    "eat": "eating",
    "walk": "walking",
    "sit": "sitting",
    "jumping_jacks": "jumping jacks",
}


def normalize_label(label: str) -> str:
    s = str(label).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_aliases(alias_json: Optional[str]) -> Dict[str, str]:
    alias_map = {normalize_label(k): normalize_label(v) for k, v in DEFAULT_ALIASES.items()}
    if alias_json:
        with open(alias_json, "r", encoding="utf-8") as f:
            user_aliases = json.load(f)
        for k, v in user_aliases.items():
            alias_map[normalize_label(k)] = normalize_label(v)
    return alias_map


def canonicalize_label(label: str, alias_map: Dict[str, str]) -> str:
    x = normalize_label(label)
    return alias_map.get(x, x)


# =========================================================
# kinetics classes / model
# =========================================================

def load_kinetics_categories(local_json: Optional[str] = None) -> List[str]:
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
    return model.eval().to(device)


# =========================================================
# experiment config
# =========================================================

@dataclass
class EvalConfig:
    name: str
    model_name: str = "x3d_xs"

    input_resize: int = 182
    input_crop: int = 182

    chunk_frames: int = 4
    sampling_rate: int = 12
    stride_frames: int = 4

    bandwidth_mbps: float = 8.0
    network_delay_ms: float = 20.0
    packet_loss: float = 0.0
    queue_delay_ms: float = 0.0

    jpeg_quality: int = 80

    max_videos: Optional[int] = None
    max_frames_per_video: Optional[int] = None
    seed: int = 42

    def to_dict(self) -> Dict:
        return asdict(self)


def parse_list(s: Optional[str], cast_type):
    if not s:
        return None
    return [cast_type(x.strip()) for x in s.split(",") if x.strip() != ""]


def build_configs(
    base: EvalConfig,
    mode: str,
    chunk_frames_list: Optional[List[int]],
    input_resize_list: Optional[List[int]],
    sampling_rate_list: Optional[List[int]],
    stride_frames_list: Optional[List[int]],
    bandwidth_list: Optional[List[float]],
    delay_list: Optional[List[float]],
    loss_list: Optional[List[float]],
    jpeg_quality_list: Optional[List[int]],
) -> List[EvalConfig]:
    factor_map = {
        "chunk_frames": chunk_frames_list,
        "input_resize": input_resize_list,
        "sampling_rate": sampling_rate_list,
        "stride_frames": stride_frames_list,
        "bandwidth_mbps": bandwidth_list,
        "network_delay_ms": delay_list,
        "packet_loss": loss_list,
        "jpeg_quality": jpeg_quality_list,
    }

    if mode == "single":
        return [base]

    if mode == "one_factor":
        configs = [base]
        for key, vals in factor_map.items():
            if not vals:
                continue
            for v in vals:
                if getattr(base, key) == v:
                    continue
                cfg = EvalConfig(**base.to_dict())
                setattr(cfg, key, v)
                cfg.name = f"{key}_{str(v).replace('.', '_')}"
                configs.append(cfg)
        return configs

    if mode == "full_grid":
        active = {k: v for k, v in factor_map.items() if v}
        if not active:
            return [base]

        keys = list(active.keys())
        value_lists = [active[k] for k in keys]
        configs = []

        for vals in itertools.product(*value_lists):
            cfg = EvalConfig(**base.to_dict())
            parts = []
            for k, v in zip(keys, vals):
                setattr(cfg, k, v)
                parts.append(f"{k}_{str(v).replace('.', '_')}")
            cfg.name = "__".join(parts)
            configs.append(cfg)

        return configs

    raise ValueError(f"Unsupported mode: {mode}")


# =========================================================
# video helpers
# =========================================================

def load_video_frames_cv2(video_path: str, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 25.0

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded: {video_path}")

    return frames, float(fps)


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
    x = torch.from_numpy(frames_thwc).float() / 255.0
    x = x.permute(3, 0, 1, 2)  # C,T,H,W

    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)

    x = (x - mean) / std
    return x.unsqueeze(0)


def make_chunk_indices(total_frames: int, chunk_frames: int, sampling_rate: int, stride_frames: int) -> List[List[int]]:
    need_span = chunk_frames * sampling_rate

    if total_frames <= 0:
        return []

    if total_frames < need_span:
        idxs = np.linspace(0, total_frames - 1, chunk_frames).astype(int).tolist()
        return [idxs]

    clips = []
    start = 0
    while start + need_span <= total_frames:
        idxs = [start + i * sampling_rate for i in range(chunk_frames)]
        clips.append(idxs)
        start += max(1, stride_frames)

    if not clips:
        idxs = np.linspace(0, total_frames - 1, chunk_frames).astype(int).tolist()
        clips.append(idxs)

    return clips


def estimate_chunk_payload_kb(frames_thwc: np.ndarray, jpeg_quality: int) -> float:
    sizes = []
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    for img in frames_thwc:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", bgr, encode_param)
        if ok:
            sizes.append(len(enc))

    return float(sum(sizes) / 1024.0) if sizes else 0.0


def estimate_tx_ms(payload_kb: float, bandwidth_mbps: float) -> float:
    kb_per_sec = max(1e-6, bandwidth_mbps * 125.0)
    return float(payload_kb / kb_per_sec * 1000.0)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def safe_mean(values: List[float]) -> float:
    vals = []
    for v in values:
        try:
            fv = float(v)
            if not math.isnan(fv):
                vals.append(fv)
        except Exception:
            pass
    return float(sum(vals) / len(vals)) if vals else float("nan")


def video_level_aggregate(chunk_rows: List[Dict]) -> Tuple[str, float]:
    valid = [r for r in chunk_rows if int(r.get("dropped", 0)) == 0 and str(r.get("pred_label", "")) != ""]
    if not valid:
        return "no_prediction", 0.0

    vote_counter = Counter([r["pred_label"] for r in valid])
    max_vote = max(vote_counter.values())
    candidates = [k for k, v in vote_counter.items() if v == max_vote]

    if len(candidates) == 1:
        winner = candidates[0]
    else:
        avg_conf = {}
        for c in candidates:
            confs = [float(r["confidence"]) for r in valid if r["pred_label"] == c]
            avg_conf[c] = sum(confs) / len(confs)
        winner = sorted(avg_conf.items(), key=lambda x: (-x[1], x[0]))[0][0]

    confs = [float(r["confidence"]) for r in valid if r["pred_label"] == winner]
    return winner, float(sum(confs) / len(confs)) if confs else 0.0


def compute_switch_rate(chunk_rows: List[Dict]) -> float:
    seq = [r["pred_label"] for r in chunk_rows if int(r.get("dropped", 0)) == 0 and r.get("pred_label")]
    if len(seq) <= 1:
        return 0.0
    switches = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    return float(switches / (len(seq) - 1))


# =========================================================
# inference core
# =========================================================

def predict_chunk(
    model,
    categories: List[str],
    chunk_frames_thwc: np.ndarray,
    device: torch.device,
    alias_map: Dict[str, str],
) -> Dict:
    x = to_model_input(chunk_frames_thwc).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    pred_idx = int(pred_idx.item())
    conf = float(conf.item())
    raw_label = categories[pred_idx]
    pred_label = canonicalize_label(raw_label, alias_map)

    return {
        "pred_label_raw": raw_label,
        "pred_label": pred_label,
        "confidence": conf,
        "infer_ms": float(infer_ms),
    }


def evaluate_one_video(
    model,
    categories: List[str],
    video_path: str,
    gt_label: str,
    cfg: EvalConfig,
    device: torch.device,
    alias_map: Dict[str, str],
    rng: random.Random,
) -> Tuple[List[Dict], Dict]:
    frames, src_fps = load_video_frames_cv2(video_path, max_frames=cfg.max_frames_per_video)

    clips = make_chunk_indices(
        total_frames=len(frames),
        chunk_frames=cfg.chunk_frames,
        sampling_rate=cfg.sampling_rate,
        stride_frames=cfg.stride_frames,
    )

    rows: List[Dict] = []
    first_result_ms: Optional[float] = None

    for chunk_id, idxs in enumerate(clips):
        start_idx = idxs[0]
        end_idx = idxs[-1]
        start_sec = float(start_idx / src_fps)
        end_sec = float(end_idx / src_fps)

        raw_chunk = np.stack([frames[i] for i in idxs], axis=0)
        proc_chunk = resize_short_and_center_crop(raw_chunk, cfg.input_resize, cfg.input_crop)

        payload_kb = estimate_chunk_payload_kb(proc_chunk, cfg.jpeg_quality)
        tx_ms = estimate_tx_ms(payload_kb, cfg.bandwidth_mbps)

        dropped = 1 if rng.random() < cfg.packet_loss else 0

        if dropped:
            rows.append(
                {
                    "video_path": video_path,
                    "gt_label": gt_label,
                    "chunk_id": chunk_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "pred_label": "",
                    "pred_label_raw": "",
                    "confidence": 0.0,
                    "infer_ms": float("nan"),
                    "payload_kb": payload_kb,
                    "tx_ms": tx_ms,
                    "queue_ms": float(cfg.queue_delay_ms),
                    "network_delay_ms": float(cfg.network_delay_ms),
                    "first_result_ms": float("nan"),
                    "end_to_end_ms": float("nan"),
                    "dropped": 1,
                    "correct": 0,
                }
            )
            continue

        pred = predict_chunk(model, categories, proc_chunk, device, alias_map)
        end_to_end_ms = float(cfg.queue_delay_ms + tx_ms + cfg.network_delay_ms + pred["infer_ms"])

        if first_result_ms is None:
            first_result_ms = end_to_end_ms

        rows.append(
            {
                "video_path": video_path,
                "gt_label": gt_label,
                "chunk_id": chunk_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "pred_label": pred["pred_label"],
                "pred_label_raw": pred["pred_label_raw"],
                "confidence": pred["confidence"],
                "infer_ms": pred["infer_ms"],
                "payload_kb": payload_kb,
                "tx_ms": tx_ms,
                "queue_ms": float(cfg.queue_delay_ms),
                "network_delay_ms": float(cfg.network_delay_ms),
                "first_result_ms": first_result_ms,
                "end_to_end_ms": end_to_end_ms,
                "dropped": 0,
                "correct": int(pred["pred_label"] == gt_label),
            }
        )

    video_pred, video_conf = video_level_aggregate(rows)
    valid_rows = [r for r in rows if int(r["dropped"]) == 0]

    sample_summary = {
        "video_path": video_path,
        "gt_label": gt_label,
        "pred_label": video_pred,
        "confidence": video_conf,
        "video_correct": int(video_pred == gt_label),
        "n_chunks": len(rows),
        "n_effective_chunks": len(valid_rows),
        "n_dropped_chunks": int(sum(int(r["dropped"]) for r in rows)),
        "chunk_accuracy": safe_mean([r["correct"] for r in valid_rows]) if valid_rows else float("nan"),
        "switch_rate": compute_switch_rate(rows),
        "infer_avg_ms": safe_mean([r["infer_ms"] for r in valid_rows]) if valid_rows else float("nan"),
        "infer_p95_ms": percentile([r["infer_ms"] for r in valid_rows], 95) if valid_rows else float("nan"),
        "tx_avg_ms": safe_mean([r["tx_ms"] for r in rows]),
        "first_result_ms": float(first_result_ms) if first_result_ms is not None else float("nan"),
        "end_to_end_avg_ms": safe_mean([r["end_to_end_ms"] for r in valid_rows]) if valid_rows else float("nan"),
    }

    return rows, sample_summary


# =========================================================
# run orchestration
# =========================================================

def evaluate_config(
    manifest_df: pd.DataFrame,
    cfg: EvalConfig,
    categories: List[str],
    device: torch.device,
    alias_map: Dict[str, str],
    run_dir: Path,
) -> Dict:
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)

    rng = random.Random(cfg.seed)
    model = load_x3d_model(cfg.model_name, device)

    if cfg.max_videos is not None:
        manifest_df = manifest_df.head(cfg.max_videos).copy()

    all_chunk_rows: List[Dict] = []
    all_sample_rows: List[Dict] = []
    skipped_rows: List[Dict] = []

    run_start = time.perf_counter()

    for _, row in manifest_df.iterrows():
        video_path = str(row["video_path"])
        gt_label = canonicalize_label(row["canonical_label"], alias_map)

        try:
            chunk_rows, sample_summary = evaluate_one_video(
                model=model,
                categories=categories,
                video_path=video_path,
                gt_label=gt_label,
                cfg=cfg,
                device=device,
                alias_map=alias_map,
                rng=rng,
            )

            all_chunk_rows.extend(chunk_rows)
            all_sample_rows.append(sample_summary)

            print(
                f"[{cfg.name}] "
                f"{len(all_sample_rows)+len(skipped_rows)}/{len(manifest_df)} | "
                f"{Path(video_path).name} | gt={gt_label} | pred={sample_summary['pred_label']} | ok={sample_summary['video_correct']}"
            )

        except Exception as e:
            skipped_rows.append(
                {
                    "video_path": video_path,
                    "gt_label": gt_label,
                    "error": str(e),
                }
            )
            print(f"[{cfg.name}] [SKIP] {video_path} | {e}")

    run_wall_time_sec = float(time.perf_counter() - run_start)

    pd.DataFrame(all_chunk_rows).to_csv(run_dir / "chunk_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_sample_rows).to_csv(run_dir / "sample_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped_rows).to_csv(run_dir / "skipped_videos.csv", index=False, encoding="utf-8-sig")

    if all_sample_rows:
        sdf = pd.DataFrame(all_sample_rows)

        labels = sorted(set(sdf["gt_label"].tolist()) | set(sdf["pred_label"].tolist()))
        cm = pd.crosstab(
            sdf["gt_label"],
            sdf["pred_label"],
            rownames=["gt"],
            colnames=["pred"],
            dropna=False,
        )
        cm = cm.reindex(index=labels, columns=labels, fill_value=0)
        cm.to_csv(run_dir / "confusion_matrix.csv", encoding="utf-8-sig")

        per_class = []
        for lb, sub in sdf.groupby("gt_label"):
            per_class.append(
                {
                    "label": lb,
                    "n_samples": int(len(sub)),
                    "video_accuracy": float(sub["video_correct"].mean()),
                    "avg_confidence": float(sub["confidence"].mean()),
                }
            )
        pd.DataFrame(per_class).to_csv(run_dir / "per_class_accuracy.csv", index=False, encoding="utf-8-sig")
    else:
        sdf = pd.DataFrame(columns=["video_correct", "confidence", "gt_label", "pred_label"])

    cdf = pd.DataFrame(all_chunk_rows) if all_chunk_rows else pd.DataFrame()

    summary = {
        "name": cfg.name,
        "model_name": cfg.model_name,
        "backend_used": "pytorchvideo_hub",

        "effective_clip_frames": int(cfg.chunk_frames),
        "effective_sampling_rate": int(cfg.sampling_rate),
        "input_resize": int(cfg.input_resize),
        "input_crop": int(cfg.input_crop),
        "stride_frames": int(cfg.stride_frames),

        "bandwidth_mbps": float(cfg.bandwidth_mbps),
        "network_delay_ms": float(cfg.network_delay_ms),
        "packet_loss": float(cfg.packet_loss),
        "queue_delay_ms": float(cfg.queue_delay_ms),
        "jpeg_quality": int(cfg.jpeg_quality),

        "n_total_manifest_rows": int(len(manifest_df)),
        "n_evaluated_videos": int(len(all_sample_rows)),
        "n_skipped_videos": int(len(skipped_rows)),

        "video_top1_accuracy": float(sdf["video_correct"].mean()) if not sdf.empty else float("nan"),
        "avg_confidence": float(sdf["confidence"].mean()) if not sdf.empty else float("nan"),
        "n_gt_classes_in_eval": int(sdf["gt_label"].nunique()) if not sdf.empty else 0,
        "n_pred_classes_in_eval": int(sdf["pred_label"].nunique()) if not sdf.empty else 0,

        "n_chunks": int(len(cdf)) if not cdf.empty else 0,
        "n_effective_chunks": int((cdf["dropped"] == 0).sum()) if not cdf.empty else 0,
        "n_dropped_chunks": int((cdf["dropped"] == 1).sum()) if not cdf.empty else 0,

        "chunk_top1_accuracy": (
            float(cdf.loc[cdf["dropped"] == 0, "correct"].mean())
            if not cdf.empty and (cdf["dropped"] == 0).any()
            else float("nan")
        ),
        "infer_avg_ms": safe_mean(cdf.loc[cdf["dropped"] == 0, "infer_ms"].tolist()) if not cdf.empty else float("nan"),
        "infer_p95_ms": percentile(cdf.loc[cdf["dropped"] == 0, "infer_ms"].tolist(), 95) if not cdf.empty else float("nan"),
        "tx_avg_ms": safe_mean(cdf["tx_ms"].tolist()) if not cdf.empty else float("nan"),
        "first_result_ms": safe_mean([r["first_result_ms"] for r in all_sample_rows]) if all_sample_rows else float("nan"),
        "end_to_end_avg_ms": safe_mean(cdf.loc[cdf["dropped"] == 0, "end_to_end_ms"].tolist()) if not cdf.empty else float("nan"),
        "prediction_switch_rate_avg": safe_mean([r["switch_rate"] for r in all_sample_rows]) if all_sample_rows else float("nan"),

        "run_wall_time_sec": run_wall_time_sec,
        "run_wall_time_min": run_wall_time_sec / 60.0,
    }

    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# =========================================================
# main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Grid experiment runner for X3D on Kinetics-style dataset manifests.")

    parser.add_argument("--manifest-csv", required=True, help="CSV containing at least: video_path, canonical_label")
    parser.add_argument("--out-dir", default="outputs_grid")
    parser.add_argument("--alias-json", default=None)
    parser.add_argument("--kinetics-json", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--mode", default="one_factor", choices=["single", "one_factor", "full_grid"])
    parser.add_argument("--model-name", default="x3d_xs", choices=["x3d_xs", "x3d_s", "x3d_m"])

    parser.add_argument("--base-chunk-frames", type=int, default=None)
    parser.add_argument("--base-sampling-rate", type=int, default=None)
    parser.add_argument("--base-input-resize", type=int, default=None)
    parser.add_argument("--base-input-crop", type=int, default=None)
    parser.add_argument("--base-stride-frames", type=int, default=None)

    parser.add_argument("--base-bandwidth-mbps", type=float, default=8.0)
    parser.add_argument("--base-network-delay-ms", type=float, default=20.0)
    parser.add_argument("--base-packet-loss", type=float, default=0.0)
    parser.add_argument("--base-queue-delay-ms", type=float, default=0.0)
    parser.add_argument("--base-jpeg-quality", type=int, default=80)

    parser.add_argument("--chunk-frames-list", default=None)
    parser.add_argument("--sampling-rate-list", default=None)
    parser.add_argument("--input-resize-list", default=None)
    parser.add_argument("--stride-frames-list", default=None)
    parser.add_argument("--bandwidth-list", default=None)
    parser.add_argument("--delay-list", default=None)
    parser.add_argument("--loss-list", default=None)
    parser.add_argument("--jpeg-quality-list", default=None)

    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    alias_map = load_aliases(args.alias_json)
    categories = load_kinetics_categories(args.kinetics_json)
    model_defaults = get_model_params(args.model_name)

    base_chunk_frames = int(args.base_chunk_frames or model_defaults["num_frames"])

    base_cfg = EvalConfig(
        name="base",
        model_name=args.model_name,

        input_resize=int(args.base_input_resize or model_defaults["side_size"]),
        input_crop=int(args.base_input_crop or model_defaults["crop_size"]),

        chunk_frames=base_chunk_frames,
        sampling_rate=int(args.base_sampling_rate or model_defaults["sampling_rate"]),
        stride_frames=int(args.base_stride_frames or base_chunk_frames),

        bandwidth_mbps=float(args.base_bandwidth_mbps),
        network_delay_ms=float(args.base_network_delay_ms),
        packet_loss=float(args.base_packet_loss),
        queue_delay_ms=float(args.base_queue_delay_ms),

        jpeg_quality=int(args.base_jpeg_quality),

        max_videos=args.max_videos,
        max_frames_per_video=args.max_frames_per_video,
        seed=int(args.seed),
    )

    configs = build_configs(
        base=base_cfg,
        mode=args.mode,
        chunk_frames_list=parse_list(args.chunk_frames_list, int),
        input_resize_list=parse_list(args.input_resize_list, int),
        sampling_rate_list=parse_list(args.sampling_rate_list, int),
        stride_frames_list=parse_list(args.stride_frames_list, int),
        bandwidth_list=parse_list(args.bandwidth_list, float),
        delay_list=parse_list(args.delay_list, float),
        loss_list=parse_list(args.loss_list, float),
        jpeg_quality_list=parse_list(args.jpeg_quality_list, int),
    )

    manifest_df = pd.read_csv(args.manifest_csv)
    if "video_path" not in manifest_df.columns or "canonical_label" not in manifest_df.columns:
        raise ValueError("manifest csv must contain columns: video_path, canonical_label")

    device = torch.device(args.device)

    base_out_dir = Path(args.out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = base_out_dir / f"grid_{args.model_name}_{args.mode}_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "manifest_csv": str(Path(args.manifest_csv).resolve()),
        "resolved_batch_dir": str(batch_dir.resolve()),
        "device": str(device),
        "mode": args.mode,
        "n_configs": len(configs),
        "alias_json": args.alias_json,
        "kinetics_json": args.kinetics_json,
    }

    with open(batch_dir / "batch_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    summaries = []

    for i, cfg in enumerate(configs):
        run_dir = batch_dir / f"run_{i:04d}"
        print(f"\n========== [{i+1}/{len(configs)}] {cfg.name} ==========")

        summary = evaluate_config(
            manifest_df=manifest_df,
            cfg=cfg,
            categories=categories,
            device=device,
            alias_map=alias_map,
            run_dir=run_dir,
        )
        summaries.append(summary)

    runs_df = pd.DataFrame(summaries)
    runs_df.to_csv(batch_dir / "runs_summary.csv", index=False, encoding="utf-8-sig")
    runs_df.to_excel(batch_dir / "runs_summary.xlsx", index=False)

    with open(batch_dir / "runs_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] batch dir: {batch_dir.resolve()}")


if __name__ == "__main__":
    main()