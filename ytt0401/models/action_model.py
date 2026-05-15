import json
import urllib.request
from dataclasses import dataclass
from typing import List, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F


KINETICS_JSON_URL = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"


@dataclass
class ActionPrediction:
    label: str
    confidence: float
    infer_ms: float
    raw_index: int
    description: str
    topk_labels: List[str]
    topk_scores: List[float]
    # clip_start_ts_ms: float
    # clip_end_ts_ms: float
    # clip_center_ts_ms: float


class ActionRecognizer:
    def __init__(
        self,
        model_name: str = "x3d_s",
        device: str = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 官方 TorchHub 示例里的 X3D transform 参数
        self.model_transform_params = {
            "x3d_xs": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 4,
                "sampling_rate": 12,
            },
            "x3d_s": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 13,
                "sampling_rate": 6,
            },
            "x3d_m": {
                "side_size": 256,
                "crop_size": 256,
                "num_frames": 16,
                "sampling_rate": 5,
            },
        }

        if self.model_name not in self.model_transform_params:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

        self.params = self.model_transform_params[self.model_name]
        self.model = self._load_model()
        self.id_to_label = self._load_kinetics_labels()

    def _load_model(self):
        model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            self.model_name,
            pretrained=True,
        )
        model = model.eval().to(self.device)
        return model

    def _load_kinetics_labels(self) -> Dict[int, str]:
        with urllib.request.urlopen(KINETICS_JSON_URL) as resp:
            kinetics_classnames = json.loads(resp.read().decode("utf-8"))
        # 官方示例里是 name -> id，这里转成 id -> name
        out = {}
        for class_name, idx in kinetics_classnames.items():
            out[int(idx)] = class_name
        return out

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.45, 0.45, 0.45], device=x.device).view(3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225], device=x.device).view(3, 1, 1, 1)
        return (x - mean) / std

    def _center_crop(self, image: np.ndarray, size: int) -> np.ndarray:
        h, w = image.shape[:2]
        y1 = max(0, (h - size) // 2)
        x1 = max(0, (w - size) // 2)
        return image[y1:y1 + size, x1:x1 + size]

    def _resize_short_side(self, image: np.ndarray, short_side: int) -> np.ndarray:
        h, w = image.shape[:2]
        if h < w:
            new_h = short_side
            new_w = int(w * short_side / h)
        else:
            new_w = short_side
            new_h = int(h * short_side / w)
        return cv2.resize(image, (new_w, new_h))

    def _uniform_temporal_subsample(self, frames: List[np.ndarray], num_frames: int) -> List[np.ndarray]:
        if len(frames) == 0:
            return []
        if len(frames) <= num_frames:
            out = list(frames)
            while len(out) < num_frames:
                out.append(out[-1])
            return out
        idx = np.linspace(0, len(frames) - 1, num=num_frames).astype(int)
        return [frames[i] for i in idx]

    def _prepare_clip_tensor(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        frames_bgr = self._uniform_temporal_subsample(frames_bgr, self.params["num_frames"])

        processed = []
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = self._resize_short_side(rgb, self.params["side_size"])
            rgb = self._center_crop(rgb, self.params["crop_size"])
            processed.append(rgb)

        arr = np.stack(processed, axis=0).astype(np.float32) / 255.0  # [T,H,W,C]
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()  # [C,T,H,W]
        tensor = tensor.to(self.device)
        tensor = self._normalize(tensor)
        tensor = tensor.unsqueeze(0)  # [1,C,T,H,W]
        return tensor

    def _to_zh(self, s: str) -> str:
        # 简单把英文标签转成更易读形式
        return s.replace("_", " ").replace('"', "")

    def _build_description(self, topk_labels: List[str], topk_scores: List[float]) -> str:
        if not topk_labels:
            return "当前未形成稳定识别结果。"

        top1 = self._to_zh(topk_labels[0])
        top1_score = topk_scores[0]

        if len(topk_labels) >= 3:
            others = f"{self._to_zh(topk_labels[1])}、{self._to_zh(topk_labels[2])}"
        elif len(topk_labels) == 2:
            others = self._to_zh(topk_labels[1])
        else:
            others = ""

        if top1_score >= 0.35:
            confidence_text = "当前主判定相对明确"
        elif top1_score >= 0.20:
            confidence_text = "当前主判定具有一定可信度"
        else:
            confidence_text = "当前动作区分度较低，结果仅供参考"

        if others:
            return f"当前画面更接近“{top1}”类动作，同时与“{others}”存在一定相似性；{confidence_text}。"
        return f"当前画面更接近“{top1}”类动作；{confidence_text}。"

    def infer_clip(self, frames_bgr: List[np.ndarray]) -> ActionPrediction:
        if len(frames_bgr) == 0:
            return ActionPrediction(
                label="warming_up",
                confidence=0.0,
                infer_ms=0.0,
                raw_index=-1,
                description="系统正在预热，尚未形成稳定识别结果。",
                topk_labels=[],
                topk_scores=[],
            )

        import time
        t0 = time.perf_counter()

        x = self._prepare_clip_tensor(frames_bgr)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]
            topk = torch.topk(probs, k=5)

        infer_ms = (time.perf_counter() - t0) * 1000.0

        topk_indices = topk.indices.cpu().tolist()
        topk_scores = topk.values.cpu().tolist()
        topk_labels = [self.id_to_label.get(i, f"class_{i}") for i in topk_indices]

        pred_idx = topk_indices[0]
        pred_label = topk_labels[0]
        pred_conf = topk_scores[0]
        desc = self._build_description(topk_labels[:3], topk_scores[:3])

        return ActionPrediction(
            label=self._to_zh(pred_label),
            confidence=float(pred_conf),
            infer_ms=infer_ms,
            raw_index=int(pred_idx),
            description=desc,
            topk_labels=topk_labels,
            topk_scores=[float(x) for x in topk_scores],
        )