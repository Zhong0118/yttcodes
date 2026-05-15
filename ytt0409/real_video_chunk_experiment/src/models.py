from __future__ import annotations
from typing import Dict, List, Union
import time
import cv2
import numpy as np

ResultValue = Union[float, str]


class BaseModel:
    def predict_chunk(self, chunk_frames: List[np.ndarray]) -> Dict[str, ResultValue]:
        raise NotImplementedError


class PlaceholderModel(BaseModel):
    '''
    当前默认模型：只是为了把真实视频 chunk 分析流程跑通。
    你后面只需要替换这个类，不需要动其他代码。
    '''

    def predict_chunk(self, chunk_frames: List[np.ndarray]) -> Dict[str, ResultValue]:
        t0 = time.perf_counter()

        if not chunk_frames:
            return {"pred_label": "none", "confidence": 0.0, "infer_ms": 0.0}

        h, w = chunk_frames[0].shape[:2]
        pseudo_compute = (h * w * len(chunk_frames)) / 2500000.0
        time.sleep(min(0.03, 0.002 + pseudo_compute * 0.004))

        brightness = []
        motion_vals = []
        prev = None

        for frame in chunk_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(float(gray.mean()))
            if prev is not None:
                motion_vals.append(float(np.mean(np.abs(gray.astype(np.float32) - prev.astype(np.float32)))))
            prev = gray

        mean_brightness = float(np.mean(brightness)) if brightness else 0.0
        mean_motion = float(np.mean(motion_vals)) if motion_vals else 0.0

        if mean_motion > 20:
            pred = "high_motion"
            confidence = min(0.95, 0.6 + mean_motion / 100.0)
        elif mean_brightness < 70:
            pred = "dark_scene"
            confidence = min(0.90, 0.55 + (70 - mean_brightness) / 120.0)
        elif mean_brightness > 160:
            pred = "bright_scene"
            confidence = min(0.90, 0.55 + (mean_brightness - 160) / 150.0)
        else:
            pred = "normal_scene"
            confidence = 0.60

        infer_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "pred_label": pred,
            "confidence": float(confidence),
            "infer_ms": float(infer_ms),
        }


def build_model(model_name: str) -> BaseModel:
    if model_name == "placeholder":
        return PlaceholderModel()
    raise ValueError(f"unknown model_name: {model_name}")
