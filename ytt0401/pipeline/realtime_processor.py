import time
import threading
from dataclasses import dataclass, asdict
from collections import deque
from typing import Deque, Optional, Tuple, List

import cv2
import numpy as np

from models.action_model import ActionRecognizer, ActionPrediction
from utils.overlay import draw_multiline_info, draw_title_bar, current_time_str


@dataclass
class RuntimeConfig:
    bandwidth_kbps: int = 4000
    network_delay_ms: int = 30
    ai_extra_delay_ms: int = 0
    ai_input_size: int = 256
    ai_fps: int = 12
    clip_len: int = 16
    model_name: str = "x3d_s"
    jpeg_quality: int = 85


@dataclass
class RuntimeMetrics:
    user_latency_ms: float = 0.0
    ai_latency_ms: float = 0.0
    overlay_skew_ms: float = 0.0
    infer_ms: float = 0.0
    infer_avg_ms: float = 0.0
    confidence: float = 0.0
    pred_label: str = "warming_up"
    description: str = "系统正在预热。"
    top3_text: str = "-"
    model_name: str = "x3d_s"
    resolution: str = "256x256"
    bandwidth_kbps: int = 4000
    device: str = "cpu"
    network_delay_ms: int = 30
    ai_extra_delay_ms: int = 0
    ai_fps: int = 12
    clip_len: int = 16
    ts_capture_ms: float = 0.0
    ts_user_show_ms: float = 0.0
    ts_ai_done_ms: float = 0.0


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.client_frame: Optional[np.ndarray] = None
        self.ai_frame: Optional[np.ndarray] = None
        self.metrics = RuntimeMetrics()
        self.running = False

        self.clip_frames: Deque[np.ndarray] = deque(maxlen=64)
        self.clip_ts_ms: Deque[float] = deque(maxlen=64)
        self.infer_times: Deque[float] = deque(maxlen=100)

        self.last_pred = ActionPrediction(
            label="warming_up",
            confidence=0.0,
            infer_ms=0.0,
            raw_index=-1,
            description="系统正在预热，尚未形成稳定识别结果。",
            topk_labels=[],
            topk_scores=[],
        )
        self.last_ai_done_ms = 0.0
        self.last_sample_wall_time = 0.0

        self.recognizer: Optional[ActionRecognizer] = None
        self.config = RuntimeConfig()

    def reset_runtime_cache(self):
        with self.lock:
            self.clip_frames.clear()
            self.clip_ts_ms.clear()
            self.infer_times.clear()
            self.last_pred = ActionPrediction(
                label="warming_up",
                confidence=0.0,
                infer_ms=0.0,
                raw_index=-1,
                description="系统正在预热，尚未形成稳定识别结果。",
                topk_labels=[],
                topk_scores=[],
            )
            self.last_ai_done_ms = 0.0
            self.last_sample_wall_time = 0.0
            self.client_frame = None
            self.ai_frame = None


def now_ms() -> float:
    return time.time() * 1000.0


def simulate_transport(
    frame_bgr: np.ndarray,
    bandwidth_kbps: int,
    network_delay_ms: int,
    jpeg_quality: int,
) -> Tuple[np.ndarray, float]:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    ok, enc = cv2.imencode(".jpg", frame_bgr, encode_param)
    if not ok:
        return frame_bgr.copy(), float(network_delay_ms)

    payload_bytes = len(enc.tobytes())
    bandwidth_Bps = max(1.0, bandwidth_kbps * 1000.0 / 8.0)
    tx_ms = payload_bytes / bandwidth_Bps * 1000.0
    total_delay_ms = network_delay_ms + tx_ms

    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec is None:
        dec = frame_bgr.copy()
    return dec, total_delay_ms


def wrap_text_lines(text: str, max_chars: int = 26) -> List[str]:
    text = text.strip()
    if not text:
        return []
    lines = []
    start = 0
    while start < len(text):
        lines.append(text[start:start + max_chars])
        start += max_chars
    return lines


class RealtimeProcessor:
    def __init__(self, shared_state: SharedState):
        self.state = shared_state

    def ensure_model(self):
        cfg = self.state.config
        need_new = False
        with self.state.lock:
            if self.state.recognizer is None:
                need_new = True
            elif self.state.recognizer.model_name != cfg.model_name:
                need_new = True

        if need_new:
            recognizer = ActionRecognizer(model_name=cfg.model_name)
            with self.state.lock:
                self.state.recognizer = recognizer
                self.state.metrics.device = recognizer.device

    def set_running(self, flag: bool):
        with self.state.lock:
            self.state.running = flag

    def update_config(self, cfg: RuntimeConfig):
        with self.state.lock:
            self.state.config = cfg
            self.state.metrics.model_name = cfg.model_name
            self.state.metrics.resolution = f"{cfg.ai_input_size}x{cfg.ai_input_size}"
            self.state.metrics.bandwidth_kbps = cfg.bandwidth_kbps
            self.state.metrics.network_delay_ms = cfg.network_delay_ms
            self.state.metrics.ai_extra_delay_ms = cfg.ai_extra_delay_ms
            self.state.metrics.ai_fps = cfg.ai_fps
            self.state.metrics.clip_len = cfg.clip_len

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        self.ensure_model()

        with self.state.lock:
            cfg = self.state.config
            running = self.state.running

        ts_capture = now_ms()

        if not running:
            idle = frame_bgr.copy()
            draw_title_bar(idle, "摄像头待机中")
            with self.state.lock:
                self.state.client_frame = idle
                self.state.ai_frame = idle.copy()
            return idle

        client_frame, simulated_user_delay_ms = simulate_transport(
            frame_bgr,
            cfg.bandwidth_kbps,
            cfg.network_delay_ms,
            cfg.jpeg_quality,
        )
        ts_user_show = ts_capture + simulated_user_delay_ms
        draw_title_bar(client_frame, "客户端收到的视频流")

        self._append_ai_sample_if_needed(frame_bgr, ts_capture)

        with self.state.lock:
            ready = len(self.state.clip_frames) >= cfg.clip_len

        if ready:
            self._run_inference_if_ready(cfg)

        with self.state.lock:
            pred = self.state.last_pred
            infer_avg_ms = float(np.mean(self.state.infer_times)) if self.state.infer_times else 0.0
            ts_ai_done = self.state.last_ai_done_ms

        top3_pairs = list(zip(pred.topk_labels[:3], pred.topk_scores[:3]))
        top3_text = " / ".join([f"{lbl.replace('_', ' ')}:{score:.2f}" for lbl, score in top3_pairs]) if top3_pairs else "-"

        ai_frame = client_frame.copy()
        draw_title_bar(ai_frame, "AI 返回结果后的视频流")

        desc_lines = wrap_text_lines(pred.description, max_chars=26)
        lines = [
            f"当前动作: {pred.label}",
            f"置信度: {pred.confidence:.3f}",
            f"时间戳: {current_time_str()}",
            f"推理耗时: {pred.infer_ms:.1f} ms",
            "Top-3:",
            top3_text[:42],
            "分析描述:",
        ] + desc_lines

        draw_multiline_info(
            ai_frame,
            lines,
            start_xy=(10, 72),
            line_gap=32,
            color=(0, 255, 0),
        )

        user_latency_ms = max(0.0, ts_user_show - ts_capture)
        ai_latency_ms = max(0.0, ts_ai_done - ts_capture) if ts_ai_done > 0 else 0.0
        overlay_skew_ms = ai_latency_ms - user_latency_ms if ts_ai_done > 0 else 0.0

        with self.state.lock:
            self.state.client_frame = client_frame
            self.state.ai_frame = ai_frame
            self.state.metrics = RuntimeMetrics(
                user_latency_ms=user_latency_ms,
                ai_latency_ms=ai_latency_ms,
                overlay_skew_ms=overlay_skew_ms,
                infer_ms=pred.infer_ms,
                infer_avg_ms=infer_avg_ms,
                confidence=pred.confidence,
                pred_label=pred.label,
                description=pred.description,
                top3_text=top3_text,
                model_name=cfg.model_name,
                resolution=f"{cfg.ai_input_size}x{cfg.ai_input_size}",
                bandwidth_kbps=cfg.bandwidth_kbps,
                device=self.state.recognizer.device if self.state.recognizer else "cpu",
                network_delay_ms=cfg.network_delay_ms,
                ai_extra_delay_ms=cfg.ai_extra_delay_ms,
                ai_fps=cfg.ai_fps,
                clip_len=cfg.clip_len,
                ts_capture_ms=ts_capture,
                ts_user_show_ms=ts_user_show,
                ts_ai_done_ms=ts_ai_done,
            )

        return ai_frame

    def _append_ai_sample_if_needed(self, frame_bgr: np.ndarray, ts_capture: float):
        with self.state.lock:
            cfg = self.state.config
            now_wall = time.time()
            interval = 1.0 / max(1, cfg.ai_fps)
            should_sample = (now_wall - self.state.last_sample_wall_time) >= interval

            if should_sample:
                self.state.last_sample_wall_time = now_wall
                resized = cv2.resize(frame_bgr, (cfg.ai_input_size, cfg.ai_input_size))
                self.state.clip_frames.append(resized)
                self.state.clip_ts_ms.append(ts_capture)

    def _run_inference_if_ready(self, cfg: RuntimeConfig):
        with self.state.lock:
            frames = list(self.state.clip_frames)[-cfg.clip_len:]
            recognizer = self.state.recognizer

        pred = recognizer.infer_clip(frames)

        if cfg.ai_extra_delay_ms > 0:
            time.sleep(cfg.ai_extra_delay_ms / 1000.0)

        ts_ai_done = now_ms()

        with self.state.lock:
            self.state.last_pred = pred
            self.state.last_ai_done_ms = ts_ai_done
            self.state.infer_times.append(pred.infer_ms)

    def get_display_frames(self):
        with self.state.lock:
            c = None if self.state.client_frame is None else self.state.client_frame.copy()
            a = None if self.state.ai_frame is None else self.state.ai_frame.copy()
        return c, a

    def get_metrics_dict(self):
        with self.state.lock:
            return asdict(self.state.metrics)