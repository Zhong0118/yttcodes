from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time
import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None


@dataclass
class WindowAnalysisResult:
    window_id: int
    start_ms: float
    end_ms: float
    infer_ms: float
    label: str
    confidence: float
    pose_detect_rate: float
    hand_detect_rate: float
    keypoint_completeness: float
    motion_energy: float
    stability_score: float
    embedding: np.ndarray
    n_frames: int


class KeypointAnalyzer:
    def __init__(self, min_det_conf: float = 0.3, min_track_conf: float = 0.3):
        if mp is None:
            raise ImportError("mediapipe is not installed. Please `pip install mediapipe`.")
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )

    def analyze_window(self, frames_bgr: list[np.ndarray], start_ms: float, end_ms: float, window_id: int) -> WindowAnalysisResult:
        t0 = time.perf_counter()
        pose_hits = 0
        hand_hits = 0
        completeness_values: list[float] = []
        frame_embeddings: list[np.ndarray] = []
        prev_vec: np.ndarray | None = None
        motion_values: list[float] = []
        smoothness_values: list[float] = []

        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_res = self.pose.process(rgb)
            hands_res = self.hands.process(rgb)
            pose_vec = self._pose_vector(pose_res)
            hand_vec = self._hands_vector(hands_res)
            full_vec = np.concatenate([pose_vec, hand_vec]).astype(np.float32)
            frame_embeddings.append(full_vec)

            pose_valid = float(np.count_nonzero(pose_vec)) / max(len(pose_vec), 1)
            hand_valid = float(np.count_nonzero(hand_vec)) / max(len(hand_vec), 1)
            completeness_values.append((pose_valid + hand_valid) / 2.0)
            if pose_valid > 0.02:
                pose_hits += 1
            if hand_valid > 0.02:
                hand_hits += 1

            if prev_vec is not None:
                delta = np.linalg.norm(full_vec - prev_vec)
                motion_values.append(float(delta))
                smoothness_values.append(float(np.mean(np.abs(full_vec - prev_vec))))
            prev_vec = full_vec

        infer_ms = (time.perf_counter() - t0) * 1000.0
        if frame_embeddings:
            emb = np.mean(np.stack(frame_embeddings, axis=0), axis=0)
        else:
            emb = np.zeros((33 * 4 + 42 * 3,), dtype=np.float32)

        motion_energy = float(np.mean(motion_values)) if motion_values else 0.0
        stability_score = 1.0 / (1.0 + (float(np.mean(smoothness_values)) if smoothness_values else 0.0))
        pose_rate = pose_hits / max(len(frames_bgr), 1)
        hand_rate = hand_hits / max(len(frames_bgr), 1)
        completeness = float(np.mean(completeness_values)) if completeness_values else 0.0
        label, conf = self._heuristic_label(motion_energy, pose_rate, hand_rate, completeness)

        return WindowAnalysisResult(
            window_id=window_id,
            start_ms=start_ms,
            end_ms=end_ms,
            infer_ms=infer_ms,
            label=label,
            confidence=conf,
            pose_detect_rate=pose_rate,
            hand_detect_rate=hand_rate,
            keypoint_completeness=completeness,
            motion_energy=motion_energy,
            stability_score=stability_score,
            embedding=emb,
            n_frames=len(frames_bgr),
        )

    @staticmethod
    def _pose_vector(pose_res: Any) -> np.ndarray:
        n = 33
        if pose_res.pose_landmarks is None:
            return np.zeros((n * 4,), dtype=np.float32)
        vals = []
        for lm in pose_res.pose_landmarks.landmark:
            vals.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.asarray(vals, dtype=np.float32)

    @staticmethod
    def _hands_vector(hands_res: Any) -> np.ndarray:
        n = 42 * 3
        if hands_res.multi_hand_landmarks is None:
            return np.zeros((n,), dtype=np.float32)
        hands = hands_res.multi_hand_landmarks[:2]
        vals = []
        for hand in hands:
            for lm in hand.landmark:
                vals.extend([lm.x, lm.y, lm.z])
        while len(vals) < n:
            vals.append(0.0)
        return np.asarray(vals[:n], dtype=np.float32)

    @staticmethod
    def _heuristic_label(motion_energy: float, pose_rate: float, hand_rate: float, completeness: float) -> tuple[str, float]:
        if completeness < 0.05:
            return "no_detection", 0.2
        if hand_rate > 0.45 and motion_energy > 0.12:
            return "active_signing", min(0.99, 0.55 + hand_rate * 0.25 + min(motion_energy, 1.0) * 0.2)
        if hand_rate > 0.25:
            return "hand_activity", min(0.9, 0.45 + hand_rate * 0.35)
        if pose_rate > 0.35:
            return "body_present_low_motion", min(0.85, 0.4 + pose_rate * 0.3)
        return "uncertain", 0.3
