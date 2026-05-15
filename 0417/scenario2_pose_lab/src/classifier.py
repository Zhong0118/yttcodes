from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional


class ShortPoseUnderstandingModel:
    """A lightweight short-sequence pose understanding model for scenario 2.

    It is intentionally simple and interpretable. You can later replace it with
    a TCN/GRU or a stronger model without changing the rest of the pipeline.
    """

    @staticmethod
    def _mean_xy(pose: List[List[float]]) -> Optional[np.ndarray]:
        if not pose:
            return None
        arr = np.asarray(pose, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        return arr[:, :2]

    @staticmethod
    def _hand_spread(hands: List[List[List[float]]]) -> float:
        if not hands:
            return 0.0
        spreads = []
        for hand in hands:
            arr = np.asarray(hand, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            xy = arr[:, :2]
            center = np.mean(xy, axis=0)
            spreads.append(float(np.mean(np.linalg.norm(xy - center, axis=1))))
        return float(np.mean(spreads)) if spreads else 0.0

    def infer(self, window_seq: List[Dict[str, Any]]) -> Dict[str, float | str]:
        pose_detect_rate = float(np.mean([1.0 if x["pose_detected"] else 0.0 for x in window_seq]))
        hand_detect_rate = float(np.mean([1.0 if x["hands_detected"] else 0.0 for x in window_seq]))
        face_detect_rate = float(np.mean([1.0 if x.get("face_detected", False) else 0.0 for x in window_seq]))
        completeness = float(np.mean([x.get("keypoint_completeness", 0.0) for x in window_seq]))

        motion_energy_vals = []
        jerk_vals = []
        spreads = []
        prev_xy = None
        prev_vel = None

        for item in window_seq:
            xy = self._mean_xy(item.get("pose", []))
            spreads.append(self._hand_spread(item.get("hands", [])))
            if xy is None:
                continue
            if prev_xy is not None and prev_xy.shape == xy.shape:
                vel = xy - prev_xy
                motion_energy_vals.append(float(np.mean(np.abs(vel))))
                if prev_vel is not None and prev_vel.shape == vel.shape:
                    jerk_vals.append(float(np.mean(np.abs(vel - prev_vel))))
                prev_vel = vel
            prev_xy = xy

        motion_energy = float(np.mean(motion_energy_vals)) if motion_energy_vals else 0.0
        jitter = float(np.mean(jerk_vals)) if jerk_vals else 0.0
        hand_spread = float(np.mean(spreads)) if spreads else 0.0
        stability_score = float(max(0.0, 1.0 - min(1.0, jitter * 50.0)))

        # Interpretative scene-2 labels
        if pose_detect_rate < 0.35:
            label = "pose_missing"
        elif hand_detect_rate > 0.7 and motion_energy > 0.02:
            label = "active_gesture"
        elif motion_energy > 0.01:
            label = "pose_transition"
        elif face_detect_rate > 0.5 and hand_spread < 0.015:
            label = "attentive_idle"
        else:
            label = "stable_pose"

        confidence = float(min(1.0, 0.35 + motion_energy * 25.0 + hand_detect_rate * 0.3 + completeness * 0.2))

        return {
            "label": label,
            "confidence": confidence,
            "pose_detect_rate": pose_detect_rate,
            "hand_detect_rate": hand_detect_rate,
            "face_detect_rate": face_detect_rate,
            "keypoint_completeness": completeness,
            "motion_energy": motion_energy,
            "stability_score": stability_score,
            "hand_spread": hand_spread,
        }
