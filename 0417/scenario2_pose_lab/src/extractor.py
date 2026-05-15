from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Any


class PoseStickExtractor:
    """Extracts pose, hands and coarse face landmarks for scenario 2."""

    def __init__(self):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Current mediapipe version has no mp.solutions. Please install mediapipe==0.10.15"
            )

        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Select a small set of face landmarks: lips + eye corners + brow center like points
        self.face_keep = [13, 14, 78, 308, 61, 291, 33, 263, 1, 159, 386]

    @staticmethod
    def _lm_pose_to_list(landmarks) -> List[List[float]]:
        out = []
        for lm in landmarks.landmark:
            out.append([float(lm.x), float(lm.y), float(lm.z), float(lm.visibility)])
        return out

    @staticmethod
    def _lm_generic_to_list(landmarks) -> List[List[float]]:
        out = []
        for lm in landmarks.landmark:
            out.append([float(lm.x), float(lm.y), float(lm.z)])
        return out

    def extract(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        pose_res = self.pose.process(rgb)
        hand_res = self.hands.process(rgb)
        face_res = self.face.process(rgb)

        pose_kps: List[List[float]] = []
        hands_kps: List[List[List[float]]] = []
        face_kps: List[List[float]] = []

        if pose_res.pose_landmarks:
            pose_kps = self._lm_pose_to_list(pose_res.pose_landmarks)

        if hand_res.multi_hand_landmarks:
            for hand_lms in hand_res.multi_hand_landmarks:
                hands_kps.append(self._lm_generic_to_list(hand_lms))

        if face_res.multi_face_landmarks:
            lms = face_res.multi_face_landmarks[0].landmark
            for idx in self.face_keep:
                lm = lms[idx]
                face_kps.append([float(lm.x), float(lm.y), float(lm.z)])

        total_expected = 33 + 42 + len(self.face_keep)
        total_found = len(pose_kps) + sum(len(h) for h in hands_kps) + len(face_kps)
        completeness = float(total_found / total_expected) if total_expected > 0 else 0.0

        return {
            "pose": pose_kps,
            "hands": hands_kps,
            "face": face_kps,
            "pose_detected": len(pose_kps) > 0,
            "hands_detected": len(hands_kps) > 0,
            "face_detected": len(face_kps) > 0,
            "keypoint_completeness": completeness,
        }


def resize_keep_aspect(frame: np.ndarray, target_short: int) -> np.ndarray:
    h, w = frame.shape[:2]
    short_side = min(h, w)
    if short_side <= 0:
        return frame
    scale = target_short / float(short_side)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
