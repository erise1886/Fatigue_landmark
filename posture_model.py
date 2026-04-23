
from collections import deque
import numpy as np


class PostureModel:
    def __init__(
        self,
        lean_threshold=0.12,
        shoulder_slope_threshold=0.08,
        shoulder_drop_threshold=0.10,
        motion_low_threshold=0.015,
        head_drop_threshold=0.08,
        score_smoothing=10,
    ):
        self.lean_threshold = lean_threshold
        self.shoulder_slope_threshold = shoulder_slope_threshold
        self.shoulder_drop_threshold = shoulder_drop_threshold
        self.motion_low_threshold = motion_low_threshold
        self.head_drop_threshold = head_drop_threshold

        self.prev_nose = None
        self.motion_buffer = deque(maxlen=30)
        self.score_buffer = deque(maxlen=score_smoothing)

    def smooth_score(self, score):
        self.score_buffer.append(score)
        return float(np.mean(self.score_buffer))

    def _calc_motion(self, nose_xy):
        if self.prev_nose is None:
            self.prev_nose = nose_xy
            self.motion_buffer.append(0.0)
            return 0.0

        dx = nose_xy[0] - self.prev_nose[0]
        dy = nose_xy[1] - self.prev_nose[1]
        motion = float((dx ** 2 + dy ** 2) ** 0.5)

        self.prev_nose = nose_xy
        self.motion_buffer.append(motion)
        return motion

    def _avg_motion(self):
        if len(self.motion_buffer) == 0:
            return 0.0
        return float(np.mean(self.motion_buffer))

    def _normalize_ratio(self, value, threshold, max_scale=2.0):
        if threshold <= 1e-6:
            return 0.0
        score = value / (threshold * max_scale)
        return float(np.clip(score, 0.0, 1.0))

    def update(self, posture_features):
        if not posture_features["visibility_ok"]:
            return {
                "posture_score_raw": 0.0,
                "posture_score": 0.0,
                "state": "not_visible",
                "lean": 0.0,
                "shoulder_slope": 0.0,
                "shoulder_drop": 0.0,
                "head_drop": 0.0,
                "avg_motion": 0.0,
            }

        lean = posture_features["lean"]
        shoulder_slope = posture_features["shoulder_slope"]
        shoulder_drop = posture_features["shoulder_drop"]
        head_drop = posture_features["head_drop"]
        nose_xy = posture_features["nose"]

        _ = self._calc_motion(nose_xy)
        avg_motion = self._avg_motion()

        lean_score = self._normalize_ratio(lean, self.lean_threshold)
        slope_score = self._normalize_ratio(shoulder_slope, self.shoulder_slope_threshold)
        drop_score = self._normalize_ratio(shoulder_drop, self.shoulder_drop_threshold)
        head_score = self._normalize_ratio(head_drop, self.head_drop_threshold)

        if avg_motion <= self.motion_low_threshold:
            low_motion_score = 1.0 - (avg_motion / max(self.motion_low_threshold, 1e-6))
        else:
            low_motion_score = 0.0

        low_motion_score = float(np.clip(low_motion_score, 0.0, 1.0))

        posture_score = (
            0.30 * lean_score +
            0.20 * slope_score +
            0.20 * drop_score +
            0.20 * head_score +
            0.10 * low_motion_score
        )

        posture_score = float(np.clip(posture_score, 0.0, 1.0))
        smoothed_posture_score = self.smooth_score(posture_score)

        if smoothed_posture_score >= 0.7:
            state = "warning"
        elif smoothed_posture_score >= 0.4:
            state = "caution"
        else:
            state = "normal"

        return {
            "posture_score_raw": float(posture_score),
            "posture_score": float(smoothed_posture_score),
            "state": state,
            "lean": float(lean),
            "shoulder_slope": float(shoulder_slope),
            "shoulder_drop": float(shoulder_drop),
            "head_drop": float(head_drop),
            "avg_motion": float(avg_motion),
            "lean_score": float(lean_score),
            "slope_score": float(slope_score),
            "drop_score": float(drop_score),
            "head_score": float(head_score),
            "low_motion_score": float(low_motion_score),
        }
