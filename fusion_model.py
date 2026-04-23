
import numpy as np


class FatigueFusionModel:
    def __init__(
        self,
        eye_weight=0.4,
        yawn_weight=0.25,
        posture_weight=0.35,
        fatigue_threshold=0.6,
    ):
        total = eye_weight + yawn_weight + posture_weight
        self.eye_weight = eye_weight / total
        self.yawn_weight = yawn_weight / total
        self.posture_weight = posture_weight / total
        self.fatigue_threshold = fatigue_threshold

    def update(self, eye_score, yawn_score, posture_score):
        fatigue_score = (
            self.eye_weight * eye_score +
            self.yawn_weight * yawn_score +
            self.posture_weight * posture_score
        )

        fatigue_score = float(np.clip(fatigue_score, 0.0, 1.0))

        if fatigue_score >= self.fatigue_threshold:
            state = "fatigued"
        elif fatigue_score >= 0.4:
            state = "warning"
        else:
            state = "normal"

        return {
            "fatigue_score": fatigue_score,
            "state": state,
        }
