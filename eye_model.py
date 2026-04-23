
from collections import deque
import numpy as np


class EyeFatigueModel:
    def __init__(
        self,
        ear_threshold=0.21,
        closed_consec_threshold=2,
        drowsy_consec_threshold=10,
        perclos_window_sec=5.0,
        perclos_threshold=0.4,
        score_smoothing=10,
    ):
        self.ear_threshold = ear_threshold
        self.closed_consec_threshold = closed_consec_threshold
        self.drowsy_consec_threshold = drowsy_consec_threshold
        self.perclos_window_sec = perclos_window_sec
        self.perclos_threshold = perclos_threshold

        self.closed_run = 0
        self.eye_state_buffer = deque()
        self.score_buffer = deque(maxlen=score_smoothing)

    def smooth_score(self, score):
        self.score_buffer.append(score)
        return float(np.mean(self.score_buffer))

    def _update_perclos_buffer(self, timestamp_ms, eye_closed):
        self.eye_state_buffer.append((timestamp_ms, int(eye_closed)))

        min_time = timestamp_ms - int(self.perclos_window_sec * 1000)
        while self.eye_state_buffer and self.eye_state_buffer[0][0] < min_time:
            self.eye_state_buffer.popleft()

    def _calc_perclos(self):
        if len(self.eye_state_buffer) == 0:
            return 0.0
        closed_ratio = np.mean([state for _, state in self.eye_state_buffer])
        return float(closed_ratio)

    def update(self, avg_ear, timestamp_ms):
        eye_closed = avg_ear < self.ear_threshold

        if eye_closed:
            self.closed_run += 1
        else:
            self.closed_run = 0

        self._update_perclos_buffer(timestamp_ms, eye_closed)
        perclos = self._calc_perclos()

        eye_score = 0.0

        if eye_closed:
            eye_score += 0.2
        if self.closed_run >= self.closed_consec_threshold:
            eye_score += 0.2
        if self.closed_run >= self.drowsy_consec_threshold:
            eye_score += 0.3
        if perclos >= self.perclos_threshold:
            eye_score += 0.3

        eye_score = min(1.0, eye_score)
        smoothed_eye_score = self.smooth_score(eye_score)

        if self.closed_run >= self.drowsy_consec_threshold:
            state = "drowsy"
        elif eye_closed:
            state = "closed"
        else:
            state = "open"

        return {
            "eye_score_raw": float(eye_score),
            "eye_score": float(smoothed_eye_score),
            "state": state,
            "avg_ear": float(avg_ear),
            "perclos": float(perclos),
            "closed_run": int(self.closed_run),
            "eye_closed": bool(eye_closed),
        }
