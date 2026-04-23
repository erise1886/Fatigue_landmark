
from collections import deque
import numpy as np


class YawnModel:
    def __init__(
        self,
        mar_threshold=0.35,
        yawn_consec_frames=15,
        event_window_sec=10.0,
        score_smoothing=10,
    ):
        self.mar_threshold = mar_threshold
        self.yawn_consec_frames = yawn_consec_frames
        self.event_window_sec = event_window_sec

        self.open_run = 0
        self.yawn_events = deque()
        self.score_buffer = deque(maxlen=score_smoothing)
        self.prev_yawning = False

    def smooth_score(self, score):
        self.score_buffer.append(score)
        return float(np.mean(self.score_buffer))

    def _remove_old_events(self, timestamp_ms):
        min_time = timestamp_ms - int(self.event_window_sec * 1000)
        while self.yawn_events and self.yawn_events[0] < min_time:
            self.yawn_events.popleft()

    def update(self, mar, timestamp_ms):
        mouth_open = mar >= self.mar_threshold

        if mouth_open:
            self.open_run += 1
        else:
            self.open_run = 0

        is_yawning = self.open_run >= self.yawn_consec_frames

        if is_yawning and not self.prev_yawning:
            self.yawn_events.append(timestamp_ms)

        self.prev_yawning = is_yawning
        self._remove_old_events(timestamp_ms)

        event_count = len(self.yawn_events)

        yawn_score = 0.0

        if mouth_open:
            yawn_score += 0.4
        if self.open_run >= max(1, self.yawn_consec_frames // 2):
            yawn_score += 0.3
        if is_yawning:
            yawn_score += 0.2
        if event_count >= 1:
            yawn_score += 0.1

        yawn_score = min(1.0, yawn_score)
        smoothed_yawn_score = self.smooth_score(yawn_score)

        if is_yawning:
            state = "yawning"
        elif mouth_open:
            state = "mouth_open"
        else:
            state = "normal"

        return {
            "yawn_score_raw": float(yawn_score),
            "yawn_score": float(smoothed_yawn_score),
            "state": state,
            "mar": float(mar),
            "mouth_open": bool(mouth_open),
            "open_run": int(self.open_run),
            "recent_yawn_events": int(event_count),
            "is_yawning": bool(is_yawning),
        }
