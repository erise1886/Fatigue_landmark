
EYE_CONFIG = {
    "ear_threshold": 0.21,
    "closed_consec_threshold": 2,
    "drowsy_consec_threshold": 10,
    "perclos_window_sec": 5.0,
    "perclos_threshold": 0.4,
    "score_smoothing": 10,
}

YAWN_CONFIG = {
    "mar_threshold": 0.35,
    "yawn_consec_frames": 15,
    "event_window_sec": 10.0,
    "score_smoothing": 10,
}

POSTURE_CONFIG = {
    "lean_threshold": 0.12,
    "shoulder_slope_threshold": 0.08,
    "shoulder_drop_threshold": 0.10,
    "motion_low_threshold": 0.015,
    "head_drop_threshold": 0.08,
    "score_smoothing": 10,
}

FUSION_CONFIG = {
    "eye_weight": 0.4,
    "yawn_weight": 0.25,
    "posture_weight": 0.35,
    "fatigue_threshold": 0.6,
}
