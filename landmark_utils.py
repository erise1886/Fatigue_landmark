
import numpy as np


def euclidean(p1, p2):
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    return np.linalg.norm(p1 - p2)


LEFT_EYE_IDX = {
    "p1": 33,
    "p2": 160,
    "p3": 158,
    "p4": 133,
    "p5": 153,
    "p6": 144,
}

RIGHT_EYE_IDX = {
    "p1": 362,
    "p2": 385,
    "p3": 387,
    "p4": 263,
    "p5": 373,
    "p6": 380,
}

MOUTH_IDX = {
    "left": 61,
    "right": 291,
    "top": 13,
    "bottom": 14,
}


def get_face_point(face_landmarks, idx, image_w, image_h):
    lm = face_landmarks[idx]
    return (lm.x * image_w, lm.y * image_h)


def get_pose_point(pose_landmarks, idx, image_w, image_h):
    lm = pose_landmarks[idx]
    return (lm.x * image_w, lm.y * image_h, lm.visibility)


def calc_ear(face_landmarks, eye_indices, image_w, image_h):
    p1 = get_face_point(face_landmarks, eye_indices["p1"], image_w, image_h)
    p2 = get_face_point(face_landmarks, eye_indices["p2"], image_w, image_h)
    p3 = get_face_point(face_landmarks, eye_indices["p3"], image_w, image_h)
    p4 = get_face_point(face_landmarks, eye_indices["p4"], image_w, image_h)
    p5 = get_face_point(face_landmarks, eye_indices["p5"], image_w, image_h)
    p6 = get_face_point(face_landmarks, eye_indices["p6"], image_w, image_h)

    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(ear)


def calc_both_ears(face_landmarks, image_w, image_h):
    left_ear = calc_ear(face_landmarks, LEFT_EYE_IDX, image_w, image_h)
    right_ear = calc_ear(face_landmarks, RIGHT_EYE_IDX, image_w, image_h)
    avg_ear = (left_ear + right_ear) / 2.0
    return float(left_ear), float(right_ear), float(avg_ear)


def calc_mar(face_landmarks, image_w, image_h):
    left = get_face_point(face_landmarks, MOUTH_IDX["left"], image_w, image_h)
    right = get_face_point(face_landmarks, MOUTH_IDX["right"], image_w, image_h)
    top = get_face_point(face_landmarks, MOUTH_IDX["top"], image_w, image_h)
    bottom = get_face_point(face_landmarks, MOUTH_IDX["bottom"], image_w, image_h)

    mouth_width = euclidean(left, right)
    mouth_height = euclidean(top, bottom)

    if mouth_width < 1e-6:
        return 0.0

    mar = mouth_height / mouth_width
    return float(mar)


def calc_posture_features(pose_landmarks, image_w, image_h):
    nose_x, nose_y, nose_vis = get_pose_point(pose_landmarks, 0, image_w, image_h)
    l_sh_x, l_sh_y, l_vis = get_pose_point(pose_landmarks, 11, image_w, image_h)
    r_sh_x, r_sh_y, r_vis = get_pose_point(pose_landmarks, 12, image_w, image_h)

    shoulder_center_x = (l_sh_x + r_sh_x) / 2.0
    shoulder_center_y = (l_sh_y + r_sh_y) / 2.0
    shoulder_width = max(abs(r_sh_x - l_sh_x), 1e-6)

    lean = abs(nose_x - shoulder_center_x) / shoulder_width
    shoulder_slope = abs(l_sh_y - r_sh_y) / shoulder_width
    shoulder_drop = abs(l_sh_y - r_sh_y) / shoulder_width
    head_drop = max(0.0, (nose_y - shoulder_center_y)) / max(image_h, 1e-6)

    return {
        "lean": float(lean),
        "shoulder_slope": float(shoulder_slope),
        "shoulder_drop": float(shoulder_drop),
        "head_drop": float(head_drop),
        "visibility_ok": bool(nose_vis > 0.5 and l_vis > 0.5 and r_vis > 0.5),
        "nose": (nose_x, nose_y),
        "left_shoulder": (l_sh_x, l_sh_y),
        "right_shoulder": (r_sh_x, r_sh_y),
        "shoulder_center": (shoulder_center_x, shoulder_center_y),
    }
