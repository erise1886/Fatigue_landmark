import sys
sys.path.append("/usr/lib/python3/dist-packages")

import time
from collections import defaultdict

import cv2
import mediapipe as mp
from picamera2 import Picamera2
from ultralytics import YOLO

from config import EYE_CONFIG, YAWN_CONFIG, POSTURE_CONFIG, FUSION_CONFIG
from eye_model import EyeFatigueModel
from yawn_model import YawnModel
from posture_model import PostureModel
from fusion_model import FatigueFusionModel
from landmark_utils import calc_both_ears, calc_mar, calc_posture_features


MODEL_PATH = "/home/erise/fatigue_project/models/yolov8n.pt"
FRAME_SIZE = (640, 480)
YOLO_IMGSZ = 320
CONF_THRES = 0.4
MAX_MISSING_FRAMES = 30


def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def init_person_state():
    return {
        "eye_model": EyeFatigueModel(**EYE_CONFIG),
        "yawn_model": YawnModel(**YAWN_CONFIG),
        "posture_model": PostureModel(**POSTURE_CONFIG),
        "fusion_model": FatigueFusionModel(**FUSION_CONFIG),
        "missing_frames": 0,
        "fatigue_score": 0.0,
        "fatigue_state": "normal",
        "eye_score": 0.0,
        "eye_state": "unknown",
        "yawn_score": 0.0,
        "yawn_state": "unknown",
        "posture_score": 0.0,
        "posture_state": "unknown",
    }


def main():
    model = YOLO(MODEL_PATH)

    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": FRAME_SIZE, "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()

    person_data = defaultdict(init_person_state)
    prev_time = time.time()

    try:
        while True:
            frame = picam2.capture_array()
            frame_h, frame_w = frame.shape[:2]
            annotated = frame.copy()
            timestamp_ms = int(time.time() * 1000)

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],          # person only
                conf=CONF_THRES,
                imgsz=YOLO_IMGSZ,
                verbose=False,
            )

            active_ids = set()

            for r in results:
                if r.boxes is None:
                    continue

                for box in r.boxes:
                    if box.id is None:
                        continue

                    track_id = int(box.id[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    bbox = clamp_bbox(x1, y1, x2, y2, frame_w, frame_h)
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = bbox

                    active_ids.add(track_id)
                    pdata = person_data[track_id]
                    pdata["missing_frames"] = 0

                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue

                    person_h, person_w = person_crop.shape[:2]
                    person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

                    face_ok = False
                    pose_ok = False

                    # 기본값 유지
                    eye_result = {
                        "eye_score": pdata["eye_score"],
                        "state": pdata["eye_state"],
                    }
                    yawn_result = {
                        "yawn_score": pdata["yawn_score"],
                        "state": pdata["yawn_state"],
                    }
                    posture_result = {
                        "posture_score": pdata["posture_score"],
                        "state": pdata["posture_state"],
                    }

                    # FaceMesh on cropped person
                    face_results = face_mesh.process(person_rgb)
                    if face_results.multi_face_landmarks:
                        try:
                            face_landmarks = face_results.multi_face_landmarks[0].landmark

                            _, _, avg_ear = calc_both_ears(
                                face_landmarks, person_w, person_h
                            )
                            mar = calc_mar(face_landmarks, person_w, person_h)

                            eye_result = pdata["eye_model"].update(avg_ear, timestamp_ms)
                            yawn_result = pdata["yawn_model"].update(mar, timestamp_ms)

                            pdata["eye_score"] = float(eye_result["eye_score"])
                            pdata["eye_state"] = str(eye_result["state"])

                            pdata["yawn_score"] = float(yawn_result["yawn_score"])
                            pdata["yawn_state"] = str(yawn_result["state"])

                            face_ok = True

                        except Exception as e:
                            print(f"[FACE ERROR][ID {track_id}] {e}")
                            cv2.putText(
                                annotated,
                                f"Face err {track_id}",
                                (x1, max(20, y1 - 35)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1,
                            )

                    # Pose on cropped person
                    pose_results = pose.process(person_rgb)
                    if pose_results.pose_landmarks:
                        try:
                            pose_landmarks = pose_results.pose_landmarks.landmark
                            posture_features = calc_posture_features(
                                pose_landmarks, person_w, person_h
                            )

                            posture_result = pdata["posture_model"].update(posture_features)

                            pdata["posture_score"] = float(posture_result["posture_score"])
                            pdata["posture_state"] = str(posture_result["state"])

                            pose_ok = True

                        except Exception as e:
                            print(f"[POSE ERROR][ID {track_id}] {e}")
                            cv2.putText(
                                annotated,
                                f"Pose err {track_id}",
                                (x1, max(20, y1 - 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1,
                            )

                    fusion_result = pdata["fusion_model"].update(
                        pdata["eye_score"],
                        pdata["yawn_score"],
                        pdata["posture_score"],
                    )

                    pdata["fatigue_score"] = float(fusion_result["fatigue_score"])
                    pdata["fatigue_state"] = str(fusion_result["state"])

                    color = (0, 255, 0)
                    if pdata["fatigue_score"] >= 0.7:
                        color = (0, 0, 255)
                    elif pdata["fatigue_score"] >= 0.4:
                        color = (0, 255, 255)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"ID {track_id} {conf:.2f}",
                        (x1, max(20, y1 - 45)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )
                    cv2.putText(
                        annotated,
                        f"fatigue: {pdata['fatigue_score']:.2f} | {pdata['fatigue_state']}",
                        (x1, max(20, y1 - 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
                    cv2.putText(
                        annotated,
                        f"E:{pdata['eye_score']:.2f} Y:{pdata['yawn_score']:.2f} P:{pdata['posture_score']:.2f}",
                        (x1, min(frame_h - 10, y2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                    )

                    status = []
                    if face_ok:
                        status.append("face")
                    if pose_ok:
                        status.append("pose")

                    if status:
                        cv2.putText(
                            annotated,
                            " / ".join(status),
                            (x1, min(frame_h - 30, y2 + 38)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1,
                        )

            remove_ids = []
            for tid, pdata in person_data.items():
                if tid not in active_ids:
                    pdata["missing_frames"] += 1
                if pdata["missing_frames"] > MAX_MISSING_FRAMES:
                    remove_ids.append(tid)

            for tid in remove_ids:
                del person_data[tid]

            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            cv2.putText(
                annotated,
                f"FPS: {fps:.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Fatigue Monitoring", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        picam2.stop()
        face_mesh.close()
        pose.close()


if __name__ == "__main__":
    main()