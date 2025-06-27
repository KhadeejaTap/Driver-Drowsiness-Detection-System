import cv2
import numpy as np
import mediapipe as mp
from playsound import playsound
import threading
import math

# Settings
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
HEAD_PITCH_THRESHOLD = 15

# Alarm
alarm_on = False
def play_alarm():
    playsound("alarm.wav")

# Eye EAR Function
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def compute_ear(eye_points):
    A = euclidean(eye_points[1], eye_points[5])
    B = euclidean(eye_points[2], eye_points[4])
    C = euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Landmark IDs 
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
HEAD_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 263,
    "right_eye_corner": 33,
    "left_mouth": 61,
    "right_mouth": 291
}

# MediaPipe 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Webcam 
cap = cv2.VideoCapture(0)
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    trigger_alarm = False

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            # EAR for Drowsiness Detection
            left = [(int(face.landmark[i].x * w), int(face.landmark[i].y * h)) for i in LEFT_EYE]
            right = [(int(face.landmark[i].x * w), int(face.landmark[i].y * h)) for i in RIGHT_EYE]
            left_ear = compute_ear(left)
            right_ear = compute_ear(right)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (30, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                    trigger_alarm = True
            else:
                frame_counter = 0

            # Head Pose Detection
            image_points = np.array([
                (face.landmark[HEAD_LANDMARKS["nose_tip"]].x * w,
                 face.landmark[HEAD_LANDMARKS["nose_tip"]].y * h),
                (face.landmark[HEAD_LANDMARKS["chin"]].x * w,
                 face.landmark[HEAD_LANDMARKS["chin"]].y * h),
                (face.landmark[HEAD_LANDMARKS["left_eye_corner"]].x * w,
                 face.landmark[HEAD_LANDMARKS["left_eye_corner"]].y * h),
                (face.landmark[HEAD_LANDMARKS["right_eye_corner"]].x * w,
                 face.landmark[HEAD_LANDMARKS["right_eye_corner"]].y * h),
                (face.landmark[HEAD_LANDMARKS["left_mouth"]].x * w,
                 face.landmark[HEAD_LANDMARKS["left_mouth"]].y * h),
                (face.landmark[HEAD_LANDMARKS["right_mouth"]].x * w,
                 face.landmark[HEAD_LANDMARKS["right_mouth"]].y * h)
            ], dtype="double")

            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1)
            ])

            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))

            success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

            if success:
                rot_mat, _ = cv2.Rodrigues(rvec)
                pose_mat = cv2.hconcat((rot_mat, tvec))
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
                pitch, yaw, roll = euler_angles.flatten()

                cv2.putText(frame, f"Pitch: {pitch:.1f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Yaw: {yaw:.1f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Roll: {roll:.1f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                if pitch > HEAD_PITCH_THRESHOLD:
                    cv2.putText(frame, "HEAD NOD ALERT!", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                    trigger_alarm = True

    # Alarm Trigger Logic 
    if trigger_alarm:
        if not alarm_on:
            alarm_on = True
            threading.Thread(target=play_alarm, daemon=True).start()
    else:
        alarm_on = False

    cv2.imshow("Drowsiness + Head Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
