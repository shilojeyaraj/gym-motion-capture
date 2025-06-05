import cv2
import os
import numpy as np
import mediapipe as mp

# Define label keys and output directory
LABELS = {
    'p': 'pushup',
    's': 'squat',
    'j': 'jumping_jack',
}
output_dir = 'data'
for label in LABELS.values():
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Angle calculation utility
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Start webcam
cap = cv2.VideoCapture(0)
frame_num = 0

print("Press a key to label the frame and save it:")
for key, label in LABELS.items():
    print(f"  '{key}' for {label}")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    angles = {}

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        def get_coord(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        try:
            # Right Elbow
            a = get_coord(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            b = get_coord(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            c = get_coord(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            angles['Right Elbow'] = (b, calculate_angle(a, b, c))

            # Right Knee
            a = get_coord(mp_pose.PoseLandmark.RIGHT_HIP.value)
            b = get_coord(mp_pose.PoseLandmark.RIGHT_KNEE.value)
            c = get_coord(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
            angles['Right Knee'] = (b, calculate_angle(a, b, c))

            # Right Wrist
            a = get_coord(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            b = get_coord(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            c = get_coord(mp_pose.PoseLandmark.RIGHT_INDEX.value)
            angles['Right Wrist'] = (b, calculate_angle(a, b, c))
        except:
            pass

        # Draw skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Draw angle text and joint dots
        for name, (pt, ang) in angles.items():
            x, y = int(pt[0]), int(pt[1])
            cv2.putText(frame, f"{name}: {int(ang)}°", (x - 50, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)

        # Draw wrist vectors
        if 'Right Wrist' in angles:
            wrist = get_coord(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            elbow = get_coord(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            index = get_coord(mp_pose.PoseLandmark.RIGHT_INDEX.value)

            cv2.line(frame, tuple(elbow.astype(int)), tuple(wrist.astype(int)), (0, 255, 255), 2)
            cv2.line(frame, tuple(index.astype(int)), tuple(wrist.astype(int)), (0, 255, 255), 2)
            cv2.circle(frame, tuple(elbow.astype(int)), 5, (255, 0, 0), -1)
            cv2.circle(frame, tuple(index.astype(int)), 5, (0, 0, 255), -1)

    # Display keypress instructions
    instructions = " | ".join([f"'{k}'={v}" for k, v in LABELS.items()])
    cv2.putText(frame, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)

    cv2.imshow("Webcam - Joint Angles & Labeling", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    for k, label in LABELS.items():
        if key == ord(k):
            filename = os.path.join(output_dir, label, f"{label}_{frame_num:05d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            frame_num += 1

cap.release()
cv2.destroyAllWindows()
