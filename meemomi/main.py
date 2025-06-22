import cv2
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
import time
import pygame

# === CONFIG ===
pygame.mixer.init()

# Sound map for keys
sound_map = {
    'C3': pygame.mixer.Sound("sounds/C3.wav"),
    'D3': pygame.mixer.Sound("sounds/D3.wav"),
    'E3': pygame.mixer.Sound("sounds/E3.wav"),
    'F3': pygame.mixer.Sound("sounds/F3.wav"),
    'G3': pygame.mixer.Sound("sounds/G3.wav"),
}

# === Load YOLOv8 Model ===
model = YOLO("best.pt")

# === Finger Config ===
fingertip_indices = {
    'Thumb': 4,
    'Index': 8,
    'Middle': 12,
    'Ring': 16,
    'Pinky': 20,
}

finger_histories = {name: deque(maxlen=5) for name in fingertip_indices}
last_press_time = {name: 0 for name in fingertip_indices}
press_threshold = 18
press_cooldown = 0.6

# === Init MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Webcam ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # === Run YOLOv8 Detection ===
    yolo_results = model.predict(source=frame, conf=0.5, verbose=False)[0]
    detections = []
    for box in yolo_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append({'box': (x1, y1, x2, y2), 'class': cls, 'conf': conf})
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

    # === Hand Tracking ===
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for finger_name, idx in fingertip_indices.items():
                x = int(hand_landmarks.landmark[idx].x * w)
                y = int(hand_landmarks.landmark[idx].y * h)
                finger_histories[finger_name].append(y)

                if len(finger_histories[finger_name]) == 5:
                    y_vals = list(finger_histories[finger_name])
                    avg_old = sum(y_vals[:3]) / 3
                    avg_new = sum(y_vals[-2:]) / 2
                    dy = avg_new - avg_old

                    if dy > press_threshold and time.time() - last_press_time[finger_name] > press_cooldown:
                        last_press_time[finger_name] = time.time()
                        print(f"{finger_name} press!")

                        for det in detections:
                            x1, y1, x2, y2 = det['box']
                            if x1 <= x <= x2 and y1 <= y <= y2:
                                key_name = model.names[det['class']]
                                print(f"{finger_name} touched {key_name}!")

                                if key_name in sound_map:
                                    sound_map[key_name].play()

                                cv2.putText(frame, f"{finger_name} -> {key_name}", (10, 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

    cv2.imshow("YOLO + Finger Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
