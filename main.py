# === main.py ===
import cv2
import numpy as np
import pygame
import time
from roboflow import Roboflow
import mediapipe as mp
# === Roboflow Load ===
rf = Roboflow(api_key="5iuvSj8loZZq8zpxpdEJ")  # your actual key
project = rf.workspace("nsnrr-ysaip").project("meemomi")
model = project.version(3).model  # Use the trained version

# === Pygame Init ===
pygame.init()

# Load piano note sounds
note_sounds = {
    "C3": pygame.mixer.Sound("sounds/C3.wav"),
    "D3": pygame.mixer.Sound("sounds/D3.wav"),
    "E3": pygame.mixer.Sound("sounds/E3.wav"),
    "F3": pygame.mixer.Sound("sounds/F3.wav"),
    "G3": pygame.mixer.Sound("sounds/G3.wav"),
}
# === Mediapipe Hands Init ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
# === CAMERA ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera failed to open")
    exit()
print("✅ Camera opened")

