import numpy as np
import cv2
import streamlit as st
import mediapipe as mp

@st.cache(allow_output_mutation=True)
def load_mediapose():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    return mp_drawing, mp_pose

mp_drawing, mp_pose = load_mediapose()

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        _, image = cap.read()
        image = cv2.flip(image, 1)
        results = pose.process(image)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        FRAME_WINDOW.image(image_bgr)
