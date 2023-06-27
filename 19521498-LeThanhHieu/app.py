import numpy as np
import cv2
import streamlit as st
import mediapipe as mp

@st.cache_resource()
def load_mediapose():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    return mp_drawing, mp_drawing_styles, mp_pose

with st.spinner("Loading Model...."):
    mp_drawing, mp_drawing_styles, mp_pose = load_mediapose()

FRAME_WINDOW = st.image([])

effect_frame_plasma_ball = cv2.VideoCapture('effect-video/glove.mp4')
effect_frame_punch = cv2.VideoCapture('effect-video/fire-burning.mp4')
effect_frame_explosion = cv2.VideoCapture('effect-video/fireball.mp4')
effect_frame_storm= cv2.VideoCapture('effect-video/punch.mp4')

cap = cv2.VideoCapture(0)

effect_name = 'left'

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        _, image = cap.read()
        image = cv2.flip(image, 1)

        if effect_name == 'left':
            success1, effect_left_initial = effect_frame_plasma_ball.read()
            success2, effect_punch = effect_frame_storm.read()

            if success1 and success2:
                height_frame, width_frame, _ = image.shape
                results = pose.process(image)

                left_pinky = results.pose_landmarks.landmark[17]
                left_elbow = results.pose_landmarks.landmark[13]
                left_shoulder = results.pose_landmarks.landmark[11]

                if left_pinky.y >= 1 or left_elbow.y >= 1 and left_shoulder.y >= 1:
                    result = image
                else:
                    left_pinky = [int(left_pinky.x * width_frame), int(left_pinky.y * height_frame)]
                    left_elbow = [int(left_elbow.x * width_frame), int(left_elbow.y * height_frame)]
                    left_shoulder = [int(left_shoulder.x * width_frame), int(left_shoulder.y * height_frame)]

                    distance = np.linalg.norm(np.array(left_pinky) - np.array(left_elbow))

                    if left_elbow[1] - left_pinky[1] > 50 and left_elbow[1] - left_shoulder[1] > 50:
                        new_height = int(0.5 * distance)
                        new_width = int(new_height * 1280 / 720)
                        effect_left = cv2.resize(effect_left_initial, (new_width, int(0.7 * distance)), interpolation=cv2.INTER_AREA)

                        effect_left = effect_left[5:-5, 5:-5]
                        top = bot = 1000 - effect_left.shape[0] // 2
                        left = right = 1000 - effect_left.shape[1] // 2

                        background = cv2.copyMakeBorder(src=effect_left, top=top, bottom=bot, left=left, right=right, borderType=cv2.BORDER_CONSTANT)

                        background = background[1000 - left_pinky[1]:1000 - left_pinky[1] + 480, 1000 - left_pinky[0]:1000 - left_pinky[0] + 640, :]
                    else:
                        background = cv2.resize(effect_punch, (640, 480), interpolation=cv2.INTER_AREA)

                    result = cv2.addWeighted(image, 0.3, background, 0.7, 0)
            else:
                effect_name = 'flame'
                result = image

        elif effect_name == 'flame':
            success1, effect_punch = effect_frame_punch.read()
            success2, effect_right_initial = effect_frame_explosion.read()

            if success1 and success2:
                height_frame, width_frame, _ = image.shape
                results = pose.process(image)

                right_pinky = results.pose_landmarks.landmark[18]
                right_elbow = results.pose_landmarks.landmark[14]
                right_shoulder = results.pose_landmarks.landmark[12]

                if right_pinky.y >= 1 or right_elbow.y >= 1 and right_shoulder.y >= 1:
                    result = image
                else:
                    right_pinky = [int(right_pinky.x * width_frame), int(right_pinky.y * height_frame)]
                    right_elbow = [int(right_elbow.x * width_frame), int(right_elbow.y * height_frame)]
                    right_shoulder = [int(right_shoulder.x * width_frame), int(right_shoulder.y * height_frame)]

                    distance = np.linalg.norm(np.array(right_pinky) - np.array(right_elbow))

                    if right_elbow[1] - right_pinky[1] > 50 and right_elbow[1] - right_shoulder[1] > 50:
                        new_height = int(0.5 * distance)
                        new_width = int(new_height * 1280 / 720)
                        effect_right = cv2.resize(effect_right_initial, (new_width, int(0.7 * distance)), interpolation=cv2.INTER_AREA)

                        effect_right = effect_right[5:-5, 5:-5]
                        top = bot = 1000 - effect_right.shape[0] // 2
                        left = right = 1000 - effect_right.shape[1] // 2

                        background = cv2.copyMakeBorder(src=effect_right, top=top, bottom=bot, left=left, right=right, borderType=cv2.BORDER_CONSTANT)

                        background = background[1000 - right_pinky[1]:1000 - right_pinky[1] + 480, 1000 - right_pinky[0]:1000 - right_pinky[0] + 640, :]
                    else:
                        background = cv2.resize(effect_punch, (640, 480), interpolation=cv2.INTER_AREA)

                    result = cv2.addWeighted(image, 0.3, background, 0.7, 0)
            else:
                effect_name = 'flame'
                result = image
        else:
            result = image
            st.write('End code')
            break

        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(result)
