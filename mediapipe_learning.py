import cv2
import numpy as np
import math

# 导入mediapipe：https://google.github.io/mediapipe/solutions/hands
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

Pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5)
# 读取视频流
img = pyautogui.screenshot()
cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width,height)


while True:

    ret,frame = cap.read()
    # 镜像
    frame = cv2.flip(frame,1)

    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = Pose.process(frame)

    # Draw the pose annotation on the image.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        x_list = []
        y_list = []
        # print(results.pose_landmarks)
        for landmark in results.pose_landmarks.landmark:
            x_list.append(landmark.x)
            y_list.append(landmark.y)
        index_head_x = x_list[0] * width
        index_head_y = y_list[0] * height
        print(index_head_x, index_head_y)
        # pyautogui.moveTo(index_head_x, index_head_y, duration=1)
    cv2.imshow('drag',frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyWindow()
