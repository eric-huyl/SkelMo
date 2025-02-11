import mediapipe as mp
import json
import os
import cv2
import numpy as np
from draw_result import draw_landmarks_mediapipe
from Pose import Pose

FRAME_INTERVAL = 30


def read_video_as_numpy(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return None

    # 获取原始帧率
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {original_fps}")

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    # 存储视频帧的列表
    frames = []

    frame_idx = 0
    while True:
        # 设置视频读取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # 读取一帧
        ret, frame = cap.read()

        if not ret:
            break

        # 将帧添加到数组中
        frames.append(frame)

        # 按指定的帧率调整读取帧的位置
        frame_idx += FRAME_INTERVAL

    # 转换为 NumPy 数组
    frames_np = np.array(frames)

    # 释放资源
    cap.release()

    return frames_np


def main():
    model_path = 'pose_landmarker_heavy.task'
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=3)

    with PoseLandmarker.create_from_options(options) as landmarker:
        frames = read_video_as_numpy("input.mp4")
        for idx, frame in enumerate(frames):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=frame)
            detection_result = landmarker.detect(mp_image)
            annotated_image = draw_landmarks_mediapipe(frame,
                                                       detection_result)
            cv2.imshow("annotated", annotated_image)
            cv2.waitKey(0)
            pose = Pose(idx * FRAME_INTERVAL, detection_result)
            with open(f'output/frame_{idx}.json', 'w') as f:
                f.write(pose.to_json())


if __name__ == '__main__':
    main()
