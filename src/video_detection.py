import mediapipe as mp
import json
import os
import cv2
import numpy as np
from utils import delete_all_files_in_folder





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

    # 计算读取每帧的时间间隔
    frame_interval = 60

    frame_idx = 0
    while True:
        # 设置视频读取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # 读取一帧
        ret, frame = cap.read()

        if not ret:
            break

        frame_filename = os.path.join('tmp/', f"frame_{frame_idx}.npy")
        np.save(frame_filename, frame)

        # 将帧添加到数组中
        frames.append(frame)

        # 按指定的帧率调整读取帧的位置
        frame_idx += frame_interval

    # 转换为 NumPy 数组
    frames_np = np.array(frames)

    # 释放资源
    cap.release()

    return frames_np


def load_numpy_images_from_folder(folder_path="tmp"):
    """
    读取指定文件夹内的所有 .npy 文件并将它们作为 NumPy 数组加载到列表中。

    参数:
    - folder_path: 文件夹路径，包含 .npy 文件。

    返回:
    - images: 一个列表，包含所有加载的 NumPy 数组。
    """
    images = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 只处理 .npy 文件
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)

            # 加载 NumPy 数组文件
            image = np.load(file_path)
            images.append(image)
            print(f"加载了 {filename}")

    return images


if __name__ == '__main__':
    delete_all_files_in_folder('tmp/')
    model_path = 'pose_landmarker_heavy.task'
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)

    with PoseLandmarker.create_from_options(options) as landmarker:
        read_video_as_numpy("input.mp4")
        images = load_numpy_images_from_folder()
        video_landmarks = []
        for idx, numpy_image in enumerate(images):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=numpy_image)
            detection_result = landmarker.detect(mp_image)
            try:
                frame_landmarks = []
                for landmark_object in detection_result.pose_landmarks[0]:
                    landmark_dict = {
                        'x': landmark_object.x,
                        'y': landmark_object.y,
                        'z': landmark_object.z,
                        'visibility': landmark_object.visibility
                    }
                    frame_landmarks.append(landmark_dict)
            except Exception as e:
                print(e)
        with open('output/test.json', 'w') as f:
            json.dump(result_list, f)
    delete_all_files_in_folder('tmp/')
