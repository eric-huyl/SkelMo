import mediapipe as mp
from mediapipe import solutions
import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from mediapipe.framework.formats import landmark_pb2


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x,
                                            y=landmark.y,
                                            z=landmark.z)
            for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image, pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def delete_all_files_in_folder(folder_path):
    # 列出文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 如果是文件，则删除它
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")


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

        frame_filename = os.path.join('tmp/',
                                          f"frame_{frame_idx}.npy")
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
        for idx, numpy_image in enumerate(images):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
            detection_result = landmarker.detect(mp_image)
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(),
                                                      detection_result)
            try:
                cv2.imshow(f'title{idx}', annotated_image)
                cv2.waitKey(0)
            except Exception as e:
                print(e)
