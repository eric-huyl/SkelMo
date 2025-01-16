import mediapipe as mp
from mediapipe import solutions
import os
import cv2
import numpy as np
import time

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


def save_frames_as_numpy(video_path, output_folder="tmp", interval=1):
    """
    打开视频文件并每隔指定时间间隔截图一次，保存为 NumPy 格式文件。

    参数:
    - video_path: 视频文件的路径
    - output_folder: 输出保存截图的文件夹，默认为 "frames"
    - interval: 截图的时间间隔（秒），默认为 1 秒
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # 获取视频的帧率（fps）
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频的帧率: {fps} FPS")

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 初始化计时器
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()

        # 如果读取失败，退出循环
        if not ret:
            print("Error: Failed to read frame from video")
            break

        # 获取当前时间
        current_time = time.time()

        # 每隔指定时间间隔截图一次
        if current_time - start_time >= interval:
            # 保存当前帧为 NumPy 格式
            frame_filename = os.path.join(output_folder,
                                          f"frame_{frame_count}.npy")
            np.save(frame_filename, frame)
            print(f"保存截图: {frame_filename}")

            # 更新计时器
            start_time = current_time
            frame_count += 1

        # 如果已处理完所有帧，退出
        if frame_count >= total_frames:
            break

    # 释放资源并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()


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


model_path = 'pose_landmarker_heavy.task'
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

with PoseLandmarker.create_from_options(options) as landmarker:
    save_frames_as_numpy("input.mp4")
    images = load_numpy_images_from_folder()
    for numpy_image in images:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        detection_result = landmarker.detect(mp_image)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(),
                                                  detection_result)
        cv2.imshow(annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
