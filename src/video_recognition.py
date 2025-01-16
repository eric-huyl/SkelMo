import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
from video_capture import open_camera, show_video_stream
import time


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

model_path = 'pose_landmarker_heavy.task'


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

annotated_image = None

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print('pose landmarker result: {}'.format(result))
    global annotated_image
    annotated_image = draw_landmarks_on_image(output_image.numpy_view(),
                                              result)
    


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
landmarker = PoseLandmarker.create_from_options(options)

def main():
    cap = open_camera(0)
    prev_time = time.time()  # 初始化时间
    while True:
        ret, frame = cap.read()  # 捕获每一帧
        if not ret:
            print("无法读取视频帧")
            break

        current_time = time.time()  # 获取当前时间
        if current_time - prev_time >= 1:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(mp_image, int(current_time*1000))

        # 显示视频流
        global annotated_image

        cv2.imshow('Live Video', frame)
        try:
            cv2.imshow('annotated', annotated_image)
        except Exception as e:
            print(e)
        # 如果按下 'q' 键，退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # 释放摄像头资源
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口


if __name__ == '__main__':
    main()
