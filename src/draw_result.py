from mediapipe.framework.formats import landmark_pb2
import numpy as np
from mediapipe import solutions
import matplotlib.pyplot as plt
import json


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


def draw_landmarks_lite(image_landmarks):
    list_x = []
    list_y = []
    for landmark in image_landmarks:
        list_x.append(landmark.x)
        list_y.append(landmark.y)
    plt.scatter(list_x, list_y)
    plt.xlabel('Normalized X')

    plt.ylabel('Normalized Y')

    # 设置图形标题
    plt.title('Scatter plot of normalized coordinates')

    # 显示图形
    plt.show()

def test():
    with open('output/test.json', 'r') as f:
        video_landmarks = json.load(f)
        draw_landmarks_lite(video_landmarks)
