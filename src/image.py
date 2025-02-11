import mediapipe as mp
import cv2
import numpy as np
from draw_result import draw_landmarks_mediapipe
from Pose import Pose
from utils import write_to_json



def image_recognition(input_file_path='input.mp4', frame_interval=10):
    
    try:
        frames = read_video_as_numpy(input_file_path, frame_interval=frame_interval)
    except Exception as e:
        print(f'Error reading video file: {e}')
        return
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
        poses = []
        for idx, frame in enumerate(frames):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = landmarker.detect(mp_image)
            annotated_image = draw_landmarks_mediapipe(frame, detection_result)
            cv2.imshow("annotated", annotated_image)
            cv2.waitKey(10)
            try:
                pose = Pose.init_from_detection_result(idx * frame_interval,
                                                       detection_result)
                poses.append(pose.to_dict)
            except Exception as e:
                print(e)
                continue
        write_to_json('output/poses.json', poses)


if __name__ == '__main__':
    video_recognition()
