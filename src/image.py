import mediapipe as mp
import cv2
from draw_result import draw_landmarks_mediapipe
from Pose import Pose
from utils import write_to_json


def image_recognition(input_file_path: str, output_file_path: str):

    try:
        # Load the input image from an image file.
        mp_image = mp.Image.create_from_file(input_file_path)
    except Exception as e:
        print(f'Error reading image file: {e}')
        return
    model_path = 'pose_landmarker_heavy.task'
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        detection_result = landmarker.detect(mp_image)
        annotated_image = draw_landmarks_mediapipe(mp_image, detection_result)
        cv2.imshow("annotated", annotated_image)
        cv2.waitKey(10)
        try:
            pose = Pose.init_from_detection_result(0,
                                                   detection_result)
        except Exception as e:
            print(f'Error occured:{e}')
            return
        write_to_json(f'output/{output_file_path}.json', pose)
