import math
import json


class Pose:

    def __init__(self, timestamp=None, landmarks=None):
        self.timestamp = timestamp
        self.landmarks = landmarks

    @staticmethod
    def resolve_detection_result(detection_result) -> dict:
        landmarks = []
        try:
            for landmark_object in detection_result.pose_world_landmarks[0]:
                landmark = {
                    'x': landmark_object.x,
                    'y': landmark_object.y,
                    'z': landmark_object.z,
                    'visibility': landmark_object.visibility
                }
                landmarks.append(landmark)
        except Exception as e:
            raise ValueError(
                f"An error occured resolving dectection result: {e}")
        resolved_landmarks = {}
        resolved_landmarks['left_shoulder'] = landmarks[11]
        resolved_landmarks['right_shoulder'] = landmarks[12]
        resolved_landmarks['left_hip'] = landmarks[23]
        resolved_landmarks['right_hip'] = landmarks[24]
        resolved_landmarks['left_knee'] = landmarks[25]
        resolved_landmarks['right_knee'] = landmarks[26]
        resolved_landmarks['left_ankle'] = landmarks[27]
        resolved_landmarks['right_ankle'] = landmarks[28]
        resolved_landmarks['left_heel'] = landmarks[29]
        resolved_landmarks['right_heel'] = landmarks[30]
        resolved_landmarks['left_foot_index'] = landmarks[31]
        resolved_landmarks['right_foot_index'] = landmarks[32]
        return resolved_landmarks

    @property
    def angles(self):
        return {
            'left_knee': self._calculate_joint_angle('left_knee'),
            'right_knee': self._calculate_joint_angle('right_knee'),
            'left_hip': self._calculate_joint_angle('left_hip'),
            'left_ankle': self._calculate_joint_angle('left_ankle'),
            'right_ankle': self._calculate_joint_angle('right_ankle')
        }

    def _calculate_joint_angle(self, joint: str) -> float:
        joint_mapping = {
            'left_knee': ('left_hip', 'left_knee', 'left_ankle'),
            'right_knee': ('right_hip', 'right_knee', 'right_ankle'),
            'left_hip': ('left_shoulder', 'left_hip', 'left_knee'),
            'left_ankle': ('left_knee', 'left_ankle', 'left_foot_index'),
            'right_ankle': ('right_knee', 'right_ankle', 'right_foot_index')
        }
        if joint not in joint_mapping:
            raise ValueError(f"Unknown joint: {joint}")
        points = joint_mapping[joint]
        return self.calculate_angle(self.landmarks[points[0]],
                                    self.landmarks[points[1]],
                                    self.landmarks[points[2]])

    @staticmethod
    def calculate_angle(pointA, pointB, pointC) -> float:
        # Calculate the angle between three points (A, B, C)
        AB = (pointB['x'] - pointA['x'], pointB['y'] - pointA['y'])
        BC = (pointC['x'] - pointB['x'], pointC['y'] - pointB['y'])

        dot_product = AB[0] * BC[0] + AB[1] * BC[1]
        magnitude_AB = math.sqrt(AB[0]**2 + AB[1]**2)
        magnitude_BC = math.sqrt(BC[0]**2 + BC[1]**2)

        if magnitude_AB == 0 or magnitude_BC == 0:
            return 0

        cos_angle = dot_product / (magnitude_AB * magnitude_BC)
        angle = math.acos(cos_angle)
        if angle > 0.5 * math.pi:
            angle = math.pi - angle
        return math.degrees(angle)

    @property
    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'landmarks': self.landmarks,
            'angles': self.angles
        }

    @staticmethod
    def init_from_dict(data):
        pose = Pose()
        pose.timestamp = data['timestamp']
        pose.landmarks = data['landmarks']
        return pose

    @staticmethod
    def init_from_detection_result(timestamp, detection_result):
        pose = Pose()
        pose.timestamp = timestamp
        pose.landmarks = Pose.resolve_detection_result(detection_result)
        return pose


if __name__ == '__main__':
    print(Pose.calculate_angle((0, 0), (1, 1), (1, 0)))
