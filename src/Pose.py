import math
import json


class Pose:

    def __init__(self, timestamp, detection_result):
        self.timestamp = timestamp
        self.landmarks = self._resolve_detection_result(detection_result)

    def _resolve_detection_result(self, detection_result) -> dict:
        landmarks = []
        for landmark_object in detection_result.pose_world_landmarks[0]:
            landmark = {
                'x': landmark_object.x,
                'y': landmark_object.y,
                'z': landmark_object.z,
                'visibility': landmark_object.visibility
            }
            landmarks.append(landmark)
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
            'left_knee': self.calculate_angle('left_knee'),
            'right_knee': self.calculate_angle('right_knee'),
            'left_hip': self.calculate_angle('left_hip'),
            'left_ankle': self.calculate_angle('left_ankle'),
            'right_ankle': self.calculate_angle('right_ankle')
        }

    def calculate_angle(self, joint):
        if joint == 'left_knee':
            return self._calculate_joint_angle(
                self.landmarks['left_hip'],
                self.landmarks['left_knee'],
                self.landmarks['left_ankle'])
        elif joint == 'right_knee':
            return self._calculate_joint_angle(
                self.landmarks['right_hip'],
                self.landmarks['right_knee'],
                self.landmarks['right_ankle'])
        elif joint == 'left_hip':
            return self._calculate_joint_angle(
                self.landmarks['left_shoulder'],
                self.landmarks['left_hip'],
                self.landmarks['left_knee'])
        elif joint == 'left_ankle':
            return self._calculate_joint_angle(
                self.landmarks['left_knee'],
                self.landmarks['left_ankle'],
                self.landmarks['left_foot_index'])
        elif joint == 'right_ankle':
            return self._calculate_joint_angle(
                self.landmarks['right_knee'],
                self.landmarks['right_ankle'],
                self.landmarks['right_foot_index'])
        else:
            raise ValueError(f"Unknown joint: {joint}")

    def _calculate_joint_angle(self, pointA, pointB, pointC) -> float:
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

    def to_json(self):
        return json.dumps({
            'timestamp': self.timestamp,
            'landmarks': self.landmarks,
            'angles': self.angles
        })


if __name__ == '__main__':
    print(Pose._calculate_joint_angle(None, (0, 0), (1, 1), (1, 0)))
