import time
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    def __init__(self, model_path="models/hand_landmarker.task"):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_timestamp_ms = 0

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        timestamp_ms = int(time.monotonic() * 1000)
        timestamp_ms = max(timestamp_ms, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms

        return self.detector.detect_for_video(image, timestamp_ms)

    def draw_landmarks(self, frame, results):
        height, width = frame.shape[:2]
        connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

        for hand in results.hand_landmarks:
            points = [
                (int(point.x * width), int(point.y * height))
                for point in hand
            ]

            for connection in connections:
                start = connection.start
                end = connection.end
                cv2.line(
                    frame, points[start], points[end],
                    (100, 220, 100), 2
                )

            for point in points:
                cv2.circle(frame, point, 4, (0, 255, 255), -1)

    def close(self):
        self.detector.close()
