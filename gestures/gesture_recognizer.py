
import math
import time


class GestureRecognizer:
    HOLD_SECONDS = 1.0
    FINGERTIP_DISTANCE_THRESHOLD = 0.06

    def __init__(self):
        self.scaling_mode = False
        self.hold_started = None
        self.must_separate = False
        self.exploded_mode = False

    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def finger_extended(hand, tip, pip):
        return hand[tip].y < hand[pip].y

    @staticmethod
    def thumb_extended(hand):
        # Keep the existing thumb classification behavior.
        thumb = (hand[4].x, hand[4].y)
        other_tips = [
            (hand[i].x, hand[i].y) for i in (12, 16, 20)
        ]

        average_distance = sum(
            math.hypot(thumb[0] - tip[0], thumb[1] - tip[1])
            for tip in other_tips
        ) / len(other_tips)

        palm_size = math.hypot(
            hand[9].x - hand[0].x,
            hand[9].y - hand[0].y,
        )

        if palm_size < 0.001:
            return False

        return average_distance > palm_size * 0.70

    def classify_hand(self, hand):
        # Existing gesture logic retained.
        index = self.finger_extended(hand, 8, 6)
        middle = self.finger_extended(hand, 12, 10)
        ring = self.finger_extended(hand, 16, 14)
        pinky = self.finger_extended(hand, 20, 18)
        thumb = self.thumb_extended(hand)

        other_fingers_curled = not middle and not ring and not pinky

        if thumb and other_fingers_curled:
            return "SCALE"

        if index and not thumb and other_fingers_curled:
            return "ROTATE_X"

        if not index and not middle and not ring and not pinky:
            return "MOVE"

        return "IDLE"

    def update_mode_switch(self, landmarks):
        """Toggle Normal/Exploded mode after both index tips touch for 1 second."""
        touching = False

        if len(landmarks) == 2:
            first = (landmarks[0][8].x, landmarks[0][8].y)
            second = (landmarks[1][8].x, landmarks[1][8].y)

            touching = (
                self.distance(first, second)
                < self.FINGERTIP_DISTANCE_THRESHOLD
            )

        now = time.monotonic()

        if touching:
            if self.touch_started is None and not self.must_separate:
                self.touch_started = now

            if (
                self.touch_started is not None
                and now - self.touch_started >= self.HOLD_SECONDS
                and not self.must_separate
            ):
                self.exploded_mode = not self.exploded_mode
                self.must_separate = True
                self.touch_started = None
        else:
            self.touch_started = None
            self.must_separate = False

        progress = 0.0
        if touching and self.touch_started is not None:
            progress = min(
                1.0,
                (now - self.touch_started) / self.HOLD_SECONDS,
            )

        return touching, progress

    def recognize(self, results):
        landmarks = results.hand_landmarks

        switching_touch, hold_progress = self.update_mode_switch(
            landmarks
        )

        hands = []

        for hand in landmarks:
            hands.append({
                "gesture": self.classify_hand(hand),
                "index_tip": (hand[8].x, hand[8].y),
                "thumb_tip": (hand[4].x, hand[4].y),
            })

        fingertip_distance = None

        if len(landmarks) == 1:
            hand = landmarks[0]
            fingertip_distance = self.distance(
                (hand[4].x, hand[4].y),
                (hand[8].x, hand[8].y),
            )

        scaling_mode = (
            len(hands) == 1
            and hands[0]["gesture"] == "SCALE"
        )

        if switching_touch:
            gesture_mode = "MODE SWITCH"
            status = "Hold both index fingertips together"
        elif scaling_mode:
            gesture_mode = "PINCH SCALING"
            status = (
                "Adjust component separation"
                if self.exploded_mode
                else "Spread/pinch thumb and index to resize"
            )
        elif len(hands) == 2:
            active = next(
                (
                    item["gesture"] for item in hands
                    if item["gesture"] in (
                        "ROTATE_X", "ROTATE_Y", "MOVE"
                    )
                ),
                None,
            )
            gesture_mode = active or "TWO HANDS: READY"
            status = "Touch both index fingertips to switch modes"
        elif len(hands) == 1:
            gesture_mode = hands[0]["gesture"]
            status = "Index: rotate | Fist: move"
        else:
            gesture_mode = "IDLE"
            status = "Show your hand"

        mode_name = (
            "EXPLODED"
            if self.exploded_mode
            else "NORMAL"
        )

        return {
            "mode": gesture_mode,
            "display_mode": mode_name,
            "exploded_mode": self.exploded_mode,
            "mode_switch_touch": switching_touch,
            "scaling_mode": scaling_mode,
            "hands": hands,
            "activation_gesture": scaling_mode,
            "activation_status": status,
            "hold_progress": hold_progress,
            "fingertip_distance": fingertip_distance,
            "must_separate": self.must_separate,
        }
