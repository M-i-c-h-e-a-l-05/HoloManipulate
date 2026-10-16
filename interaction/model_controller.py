
import math
import numpy as np
import open3d as o3d


class ModelController:
    POSITION_SMOOTHING = 0.25
    MOTION_SMOOTHING = 0.30
    ROTATION_GAIN = 3.0
    MOVE_GAIN = 8.0
    DEAD_ZONE = 0.0015
    SCALE_SMOOTHING = 0.25

    def __init__(self, components):
        if isinstance(components, list):
            self.components = components
        else:
            self.components = [{
                "mesh": components,
                "_base_vertices": np.asarray(
                    components.vertices
                ).copy(),
                "_explode_direction": np.array([0.0, 1.0, 0.0]),
                "_explode_distance": 0.0,
            }]

        self.geometries = [
            item["mesh"] for item in self.components
        ]

        all_vertices = [
            item["_base_vertices"]
            for item in self.components
            if len(item["_base_vertices"])
        ]

        self.center = (
            np.min(np.vstack([v.min(axis=0) for v in all_vertices]), axis=0)
            + np.max(np.vstack([v.max(axis=0) for v in all_vertices]), axis=0)
        ) / 2.0

        self.linear = np.eye(3)
        self.translation = np.zeros(3)
        self.exploded_amount = 0.0
        self.filtered_exploded_amount = None
        self.filtered_pinch_distance = None

        self.previous_mode = None
        self.previous_wrist = None
        self.filtered_wrist = None
        self.previous_index_tip = None
        self.previous_wrist_distance = None
        self.filtered_wrist_distance = None
        self.filtered_motion = [0.0, 0.0]

    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def reset_motion(self):
        self.previous_wrist = None
        self.filtered_wrist = None
        self.previous_index_tip = None
        self.filtered_motion = [0.0, 0.0]

    def rebuild_meshes(self):
        """Rebuild each mesh from its original vertices and shared transform."""
        for component in self.components:
            mesh = component["mesh"]
            base = component["_base_vertices"]
            direction = component["_explode_direction"]
            distance = component["_explode_distance"]

            vertices = base + (
                direction * distance * self.exploded_amount
            )

            transformed = (
                (vertices - self.center) @ self.linear.T
                + self.center
                + self.translation
            )

            mesh.vertices = o3d.utility.Vector3dVector(transformed)
            mesh.compute_vertex_normals()

    def apply_scale(self, factor):
        factor = max(0.95, min(1.05, factor))
        self.linear *= factor
        self.rebuild_meshes()

    def apply_rotation(self, angle_x, angle_y):
        rotation = o3d.geometry.get_rotation_matrix_from_xyz(
            (angle_x, angle_y, 0.0)
        )
        self.linear = rotation @ self.linear
        self.rebuild_meshes()

    def apply_translation(self, delta):
        self.translation += np.asarray(delta, dtype=float)
        self.rebuild_meshes()

    def update_exploded_intensity(self, gap):
        # Wider thumb-index gap -> more separation.
        target = max(0.0, min(2.5, (gap - 0.035) / 0.09))

        if self.filtered_exploded_amount is None:
            self.filtered_exploded_amount = target
        else:
            self.filtered_exploded_amount += 0.18 * (
                target - self.filtered_exploded_amount
            )

        self.exploded_amount = self.filtered_exploded_amount
        self.rebuild_meshes()

    def update(self, state, landmarks):
        mode = state["mode"]
        hands = state["hands"]

        # Prevent mode-switching fingertip contact from moving the model.
        if state.get("mode_switch_touch", False):
            self.reset_motion()
            return

        if mode != self.previous_mode:
            self.reset_motion()
            self.previous_wrist_distance = None
            self.filtered_wrist_distance = None
            self.filtered_pinch_distance = None
            self.previous_mode = mode

        # One-hand thumb-index control:
        # Normal mode = resize whole assembly.
        # Exploded mode = change component separation.
        if state["scaling_mode"] and len(landmarks) == 1:
            hand = landmarks[0]
            gap = self.distance(
                (hand[4].x, hand[4].y),
                (hand[8].x, hand[8].y),
            )

            if state["exploded_mode"]:
                self.update_exploded_intensity(gap)
            else:
                if self.filtered_pinch_distance is None:
                    self.filtered_pinch_distance = gap
                else:
                    self.filtered_pinch_distance += 0.25 * (
                        gap - self.filtered_pinch_distance
                    )

                gap = self.filtered_pinch_distance

                if gap > 0.12:
                    amount = min(0.012, (gap - 0.12) * 0.06)
                    self.apply_scale(1.0 + amount * 1.25)
                elif gap < 0.08:
                    amount = min(0.012, (0.08 - gap) * 0.06)
                    amount = min(amount * 4.5, 0.05)
                    self.apply_scale(1.0 - amount)

            self.reset_motion()
            return

        # Existing two-hand scaling support, applied to the whole assembly.
        if state["scaling_mode"] and len(landmarks) == 2:
            a = (landmarks[0][0].x, landmarks[0][0].y)
            b = (landmarks[1][0].x, landmarks[1][0].y)
            raw_distance = self.distance(a, b)

            if self.filtered_wrist_distance is None:
                self.filtered_wrist_distance = raw_distance
            else:
                self.filtered_wrist_distance += self.SCALE_SMOOTHING * (
                    raw_distance - self.filtered_wrist_distance
                )

            if self.previous_wrist_distance is not None:
                old = self.previous_wrist_distance

                if old > 0.01:
                    factor = self.filtered_wrist_distance / old
                    change = factor - 1.0
                    intensity = 1.0 + min(3.0, abs(change) * 100.0)
                    factor = 1.0 + change * intensity
                    self.apply_scale(factor)

            self.previous_wrist_distance = self.filtered_wrist_distance
            return

        # Pick a hand performing an existing supported gesture.
        active_index = next(
            (
                i for i, item in enumerate(hands)
                if item["gesture"] in (
                    "ROTATE_X", "ROTATE_Y", "MOVE"
                )
            ),
            None,
        )

        if active_index is None or active_index >= len(landmarks):
            self.reset_motion()
            return

        hand = landmarks[active_index]
        gesture = hands[active_index]["gesture"]

        wrist = (hand[0].x, hand[0].y)

        if self.filtered_wrist is None:
            self.filtered_wrist = wrist
        else:
            self.filtered_wrist = (
                self.filtered_wrist[0] + self.POSITION_SMOOTHING
                * (wrist[0] - self.filtered_wrist[0]),
                self.filtered_wrist[1] + self.POSITION_SMOOTHING
                * (wrist[1] - self.filtered_wrist[1]),
            )

        point = self.filtered_wrist

        if self.previous_wrist is None:
            self.previous_wrist = point
            return

        dx = point[0] - self.previous_wrist[0]
        dy = point[1] - self.previous_wrist[1]
        self.previous_wrist = point

        if abs(dx) < self.DEAD_ZONE:
            dx = 0.0
        if abs(dy) < self.DEAD_ZONE:
            dy = 0.0

        if gesture in ("ROTATE_X", "ROTATE_Y"):
            tip = (hand[8].x, hand[8].y)

            if self.previous_index_tip is not None:
                tip_dx = tip[0] - self.previous_index_tip[0]
                tip_dy = tip[1] - self.previous_index_tip[1]

                if abs(tip_dx) < self.DEAD_ZONE:
                    tip_dx = 0.0
                if abs(tip_dy) < self.DEAD_ZONE:
                    tip_dy = 0.0

                speed = math.hypot(tip_dx, tip_dy)
                intensity = 1.0 + min(3.0, speed * 100.0)

                self.apply_rotation(
                    tip_dy * self.ROTATION_GAIN * intensity,
                    tip_dx * self.ROTATION_GAIN * intensity,
                )

            self.previous_index_tip = tip

        elif gesture == "MOVE":
            self.filtered_motion[0] += self.MOTION_SMOOTHING * (
                dx - self.filtered_motion[0]
            )
            self.filtered_motion[1] += self.MOTION_SMOOTHING * (
                dy - self.filtered_motion[1]
            )

            speed = math.hypot(dx, dy)
            intensity = 1.0 + min(3.0, speed * 100.0)

            self.apply_translation([
                self.filtered_motion[0] * self.MOVE_GAIN * intensity,
                -self.filtered_motion[1] * self.MOVE_GAIN * intensity,
                0.0,
            ])
