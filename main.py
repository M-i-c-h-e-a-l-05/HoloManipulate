
import cv2
import open3d as o3d

from tracking.hand_tracker import HandTracker
from gestures.gesture_recognizer import GestureRecognizer
from scene.model_loader import load_multipart_model
from scene.exploded_view import prepare_exploded_view
from interaction.model_controller import ModelController


def main():
    model_path = "/home/micheal-leveiro/Projects/HoloManipulate/tests/3d_printable_radial_pneumatic_engine.glb"

    tracker = HandTracker(model_path="models/hand_landmarker.task")
    recognizer = GestureRecognizer()
    camera = cv2.VideoCapture(0)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        tracker.close()
        raise RuntimeError("Could not open webcam.")

    preview_name = "HoloManipulate - Webcam Controls"
    cv2.namedWindow(preview_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(preview_name, 360, 270)

    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()

        cv2.moveWindow(
            preview_name,
            max(0, screen_width - 380),
            max(0, screen_height - 340),
        )
    except Exception:
        pass

    viewer = None

    try:
        components = load_multipart_model(model_path)
        prepare_exploded_view(components)

        controller = ModelController(components)

        viewer = o3d.visualization.Visualizer()

        if not viewer.create_window(
            window_name="HoloManipulate - Engine Workspace",
            width=1000,
            height=750,
        ):
            raise RuntimeError("Could not create the 3D window.")

        for component in components:
            viewer.add_geometry(component["mesh"])

        render = viewer.get_render_option()
        render.background_color = [0.035, 0.045, 0.07]
        render.mesh_show_back_face = True

        viewer.reset_view_point(True)
        viewer.get_view_control().set_zoom(0.75)

        print(f"Loaded {len(components)} engine components.")
        print("NORMAL: index rotates; thumb-index distance scales.")
        print("EXPLODED: thumb-index distance controls separation.")
        print("Touch both index fingertips for 1 second to switch modes.")
        print("Press Q or ESC in the webcam window to quit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            results = tracker.detect(frame)
            state = recognizer.recognize(results)

            tracker.draw_landmarks(frame, results)

            controller.update(state, results.hand_landmarks)

            for component in components:
                viewer.update_geometry(component["mesh"])

            mode = state["display_mode"]
            color = (
                (0, 255, 0)
                if mode == "EXPLODED"
                else (0, 220, 255)
            )

            cv2.putText(
                frame,
                f"Mode: {mode}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

            cv2.putText(
                frame,
                f"Gesture: {state['mode']}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Hands detected: {len(results.hand_landmarks)}",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                state["activation_status"],
                (15, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 200, 255),
                1,
            )

            if state["mode_switch_touch"]:
                progress = state["hold_progress"]
                cv2.putText(
                    frame,
                    f"Switching mode: {progress * 100:.0f}%",
                    (15, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(preview_name, frame)

            if not viewer.poll_events():
                break

            viewer.update_renderer()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        camera.release()
        tracker.close()

        if viewer is not None:
            viewer.destroy_window()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
