# HoloManipulate

HoloManipulate is a gesture-controlled 3D model manipulation application built with Python, OpenCV, MediaPipe, and Open3D.

The project uses real-time hand tracking to interact with 3D objects through webcam-based hand gestures, providing an intuitive way to rotate, move, scale, and explore 3D models.

## Features

- **Real-time hand tracking:** Tracks hand landmarks using MediaPipe.
- **3D model rotation:** Rotate the model using index-finger movements.
- **Model movement:** Move the model using the existing fist gesture.
- **Interactive scaling:** Adjust model size using thumb and index-finger distance.
- **Normal and Exploded modes:** Switch modes by bringing both index fingertips together and holding them for approximately one second.
- **3D visualization:** Renders the model using Open3D.
- **CPU-friendly design:** Designed to run without requiring an NVIDIA GPU.

## Technology Stack

- Python
- OpenCV
- MediaPipe Tasks
- Open3D
- NumPy
- Trimesh

## Project Structure

```text
HoloManipulate/
├── main.py
├── camera/
├── tracking/
│   └── hand_tracker.py
├── gestures/
│   └── gesture_recognizer.py
├── interaction/
│   └── model_controller.py
├── scene/
│   ├── model_loader.py
│   ├── exploded_view.py
│   └── city_scene.py
├── models/
│   └── hand_landmarker.task
├── tests/
└── README.md
```

*The directory listing reflects the main modules; additional files may exist in the project.*

## Requirements

- Python 3.10 or a compatible version supported by the installed dependencies
- A working webcam
- OpenCV-compatible camera access
- MediaPipe hand-landmarker model file
- Open3D-compatible graphics environment

## Setup

Activate your existing HVC virtual environment:

```bash
source ~/Projects/human_video_composition/HVC/bin/activate
```

Navigate to the project directory:

```bash
cd ~/Projects/HoloManipulate
```

Install any missing dependencies in that environment:

```bash
python -m pip install opencv-python mediapipe open3d numpy trimesh
```

Ensure the MediaPipe hand-landmarker model is available at:

```text
models/hand_landmarker.task
```

## Running the Application

From the project directory, run:

```bash
python main.py
```

Show your hand in front of the webcam and use the gestures supported by your current configuration.

Press `Q` or `Esc` in the webcam window to quit, if supported by the running version.

## Notes

- Keep your virtual environment outside the repository.
- The hand-landmarker model is a required runtime asset.
- Imported models must be compatible with the project's current loading implementation.
- Exploded-view behavior depends on the structure of the imported model.
- Gesture thresholds and controls are implemented in the existing tracking, gesture-recognition, and interaction modules.

## Future Improvements

- Improve gesture stability and motion smoothing.
- Support additional 3D model formats.
- Improve component-level interaction and exploded-view visualization.
- Enhance the interface and model-loading workflow.

## License

No license has been specified yet. Add a license before distributing the project if you intend to permit reuse.
