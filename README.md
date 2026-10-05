# Self-driving Robot

Control and computer-vision software for an autonomous line-following robot, running on a Raspberry Pi with a camera and an Arduino for motor and steering control.

The project combines real-time line detection, steering control, AprilTag/ArUco marker detection, traffic-sign detection, and traffic-light detection for autonomous navigation.

---

## Features

- Real-time line/lane detection using OpenCV
- Left/right lane detection using dedicated regions of interest (ROIs)
- Steering-angle calculation based on lane position
- Steering smoothing using EMA filtering and rate limiting
- Serial communication between Raspberry Pi and Arduino
- AprilTag/ArUco-based marker detection
- Traffic-sign detection using a custom YOLOv8 model
- Traffic-light detection using HSV color segmentation
- Traffic-light state stabilization using a history/voting mechanism
- HSV parameter testing tool for camera calibration
- Raspberry Pi Camera (`Picamera2`) support
- OpenCV webcam/DroidCam support for development and testing

---

## Project Structure

```text
Self_driving_-_Line-Following-Robot/
│
├── main.py
├── vehicle.py
├── vision.py
├── robot.py
├── config.py
├── requirements.txt
│
├── traffic_sign/
│   │
│   └── train_sign_detector/
│       │
│       ├── assets/
│       │   └── yolov8n_fira_best.pt
│       │
│       ├── configs/
│       │   └── data.yaml
│       │
│       ├── tools/
│       │   ├── augment_dataset.py
│       │   ├── prepare_dataset.py
│       │   ├── count_labels.py
│       │   └── split_dataset.py
│       │
│       ├── raw/
│       │   ├── images/
│       │   └── labels/
│       │
│       ├── dataset/
│       │   ├── images/
│       │   │   ├── train/
│       │   │   └── val/
│       │   │
│       │   └── labels/
│       │       ├── train/
│       │       └── val/
│       │
│       ├── runs/
│       │   └── fira_yolov8n_park/
│       │       └── weights/
│       │           ├── best.pt
│       │           └── best.onnx
│       │
│       └── 01_training_and_export.ipynb
│
└── traffic_light/
    ├── detect_light(1).py
    └── test_HSV_parameters.py
```

> **Dataset note:** The traffic-sign dataset itself is not included in the GitHub repository. The `raw/` and generated `dataset/` directories above show the expected local dataset structure used for training.

---

# Main Robot System

The main robot software is separated into modules so that vision processing, vehicle control, serial communication, and configuration can be developed and tested independently.

```text
main.py
   │
   ▼
vehicle.py
   │
   ├──► vision.py
   │       ├── Camera
   │       ├── Line detection
   │       └── AprilTag/ArUco detection
   │
   └──► robot.py
           └── Serial communication
                    │
                    ▼
                 Arduino
                    │
              ┌─────┴─────┐
              ▼           ▼
            Motors       Servo
```

### Main files

| File | Description |
|---|---|
| `main.py` | Main entry point |
| `vehicle.py` | Main vehicle state machine and control loop |
| `vision.py` | Camera handling, line detection, and AprilTag detection |
| `robot.py` | Arduino serial communication and steering/speed control |
| `config.py` | Tunable vision and control parameters |
| `requirements.txt` | Python dependencies |

---

# Traffic Sign Detection

The project contains a dedicated traffic-sign detection and training pipeline based on **YOLOv8**.

The training project is located in:

```text
traffic_sign/train_sign_detector/
```

## Training Project Structure

```text
train_sign_detector/
│
├── assets/
│   └── yolov8n_fira_best.pt
│
├── configs/
│   └── data.yaml
│
├── tools/
│   ├── augment_dataset.py
│   ├── prepare_dataset.py
│   ├── count_labels.py
│   └── split_dataset.py
│
├── raw/
│   ├── images/
│   └── labels/
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   │
│   └── labels/
│       ├── train/
│       └── val/
│
├── runs/
│   └── fira_yolov8n_park/
│       └── weights/
│           ├── best.pt
│           └── best.onnx
│
└── 01_training_and_export.ipynb
```

### Dataset

The original traffic-sign dataset is intentionally not included in the repository because of its size.

For local training, the raw dataset is expected at:

```text
raw/
├── images/
└── labels/
```

The dataset preparation workflow generates:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
│
└── labels/
    ├── train/
    └── val/
```

### Training tools

| File | Purpose |
|---|---|
| `augment_dataset.py` | Augments the dataset |
| `prepare_dataset.py` | Prepares the dataset for training |
| `count_labels.py` | Counts and analyzes labels |
| `split_dataset.py` | Dataset splitting utility; currently not used |

### Model

The project includes the trained YOLOv8 model:

```text
assets/yolov8n_fira_best.pt
```

The training/export workflow also produces:

```text
runs/fira_yolov8n_park/weights/
├── best.pt
└── best.onnx
```

The ONNX model can be used for deployment with ONNX Runtime.

---

# Traffic Light Detection

The project also contains a dedicated traffic-light detection module:

```text
traffic_light/
├── detect_light(1).py
└── test_HSV_parameters.py
```

The current implementation detects **red and green traffic lights** using HSV color segmentation.

> The current detector explicitly uses red and green. Yellow is not included because the target traffic-light board used for this implementation does not have a yellow LED. fileciteturn1file0L10-L18

## `detect_light(1).py`

The main detector performs the following pipeline:

```text
Camera Frame
     │
     ▼
ROI Selection
     │
     ▼
Median Blur
     │
     ▼
BGR → HSV
     │
     ▼
Red / Green Masks
     │
     ▼
Morphological Filtering
     │
     ▼
Contour Detection
     │
     ▼
Area + Circularity Filtering
     │
     ▼
Brightness Verification
     │
     ▼
Red/Green Conflict Resolution
     │
     ▼
Best Candidate Selection
     │
     ▼
State History / Voting
     │
     ▼
Confirmed Traffic-Light State
```

The detector uses:

- Separate HSV ranges for red and green
- Brightness verification using the HSV Value channel
- Minimum and maximum contour-area limits
- Circularity filtering to reject irregular objects
- Morphological operations to reduce noise and reconnect real regions
- Distance-based conflict resolution when red and green detections overlap
- A temporal state history to stabilize the detected color

The detector keeps a history of recent observations and confirms a state after enough votes are received. fileciteturn1file0L26-L38

The default implementation uses an ROI covering approximately:

```text
Top:    10%
Bottom: 35%
Left:   55%
Right:  80%
```

of the camera frame. fileciteturn1file0L5-L8

---

## `test_HSV_parameters.py`

This script is a small calibration/debugging utility for finding suitable HSV values for the traffic-light detector.

Clicking a pixel in the camera window prints its HSV values to the terminal:

```text
Pixel (x,y) -> H=..., S=..., V=...
```

This makes it easier to determine the actual HSV ranges produced by the camera under different lighting conditions. fileciteturn1file1L3-L13

The script can be used when the traffic-light detector needs to be calibrated for a different camera, environment, or lighting condition.

---

# Camera Support

The main robot system is designed to work with a Raspberry Pi camera through `Picamera2`.

If `Picamera2` is unavailable, OpenCV's `cv2.VideoCapture()` can be used for testing with a regular webcam.

The traffic-light testing scripts can also be used with a network camera such as DroidCam by replacing the camera URL in the script.

---

# Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

On Raspberry Pi OS, `Picamera2` may need to be installed separately:

```bash
sudo apt install -y python3-picamera2
```

For the traffic-sign training/inference pipeline, install the dependencies required by the YOLOv8 workflow as specified by the project environment.

---

# Running the Robot

From the main project directory:

```bash
python main.py
```

The main control system starts the camera, processes incoming frames, detects the lane/line, calculates the steering command, and communicates with the Arduino.

---

# Running Traffic-Light Detection

The traffic-light detector can be run independently for testing:

```bash
python traffic_light/detect_light(1).py
```

The detector opens a window named:

```text
Traffic Light Detection
```

and draws the detected red or green state on the frame.

Press:

```text
Q
```

to exit the program.

> The camera URL inside the script must be configured for the camera being used.

---

# Calibrating HSV Parameters

To inspect HSV values from the camera:

```bash
python traffic_light/test_HSV_parameters.py
```

Click on pixels belonging to the traffic-light LEDs in the displayed camera window.

The HSV values are printed in the terminal and can be used to adjust:

```python
RED_LOWER_1
RED_UPPER_1
RED_LOWER_2
RED_UPPER_2

GREEN_LOWER
GREEN_UPPER
```

in the traffic-light detector.

---

# Lane Detection

The lane-detection pipeline uses OpenCV image processing.

The camera frame is processed inside dedicated regions of interest for the left and right lane boundaries.

The detected lane information is used to estimate the lane center and calculate a steering angle.

```text
Camera Frame
     │
     ▼
ROI Selection
     │
     ▼
Edge Detection
     │
     ▼
Line Detection
     │
     ▼
Lane Position
     │
     ▼
Steering Error
     │
     ▼
Steering Angle
     │
     ▼
Filtering + Rate Limiting
     │
     ▼
Arduino
```

---

# Steering Control

The steering pipeline applies several stages of processing to prevent sudden servo movements.

```text
Raw Steering Angle
        │
        ▼
EMA Low-pass Filter
        │
        ▼
Rate Limiter
        │
        ▼
Servo Range Clamp
        │
        ▼
Final Steering Command
```

Important configurable parameters include:

- `ROI_*_RL` / `ROI_*_LL` — right/left lane detection regions
- `CANNY_LOW` / `CANNY_HIGH` — Canny edge thresholds
- `K_MIN` / `K_MAX` — adaptive steering gain
- `STEERING_ALPHA` — EMA smoothing coefficient
- `MAX_STEERING_STEP` — maximum steering-angle change per frame
- `APRILTAG_DETECT_EVERY_N_FRAMES` — AprilTag detection interval

---

# AprilTag / ArUco Detection

AprilTag-style markers are used as visual landmarks for specific navigation events.

Detection can be throttled so that it does not need to run on every camera frame. This reduces CPU usage on the Raspberry Pi while maintaining the latest valid detection result between detection cycles.

---

# Raspberry Pi + Arduino Architecture

The robot uses a two-level control architecture.

```text
                 Raspberry Pi
        ┌─────────────────────────┐
        │ Camera                  │
        │                         │
        │ OpenCV                 │
        │ Line Detection         │
        │ Traffic Signs          │
        │ Traffic Lights         │
        │ AprilTags / ArUco       │
        │                         │
        │ Vehicle Control         │
        └────────────┬────────────┘
                     │
                Serial / USB
                     │
                     ▼
                ┌─────────┐
                │ Arduino │
                └────┬────┘
                     │
             ┌───────┴───────┐
             ▼               ▼
           Motors           Servo
```

The Raspberry Pi performs high-level computer-vision and navigation processing, while the Arduino handles low-level motor and steering control.

---

# Technical Notes

### Steering smoothing

The raw steering angle is smoothed using an exponential moving average and then constrained by a rate limiter before being sent to the servo.

### AprilTag throttling

AprilTag detection can run only every N frames because marker detection is more computationally expensive than the line-detection pipeline.

### Traffic-light temporal filtering

The traffic-light detector keeps a short history of detected states and uses majority voting before confirming a state. This reduces rapid red/green state changes caused by individual noisy frames. fileciteturn1file0L37-L38

### Traffic-sign inference

The custom YOLOv8 model provides object-detection results that can be used by the robot's navigation and decision-making logic.

---

# Hardware Warning

This software can send commands directly to the robot's motors and steering servo.

Before running the complete system on the physical robot:

- Test on a safe and obstacle-free surface.
- Keep immediate access to the STOP command.
- Verify motor direction.
- Verify servo center position.
- Check `STEER_MIN` and `STEER_MAX`.
- Verify the configured motor speed matches the actual hardware.
- Test each perception module independently before enabling autonomous movement.

---

# Project Status

The project currently contains several autonomous-navigation and computer-vision components:

- Line/lane following
- Steering and motor control
- AprilTag/ArUco detection
- Traffic-sign detection
- YOLOv8 traffic-sign training/export pipeline
- Traffic-light detection
- HSV traffic-light calibration tools

The datasets used for traffic-sign training are intentionally excluded from the repository due to their size.

---

# License

MIT License.
