# Self_driving Robot

Control code for an autonomous line-following robot, running on a Raspberry Pi + camera, with an Arduino handling motor/servo control.

The robot uses OpenCV to detect the left and right lane lines in two separate regions of interest (ROIs) of the frame, computes a steering angle from how far the lane center is from the frame center, and sends motion commands to the Arduino over a serial connection. AprilTags (e.g. for stopping at a marked point) are detected with ArUco.

## Project structure

```
race/
├── main.py         # Entry point
├── vehicle.py       # Main state machine (WAITING / LINE_FOLLOWING) and control loop
├── vision.py         # Camera (Picamera2 / webcam) + lane and AprilTag detection
├── robot.py          # Serial link to the Arduino + steering/speed control with filtering & smoothing
├── config.py          # All tunable parameters (ROIs, thresholds, gains, etc.)
└── requirements.txt
```

Splitting the code this way keeps each concern (vision, motor control, configuration) separately testable and editable, instead of everything living in one several-hundred-line file.

## Installation

```bash
pip install -r requirements.txt
```

On a Raspberry Pi, `picamera2` needs to be installed separately via apt (it's usually already present on Raspberry Pi OS):

```bash
sudo apt install -y python3-picamera2
```

If `picamera2` isn't available, the code automatically falls back to `cv2.VideoCapture` (a regular webcam) — so it can also be run on a laptop for testing (without real motor control, since no serial device is connected).

## Running

```bash
python main.py
```

## Key settings (`config.py`)

- `ROI_*_RL` / `ROI_*_LL`: right/left lane detection regions (as percentages of frame width/height)
- `CANNY_LOW` / `CANNY_HIGH`: Canny edge detection thresholds
- `K_MIN` / `K_MAX`: adaptive steering gain range, scaled by error magnitude
- `STEERING_ALPHA`: EMA filter coefficient for smoothing frame-to-frame noise
- `MAX_STEERING_STEP`: max servo angle change per frame (rate limiter)
- `APRILTAG_DETECT_EVERY_N_FRAMES`: how often AprilTag detection runs (to save processing time)

## Technical notes

- **Steering pipeline:** the raw angle computed from the detected lines is first smoothed with an EMA (low-pass) filter, then limited by a rate limiter so the servo can't jump suddenly, and finally clamped to the servo's valid range (`STEER_MIN`..`STEER_MAX`).
- **Duplicate command suppression:** `SerialLink.send()` won't resend an identical command back-to-back, which keeps the Arduino's UART buffer from filling up.
- **AprilTag throttling:** since AprilTag detection is heavier than lane detection, it only runs every N frames; the last valid result is kept between detections.

## Hardware warning

This code sends motion commands directly to the motors. Before running on the real robot:
- Test on a safe, obstacle-free surface
- Keep quick access to the STOP command (`s` key)
- Check that `STEER_MIN` / `STEER_MAX` and `current_speed` match your hardware

## License

MIT (or whichever license you choose for the repo)
