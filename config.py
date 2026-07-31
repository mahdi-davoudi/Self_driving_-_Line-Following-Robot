"""
Central configuration for the FIRA line-following robot.

All tunable parameters (camera size, lane-detection thresholds, steering
gains and smoothing) live here so they can be adjusted without touching
the logic in the other modules.
"""

# --------------------------------------------------------------------------- #
# Serial / robot
# --------------------------------------------------------------------------- #
SERIAL_PORT = '/dev/ttyUSB0'
SERIAL_BAUDRATE = 115200

# Robot control defaults
DEFAULT_ANGLE = 100
DEFAULT_SPEED = 180

# --------------------------------------------------------------------------- #
# Lane detection (Hough transform)
# --------------------------------------------------------------------------- #
CANNY_LOW = 60
CANNY_HIGH = 160
HOUGH_THRESHOLD = 50           # Lowered for speed
MIN_LINE_LENGTH = 40
MAX_LINE_GAP = 60

# --------------------------------------------------------------------------- #
# Steering gain parameters
# --------------------------------------------------------------------------- #
K_MIN = 0.4
K_MAX = 1.6
STEER_CENTER = 100
STEER_MIN = 55
STEER_MAX = 135

# --------------------------------------------------------------------------- #
# Steering smoothing (applied after the steering angle is computed,
# before the command is sent to the Arduino)
# --------------------------------------------------------------------------- #
STEERING_ALPHA = 0.4        # Low-pass filter (EMA) smoothing factor, 0 < alpha <= 1
MAX_STEERING_STEP = 9        # Max degrees the servo angle may change per frame (rate limiter)

# --------------------------------------------------------------------------- #
# Frame size
# --------------------------------------------------------------------------- #
FRAME_WIDTH = 384
FRAME_HEIGHT = 216

# --------------------------------------------------------------------------- #
# ROI (Region Of Interest) definitions, expressed as percentages of the frame
# --------------------------------------------------------------------------- #
# Right lane ROI
ROI_TOP_RL = 0.65
ROI_BOTTOM_RL = 1.0
ROI_LEFT_RL = 0.6
ROI_RIGHT_RL = 0.9

# Left lane ROI
ROI_TOP_LL = 0.65
ROI_BOTTOM_LL = 1.0
ROI_LEFT_LL = 0.0
ROI_RIGHT_LL = 0.4

# AprilTag ROI
ROI_TOP_AT = 0.0
ROI_BOTTOM_AT = 0.5
ROI_LEFT_AT = 0.0
ROI_RIGHT_AT = 1.0

# How often (in frames) to run AprilTag detection - it's expensive, so it
# doesn't need to run on every single frame.
APRILTAG_DETECT_EVERY_N_FRAMES = 6


def build_vision_config() -> dict:
    """Bundle the parameters that VisionProcessor needs into a single dict."""
    return {
        'F_Width': FRAME_WIDTH,
        'F_Height': FRAME_HEIGHT,
        'CANNY_LOW': CANNY_LOW,
        'CANNY_HIGH': CANNY_HIGH,
        'HOUGH_THRESHOLD': HOUGH_THRESHOLD,
        'MIN_LINE_LENGTH': MIN_LINE_LENGTH,
        'MAX_LINE_GAP': MAX_LINE_GAP,
        'ROI_TOP_RL': ROI_TOP_RL,
        'ROI_BOTTOM_RL': ROI_BOTTOM_RL,
        'ROI_LEFT_RL': ROI_LEFT_RL,
        'ROI_RIGHT_RL': ROI_RIGHT_RL,
        'ROI_TOP_LL': ROI_TOP_LL,
        'ROI_BOTTOM_LL': ROI_BOTTOM_LL,
        'ROI_LEFT_LL': ROI_LEFT_LL,
        'ROI_RIGHT_LL': ROI_RIGHT_LL,
        'ROI_TOP_AT': ROI_TOP_AT,
        'ROI_BOTTOM_AT': ROI_BOTTOM_AT,
        'ROI_LEFT_AT': ROI_LEFT_AT,
        'ROI_RIGHT_AT': ROI_RIGHT_AT,
    }
