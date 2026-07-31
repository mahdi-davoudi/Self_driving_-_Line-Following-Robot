"""
Vision pipeline: camera capture (Picamera2 or generic webcam) and the
lane / AprilTag detection logic that turns a raw frame into a steering
angle.
"""

import math
import time
from typing import Dict, Optional, Tuple

import cv2
import cv2.aruco as aruco
import numpy as np

import config


class CameraConfig:
    """Handles camera initialization and frame capture."""

    def __init__(self, width: int = config.FRAME_WIDTH, height: int = config.FRAME_HEIGHT):
        self.width = width
        self.height = height
        self.use_rpi_cam = False
        self.cap = None
        self.picam2 = None
        self._initialize_camera()

    def _initialize_camera(self) -> None:
        try:
            from picamera2 import Picamera2
            self.use_rpi_cam = True
            self.picam2 = Picamera2()
            cam_config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(cam_config)
            self.picam2.set_controls({
                "AeEnable": False,
                "AwbEnable": False,
                "AnalogueGain": 1.0,
                "ExposureTime": 35000,
                "Brightness": 0.4,
                "Contrast": 1.0,
                "Saturation": 1.0,
                "Sharpness": 0.0,
                "NoiseReductionMode": 0,
                "FrameDurationLimits": (50000, 50000),
            })
            self.picam2.start()
            time.sleep(1)
        except ImportError:
            self.use_rpi_cam = False
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def capture_frame(self) -> Optional[np.ndarray]:
        if self.use_rpi_cam:
            frame = self.picam2.capture_array()
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None
        # Ensure frame is resized to configured size
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

    def release(self) -> None:
        if not self.use_rpi_cam and self.cap:
            self.cap.release()


class VisionProcessor:
    """Handles lane detection and AprilTag detection."""

    def __init__(self, vision_config: dict):
        self.config = vision_config
        self.last_steering_angle = 100

        # AprilTag / ArUco detector setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.last_center_left_x = None
        self.last_center_right_x = None

        # AprilTag detection is run every N frames only (it's expensive);
        # these hold state between detections.
        self.frame_counter = 0
        self.last_ids_list = []

    def detect_lines(self, frame: np.ndarray) -> Tuple[float, np.ndarray, Dict[str, float]]:
        height, width = frame.shape[:2]
        frame_center = width // 2
        error_max = frame_center  # max possible pixel error

        steering_angle = self.last_steering_angle
        line_center_x = frame_center
        target_x = frame_center
        error = 0
        right_lines_info = []
        left_lines_info = []
        longest_line_length = 0
        lane_type = "none"

        # Right lane ROI
        top_rl = int(height * self.config['ROI_TOP_RL'])
        bottom_rl = int(height * self.config['ROI_BOTTOM_RL'])
        left_rl = int(width * self.config['ROI_LEFT_RL'])
        right_rl = int(width * self.config['ROI_RIGHT_RL'])
        roi_frame_rl = frame[top_rl:bottom_rl, left_rl:right_rl]

        # Left lane ROI
        top_ll = int(height * self.config['ROI_TOP_LL'])
        bottom_ll = int(height * self.config['ROI_BOTTOM_LL'])
        left_ll = int(width * self.config['ROI_LEFT_LL'])
        right_ll = int(width * self.config['ROI_RIGHT_LL'])
        roi_frame_ll = frame[top_ll:bottom_ll, left_ll:right_ll]

        # ---- Process right lane ----
        gray_rl = cv2.cvtColor(roi_frame_rl, cv2.COLOR_RGB2GRAY)
        edge_rl = cv2.Canny(gray_rl, self.config['CANNY_LOW'], self.config['CANNY_HIGH'])
        edge_dilated_rl = cv2.dilate(edge_rl, np.ones((3, 3), np.uint8), iterations=1)
        lines_rl = cv2.HoughLinesP(edge_dilated_rl, 1, np.pi / 180, self.config['HOUGH_THRESHOLD'],
                                    minLineLength=self.config['MIN_LINE_LENGTH'],
                                    maxLineGap=self.config['MAX_LINE_GAP'])

        center_right_x = right_rl + 10
        if lines_rl is not None:
            right_lines = []
            roi_height_rl = roi_frame_rl.shape[0]
            for line in lines_rl:
                x1, y1, x2, y2 = line.flatten()
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    if abs(slope) > 0.4:
                        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        x_bottom = x1 + (roi_height_rl - y1) / slope if abs(slope) > 1e-5 else x1
                        x_bottom = max(0, min(right_rl - left_rl - 1, x_bottom))
                        x_bottom_full = x_bottom + left_rl
                        if x_bottom_full >= frame_center:
                            right_lines.append((x1, y1, x2, y2, x_bottom, length))
                            right_lines_info.append((x1, y1, x2, y2, length))

            if right_lines:
                x_bottom_sum = sum(line[4] + left_rl for line in right_lines)
                center_right_x = x_bottom_sum / len(right_lines)
                longest_line_length = max(longest_line_length, int(max(line[5] for line in right_lines)))

        # ---- Process left lane ----
        gray_ll = cv2.cvtColor(roi_frame_ll, cv2.COLOR_RGB2GRAY)
        edge_ll = cv2.Canny(gray_ll, self.config['CANNY_LOW'], self.config['CANNY_HIGH'])
        edge_dilated_ll = cv2.dilate(edge_ll, np.ones((3, 3), np.uint8), iterations=1)
        lines_ll = cv2.HoughLinesP(edge_dilated_ll, 1, np.pi / 180, self.config['HOUGH_THRESHOLD'],
                                    minLineLength=self.config['MIN_LINE_LENGTH'],
                                    maxLineGap=self.config['MAX_LINE_GAP'])

        center_left_x = left_ll - 0
        if lines_ll is not None:
            left_lines = []
            roi_height_ll = roi_frame_ll.shape[0]
            for line in lines_ll:
                x1, y1, x2, y2 = line.flatten()
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    if abs(slope) > 0.3:
                        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        x_bottom = x1 + (roi_height_ll - y1) / slope if abs(slope) > 1e-5 else x1
                        x_bottom = max(0, min(right_ll - left_ll - 1, x_bottom))
                        x_bottom_full = x_bottom + left_ll
                        if x_bottom_full <= frame_center:
                            left_lines.append((x1, y1, x2, y2, x_bottom, length))
                            left_lines_info.append((x1, y1, x2, y2, length))

            if left_lines:
                x_bottom_sum = sum(line[4] + left_ll for line in left_lines)
                center_left_x = x_bottom_sum / len(left_lines)
                longest_line_length = max(longest_line_length, int(max(line[5] for line in left_lines)))

        # ---- AprilTag detection (throttled to every N frames) ----
        gray_at = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.frame_counter += 1

        ids_list = self.last_ids_list
        if self.frame_counter % config.APRILTAG_DETECT_EVERY_N_FRAMES == 0:
            corners, ids, rejected = self.detector.detectMarkers(gray_at)
            ids_list = ids.flatten().tolist() if ids is not None else []
            self.last_ids_list = ids_list

        # Placeholders: traffic-light color and AprilTag summary info are
        # not computed in this build (kept for API compatibility with the
        # rest of the pipeline / dashboard).
        tl_color = "None"
        april_tags_info = {'count': 0, 'ids': [], 'areas': []}

        # ---- Steering logic ----
        right_detected = len(right_lines_info) > 0
        left_detected = len(left_lines_info) > 0
        lane_type = "both" if right_detected and left_detected else "right" if right_detected else "left" if left_detected else "none"
        line_center_x = (center_right_x + center_left_x) / 2
        target_x = frame_center
        error = line_center_x - target_x

        if not right_detected and not left_detected:
            steering_angle = 125
        elif not right_detected:
            steering_angle = 125
        else:
            # Adaptive Gain based on error magnitude
            error_norm = min(abs(error) / error_max, 1.0)
            K = config.K_MIN + (config.K_MAX - config.K_MIN) * error_norm
            steering_angle = config.STEER_CENTER + K * error
            steering_angle = max(config.STEER_MIN, min(config.STEER_MAX, steering_angle))

        self.last_steering_angle = steering_angle

        # ---- Draw visualization ----
        lanes_frame = frame.copy()
        cv2.rectangle(lanes_frame, (left_rl, top_rl), (right_rl, bottom_rl), (0, 255, 255), 2)
        cv2.putText(lanes_frame, "RL", (left_rl + 5, top_rl + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.rectangle(lanes_frame, (left_ll, top_ll), (right_ll, bottom_ll), (255, 255, 0), 2)
        cv2.putText(lanes_frame, "LL", (left_ll + 5, top_ll + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if right_detected:
            cv2.line(lanes_frame, (int(center_right_x), top_rl), (int(center_right_x), bottom_rl), (255, 0, 0), 2)
        else:
            cv2.line(lanes_frame, (int(center_right_x), top_rl), (int(center_right_x), bottom_rl), (255, 0, 0), 1,
                     cv2.LINE_AA)
        if left_detected:
            cv2.line(lanes_frame, (int(center_left_x), top_ll), (int(center_left_x), bottom_ll), (0, 0, 255), 2)
        else:
            cv2.line(lanes_frame, (int(center_left_x), top_ll), (int(center_left_x), bottom_ll), (0, 0, 255), 1,
                     cv2.LINE_AA)

        cv2.putText(lanes_frame, f"R Lines: {len(right_lines_info)}", (left_rl + 5, top_rl + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(lanes_frame, f"L Lines: {len(left_lines_info)}", (left_ll + 5, top_ll + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(lanes_frame, f"Len: {longest_line_length} px", (left_rl + 5, top_rl + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(lanes_frame, f"Lane: {lane_type}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        line_info = {
            'line_center_x': line_center_x,
            'target_x': target_x,
            'error': error,
            'lane_type': lane_type,
            'steering_angle': steering_angle,
            'speed': getattr(self, 'current_speed', 100),
            'center_right_x': center_right_x,
            'center_left_x': center_left_x,
            'tl_color': tl_color,
            'april_tags_count': april_tags_info['count'],
            'april_tags_areas': april_tags_info['areas'],
            'april_tags_ids': ids_list,
        }
        return steering_angle, lanes_frame, line_info
