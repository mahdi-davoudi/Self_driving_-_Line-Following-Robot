"""
Top-level state machine: ties the camera, vision pipeline and robot
controller together and runs the main control loop.
"""

from typing import Dict

import cv2
import numpy as np

import config
from robot import RobotController, SerialLink
from vision import CameraConfig, VisionProcessor


class LaneTrackingVehicle:
    STATE_WAITING = "WAITING"
    STATE_LINE_FOLLOWING = "LINE_FOLLOWING"

    def __init__(self):
        self.vision_config = config.build_vision_config()

        self.camera = CameraConfig(self.vision_config['F_Width'], self.vision_config['F_Height'])
        self.serial_link = SerialLink()
        self.robot = RobotController(self.serial_link)
        self.vision = VisionProcessor(self.vision_config)

        self.state = self.STATE_WAITING
        self.move_status = False
        self.display_enabled = True
        self.tag5_detected = False

    def _check_gpio(self) -> bool:
        # Placeholder for GPIO start-button logic (disabled on this build).
        return True

    def run(self) -> None:
        while True:
            frame = self.camera.capture_frame()
            if frame is None:
                print("Failed to capture image")
                break

            steering_angle, lanes_frame, line_info = self.vision.detect_lines(frame)

            if self.state == self.STATE_WAITING:
                self._display_frame(lanes_frame, line_info, 0)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('g') or self._check_gpio():
                    self.state = self.STATE_LINE_FOLLOWING
                    self.move_status = True
                    print("Starting robot (G pressed or GPIO high)")
                elif key == ord('q'):
                    break

            elif self.state == self.STATE_LINE_FOLLOWING:
                if self.move_status:
                    if 5 in line_info['april_tags_ids']:
                        if not self.tag5_detected:
                            self.serial_link.send(b'STOP\r\n')
                            print("[TAG 5] Detected -> STOP")
                            self.tag5_detected = True
                    else:
                        if self.tag5_detected:
                            print("[TAG 5] Lost -> Resume")
                        self.tag5_detected = False

                        self.serial_link.send(b'F\r\n')
                        self.robot.control(steering_angle)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('h'):
                self.serial_link.send(b'STOP\r\n')
                self.state = self.STATE_WAITING
                self.move_status = False
                print("Stopping robot (H pressed or GPIO low)")
            elif key == ord('q'):
                break
            elif key == ord('f'):
                self.serial_link.send(b'F\r\n')
            elif key == ord('s'):
                self.serial_link.send(b'STOP\r\n')

        self._cleanup()

    def _display_frame(self, frame: np.ndarray, line_info: Dict[str, float], fps: float) -> None:
        display_frame = frame.copy()
        if line_info:
            cv2.line(display_frame, (int(line_info['line_center_x']), 0),
                     (int(line_info['line_center_x']), frame.shape[0]), (0, 0, 255), 2)
            cv2.line(display_frame, (int(line_info['target_x']), 0), (int(line_info['target_x']), frame.shape[0]),
                     (0, 255, 0), 2)

            cv2.putText(display_frame, f"Error: {line_info['error']:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            cv2.putText(display_frame, f"Steering: {line_info['steering_angle']:.2f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Speed: {line_info['speed']:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Lane: {line_info['lane_type']}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

    def _cleanup(self) -> None:
        self.camera.release()
        self.robot.close()
        cv2.destroyAllWindows()
