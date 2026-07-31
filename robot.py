"""
Low-level robot control: serial connection to the Arduino and the
steering/speed controller that turns a raw steering angle into smoothed
servo/speed commands.
"""

import serial

import config


class SerialLink:
    """Thin wrapper around a pyserial connection with duplicate-command
    suppression (never sends the same command twice in a row)."""

    def __init__(self, port: str = config.SERIAL_PORT, baudrate: int = config.SERIAL_BAUDRATE):
        self.connection = None
        self._last_cmd = None
        try:
            self.connection = serial.Serial(port, baudrate)
        except serial.SerialException:
            print("Failed to initialize serial port. Robot control disabled.")
            self.connection = None

    def send(self, command: bytes) -> bool:
        """Send a command to the Arduino, skipping it if identical to the
        last command sent. Returns True if the command was actually sent."""
        if self._last_cmd == command:
            return False
        self._last_cmd = command

        if self.connection is not None and self.connection.is_open:
            self.connection.write(command)
            return True

        print(f"Serial connection not available for command: {command}")
        return False

    def close(self) -> None:
        if self.connection is not None and self.connection.is_open:
            self.connection.close()


class RobotController:
    """Manages robot control (Servo Steering + Dual Motor Drive)."""

    def __init__(self, serial_link: SerialLink):
        self.serial_link = serial_link

        self.current_angle = config.DEFAULT_ANGLE
        self.current_speed = config.DEFAULT_SPEED
        self.angle_old = config.DEFAULT_ANGLE
        self.speed_old = config.DEFAULT_SPEED
        self.filtered_angle = 100          # Previous filtered angle, used by the Low Pass Filter
        self.alpha = config.STEERING_ALPHA  # Low Pass Filter smoothing factor (configurable)

        # Put the robot into a known, stopped state on startup
        self.serial_link.send(b'STOP\r\n')
        self.serial_link.send(f'SPD{self.current_speed}\r\n'.encode())
        self.serial_link.send(f'S{self.current_angle}\r\n'.encode())

    def control(self, steering_angle: float) -> None:
        """Calculates speed based on steering angle and sends servo/speed commands."""

        angle_diff = abs(steering_angle - 107)
        self.current_speed = max(150, 250 - 1.5 * angle_diff)

        # ================= STEERING SMOOTHING PIPELINE =================
        # Raw Steering Angle -> Low Pass Filter -> Rate Limiter -> Clamp to Servo Limits -> Send
        # Lane detection and the steering formula are computed elsewhere;
        # only the command sent to the Arduino is smoothed here.

        # ---- 1) Low Pass Filter (Exponential Moving Average) ----
        # Smooths frame-to-frame detection noise:
        #   filtered_angle = alpha * new_angle + (1 - alpha) * previous_filtered_angle
        self.filtered_angle = self.alpha * steering_angle + (1 - self.alpha) * self.filtered_angle

        # ---- 2) Steering Rate Limiter ----
        # Limits how much the (already filtered) angle can change from the
        # previously sent angle, in either direction, to MAX_STEERING_STEP degrees.
        delta = self.filtered_angle - self.angle_old
        delta = max(-config.MAX_STEERING_STEP, min(config.MAX_STEERING_STEP, delta))
        rate_limited_angle = self.angle_old + delta

        # ---- 3) Clamp to Servo Limits ----
        self.current_angle = int(max(config.STEER_MIN, min(config.STEER_MAX, rate_limited_angle)))

        # ---- 4) Send to Arduino ----
        if self.current_angle != self.angle_old:
            self.serial_link.send(f'S{self.current_angle}\r\n'.encode())
            self.angle_old = self.current_angle

        if self.current_speed != self.speed_old:
            self.serial_link.send(f'SPD{self.current_speed}\r\n'.encode())
            self.speed_old = self.current_speed

    def close(self) -> None:
        self.serial_link.send(b'STOP\r\n')
        self.serial_link.close()
