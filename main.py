"""
Entry point. Run with:  python main.py
"""

from vehicle import LaneTrackingVehicle


def main() -> None:
    vehicle = LaneTrackingVehicle()
    try:
        vehicle.run()
    finally:
        vehicle.serial_link.send(b'STOP\r\n')


if __name__ == "__main__":
    main()
