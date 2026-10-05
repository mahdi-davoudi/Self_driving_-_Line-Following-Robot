"""
Photo capture tool for building the sign dataset on the Raspberry Pi.

Controls (click the preview window first so it has keyboard focus):
  t -> save the current frame as a JPG into the output folder
  q -> quit

Images are saved WITHOUT any overlay, exactly as the camera produced them,
and with the same manual camera settings used in city.py, so the training
images look like what the robot will see at run time.
File names are millisecond timestamps (e.g. 1783841572549.jpg), so they are
always unique and never collide with images captured earlier.
"""
import os
import time

import cv2

# Folder (next to this script) where photos are stored. Created automatically.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")

# Frame size. 384x216 is what city.py feeds to the vision code.
# Increase (e.g. 640x480) if you want higher-resolution training images.
F_Width = 384
F_Height = 216

JPEG_QUALITY = 95

# Ignore extra 't' presses that arrive faster than this (seconds).
MIN_SAVE_INTERVAL = 0.3

# Draw the sign ROI used in city.py on the PREVIEW only (never on saved images),
# so you can see where the sign has to be in the frame.
SHOW_SIGN_ROI = True
ROI_TOP_SIGN = 0.0
ROI_BOTTOM_SIGN = 0.6
ROI_LEFT_SIGN = 0.6
ROI_RIGHT_SIGN = 1.0


def init_camera():
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (F_Width, F_Height), "format": "RGB888"}
        )
        picam2.configure(config)
        # Same manual settings as city.py
        picam2.set_controls({
            "AeEnable": False,
            "AwbEnable": False,
            "AnalogueGain": 1.1,
            "ExposureTime": 16000,
            "Brightness": 0.0,
            "Contrast": 1.0,
            "Saturation": 1.0,
            "Sharpness": 0.0,
            "NoiseReductionMode": 0,
            "FrameDurationLimits": (50000, 50000),
        })
        picam2.start()
        time.sleep(1)
        print("[INFO] Using Picamera2")
        return "picam2", picam2
    except ImportError:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, F_Width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, F_Height)
        print("[INFO] Picamera2 not found, using cv2.VideoCapture(0)")
        return "cv2", cap


def capture(cam_type, cam):
    if cam_type == "picam2":
        frame = cam.capture_array()
    else:
        ret, frame = cam.read()
        if not ret:
            return None
    return cv2.resize(frame, (F_Width, F_Height), interpolation=cv2.INTER_AREA)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cam_type, cam = init_camera()

    # Continue counting from photos already in the folder
    saved_total = len([f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".jpg")])
    saved_session = 0
    last_save_time = 0.0
    flash_until = 0.0

    print(f"[INFO] Saving to: {OUTPUT_DIR}")
    print(f"[INFO] Photos already in folder: {saved_total}")
    print("Press 't' to take a photo, 'q' to quit.\n")

    while True:
        frame = capture(cam_type, cam)
        if frame is None:
            print("Failed to capture frame")
            break

        # ---- preview (copy, so the saved image stays clean) ----
        preview = frame.copy()
        if SHOW_SIGN_ROI:
            h, w = preview.shape[:2]
            cv2.rectangle(
                preview,
                (int(w * ROI_LEFT_SIGN), int(h * ROI_TOP_SIGN)),
                (int(w * ROI_RIGHT_SIGN), int(h * ROI_BOTTOM_SIGN)),
                (255, 0, 255), 1,
            )
        cv2.putText(preview, f"Saved: {saved_total} (this run: {saved_session})", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if time.time() < flash_until:
            cv2.putText(preview, "SAVED!", (8, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capture - t: save, q: quit", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'):
            now = time.time()
            if now - last_save_time >= MIN_SAVE_INTERVAL:
                filename = f"{int(now * 1000)}.jpg"
                path = os.path.join(OUTPUT_DIR, filename)
                if cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                    saved_total += 1
                    saved_session += 1
                    last_save_time = now
                    flash_until = now + 0.5
                    print(f"[SAVED] {filename}  (total: {saved_total})")
                else:
                    print(f"[ERROR] Could not write {path}")
        elif key == ord('q'):
            break

    if cam_type == "cv2":
        cam.release()
    else:
        cam.stop()
    cv2.destroyAllWindows()
    print(f"Done. {saved_session} photos saved this run, {saved_total} in total.")


if __name__ == "__main__":
    main()
