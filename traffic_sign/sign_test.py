"""
Opens the camera, runs the sign detector on every frame, and prints a short
code to the terminal whenever a sign is detected (only prints on change, so
your terminal doesn't get flooded every single frame).

Codes:
  SS -> STOP
  RR -> TURN RIGHT
  LL -> TURN LEFT
  FF -> STRAIGHT
  PP -> PARK

Controls (with the preview window focused):
  q -> quit
"""
import cv2
import time
import numpy as np

from yolo_detector import TrafficSignDetector
# from svm_detector import TrafficSignDetector   # uncomment to test SVM instead

# Lower this temporarily to see weak/borderline detections that the normal
# 0.2 threshold hides. Set back to 0.2 (or whatever works) once you've
# figured out what's going on.
DEBUG_CONFIDENCE_THRESHOLD = 0.05

# Camera source switch:
#   True  -> laptop + phone: read frames from DroidCam over Wi-Fi (DROIDCAM_URL)
#   False -> Raspberry Pi: use Picamera2 (falls back to cv2.VideoCapture(0))
USE_DROIDCAM = False

# The phone's IP changes whenever it reconnects to Wi-Fi -- update it here.
# Newer DroidCam versions use /video ; older ones use /mjpegfeed
DROIDCAM_URL = "http://10.98.50.94:4747/video"

# Run the model (and print its output) only once every N frames.
# The camera preview still updates on every frame.
RUN_MODEL_EVERY_N_FRAMES = 15

F_Width = 384
F_Height = 216

# Same ROI used in city.py -- widen these if signs fall outside the box
ROI_TOP_SIGN = 0.0
ROI_BOTTOM_SIGN = 0.6
ROI_LEFT_SIGN = 0.6
ROI_RIGHT_SIGN = 1.0

SIGN_CODES = {
    "STOP": "SS",
    "TURN RIGHT": "RR",
    "TURN LEFT": "LL",
    "STRAIGHT": "FF",
    "PARK": "PP",
}


def init_camera():
    if USE_DROIDCAM:
        cap = cv2.VideoCapture(DROIDCAM_URL)
        # Keep only the newest frame so the preview doesn't lag behind the phone
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open DroidCam stream at {DROIDCAM_URL} -- "
                "check the phone IP, that both devices are on the same Wi-Fi, "
                "and that the DroidCam app is running."
            )
        print(f"[INFO] Using DroidCam: {DROIDCAM_URL}")
        return "cv2", cap

    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (F_Width, F_Height), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(1)
        print("[INFO] Using Picamera2")
        return "picam2", picam2
    except ImportError:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, F_Width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, F_Height)
        print("[INFO] Using cv2.VideoCapture(0)")
        return "cv2", cap


def capture(cam_type, cam):
    if cam_type == "picam2":
        frame = cam.capture_array()
    else:
        ret, frame = cam.read()
        if not ret:
            return None
    frame = cv2.resize(frame, (F_Width, F_Height), interpolation=cv2.INTER_AREA)
    return frame


def main():
    cam_type, cam = init_camera()
    detector = TrafficSignDetector()

    last_code = None
    frame_count = 0

    print("Press 'q' (with the preview window focused) to quit.\n")

    while True:
        frame = capture(cam_type, cam)
        if frame is None:
            print("Failed to capture frame")
            break

        height, width = frame.shape[:2]
        top = int(height * ROI_TOP_SIGN)
        bottom = int(height * ROI_BOTTOM_SIGN)
        left = int(width * ROI_LEFT_SIGN)
        right = int(width * ROI_RIGHT_SIGN)

        roi = frame[top:bottom, left:right]
        roi_bgr = roi

        frame_count += 1
        if frame_count % RUN_MODEL_EVERY_N_FRAMES == 0:
            result = detector.process_frame(roi_bgr, confidence_threshold=DEBUG_CONFIDENCE_THRESHOLD)
            text = result["text"]  # "" if nothing recognized

            # Print raw top candidates every frame regardless of threshold,
            # so we can see exactly what the model is leaning towards.
            try:
                img_tensor, r, dw, dh = detector.letterbox(roi_bgr)
                outputs = detector.session.run(None, {detector.input_name: img_tensor})
                preds = np.squeeze(outputs[0], axis=0).T
                candidates = []
                for row in preds:
                    classes_scores = row[4:]
                    class_id = int(np.argmax(classes_scores))
                    conf = float(classes_scores[class_id])
                    if conf > 0.03:
                        candidates.append((class_id, conf))
                candidates.sort(key=lambda x: -x[1])
                if candidates:
                    top = candidates[:3]
                    names = [f"id={c}({detector.SIGNS[c+1] if c+1 < len(detector.SIGNS) else '?'}):{conf:.2f}" for c, conf in top]
                    print("[RAW]", " | ".join(names))
            except Exception as e:
                print("[RAW] error computing raw confidences:", e)

            code = SIGN_CODES.get(text)

            # Only print when the detected sign changes (avoids terminal spam).
            # Change this to "if code:" instead if you want it printed every frame.
            if code != last_code:
                if code:
                    print(f"[SIGN] {code}  ({text})")
                else:
                    print("[SIGN] ---")
                last_code = code

        # Preview window so you can see what the camera sees
        display = frame.copy()
        cv2.imshow("Camera - sign detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if cam_type == "cv2":
        cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()