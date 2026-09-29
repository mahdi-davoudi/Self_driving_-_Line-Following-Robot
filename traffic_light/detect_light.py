import numpy as np
import cv2
from collections import deque, Counter

ROI_TOP_FRAC = 0.1
ROI_BOTTOM_FRAC = 0.35
ROI_LEFT_FRAC = 0.55
ROI_RIGHT_FRAC = 0.8

# --- HSV Color Ranges ---
# Only Red and Green are used - this board has no yellow LED.
RED_LOWER_1 = np.array([0, 70, 170], np.uint8)
RED_UPPER_1 = np.array([12, 255, 255], np.uint8)
RED_LOWER_2 = np.array([168, 70, 170], np.uint8)
RED_UPPER_2 = np.array([180, 255, 255], np.uint8)

GREEN_LOWER = np.array([22, 25, 170], np.uint8)
GREEN_UPPER = np.array([90, 255, 255], np.uint8)

# Brightness (Value channel) verification ---
# predictable across different rooms/backgrounds.
BRIGHTNESS_THRESHOLD = 150          # 0-255 scale

BRIGHTNESS_MODE = "max"             # "mean" or "max" (max gives more headroom)

MIN_CONTOUR_AREA = 80               # lowered - the reliably-colored rim around                                 
MAX_CONTOUR_AREA = 4000            
MIN_CIRCULARITY = 0.55             # 1.0 = perfect circle; rejects fabric/edges/irregular shapes

# Cross-color conflict resolution
CONFLICT_DISTANCE = 40              # px; detections closer than this are treated as the same light

# --- Morphology ---
OPEN_KERNEL = np.ones((3, 3), np.uint8)   # strips thin 1-2px fringing artifacts
KERNEL = np.ones((5, 5), np.uint8)        # then reconnects/grows real regions

STATE_HISTORY_LEN = 8
MIN_VOTES_TO_CONFIRM = 4

# salt-and-pepper sensor noise before it ever reaches the color threshold.
BLUR_KERNEL_SIZE = 3   # must be odd; 0 to disable

DRAW_COLOR = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
}


def build_masks(hsv_frame):
    red_mask = cv2.inRange(hsv_frame, RED_LOWER_1, RED_UPPER_1) | \
               cv2.inRange(hsv_frame, RED_LOWER_2, RED_UPPER_2)

    green_mask = cv2.inRange(hsv_frame, GREEN_LOWER, GREEN_UPPER)

    masks = {
        "Red": red_mask,
        "Green": green_mask,
    }

    for key in masks:
        masks[key] = cv2.erode(masks[key], OPEN_KERNEL)
        masks[key] = cv2.dilate(masks[key], OPEN_KERNEL)
        masks[key] = cv2.dilate(masks[key], KERNEL)

    return masks


def brightness_ok(v_channel, mask, contour):
    contour_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)


    combined_mask = cv2.bitwise_and(contour_mask, mask)

    region_values = v_channel[combined_mask == 255]
    if region_values.size == 0:
        return False, 0.0

    if BRIGHTNESS_MODE == "max":
        brightness = float(np.max(region_values))
    else:
        brightness = float(np.mean(region_values))

    return brightness > BRIGHTNESS_THRESHOLD, brightness


def detect_lights(mask, v_channel, label):
    detections = []

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= MIN_CONTOUR_AREA or area >= MAX_CONTOUR_AREA:
            continue 

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            continue 

        is_bright, brightness = brightness_ok(v_channel, mask, contour)
        if not is_bright:
            continue  

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        detections.append(((int(cx), int(cy)), int(radius), area))

    return detections


def resolve_color_conflicts(red_detections, green_detections):
    kept_red = list(red_detections)
    kept_green = list(green_detections)

    for r_center, r_radius, r_area in list(kept_red):
        for g_center, g_radius, g_area in list(kept_green):
            dist = np.hypot(r_center[0] - g_center[0], r_center[1] - g_center[1])
            if dist < CONFLICT_DISTANCE:
                if r_area >= g_area:
                    if (g_center, g_radius, g_area) in kept_green:
                        kept_green.remove((g_center, g_radius, g_area))
                else:
                    if (r_center, r_radius, r_area) in kept_red:
                        kept_red.remove((r_center, r_radius, r_area))

    return kept_red, kept_green


def draw_detection(frame, center, radius, label):
    color = DRAW_COLOR[label]

    cv2.circle(frame, center, radius, color, 2)
    cv2.circle(frame, center, 3, color, -1)
    text_pos = (center[0] - radius, center[1] - radius - 10)
    cv2.putText(frame, label, text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def offset_to_full_frame(center, left, top):
    return (center[0] + left, center[1] + top)


def main():
    # fix droidcam url if you use that
    webcam =cv2.VideoCapture("http://droidcam url:4747/video")

    state_history = deque(maxlen=STATE_HISTORY_LEN)
    confirmed_state = None

    while True:
        ret, imageFrame = webcam.read()
        if not ret:
            break

        # Convert the fractional ROI into pixel bounds for this frame's
        frame_h, frame_w = imageFrame.shape[:2]
        roi_top = int(ROI_TOP_FRAC * frame_h)
        roi_bottom = int(ROI_BOTTOM_FRAC * frame_h)
        roi_left = int(ROI_LEFT_FRAC * frame_w)
        roi_right = int(ROI_RIGHT_FRAC * frame_w)

        roiFrame = imageFrame[roi_top:roi_bottom, roi_left:roi_right]

        if BLUR_KERNEL_SIZE:
            roiFrame = cv2.medianBlur(roiFrame, BLUR_KERNEL_SIZE)

        hsvFrame = cv2.cvtColor(roiFrame, cv2.COLOR_BGR2HSV)
        v_channel = hsvFrame[:, :, 2]

        masks = build_masks(hsvFrame)

        red_detections = detect_lights(masks["Red"], v_channel, "Red")
        green_detections = detect_lights(masks["Green"], v_channel, "Green")

        red_detections, green_detections = resolve_color_conflicts(
            red_detections, green_detections
        )

        # Pick this frame's single best candidate (largest colored area)
        best_red = max(red_detections, key=lambda d: d[2], default=None)
        best_green = max(green_detections, key=lambda d: d[2], default=None)

        if best_red and (best_green is None or best_red[2] >= best_green[2]):
            current_color, current_detection = "Red", best_red
        elif best_green:
            current_color, current_detection = "Green", best_green
        else:
            current_color, current_detection = None, None

        state_history.append(current_color)
        most_common_color, votes = Counter(state_history).most_common(1)[0]
        if votes >= MIN_VOTES_TO_CONFIRM:
            confirmed_state = most_common_color

   
        if confirmed_state and current_color == confirmed_state:
            center, radius, _ = current_detection
            draw_detection(
                imageFrame,
                offset_to_full_frame(center, roi_left, roi_top),
                radius,
                confirmed_state
            )

        cv2.rectangle(imageFrame, (roi_left, roi_top), (roi_right, roi_bottom),
                      (255, 255, 0), 2)

        cv2.imshow("Traffic Light Detection", imageFrame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()