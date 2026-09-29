import cv2

# Click on any pixel in the video window to print its H, S, V values
# in the terminal. Use this to find the ACTUAL HSV range of your

webcam = cv2.VideoCapture("http://10.98.50.94:4747/video")
current_hsv_frame = None


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and current_hsv_frame is not None:
        h, s, v = current_hsv_frame[y, x]
        print(f"Pixel ({x},{y}) -> H={h}  S={s}  V={v}")


cv2.namedWindow("HSV Probe")
cv2.setMouseCallback("HSV Probe", on_mouse)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    current_hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV Probe", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()