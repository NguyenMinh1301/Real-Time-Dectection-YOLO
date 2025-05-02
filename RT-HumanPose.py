import cv2
from ultralytics import YOLO

# === Configuration ===
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# === Load YOLO pose estimation model ===
model = YOLO("/model/yolo11n-pose.pt")

# === Open webcam ===
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Webcam not accessible"
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# === Setup display window ===
cv2.namedWindow("Live - Human Pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live - Human Pose", WINDOW_WIDTH, WINDOW_HEIGHT)

# === Realtime loop ===
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame.")
        break

    results = model.predict(frame, show=False, conf=0.5, verbose=False)
    output_frame = results[0].plot()

    cv2.imshow("Live - Human Pose", output_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
