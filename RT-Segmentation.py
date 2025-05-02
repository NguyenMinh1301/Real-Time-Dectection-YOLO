import cv2
from ultralytics import solutions

# === Configuration ===
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# === Open webcam ===
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Webcam not accessible"
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# === Initialize InstanceSegmentation with YOLOv8 model ===
isegment = solutions.InstanceSegmentation(
    show=False,
    model="/model/yolo11n-seg.pt"
    # classes=[0, 2]  # Optional: segment only 'person' and 'car'
)

# === Setup display window ===
cv2.namedWindow("Live - Segmentation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live - Segmentation", WINDOW_WIDTH, WINDOW_HEIGHT)

# === Realtime loop ===
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame.")
        break

    results = isegment(frame)
    output_frame = results.plot_im

    cv2.imshow("Live - Segmentation", output_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
