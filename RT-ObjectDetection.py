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

# === Define a detection region in the center of the frame ===
region_width = 300
region_height = 300
center_x, center_y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
region_01 = [
    (center_x - region_width // 2, center_y - region_height // 2),
    (center_x + region_width // 2, center_y - region_height // 2),
    (center_x + region_width // 2, center_y + region_height // 2),
    (center_x - region_width // 2, center_y + region_height // 2),
]
region_points = {"region-01": region_01}

# === Initialize RegionCounter with YOLOv8 model ===
regioncounter = solutions.RegionCounter(
    show=False,
    region=region_points,
    model="/model/yolo11n.pt"
)

# === Setup display window ===
cv2.namedWindow("Live - Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live - Object Detection", WINDOW_WIDTH, WINDOW_HEIGHT)

# === Realtime loop ===
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame.")
        break

    results = regioncounter(frame)
    output_frame = results.plot_im

    cv2.imshow("Live - Object Detection", output_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
