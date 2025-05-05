import cv2
from ultralytics import solutions

# === Webcam configuration ===
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Cannot open webcam"
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# === Initialize AIGym with YOLO pose model ===
gym = solutions.AIGym(
    show=False,
    kpts=[6, 8, 10],  # Use [6, 8, 10] for push-up (left shoulder–elbow–wrist)
    model="/model/yolo11n-pose.pt"
)

# === Create display window ===
cv2.namedWindow("Live Workout Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Workout Tracker", WINDOW_WIDTH, WINDOW_HEIGHT)

# === Realtime processing loop ===
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from webcam.")
        break

    results = gym(frame)
    output_frame = results.plot_im

    # Optional: show live rep count on screen
    count = results.workout_count[0] if results.workout_count else 0
    cv2.putText(
        output_frame,
        f"Reps: {count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )

    cv2.imshow("Live Workout Tracker", output_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
