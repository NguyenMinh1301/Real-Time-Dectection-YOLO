# Real-Time-Dectection

> A lightweight collection of three real‑time computer‑vision tools—Object Detection, Instance Segmentation, and Human‑Pose Estimation—powered by Ultralytics YOLO models.

<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

## ✨ Features

| Module                    | What it does                             | Highlights                              |
| ------------------------- | ---------------------------------------- | --------------------------------------- |
| **RealTime‑ObjectDetect** | Counts objects inside a user‑defined ROI | Polygon region, live FPS on laptop GPUs |
| **RealTime‑Segmentation** | Generates instance masks live            | Optional class filtering                |
| **RealTime‑HumanPose**    | Draws a 17‑keypoint skeleton             | Full‑body pose estimation on‑the‑fly    |

### Common perks:
- Plug‑and‑play YOLO 11/8 .pt models (swap files in models/)
- Pure Python + OpenCV—no heavy GUI dependencies
- Identical CLI usage pattern for every script
- Concise, fully commented code—ready for hacking

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenCV
- Ultralytics (latest version)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/NguyenMinh1301/Real-Time-Dectection.git
cd Real-Time-Dectection
```

2. Create virtualenv:
```bash
python -m venv venv
```
```bash
.\venv\Scripts\activate 
```

3. Install dependencies:

```bash
pip install ultralytics opencv-python
```

### Running the Object Counter

#### Region‑based object detection & counting

```bash
python RT-ObjectDetection.py
```

#### Human‑pose estimation

```bash
python RT-HumanPose.py
```

#### Instance segmentation

```bash
python RT-Segmentation.py
```

Press q in any window to quit.

### Swap to a different model
#### Open the script and change the path, e.g.

```bash
model = "models/yolo11x.pt"          # detection
model = "models/yolo11n-seg.pt"      # segmentation
model = "models/yolo11n-pose.pt"     # pose
```

### Filter classes (segmentation)
```bash
isegment = solutions.InstanceSegmentation(
    show=False,
    model="models/yolo11n-seg.pt",
    classes=[0, 2]      # 0 = person, 2 = car
)
```

### Select another webcam / resolution

```bash
cap = cv2.VideoCapture(1)            # 0 = default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📬 Contact

- **Email**: nguyenminh1301.dev@gmail.com
- **GitHub**: [NguyenMinh1301](https://github.com/NguyenMinh1301)