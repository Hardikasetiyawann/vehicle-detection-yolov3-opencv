# Vehicle Detection and Counting using YOLOv3 and OpenCV

This project implements a **real-time vehicle detection and counting system**
using **YOLOv3 (You Only Look Once)** and **OpenCV**.
The system detects vehicles from a webcam or video input and displays
the total number of detected vehicles in real time.

This project is developed as part of a **Computer Vision / Machine Learning** task
and demonstrates the application of deep learning for object detection.

---

## 🚗 Features
- Real-time object detection using YOLOv3
- Vehicle detection for the following classes:
  - Car
  - Bus
  - Truck
  - Motorbike
- Vehicle counting based on detected objects
- Bounding box visualization with class labels
- Webcam-based detection (can be extended to video files)

---

## 🧠 Technologies Used
- **Python**
- **OpenCV (cv2)**
- **NumPy**
- **YOLOv3 (Darknet)**
- Pre-trained **COCO dataset**

---

## 📂 Project Structure
```bash
vehicle-detection-yolov3-opencv/
├── main.py
├── yolov3.cfg
├── coco.names
├── requirements.txt
├── README.md
└── yolov3.weights   # downloaded separately (not included in repository)
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Hardikasetiyawann/vehicle-detection-yolov3-opencv.git
cd vehicle-detection-yolov3-opencv
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Download YOLOv3 Files

Download the following files and place them in the project directory:

* **yolov3.weights**
  [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
* **yolov3.cfg**
  [https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
* **coco.names**
  [https://github.com/pjreddie/darknet/blob/master/data/coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

---

## ▶️ How to Run

```bash
python main.py
```

* The webcam will start automatically
* Detected vehicles will be shown with bounding boxes
* Vehicle count will be displayed on the screen
* Press **Q** to exit

---

## 🎥 Using Video File (Optional)

To use a video file instead of a webcam, modify this line in `main.py`:

```python
cap = cv2.VideoCapture("video.mp4")
```

---

## 📌 Notes

* Make sure your camera is connected properly
* If the webcam does not open, try changing the camera index:

```python
cap = cv2.VideoCapture(1)
```

* This project uses a **pre-trained YOLOv3 model** and does not require model training

---

## 📚 References

* YOLO: [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)
* OpenCV: [https://opencv.org/](https://opencv.org/)
* COCO Dataset: [https://cocodataset.org/](https://cocodataset.org/)

---

## 👤 Author

**Hardika Setiyawan**
Informatics Student | Computer Vision & Machine Learning

---

## 📄 License

This project is intended for **educational purposes only**.

```