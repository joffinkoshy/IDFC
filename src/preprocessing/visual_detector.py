import os
import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

MODEL_PATH = os.path.join(BASE_DIR, "models", "signature_stamp.pt")

model = YOLO(MODEL_PATH)

CLASS_MAP = {0: "signature", 1: "stamp"}


def detect_visuals(image_name):
    img_path = os.path.join(PROCESSED_DIR, image_name)
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Image not found: {image_name}")

    results = model(img, conf=0.25)[0]

    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({"type": CLASS_MAP[cls_id], "confidence": conf, "bbox": [x1, y1, x2, y2]})

    return detections
