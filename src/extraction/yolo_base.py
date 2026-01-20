import os
from ultralytics import YOLO


class YOLOBaseDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self.model_available = os.path.exists(model_path)
        if self.model_available:
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
        else:
            self.model = None
            print(f"Warning: Model file {model_path} not found. YOLO detection will be skipped.")

    def predict(self, image):
        """
        image: numpy array (H, W, C)
        returns: list of dicts with bbox, confidence, class_id
        """
        if not self.model_available or self.model is None:
            return []

        results = self.model.predict(source=image, conf=self.conf_threshold, verbose=False)

        detections = []
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(
                    {"class_id": int(box.cls[0]), "confidence": float(box.conf[0]), "bbox": [x1, y1, x2, y2]}
                )

        return detections
