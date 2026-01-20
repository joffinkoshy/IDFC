from src.extraction.yolo_base import YOLOBaseDetector


class DealerStampExtractor:
    """
    YOLO class_id = 1 → dealer stamp
    """

    def __init__(self, model_path: str):
        self.detector = YOLOBaseDetector(model_path)

    def extract(self, image):
        detections = self.detector.predict(image)

        stamps = [d for d in detections if d["class_id"] == 1]

        if not stamps:
            return {"present": False, "bbox": None, "confidence": 0.0}

        best = max(stamps, key=lambda x: x["confidence"])

        return {"present": True, "bbox": best["bbox"], "confidence": round(best["confidence"], 3)}
