from src.extraction.yolo_base import YOLOBaseDetector


class DealerSignatureExtractor:

    def __init__(self, model_path: str):
        self.detector = YOLOBaseDetector(model_path)

    def extract(self, image):
        detections = self.detector.predict(image)

        signatures = [d for d in detections if d["class_id"] == 0]

        if not signatures:
            return {"present": False, "bbox": None, "confidence": 0.0}

        best = max(signatures, key=lambda x: x["confidence"])

        return {"present": True, "bbox": best["bbox"], "confidence": round(best["confidence"], 3)}
