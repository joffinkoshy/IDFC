class Candidate:
    def __init__(self, value, bbox, confidence, source):
        self.value = value
        self.bbox = bbox
        self.confidence = confidence
        self.source = source  # e.g. "ocr", "regex", "heuristic"

    def to_dict(self):
        return {
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "bbox": self.bbox
        }
