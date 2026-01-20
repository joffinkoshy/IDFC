from paddleocr import PaddleOCR


class OCREngine:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")  # SAFE & STABLE

    def run(self, image):
        """
        image: numpy array (H, W, C)
        returns: list of dicts with text, bbox, confidence
        """
        raw_results = self.ocr.ocr(image)

        ocr_outputs = []

        if not raw_results:
            return ocr_outputs

        # Single image case → raw_results[0]
        for line in raw_results[0]:
            bbox = line[0]  # 4-point polygon
            text, conf = line[1]  # (text, confidence)

            if text and bbox:
                ocr_outputs.append({"text": text.strip(), "bbox": bbox, "confidence": float(conf)})

        return ocr_outputs
