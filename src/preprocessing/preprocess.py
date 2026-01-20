import cv2
from .image_normalizer import ImageNormalizer
from .ocr_engine import OCREngine


class Preprocessor:
    def __init__(self):
        self.normalizer = ImageNormalizer()
        self.ocr_engine = OCREngine()

    def run(self, image_path):
        image = cv2.imread(image_path)

        norm_image = self.normalizer.run(image)

        ocr_results = self.ocr_engine.run(norm_image)

        return {"image": norm_image, "ocr": ocr_results}
