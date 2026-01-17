import cv2
import numpy as np

class ImageNormalizer:
    def run(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adaptive contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(gray)

        # Light denoising (safe for text)
        denoised = cv2.fastNlMeansDenoising(
            enhanced,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21
        )

        # Convert back to 3-channel (OCR expects this)
        final = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

        return final
