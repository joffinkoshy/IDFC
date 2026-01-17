"""
OCR Engine Module
Handles OCR processing using PaddleOCR
"""
import os
from paddleocr import PaddleOCR

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# ---------- OCR INITIALIZATION ----------
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'  # later we will add multilingual
)

# ---------- RUN OCR ----------
for filename in os.listdir(PROCESSED_DIR):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(PROCESSED_DIR, filename)

        print("\n==============================")
        print(f"📄 OCR for image: {filename}")
        print("==============================")

        result = ocr.ocr(image_path, cls=True)

        if result is None:
            print("❌ No text detected")
            continue

        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]

            print(f"TEXT: {text} | CONF: {confidence:.2f}")
