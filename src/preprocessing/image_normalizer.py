"""
Image Normalizer Module
Handles image normalization including resizing and color correction
"""
import cv2
import os

# Resolve paths relative to repo root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DIR = os.path.join(BASE_DIR, "data", "train")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

for filename in os.listdir(RAW_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raw_path = os.path.join(RAW_DIR, filename)

        img = cv2.imread(raw_path)
        if img is None:
            print(f"❌ Could not read {filename}")
            continue

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize (DPI-equivalent)
        img = cv2.resize(
            img,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC
        )

        out_path = os.path.join(PROCESSED_DIR, filename)
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(f"✅ Processed {filename}")

print("🎉 Preprocessing completed.")
