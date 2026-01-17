"""
Image Normalizer Module
Handles image normalization including resizing and color correction
"""
import cv2
import os

RAW_DIR = r"C:\Users\HP\PycharmProjects\IDFC\data\raw"
PROCESSED_DIR = r"C:\Users\HP\PycharmProjects\IDFC\data\processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

for filename in os.listdir(RAW_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raw_path = os.path.join(RAW_DIR, filename)

        img = cv2.imread(raw_path)
        if img is None:
            print(f"❌ Could not read {filename}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(
            img, None,
            fx=1.5, fy=1.5,
            interpolation=cv2.INTER_CUBIC
        )

        out_path = os.path.join(PROCESSED_DIR, filename)
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(f"✅ Processed {filename}")

print("🎉 Done!")
