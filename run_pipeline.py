import sys
import json
import cv2

from src.preprocessing.preprocess import Preprocessor
from src.extraction.dealer_name import DealerNameResolver
from src.extraction.model_name import ModelNameResolver
from src.extraction.stamp import DealerStampExtractor
from src.extraction.signature import DealerSignatureExtractor

from src.layout.line_grouping import group_tokens_into_lines
from src.layout.block_grouping import group_lines_into_blocks
from src.layout.geometry import quad_to_rect


def main(image_path):
    # ---------- Init ----------
    preprocessor = Preprocessor()
    dealer_resolver = DealerNameResolver()
    model_resolver = ModelNameResolver()

    # 🔹 YOLO extractors (update model paths if needed)
    stamp_extractor = DealerStampExtractor(
        model_path="models/stamp_yolo.pt"
    )
    signature_extractor = DealerSignatureExtractor(
        model_path="models/signature_yolo.pt"
    )

    # ---------- Load image once ----------
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_height = image.shape[0]

    # ---------- Step 1: Preprocess (OCR etc.) ----------
    result = preprocessor.run(image_path)
    ocr_tokens = result["ocr"]

    # ---------- Step 2: Layout processing ----------
    for t in ocr_tokens:
        t["rect"] = quad_to_rect(t["bbox"])

    lines = group_tokens_into_lines(ocr_tokens)
    blocks = group_lines_into_blocks(lines)

    # ---------- Step 3: Dealer name ----------
    dealer_result = dealer_resolver.resolve(blocks, image_height)

    # ---------- Step 4: Model name ----------
    model_result = model_resolver.resolve(blocks, image_height)

    # ---------- Step 5: Stamp & Signature (YOLO) ----------
    stamp_result = stamp_extractor.extract(image)
    signature_result = signature_extractor.extract(image)

    # ---------- Final output ----------
    output = {
        "status": "ok",
        "image": image_path,

        "num_ocr_tokens": len(ocr_tokens),
        "num_lines": len(lines),
        "num_blocks": len(blocks),

        "dealer_name_result": dealer_result,
        "model_name_result": model_result,

        "stamp_result": stamp_result,
        "signature_result": signature_result
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <image_path>")
        sys.exit(1)

    main(sys.argv[1])
