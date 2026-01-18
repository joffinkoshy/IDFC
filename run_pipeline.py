import sys
import json
from src.preprocessing.preprocess import Preprocessor
from src.extraction.dealer_name import DealerNameResolver
from src.extraction.model_name import ModelNameResolver
from src.layout.line_grouping import group_tokens_into_lines
from src.layout.block_grouping import group_lines_into_blocks
from src.layout.geometry import quad_to_rect

def main(image_path):
    preprocessor = Preprocessor()
    dealer_resolver = DealerNameResolver()
    model_resolver = ModelNameResolver()

    # Step 1: Preprocess
    result = preprocessor.run(image_path)

    ocr_tokens = result["ocr"]

    # Step 2: Layout processing (normalize bboxes and create blocks)
    for t in ocr_tokens:
        t["rect"] = quad_to_rect(t["bbox"])

    lines = group_tokens_into_lines(ocr_tokens)
    blocks = group_lines_into_blocks(lines)

    # Step 3: Dealer name extraction
    # Get page height from the image
    image_height = result["image"].shape[0] if result["image"] is not None else 1000
    dealer_result = dealer_resolver.resolve(blocks, image_height)

    # Step 4: Model name extraction
    model_result = model_resolver.resolve(blocks, image_height)

    # Step 5: Final debug-friendly output
    output = {
        "status": "ok",
        "image": image_path,
        "num_ocr_tokens": len(ocr_tokens),
        "num_lines": len(lines),
        "num_blocks": len(blocks),
        "dealer_name_result": dealer_result,
        "model_name_result": model_result
    }

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <image_path>")
        sys.exit(1)

    main(sys.argv[1])
