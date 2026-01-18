from src.layout.geometry import quad_to_rect
from src.layout.line_grouping import group_tokens_into_lines
from src.layout.block_grouping import group_lines_into_blocks
from src.preprocessing.preprocess import Preprocessor
import sys

class Pipeline:

    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor or Preprocessor()

    def run(self, image_path):
        preprocess_result = self.preprocessor.run(image_path)
        tokens = preprocess_result["ocr"]

        # 1. Normalize bounding boxes
        for t in tokens:
            t["rect"] = quad_to_rect(t["bbox"])

        # 2. Group into lines
        lines = group_tokens_into_lines(tokens)

        # 3. Group lines into blocks
        blocks = group_lines_into_blocks(lines)

        # 4. PRINT FOR INSPECTION
        print("\n================ LINES ================\n")
        for i, line in enumerate(lines):
            line_text = " ".join(tok["text"] for tok in line)
            print(f"Line {i}: {line_text}")

        print("\n================ BLOCKS ================\n")
        for b, block in enumerate(blocks):
            print(f"\nBlock {b}:")
            for line in block:
                print("  " + " ".join(tok["text"] for tok in line))

        return {
            "num_tokens": len(tokens),
            "num_lines": len(lines),
            "num_blocks": len(blocks)
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    pipeline = Pipeline()
    pipeline.run(image_path)
