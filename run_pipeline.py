import json
import sys
from src.pipeline import Pipeline

if __name__ == "__main__":
    image_path = sys.argv[1]

    pipeline = Pipeline()
    output = pipeline.run(image_path)

    print(json.dumps(output, indent=2))
