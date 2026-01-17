from src.preprocessing.preprocess import Preprocessor

class Pipeline:
    def __init__(self):
        self.preprocessor = Preprocessor()

    def run(self, image_path):
        preprocess_result = self.preprocessor.run(image_path)

        ocr_tokens = preprocess_result["ocr"]

        return {
            "status": "ok",
            "num_ocr_tokens": len(ocr_tokens),
            "sample_tokens": ocr_tokens[:10]
        }

