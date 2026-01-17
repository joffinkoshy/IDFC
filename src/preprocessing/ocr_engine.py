from paddleocr import PaddleOCR

class OCREngine:
    def __init__(self):
        self.ocr = PaddleOCR(
            use_textline_orientation=True,  # Updated parameter name
            lang="en"
        )

    def run(self, image):
        raw_results = self.ocr.ocr(image)
        ocr_outputs = []

        if raw_results is None:
            return ocr_outputs

        # Handle new PaddleOCR v5 format
        if isinstance(raw_results, list) and len(raw_results) > 0:
            # Get the first result (single image case)
            result = raw_results[0]

            # Extract text detection results
            if 'rec_texts' in result and 'rec_polys' in result and 'rec_scores' in result:
                rec_texts = result['rec_texts']
                rec_polys = result['rec_polys']
                rec_scores = result['rec_scores']

                # Combine the results
                for text, bbox, conf in zip(rec_texts, rec_polys, rec_scores):
                    if text and len(bbox) > 0:
                        ocr_outputs.append({
                            "text": str(text).strip(),
                            "bbox": bbox.tolist() if hasattr(bbox, 'tolist') else bbox,
                            "confidence": float(conf)
                        })

        return ocr_outputs
