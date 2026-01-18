import re

KEYWORDS = [
    "tractor", "tractors", "motors", "agency", "agencies",
    "implements", "equipment", "enterprises"
]

EXCLUDE_PATTERNS = [
    r"gst", r"phone", r"mob", r"email", r"date",
    r"quotation", r"invoice", r"bank"
]


class DealerNameResolver:
    def __init__(self, score_threshold=0.6):
        self.score_threshold = score_threshold

    def resolve(self, blocks, page_height):
        candidates = []

        for block in blocks:
            # Current block structure: list of lines, each line is list of tokens
            for line in block:
                # Convert line (list of tokens) to text string
                text = " ".join(tok["text"] for tok in line).strip()

                # Get bbox from first token (approximation)
                bbox = line[0]["bbox"] if line else [[0,0],[1,1],[1,1],[0,0]]
                conf = max(tok.get("confidence", 0.0) for tok in line) if line else 0.0

                if not self._is_candidate(text):
                    continue

                score = self._score_candidate(
                    text=text,
                    bbox=bbox,
                    confidence=conf,
                    page_height=page_height
                )

                candidates.append({
                    "text": text,
                    "score": score
                })

        if not candidates:
            return {
                "dealer_name": None,
                "confidence": 0.0,
                "reason": "no_candidates"
            }

        best = max(candidates, key=lambda x: x["score"])

        if best["score"] < self.score_threshold:
            return {
                "dealer_name": None,
                "confidence": round(best["score"], 2),
                "reason": "low_confidence"
            }

        return {
            "dealer_name": best["text"],
            "confidence": round(best["score"], 2),
            "reason": "heuristic_match"
        }

    def _is_candidate(self, text):
        text_l = text.lower()

        if len(text.split()) < 2:
            return False

        for p in EXCLUDE_PATTERNS:
            if re.search(p, text_l):
                return False

        return True

    def _score_candidate(self, text, bbox, confidence, page_height):
        score = 0.0

        # 1. Position score (35% weight - higher is better)
        y_center = sum(p[1] for p in bbox) / 4
        vertical_ratio = y_center / page_height
        score += max(0, 1.0 - vertical_ratio) * 0.35

        # 2. Keyword score (30% max weight)
        text_l = text.lower()
        keyword_hits = sum(1 for k in KEYWORDS if k in text_l)
        score += min(keyword_hits * 0.15, 0.30)  # Max 30%

        # 3. Capitalization score (15% weight)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        score += caps_ratio * 0.15

        # 4. OCR confidence (20% weight)
        score += confidence * 0.20

        # Guaranteed to be <= 1.0: 0.35 + 0.30 + 0.15 + 0.20 = 1.0
        return round(score, 3)
