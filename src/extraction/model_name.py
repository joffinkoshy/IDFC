import re

# Model-specific keywords
MODEL_KEYWORDS = [
    "model", "hp", "h.p", "cyl", "tractor",
    "xt", "di", "fe", "eco", "dx", "sx", "wd"
]

# OCR cleanup
OCR_CORRECTIONS = {
    "D I": "DI",
    " F E ": " FE ",
    " X T ": " XT ",
    "M F": "MF",
    "J D": "JD",
}

# Known brands
BRANDS = [
    "swaraj", "mahindra", "john deere",
    "sonalika", "eicher", "escorts",
    "new holland", "kubota",
    "massey ferguson", "mf", "jd"
]


class ModelNameResolver:
    def __init__(self, score_threshold=0.5):
        self.score_threshold = score_threshold

    def resolve(self, blocks, page_height):
        candidates = self._generate_candidates(blocks, page_height)

        if not candidates:
            return {
                "model_name": None,
                "confidence": 0.0,
                "reason": "no_candidates"
            }

        scored = []
        for c in candidates:
            score = self._score_candidate(
                text=c["text"],
                bbox=c["bbox"],
                confidence=c["confidence"],
                page_height=page_height,
                block_id=c["block_id"],
                line_id=c["line_id"],
                raw_line=c["raw_line"],
                blocks=blocks
            )
            scored.append({**c, "score": score})

        best = max(scored, key=lambda x: x["score"])

        if best["score"] < self.score_threshold:
            return {
                "model_name": None,
                "confidence": round(best["score"], 2),
                "reason": "low_confidence"
            }

        return {
            "model_name": best["text"],
            "confidence": round(best["score"], 2),
            "reason": "heuristic_match",
            "original_text": best["raw_line"]
        }

    # ------------------------------------------------------------------

    def _generate_candidates(self, blocks, page_height):
        candidates = []

        for block_id, block in enumerate(blocks):
            is_table = self._is_table_like_block(block)

            for line_id, line in enumerate(block):
                raw_text = " ".join(tok["text"] for tok in line).strip()
                if len(raw_text) < 5:
                    continue

                if self._is_excluded_line(raw_text):
                    continue

                # Try extracting from table rows
                span = self._extract_model_from_table_row(raw_text) if is_table else raw_text
                core = self._extract_model_core(span)

                if not core:
                    continue

                bbox = line[0]["bbox"]
                conf = max(tok.get("confidence", 0.0) for tok in line)

                candidates.append({
                    "text": core,
                    "raw_line": raw_text,
                    "bbox": bbox,
                    "confidence": conf,
                    "block_id": block_id,
                    "line_id": line_id
                })

        return candidates

    # ------------------------------------------------------------------

    def _extract_model_from_table_row(self, text):
        text = text.upper()

        patterns = [
            r'\b(?:SWARAJ|MAHINDRA|MF|JD)\s*\d{3,4}\s*[A-Z]{1,3}\b',
            r'\b\d{3,4}\s*[A-Z]{1,3}\b'
        ]

        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(0)

        return None

    # ------------------------------------------------------------------

    def _extract_model_core(self, text):
        if not text:
            return None

        text = text.upper()

        # Kill obvious non-model words
        blacklist = [
            "ADDRESS", "IFSC", "BANK", "DATE", "FOR",
            "TOTAL", "AMOUNT", "HDFC", "GST"
        ]
        if any(b in text for b in blacklist):
            return None

        # Remove config noise
        text = re.sub(
            r'\b(HP|WD|CYLINDER|CATG|TRACTOR|PTO|TYRE|SIZE|X)\b',
            '',
            text
        )

        text = re.sub(r'\s+', ' ', text).strip()

        # Reject if no digits after cleanup
        if not re.search(r'\d', text):
            return None

        # Canonical formatting - remove brand prefixes
        text = re.sub(r'^(SWARAJ|MAHINDRA|MF|JD)\s+', '', text)

        # STRICT model core patterns
        patterns = [
            r'\b(?:MF|SWARAJ|MAHINDRA|JD)?\s*\d{3,4}\s*[A-Z]{1,3}\b'
        ]

        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(0).strip()

        return None

    # ------------------------------------------------------------------

    def _is_excluded_line(self, text):
        text_l = text.lower()
        blacklist = [
            "gst", "invoice", "quotation", "total",
            "amount", "price", "bank", "signature",
            "customer", "party", "terms"
        ]
        return any(b in text_l for b in blacklist)

    # ------------------------------------------------------------------

    def _is_table_like_block(self, block):
        if len(block) < 2:
            return False

        xs = []
        for line in block:
            for tok in line:
                x = sum(p[0] for p in tok["bbox"]) / 4
                xs.append(int(x // 50))

        return len(set(xs)) >= 2

    # ------------------------------------------------------------------

    def _score_candidate(self, text, bbox, confidence, page_height, block_id, line_id, raw_line, blocks):
        score = 0.0

        # Table context
        if self._is_table_like_block(blocks[block_id]):
            score += 0.30

        # Strong boost if extracted from table row
        if self._extract_model_from_table_row(raw_line):
            score += 0.15

        # Alphanumeric density
        density = sum(c.isalnum() for c in text) / max(len(text), 1)
        score += min(density, 1.0) * 0.25

        # Position (middle of page)
        y = sum(p[1] for p in bbox) / 4
        vr = y / page_height
        score += max(0, 1 - abs(vr - 0.5) * 2) * 0.25

        # OCR confidence
        score += min(confidence, 0.85) * 0.20

        return round(min(score, 1.0), 3)
