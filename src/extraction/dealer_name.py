from rapidfuzz import fuzz
from src.utils.text_normalize import normalize_text
import json
from pathlib import Path

class DealerNameResolver:
    def __init__(self, master_path="src/data/dealer_master.json"):
        self.dealers = self._load_master(master_path)

    def _load_master(self, path):
        with open(path, "r") as f:
            return [normalize_text(d) for d in json.load(f)]

    def extract_candidates(self, layout_blocks):
        candidates = []

        for block_id, block in enumerate(layout_blocks):
            # Only header area (top blocks)
            if block_id > 2:
                continue

            for line in block:
                # Convert line (list of tokens) to text string
                text = " ".join(tok["text"] for tok in line)
                normalized_text = normalize_text(text)

                if self._is_valid_candidate(normalized_text):
                    candidates.append((normalized_text, block_id))

        return candidates

    def _is_valid_candidate(self, text):
        if len(text) < 5:
            return False

        reject_keywords = [
            "GST", "EMAIL", "MOB", "DATE",
            "PIN", "PHONE", "FAX"
        ]

        if any(k in text for k in reject_keywords):
            return False

        if sum(c.isdigit() for c in text) > 3:
            return False

        return True

    def resolve(self, layout_blocks):
        candidates = self.extract_candidates(layout_blocks)

        best_match = None
        best_score = 0
        best_block = None

        for candidate, block_id in candidates:
            for dealer in self.dealers:
                score = fuzz.token_sort_ratio(candidate, dealer) / 100

                if score > best_score:
                    best_score = score
                    best_match = dealer
                    best_block = block_id

        if best_score >= 0.90:
            return {
                "dealer_name": best_match,
                "confidence": round(best_score, 2),
                "matched_from": f"header_block_{best_block}"
            }

        return {
            "dealer_name": None,
            "confidence": 0.0,
            "reason": "no_high_confidence_match"
        }
