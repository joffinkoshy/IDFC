import re


HP_KEYWORDS = [
    "hp", "h.p", "horse power", "horsepower",
    "engine power", "engine", "tractor power",
    "rated power"
]

PTO_KEYWORDS = [
    "pto", "pto hp", "pto power", "power take off"
]


class HPResolver:
    """
    Engine Horse Power extractor (NOT PTO HP)

    Design principles:
    - Geometry > keywords > regex
    - Penalize PTO aggressively
    - Bind HP numerically to semantic anchors
    - Indian tractor–specific sanity
    """

    def __init__(self, score_threshold=0.55):
        self.score_threshold = score_threshold

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def resolve(self, blocks, page_width, page_height):
        hp_col_x, pto_col_x = self._detect_hp_columns(blocks)
        candidates = []

        for block_id, block in enumerate(blocks):
            is_table = self._is_table_like_block(block)

            for line_id, line in enumerate(block):
                text = " ".join(tok["text"] for tok in line).lower()
                numbers = self._extract_hp_numbers(text)

                for value, pos in numbers:
                    score = self._score_candidate(
                        value=value,
                        pos=pos,
                        text=text,
                        line=line,
                        is_table=is_table,
                        hp_col_x=hp_col_x,
                        pto_col_x=pto_col_x,
                        page_height=page_height
                    )

                    candidates.append({
                        "hp": value,
                        "score": score
                    })

        if not candidates:
            return {
                "hp": None,
                "confidence": 0.0,
                "reason": "no_candidates"
            }

        best = max(candidates, key=lambda x: x["score"])

        if best["score"] < self.score_threshold:
            return {
                "hp": None,
                "confidence": round(best["score"], 2),
                "reason": "low_confidence"
            }

        return {
            "hp": best["hp"],
            "confidence": round(best["score"], 2),
            "reason": "column_aligned_match"
        }

    # ------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------

    def _extract_hp_numbers(self, text):
        """
        Returns list of (value, char_position)
        """
        values = []

        for m in re.finditer(r"\b\d{2,3}(\.\d+)?\b", text):
            val = float(m.group())
            if 20 <= val <= 100:
                values.append((int(round(val)), m.start()))

        # Handle "42 HP / 38 PTO" → engine HP first
        if "/" in text and len(values) >= 2:
            return [values[0]]

        return values

    # ------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------

    def _score_candidate(
        self,
        value,
        pos,
        text,
        line,
        is_table,
        hp_col_x,
        pto_col_x,
        page_height
    ):
        score = 0.0

        # --- Table context ---
        if is_table:
            score += 0.25

        # --- Column alignment ---
        line_x = self._line_center_x(line)

        if hp_col_x is not None:
            dist = abs(line_x - hp_col_x)
            score += max(0, 1 - dist / 300) * 0.35

        # --- PTO column hard penalty ---
        if pto_col_x is not None and abs(line_x - pto_col_x) < 60:
            score -= 0.45

        # --- Semantic binding: number ↔ "HP" ---
        if "hp" in text:
            hp_idx = text.find("hp")
            if abs(hp_idx - pos) <= 12:
                score += 0.30
            else:
                score += 0.10

        # --- Engine context boost ---
        if any(k in text for k in ["engine", "tractor", "diesel"]):
            score += 0.15

        # --- PTO semantic penalty ---
        if any(k in text for k in PTO_KEYWORDS):
            score -= 0.35

        # --- Vertical sanity (avoid headers/footers) ---
        y = self._line_center_y(line)
        vr = y / page_height
        if 0.25 <= vr <= 0.75:
            score += 0.10

        return max(0.0, min(round(score, 3), 1.0))

    # ------------------------------------------------------------
    # Column detection
    # ------------------------------------------------------------

    def _detect_hp_columns(self, blocks):
        hp_x = None
        pto_x = None

        for block in blocks:
            for line in block:
                for tok in line:
                    txt = tok["text"].lower()
                    x = sum(p[0] for p in tok["bbox"]) / 4

                    if txt.strip() == "hp":
                        hp_x = x
                    elif "pto" in txt:
                        pto_x = x

        return hp_x, pto_x

    # ------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------

    def _line_center_x(self, line):
        xs = [sum(p[0] for p in tok["bbox"]) / 4 for tok in line]
        return sum(xs) / len(xs)

    def _line_center_y(self, line):
        ys = [sum(p[1] for p in tok["bbox"]) / 4 for tok in line]
        return sum(ys) / len(ys)

    def _is_table_like_block(self, block):
        if len(block) < 2:
            return False

        xs = []
        for line in block:
            for tok in line:
                xs.append(sum(p[0] for p in tok["bbox"]) / 4)

        return len(set(int(x // 50) for x in xs)) >= 2
