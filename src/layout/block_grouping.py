BLOCK_BREAK_KEYWORDS = ["S.N.", "Particulars", "TRACTOR", "Accessories", "Total", "special Note", "Customer Signature"]


def contains_block_break(line):
    text = " ".join(t["text"].lower() for t in line)
    return any(k.lower() in text for k in BLOCK_BREAK_KEYWORDS)


def group_lines_into_blocks(lines):
    blocks = []
    current_block = [lines[0]]

    for prev, curr in zip(lines, lines[1:]):
        prev_bottom = max(t["rect"]["y_max"] for t in prev)
        curr_top = min(t["rect"]["y_min"] for t in curr)

        gap = curr_top - prev_bottom
        avg_height = sum(t["rect"]["height"] for t in prev) / len(prev)

        if gap > avg_height * 1.3 or contains_block_break(curr):
            blocks.append(current_block)
            current_block = [curr]
        else:
            current_block.append(curr)

    blocks.append(current_block)
    return blocks
