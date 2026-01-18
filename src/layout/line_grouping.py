def vertical_overlap(a, b):
    top = max(a["y_min"], b["y_min"])
    bottom = min(a["y_max"], b["y_max"])
    overlap = max(0, bottom - top)
    return overlap / min(a["height"], b["height"])


def group_tokens_into_lines(tokens):
    tokens = sorted(tokens, key=lambda t: t["rect"]["y_center"])
    lines = []

    for token in tokens:
        placed = False
        for line in lines:
            ref = line[0]["rect"]

            if vertical_overlap(token["rect"], ref) > 0.5:
                line.append(token)
                placed = True
                break

            if abs(token["rect"]["y_center"] - ref["y_center"]) < ref["height"] * 0.6:
                line.append(token)
                placed = True
                break

        if not placed:
            lines.append([token])

    # sort tokens left-to-right within each line
    for line in lines:
        line.sort(key=lambda t: t["rect"]["x_min"])

    return lines
