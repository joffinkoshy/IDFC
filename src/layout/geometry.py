def quad_to_rect(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return {
        "x_min": min(xs),
        "y_min": min(ys),
        "x_max": max(xs),
        "y_max": max(ys),
        "x_center": sum(xs) / 4,
        "y_center": sum(ys) / 4,
        "height": max(ys) - min(ys)
    }
