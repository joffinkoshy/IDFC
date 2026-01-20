import cv2 as cv


def load_image(image_path: str):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image
