import cv2
import numpy as np


def to_gray(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def clean_gray(image):
    """Light denoising and local contrast enhancement; keep it conservative."""
    gray = to_gray(image)
    if gray is None:
        return None
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def edge_map(image):
    """Return a Canny edge map from the preprocessed crop."""
    gray = clean_gray(image)
    if gray is None:
        return None
    return cv2.Canny(gray, 50, 150)
