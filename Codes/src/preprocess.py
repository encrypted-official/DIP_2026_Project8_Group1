"""Shared configurable preprocessing for orientation estimators."""
import cv2


def to_gray(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def clean_gray(image, blur_size=3, use_clahe=True):
    """Grayscale, optional odd-sized Gaussian blur, optional CLAHE."""
    gray = to_gray(image)
    if gray is None:
        return None
    if blur_size not in (1, 3, 5, 7):
        raise ValueError("blur_size must be one of 1, 3, 5, 7")
    if blur_size > 1:
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
    return gray


def edge_map(image, low_threshold=50, high_threshold=150, blur_size=3, use_clahe=True):
    """Return a Canny edge map using configurable preprocessing parameters."""
    if low_threshold < 0 or high_threshold <= low_threshold:
        raise ValueError("Require 0 <= low_threshold < high_threshold")
    gray = clean_gray(image, blur_size=blur_size, use_clahe=use_clahe)
    if gray is None:
        return None
    return cv2.Canny(gray, int(low_threshold), int(high_threshold))
