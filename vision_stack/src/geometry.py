import cv2 as cv
import numpy as np

def canny_edge(im: np.ndarray) -> np.ndarray:
    edges = cv.Canny(im, 50, 150)
    return edges
# =============================================================
# Lane Mask Pipeline

def white_mask(im: np.ndarray) -> np.ndarray:
    _, mask = cv.threshold(im, 180, 255, cv.THRESH_BINARY)
    return mask

def white_mask_adaptive(im: np.ndarray) -> np.ndarray:
    return cv.adaptiveThreshold(
        im,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        11,
        -2
    )

def filter_components(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    _, w = mask.shape

    for label in range(1, num_labels):
        x = stats[label, cv.CC_STAT_LEFT]
        bw = stats[label, cv.CC_STAT_WIDTH]
        bh = stats[label, cv.CC_STAT_HEIGHT]
        area = stats[label, cv.CC_STAT_AREA]

        bbox_area = max(1, bw * bh)
        fill_ratio = area / bbox_area

        if area < 80:
            continue

        component = np.zeros_like(mask)
        component[labels == label] = 255

        touches_right_edge = (x + bw) >= (w - 1)

        if (touches_right_edge and area > 2000) or (area > 3000 and fill_ratio > 0.45):
            edges = cv.Canny(component, 50, 150)
            cleaned = cv.bitwise_or(cleaned, edges)
        else:
            cleaned[labels == label] = 255

    return cleaned

def cleanup_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return closed