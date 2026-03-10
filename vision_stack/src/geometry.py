import cv2 as cv
import numpy as np

def canny_edge(im: np.ndarray) -> np.ndarray:
    edges = cv.Canny(im, 50, 150)
    return edges

# =============================================================
# Lane Detection Pipeline

def extract_lane_contours(
    edges: np.ndarray,
    min_arc_length: float = 40.0
) -> list:
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv.arcLength(c, closed=False) >= min_arc_length]
    return filtered

def draw_contours(
    base_img: np.ndarray,
    contours: list,
    color=(0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    vis = base_img.copy()
    cv.drawContours(vis, contours, -1, color, thickness)
    return vis