"""
test_matrix.py
--------------
Generates a synthetic camera intrinsic calibration file for use in the
vision stack pipeline.

Camera calibration maps 3D world coordinates to 2D image coordinates.
The intrinsic matrix (also called the camera matrix or K matrix) encodes
the internal optical properties of the camera, independent of where it
is placed in the world.

Output
------
vision_stack/calibration/camera_matrix.npz
    camera_matrix : (3, 3) float64
        The camera intrinsic matrix:
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        fx, fy : focal lengths in pixels (500 px here — a reasonable
                 approximation for a typical webcam at 640x480)
        cx, cy : principal point (image centre at 320, 240 for 640x480)
    dist_coeffs : (1, 5) float64
        Lens distortion coefficients [k1, k2, p1, p2, k3].
        All zeros means distortion is assumed negligible (ideal pinhole model).

Notes
-----
These are *fake* values intended only for pipeline testing.
Replace with real calibration data (e.g. from cv2.calibrateCamera())
before running on actual camera input.
"""

import os
import numpy as np

# Create the calibration directory if it does not already exist
os.makedirs("vision_stack/calibration", exist_ok=True)


# --- Camera intrinsic matrix (K) ---
# Focal lengths fx = fy = 500 px (square pixels, no skew).
# Principal point at image centre: cx = 320, cy = 240 (for 640x480 resolution).
camera_matrix = np.array([
    [500.0,   0.0, 320.0],
    [  0.0, 500.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

# --- Distortion coefficients ---
# Five-element vector: [k1, k2, p1, p2, k3]
# k1, k2, k3 : radial distortion  (barrel / pincushion)
# p1, p2     : tangential distortion (lens tilt / decentering)
# All zeros → ideal pinhole camera with no distortion.
dist_coeffs = np.zeros((1, 5), dtype=np.float64)

# Save both arrays into a single compressed NumPy archive
np.savez(
    "vision_stack/calibration/camera_matrix.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)

print("Wrote vision_stack/calibration/camera_matrix.npz")