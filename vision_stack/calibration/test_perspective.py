"""
test_perspective.py
-------------------
Generates a synthetic homography (perspective transform) calibration file
for use in the vision stack pipeline.

A homography is a 3x3 projective transformation matrix that maps points
from one 2D plane to another. In a bird's-eye-view (BEV) pipeline it is
used to warp the camera's perspective view of the ground plane into a
top-down view, making distance estimation and lane-keeping easier.

The transform is computed with cv2.getPerspectiveTransform(), which fits
an exact homography to four point correspondences (src → dst).

Output
------
vision_stack/calibration/homography_matrix.npz
    homography_matrix : (3, 3) float64
        The perspective transform H such that:
            dst_pt (homogeneous) = H @ src_pt (homogeneous)
        Applied with cv2.warpPerspective().
    output_width  : int32   Width of the warped output image  (pixels)
    output_height : int32   Height of the warped output image (pixels)
    src_points    : (4, 2) float32   Source corners in the camera image
    dst_points    : (4, 2) float32   Destination corners in the warped image

Point layout (order matters for getPerspectiveTransform)
---------------------------------------------------------
Index   Role
  0     Top-left
  1     Top-right
  2     Bottom-left
  3     Bottom-right

Source region (camera view)
    The trapezoid below represents the road region visible to the camera.
    The narrower top edge is further away; the wider bottom edge is closer.

        (220,300) -------- (420,300)    ← far edge (narrower in perspective)
           /                       \\
          /                         \\
        (100,470) -------------- (540,470)  ← near edge (wider in perspective)

Destination region (bird's-eye-view output, 640x480)
    The trapezoid is straightened into a rectangle occupying the centre
    third of the output frame (x: 160-480), full height (y: 0-479).

        (160,  0) -------------- (480,  0)
           |                         |
           |                         |
        (160,479) -------------- (480,479)

Notes
-----
These are *fake* values intended only for pipeline testing.
Real values must be derived from actual camera geometry — typically by
placing a known pattern (e.g. a checkerboard or coloured markers) on the
ground plane and manually picking the four correspondence points.
"""

import os
import cv2
import numpy as np

# Create the calibration directory if it does not already exist
os.makedirs("vision_stack/calibration", exist_ok=True)

# Output image dimensions for the warped (bird's-eye-view) frame
output_width  = 640
output_height = 480

# --- Source points (camera / perspective view) ---
# Four corners of the road region as seen by the camera, in pixel coords.
# Order: top-left, top-right, bottom-left, bottom-right.
src_points = np.array([
    [220, 300],   # far-left
    [420, 300],   # far-right
    [100, 470],   # near-left
    [540, 470],   # near-right
], dtype=np.float32)

# --- Destination points (rectified / bird's-eye-view) ---
# Corresponding corners in the output image.
# The road region is mapped to a centred vertical strip (x: 160–480).
dst_points = np.array([
    [160,   0],   # top-left
    [480,   0],   # top-right
    [160, 479],   # bottom-left
    [480, 479],   # bottom-right
], dtype=np.float32)

# Compute the 3x3 homography matrix from the four point pairs.
# getPerspectiveTransform requires exactly four correspondences and
# solves the system analytically (no least-squares approximation).
homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Save all calibration data into a single compressed NumPy archive
np.savez(
    "vision_stack/calibration/homography_matrix.npz",
    homography_matrix=homography_matrix,
    output_width=np.array(output_width,  dtype=np.int32),
    output_height=np.array(output_height, dtype=np.int32),
    src_points=src_points,
    dst_points=dst_points
)

print("Saved fake homography file.")