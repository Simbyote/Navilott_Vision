"""
preprocess.py

Preprocessing Stage

Purpose:
    Conditions the raw YUV frame from Phase 1 before any spatial analysis occurs.
    Three sequential operations are applied:

    1. Histogram equalization: redistributes pixel intensity so that features
                    remain visible under varying lighting
                    Applied per-channel in YCrCb space so that hue and saturation 
                    information is preserved for the downstream HSV color branch, and
                    intensity contrast is improved for the Canny/grayscale
                    branch simultaneously

    2. Gaussian blur: suppresses high-frequency sensor noise before edge
                    detection and HSV thresholding 
                    Reduces false contours and spurious HSV mask blobs 
                    without affecting coarse structural features
"""
import cv2
import numpy as np
import os

"""
Purpose:
    Preprocesses a single YUV frame from the GStreamer ring buffer 

Inputs:
    frame : np.ndarray
        Shape  : (H, W, 3)
        Dtype  : uint8
        Color  : YUV  (aligned with capture.py output)
        Source : one frame from the GStreamer ring buffer

    gaussian_kernel_size : tuple[int, int]  (default (5, 5))
        Can only be odd positive integers; Larger values blur more aggressively

    gaussian_sigma : float  (default 0.0)
        When 0.0, OpenCV computes sigma from kernel size automatically

Outputs
    Returns : np.ndarray
        Shape  : (H, W, 3)
        Dtype  : uint8
        Color  : YUV  (aligned with capture.py output)
    The returned array is a new allocation
"""
def preprocess_frame(
    frame: np.ndarray,
    gaussian_kernel_size: tuple = (5, 5),
    gaussian_sigma: float = 0.0
) -> np.ndarray:
    # Input validation
    if frame is None:
        raise ValueError("preprocess_frame: received None frame. appsink failure?")
    if frame.dtype != np.uint8:
        raise TypeError(f"preprocess_frame: expected a uint8 color space, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"preprocess_frame: expected (H, W, 3), got shape {frame.shape}")
    kw, kh = gaussian_kernel_size
    if kw % 2 == 0 or kh % 2 == 0:
        raise ValueError(f"preprocess_frame: gaussian_kernel_size must be odd, got {gaussian_kernel_size}")

    """ Step 1: 
    Histogram equalization in the YCrCb color space
    Conversion from YUV to YCrCb involves a conversion from YUV to BGR before YCrCb conversion
    The Y channel is is only applied in the histogram equalization step
    Cr and Cb carry the color information used by the HSV branch downstream
    Y carries the intensity information used by the Canny branch downstream
    """
    bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    bgr_equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    equalized = cv2.cvtColor(bgr_equalized, cv2.COLOR_BGR2YUV)

    """
    Step 2: 
    Gaussian blur used to suppress camera noise and focus edge detections on
    sharp features
    """
    blurred = cv2.GaussianBlur(equalized, gaussian_kernel_size, gaussian_sigma)
    return blurred

# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    """
    Standalone test (static dataset):

    Purpose:
        Loads every .jpg / .png from the dataset directory, runs preprocess_frame(), 
        and writes four debug images per input into vision_stack/frames/trackT1/results/

    Directory structure:
        vision_stack/
            frames/
                trackT1/
                    img001.jpg
                    img002.jpg
                    ...
                    results/
                        img001_1_original.png
                        img001_2_undistorted.png
                        ...
    """
    SAMPLE_DIRS = [
        "vision_stack/frames/trackT3",
        "vision_stack/frames/trackT4",
        "vision_stack/frames/trackT5"
    ]
    GAUSSIAN_KERNEL  = (5, 5)
    GAUSSIAN_SIGMA   = 0.0
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

    # Tracks processing progress
    total_processed = 0
    total_failed    = 0

    # Organize directory structure
    for sample_dir in SAMPLE_DIRS:
        if not os.path.isdir(sample_dir):
            print(f"[SKIP] Directory not found: {sample_dir}")
            continue

        results_dir = os.path.join(sample_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        image_files = sorted(
            f for f in os.listdir(sample_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )

        if not image_files:
            print(f"[SKIP] No images found in {sample_dir}")
            continue

        # Load each image, run the preprocessing pipeline, and write debug images to check quality
        for filename in image_files:
            img_path = os.path.join(sample_dir, filename)
            stem     = os.path.splitext(filename)[0]

            original = cv2.imread(img_path)
            if original is None:
                print(f"[FAIL] Could not read: {img_path}")
                total_failed += 1
                continue

            bgr = cv2.cvtColor(original, cv2.COLOR_YUV2BGR)
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            bgr_equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            equalized = cv2.cvtColor(bgr_equalized, cv2.COLOR_BGR2YUV)
            cv2.imwrite(os.path.join(results_dir, f"{stem}_1_equalized.png"), equalized)

            blurred = cv2.GaussianBlur(equalized, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
            cv2.imwrite(os.path.join(results_dir, f"{stem}_2_blurred.png"), blurred)

            print(f"[OK] {img_path}  ->  {results_dir}/{stem}_*.png")
            total_processed += 1

    print(f"\nDone. {total_processed} processed, {total_failed} failed.")