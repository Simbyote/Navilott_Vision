"""
preprocess.py

Preprocessing Stage

Purpose:
    Conditions the captured BGR frame before ROI cropping splits the pipeline
    into branches. The stage produces two outputs

    1. gray: single-channel, blurred
       Feeds the lane and sign ROIs. The geometry branch validates
       ndim == 2 and raises on a 3-channel input, so the BGR -> gray
       conversion belongs here and nowhere else.

    2. color: BGR, blurred
       Feeds the traffic ROI. The color branch thresholds in HSV and needs
       the chroma that grayscale discards.

Histogram equalization:
    Off by default. It lifted sensor noise enough to cost more in false
    contours than it bought in contrast. Kept behind PreprocessParams.equalize
    so the comparison can be regenerated for a report. Turning it on changes 
    the intensity distribution that LaneContourFilter.min_intensity was tuned against.
"""
import numpy as np
import cv2
from dataclasses import dataclass

from capture import FrameData

# =============================================================================
# Debug Names
# =============================================================================
GRAY_SUFFIX = "_1_gray.png"
EQUALIZED_SUFFIX = "_2_equalized.png"
GRAY_BLUR_SUFFIX = "_3_blurred_gray.png"
COLOR_BLUR_SUFFIX = "_5_blurred.png"

# =============================================================================
# Input Dataclass
# =============================================================================
@dataclass(frozen=True)
class PreprocessParams:
    """
    Tuning for the preprocessing stage

    gray_kernel: (width, height) Gaussian kernel for the geometry path
                 Anisotropic on purpose: more smoothing across a lane line
                 than along it, so fragments bridge without the line
                 thinning out
    gray_sigma: Gaussian sigma for the geometry path; 0.0 derives it from
                the kernel size
    color_kernel: (width, height) Gaussian kernel for the color path
    color_sigma: Gaussian sigma for the color path; 0.0 derives it
    equalize: apply histogram equalization to the grayscale path before
              blurring
    """
    gray_kernel: tuple = (9, 3)
    gray_sigma: float = 0.0
    color_kernel: tuple = (5, 5)
    color_sigma: float = 0.0
    equalize: bool = False

# =============================================================================
# Output Dataclass
# =============================================================================
@dataclass(frozen=True)
class PreprocessResult:
    """
    Output of the preprocessing stage

    gray: (H, W) uint8, blurred. Source for the lane and sign ROIs
    color: (H, W, 3) uint8 BGR, blurred. Source for the traffic ROI
    frame_id: carried from FrameData, never re-derived
    timestamp_ms: carried from FrameData, never re-derived
    """
    gray: np.ndarray
    color: np.ndarray
    frame_id: int
    timestamp_ms: int

# =============================================================================
# Validation
# =============================================================================
def _validate_frame(
        frame: np.ndarray
    ) -> None:
    """
    Purpose:
        Reject frames the stage cannot condition, with a message that names
        what arrived rather than letting OpenCV raise from inside a kernel
    """
    if frame is None:
        raise ValueError("preprocess: frame is None")
    if frame.ndim not in (2, 3):
        raise ValueError(f"preprocess: expected 2 or 3 dims, got {frame.ndim}")
    if frame.ndim == 3 and frame.shape[2] != 3:
        raise ValueError(
            f"preprocess: expected (H,W) gray or (H,W,3) BGR, got {frame.shape}"
        )
    if frame.dtype != np.uint8:
        raise TypeError(f"preprocess: expected uint8, got {frame.dtype}")

def _validate_kernel(
        kernel: tuple,
        name: str
    ) -> None:
    """
    Purpose:
        cv2.GaussianBlur requires odd, positive kernel dimensions. Catching
        it here names the offending parameter instead of surfacing an
        OpenCV assertion with no context
    """
    if len(kernel) != 2:
        raise ValueError(f"preprocess: {name} must be (width, height), got {kernel}")
    for axis, size in zip(("width", "height"), kernel):
        if size < 1 or size % 2 == 0:
            raise ValueError(
                f"preprocess: {name} {axis} must be odd and positive, got {size}"
            )

# =============================================================================
# Utility Functions
# =============================================================================
def to_grayscale(
        frame: np.ndarray
    ) -> np.ndarray:
    """
    Purpose:
        Convert BGR to single-channel. Idempotent: an already-gray frame is
        returned unchanged, so a grayscale fixture and a live BGR frame can
        travel the same path

    Inputs:
        frame: (H, W) uint8 gray or (H, W, 3) uint8 BGR

    Outputs:
        (H, W) uint8
    """
    _validate_frame(frame)
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def histogram_equalization(
        gray: np.ndarray
    ) -> np.ndarray:
    """
    Purpose:
        Redistribute intensities across the full 8-bit range to improve
        contrast. Single-channel only --- cv2.equalizeHist rejects a
        3-channel input, so the conversion must happen first

    Inputs:
        gray: (H, W) uint8

    Outputs:
        (H, W) uint8
    """
    if gray.ndim != 2:
        raise ValueError(
            f"preprocess: equalizeHist needs single-channel, got {gray.shape}. "
            "Call to_grayscale() first."
        )
    return cv2.equalizeHist(gray)

def gaussian_blur(
        frame: np.ndarray,
        kernel_size: tuple = (5, 5),
        sigma: float = 0.0
    ) -> np.ndarray:
    """
    Purpose:
        Suppress high-frequency sensor noise before edge detection and HSV
        thresholding. Reduces false contours and spurious mask blobs without
        moving coarse structural features

    Inputs:
        frame: (H, W) or (H, W, 3) uint8
        kernel_size: (width, height), both odd and positive
        sigma: 0.0 derives sigma from the kernel size

    Outputs:
        blurred array, same shape and dtype as the input
    """
    _validate_kernel(kernel_size, "kernel_size")
    return cv2.GaussianBlur(frame, kernel_size, sigma)

# =============================================================================
# Preprocessing Stage
# =============================================================================
def preprocess_frame(
        frame_data: FrameData,
        params: PreprocessParams = PreprocessParams()
    ) -> PreprocessResult:
    """
    Purpose:
        Run the preprocessing stage on one captured frame, producing both
        branch inputs and carrying the frame's identity forward unchanged

    Inputs:
        frame_data: FrameData from CameraSource.read()
        params: PreprocessParams tuning

    Outputs:
        PreprocessResult
    """
    _validate_frame(frame_data.frame)
    _validate_kernel(params.gray_kernel, "gray_kernel")
    _validate_kernel(params.color_kernel, "color_kernel")

    # Geometry path: BGR -> gray -> (optional equalize) -> blur
    gray = to_grayscale(frame_data.frame)
    if params.equalize:
        gray = histogram_equalization(gray)
    gray = gaussian_blur(gray, params.gray_kernel, params.gray_sigma)

    # Color path: BGR -> blur
    color = gaussian_blur(
        frame_data.frame, params.color_kernel, params.color_sigma
    )

    return PreprocessResult(
        gray = gray,
        color = color,
        frame_id = frame_data.frame_id,
        timestamp_ms = frame_data.timestamp_ms,
    )

# =============================================================================
# Self-test
#
#   python3 preprocess.py            logic tests only, runs anywhere
#   python3 preprocess.py --dataset  also regenerates the sample fixtures
#
# Exits non-zero on any logic failure so it can gate a commit.
# =============================================================================
if __name__ == "__main__":
    import os
    import sys
    import traceback

    H, W = 360, 480

    _results = []

    def check(name, fn):
        """Run one test, record pass/fail, never abort the suite."""
        try:
            fn()
        except Exception:
            _results.append((name, False))
            print(f"  FAIL  {name}")
            for line in traceback.format_exc().strip().splitlines()[-2:]:
                print(f"        {line.strip()}")
        else:
            _results.append((name, True))
            print(f"  ok    {name}")

    def expect_raises(exc_type, fn):
        try:
            fn()
        except exc_type:
            return
        raise AssertionError(f"expected {exc_type.__name__}, nothing was raised")

    def _bgr(seed=0):
        """A BGR frame with real structure, so a blur has something to act on."""
        rng = np.random.default_rng(seed)
        f = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        f[H // 2 :, :, :] = 255          # bright lower band, like lane tape
        return f

    def _fd(frame=None, frame_id=0, timestamp_ms=1000):
        return FrameData(_bgr() if frame is None else frame, frame_id, timestamp_ms)

    # -------------------------------------------------------------------------
    # Shape and dtype contract
    # -------------------------------------------------------------------------
    def t_gray_is_single_channel():
        r = preprocess_frame(_fd())
        assert r.gray.ndim == 2, f"geometry branch requires ndim 2, got {r.gray.ndim}"
        assert r.gray.shape == (H, W), f"got {r.gray.shape}"
        assert r.gray.dtype == np.uint8

    def t_color_keeps_three_channels():
        r = preprocess_frame(_fd())
        assert r.color.shape == (H, W, 3), f"got {r.color.shape}"
        assert r.color.dtype == np.uint8

    def t_grayscale_input_is_accepted():
        """Fixtures loaded with IMREAD_GRAYSCALE must travel the same path."""
        gray_in = np.full((H, W), 128, np.uint8)
        r = preprocess_frame(_fd(frame=gray_in))
        assert r.gray.shape == (H, W)

    def t_to_grayscale_is_idempotent():
        gray = np.full((H, W), 90, np.uint8)
        assert to_grayscale(to_grayscale(gray)).shape == (H, W)

    # -------------------------------------------------------------------------
    # Stamp passthrough
    # -------------------------------------------------------------------------
    def t_stamp_is_carried_not_rederived():
        r = preprocess_frame(_fd(frame_id=42, timestamp_ms=987654))
        assert (r.frame_id, r.timestamp_ms) == (42, 987654), (
            f"got {(r.frame_id, r.timestamp_ms)}"
        )

    def t_result_is_immutable():
        r = preprocess_frame(_fd())
        expect_raises(Exception, lambda: setattr(r, "frame_id", 9))

    # -------------------------------------------------------------------------
    # The stage must not modify the caller's frame
    # -------------------------------------------------------------------------
    def t_input_frame_is_untouched():
        frame = _bgr(seed=7)
        before = frame.copy()
        preprocess_frame(_fd(frame=frame))
        assert np.array_equal(frame, before), "preprocess wrote into its input"

    # -------------------------------------------------------------------------
    # Equalization is off unless asked for
    # -------------------------------------------------------------------------
    def t_equalize_defaults_off():
        fd = _fd()
        plain = preprocess_frame(fd)
        eq = preprocess_frame(fd, PreprocessParams(equalize=True))
        assert not np.array_equal(plain.gray, eq.gray), (
            "equalize=True produced an identical result — flag is not wired"
        )

    def t_equalize_rejects_three_channel():
        expect_raises(ValueError, lambda: histogram_equalization(_bgr()))

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    def t_even_kernel_rejected():
        expect_raises(
            ValueError,
            lambda: preprocess_frame(_fd(), PreprocessParams(gray_kernel=(8, 3))),
        )

    def t_float_frame_rejected():
        bad = np.zeros((H, W, 3), np.float32)
        expect_raises(TypeError, lambda: preprocess_frame(_fd(frame=bad)))

    def t_four_channel_frame_rejected():
        bad = np.zeros((H, W, 4), np.uint8)
        expect_raises(ValueError, lambda: preprocess_frame(_fd(frame=bad)))

    def t_none_frame_rejected():
        expect_raises(ValueError, lambda: preprocess_frame(FrameData(None, 0, 0)))

    # -------------------------------------------------------------------------
    print("\nOutput contract")
    check("gray is single-channel", t_gray_is_single_channel)
    check("color keeps 3 channels", t_color_keeps_three_channels)
    check("grayscale input accepted", t_grayscale_input_is_accepted)
    check("to_grayscale is idempotent", t_to_grayscale_is_idempotent)

    print("\nStamp")
    check("frame_id/timestamp carried through", t_stamp_is_carried_not_rederived)
    check("PreprocessResult is immutable", t_result_is_immutable)
    check("input frame is not modified", t_input_frame_is_untouched)

    print("\nEqualization")
    check("defaults off, flag is wired", t_equalize_defaults_off)
    check("rejects 3-channel input", t_equalize_rejects_three_channel)

    print("\nValidation")
    check("even kernel rejected", t_even_kernel_rejected)
    check("float32 frame rejected", t_float_frame_rejected)
    check("4-channel frame rejected", t_four_channel_frame_rejected)
    check("None frame rejected", t_none_frame_rejected)

    passed = sum(1 for _, ok in _results if ok)
    print(f"\n{passed}/{len(_results)} passed")

    # -------------------------------------------------------------------------
    # Dataset pass: regenerate the fixtures roi_crop and geometry consume
    # -------------------------------------------------------------------------
    if "--dataset" in sys.argv:
        SAMPLE_DIRS = (
            "vision_stack/frames/Sample1",
            "vision_stack/frames/Sample2",
            "vision_stack/frames/Sample3",
        )
        IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

        params = PreprocessParams()
        total_processed = 0
        total_failed = 0
        frame_id = 0

        print("\nDataset")
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

            for filename in image_files:
                img_path = os.path.join(sample_dir, filename)
                stem = os.path.splitext(filename)[0]

                original = cv2.imread(img_path)
                if original is None:
                    print(f"[FAIL] Could not read: {img_path}")
                    total_failed += 1
                    continue

                # Same entry point the live loop calls, same params. The
                # fixtures and the pipeline cannot drift apart while this
                # harness goes through preprocess_frame() rather than
                # reimplementing it with its own cv2 calls.
                fd = FrameData(original, frame_id, 0)
                try:
                    result = preprocess_frame(fd, params)
                except (ValueError, TypeError) as e:
                    print(f"[FAIL] {img_path}: {e}")
                    total_failed += 1
                    continue

                cv2.imwrite(os.path.join(results_dir, f"{stem}{GRAY_SUFFIX}"),
                            to_grayscale(original))
                cv2.imwrite(os.path.join(results_dir, f"{stem}{GRAY_BLUR_SUFFIX}"),
                            result.gray)
                cv2.imwrite(os.path.join(results_dir, f"{stem}{COLOR_BLUR_SUFFIX}"),
                            result.color)

                if params.equalize:
                    cv2.imwrite(
                        os.path.join(results_dir, f"{stem}{EQUALIZED_SUFFIX}"),
                        histogram_equalization(to_grayscale(original)),
                    )

                print(f"[OK] frame_id={frame_id} {img_path}")
                frame_id += 1
                total_processed += 1

        print(f"\nDone. {total_processed} processed, {total_failed} failed.")
    else:
        print("\nDataset: skipped (pass --dataset to regenerate fixtures)")

    sys.exit(0 if passed == len(_results) else 1)