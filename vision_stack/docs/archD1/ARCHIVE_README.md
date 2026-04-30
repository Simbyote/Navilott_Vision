# Archive Notice

The v1 documents in this archive reflect the pipeline design as originally specified, prior to hardware profiling and implementation decisions made during development.

They are kept for bookkeeping and aim to show design evolution and the reasoning behind changes.

**These documents are not as reference for the current implementation**

---

## What changed and why

**Homography / perspective transform removed.**
The original design called for a bird's-eye view transform to produce metric lateral offset. In practice, the fixed, forward-facing, low-mount camera geometry (3–4 cm
ground level) makes homography impractical: it requires precise physical calibration of mount height and tilt, costs ~3–5 ms per frame, and produces metric outputs that
need renormalizing for the steering PID anyway. Pixel-based lateral offset from boundary x-positions is simpler, faster, and sufficient for this geometry. The
calibration procedure is preserved in `operations.md` Step 3 in case metric-space features are added later.

**Undistortion demoted to optional.**
The IMX219 exhibits modest barrel distortion. Because the pipeline uses pixel-based (non-metric) offset, the practical benefit of undistortion in the center ROI is
limited. A comparative test on course frames is the recommended gate before enabling it.

**Operating point revised.**
The original target was 640×480 @ 30 FPS (33.3 ms budget). After profiling on hardware, 480×360 @ ~17–20 FPS (50 ms budget) was accepted as the operating point with ~4
ms headroom after pipeline stages. Histogram equalization is the dominant cost toggle at this resolution.

**Frame format.**
The original docs assumed BGR throughout. The live pipeline receives YUV from libcamera natively; conversion happens inside preprocessing as needed.

---

## Current implementation reference

All v2 documents in the parent directory reflect the actual implemented pipeline.
