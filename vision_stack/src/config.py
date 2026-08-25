"""
config.py

Toggle Configurations for feature fusion and vision pipeline; enables bench testing

Purpose:
    Targets areas of known failure in phase 2, making each toggeable so on-off state can be correlated 
    with appropriate resolution across different test runs. Config with all flags are False to reproduce
    current pipeline behavior

    Also manages:
        - FrameTags: Per-frame failure mode counters filled in by the stages, logged by the linker.
        - Intersection edge-ratio calculation (log only with no state transitions)

Fixes:
    1. roi_inset (roi_crop.py)
        - Horizontal margin and top trim on lane ROI
    2. trapezoid_mask (geometry.py)
        - fillPoly mask applied to canny output
    3. orientation_filt (geometry.py)
        - Reject near-vertical and top region contours
    4. dashed_dilate (geometry.py)
        - Vertical dilate on Canny pre-contour
    5. anchor_halves (lane_offset.py)
        - Left anchor on left half, right anchor on right half
"""

from dataclasses import dataclass, field
from tkinter.font import names
import numpy as np

# =============================================================================
# Configuration Parameters
# =============================================================================
@dataclass(frozen=True)
class RoiInsetParams:
    """
    Resolution 1: Lane ROI inset parameters for lane detection

    side_margin_frac: Horizontal inset applied to each side, fraction of width
        Symmetric - ROI center is the frames center while normalized semantics
        are preserved

    top_frac: Lane ROI starts at H*top_frac
    """
    side_margin_frac: float = 0.12
    top_frac: float = 0.60

@dataclass(frozen=True)
class TrapezoidMaskParams:
    """
    Resolution 2: Trapezoid mask parameters for lane detection

    corners: (x_frac, y_frac) Fractions of the lane ROIs remain as delivered to
        the geometry branch. Post-inset when resolution 1 is enabled
    """
    # TODO: REQUIRES CALIBRATION
    corners: tuple = (
        (0.05, 1.00),   # Bottom left corner
        (0.95, 1.00),   # Bottom right corner
        (0.72, 0.00),   # Top left corner
        (0.28, 0.00),   # Top right corner
    )

@dataclass(frozen=True)
class OrientationFiltParams:
    """
    Resolution 3: Orientation filter parameters for lane detection

    max_angle_from_horizontal_deg: Reject contours whose minAreaRect long axis
        is steeper than the angle threshold
        - NOTE: Could reject legitimate vertical lane markings mid-turn

    reject_top_frac: reject contours whose centroid sits in the top fraction of
        of the lane ROI (accounts for far-field clutter or wall bases)
    """
    max_angle_from_horizontal_deg: float = 65.0
    reject_top_frac: float = 0.25

@dataclass(frozen=True)
class DashedDilateParams:
    """
    Resolution 4: Dashed dilate parameters for lane detection

    Vertical structuring element dilates Canny output so vertically stacked dashed 
        segments merge into one contour before extraction

    kernel_h: vertical extent in px. Must exceed the projected inter-dash gap at
        detection range
    kernel_w: keep at 1 to avoid horizontal thickening
    """
    kernel_h: int = 50  # TODO: REQUIRES CALIBRATION
    kernel_w: int = 1

# =============================================================================
# Top Level Configurations
# =============================================================================
@dataclass(frozen=True)
class Config:
    roi_inset: bool = False
    trapezoid_mask: bool = False
    orientation_filt: bool = False
    dashed_dilate: bool = False
    anchor_halves: bool = False

    roi_params: RoiInsetParams = field(default_factory=RoiInsetParams)
    trapezoid_params: TrapezoidMaskParams = field(default_factory=TrapezoidMaskParams)
    orientation_params: OrientationFiltParams = field(default_factory=OrientationFiltParams)
    dashed_params: DashedDilateParams = field(default_factory=DashedDilateParams)

    # Names accepted by the CLI / from_names; order defines flags_str() order
    FIX_NAMES = (
        "roi_inset",
        "trapezoid_mask",
        "orientation_filt",
        "dashed_dilate",
        "anchor_halves",
    )

    def flags_str(
            self
        ) -> str:
        """
        Compact log tag
        Computes once at startup and reuses the string
        """
        return (f"R{int(self.roi_inset)} T{int(self.trapezoid_mask)} "
                f"O{int(self.orientation_filt)} D{int(self.dashed_dilate)} "
                f"A{int(self.anchor_halves)}")
    
    @classmethod
    def from_names(
            cls, 
            names
        ) -> "Config":
        """
        Build a Config with listed fixes enabled, all others disabled. Names are case-insensitive
        """
        names = set(names or ())
        unknown = names - set(cls.FIX_NAMES)
        if unknown:
            raise ValueError(
                f"Unknown fix names: {unknown}. "
                f"Valid names: {cls.FIX_NAMES}"
            )
        return cls(**{n: True for n in names})

# =============================================================================
# Contour Debugging
# =============================================================================
@dataclass
class ContourDebug:
    area: float
    aspect: float
    intensity: float
    roi_span: float
    center_in_middle_third: bool
    accepted: bool
    reject_reason: str = "-" #: "max_area", 
                             #  "min_aspect", 
                             # "max_aspect", 
                             # "min_intensity", 
                             # "max_roi_span", or 
                             # "-"
    
# =============================================================================
# Per-Frame Failure Mode Intrumentation
# =============================================================================
@dataclass
class FrameTags:
    """
    Failure-mode counters for one frame. Stages fill these in when a tags object 
    is provided. The main pipeline turns them into a single log line

    All tags are computed from data the stages already have. Tags are active regardless
    of which fixes are in place

    pole_misclassified: ACCEPTED lane candidates that violate the 3rd resolution
        (Steep angle and/or top-region centroid)
    wall_edge_detected: ACCEPTED lane candidates whose bbox touches the lane ROI top
        row
    dashed_reject_center: Contours rejected for area/elongation whose bbox center lies
        in the central third of the lane ROI
    anchor_wrong_half: both lane anchors land in the same ROI half
    contour_debug: list of ContourDebug objects for all contours that reach the
        geometry stage, regardless of accept/reject status
    """
    pole_misclassified: int = 0
    wall_edge_detected: int = 0
    dashed_reject_center: int = 0
    anchor_wrong_half: bool = False
    contour_debug: list = field(default_factory=list)

    # Minimum center-region restrictions to call a dashed-line drop
    DASHED_DROP_MIN_REJECTS = 3     # TODO: REQUIRES CALIBRATION

    def summary(
            self, 
            offset_mode: str
        ) -> str:
        """
        Return a space-separated tag string, or "" when nothing fired
        """
        parts = []
        if self.pole_misclassified:
            parts.append(f"pole_misclassified(x{self.pole_misclassified})")
        if self.wall_edge_detected:
            parts.append(f"wall_edge_detected(x{self.wall_edge_detected})")
        if self.dashed_reject_center:
            parts.append(f"dashed_reject_center(x{self.dashed_reject_center})")
        if self.anchor_wrong_half:
            parts.append("anchor_wrong_half")
        if (offset_mode in ("left_only", "right_only", "none")
            and self.dashed_reject_center >= self.DASHED_DROP_MIN_REJECTS):
            parts.append(f"dashed_line_dropped(raj={self.dashed_reject_center})")
        return " ".join(parts)
    
# =============================================================================
# Intersection Edge Ratio Detection
# =============================================================================
# Band of the lane ROI inspected for a top line (fraction of the ROIs height)
INTERSECTION_BAND_TOP_FRAC = 0.65   # TODO: REQUIRES CALIBRATION

# Triggers when band edge density exceeds this multiple of whole-ROI density
INTERSECTION_EDGE_RATIO_THRESH = 0.25  # TODO: REQUIRES CALIBRATION

_EPSILON = 1e-6

def intersection_edge_ratio(
        edges: np.ndarray
    ) -> float:
    """
    Purpose:
        Cheap stop-line indicator: a stop line adds a dense horizontal edge band
            near the ROI bottom, raising bottom-band edge density well above the
            ROI-wide average

    Inputs:
        edges: uint8 binary Canny output for the lane ROI
    
    Outputs:
        ratio: band edge density / whole-ROI edge density
            ~1.0 = uniform edges; >> 1.0 = edge mass concentrated in band
            0.0 when the ROI contains no edges at all
    """
    h = edges.shape[0]
    band = edges[int(h * INTERSECTION_BAND_TOP_FRAC):, :]

    total_density = np.count_nonzero(edges) / edges.size
    if total_density < _EPSILON:
        return 0.0
    band_density = np.count_nonzero(band) / band.size
    return band_density / total_density