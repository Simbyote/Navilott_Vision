"""
artifacts.py

Output helpers for --hardware runs: CSV, JSON, PNG, and (optionally) graphs.

Everything here is write-only and cheap. Hardware tests should buffer rows in
memory during the frame loop and call these AFTER the loop, so file I/O never
lands inside the timing window on the Pi.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def summarize(values) -> dict:
    """n / mean / std / min / p50 / p95 / p99 / max for a 1-D sequence."""
    v = np.asarray(list(values), dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "max": float(v.max()),
    }


class Artifacts:
    """One output directory per test; every method returns the path written."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def csv(self, name: str, header: list, rows: list) -> Path:
        out = self.path / name
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        return out

    def json(self, name: str, obj) -> Path:
        out = self.path / name
        with open(out, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        return out

    def image(self, name: str, img: np.ndarray) -> Path:
        out = self.path / name
        if not cv2.imwrite(str(out), img):
            raise IOError(f"cv2.imwrite failed for {out}")
        return out

    def histogram(self, name, values, title, xlabel, bins=40):
        """
        Graph output. matplotlib is optional: the Pi usually doesn't have it,
        and the CSV carries the same data for plotting on a desktop. Returns
        None when matplotlib is unavailable.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        out = self.path / name
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(list(values), bins=bins)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("frames")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out