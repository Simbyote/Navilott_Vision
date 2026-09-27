"""Shared helpers for the analysis interpreters: reading run CSVs, finding runs, statistics, output.

Purpose:
    Every interpreter reads a CSV a run or a hardware test wrote, reduces it
    to numbers, prints them and draws figures next to the input. The parts
    they share live here so each interpreter is only its own arithmetic.

Main package:
    Table        a CSV as columns: numeric() parses on demand (blank and
                 non-numeric cells become NaN), text() keeps strings.
    find_csv()   a CSV from a file, a folder, or the newest run that has one.
    stats()      NaN-aware n / mean / std / min / p50 / p95 / p99 / max.
    pyplot()     matplotlib on the headless backend, or None: every figure is
                 optional, the numbers never are.

No pandas: the Pi's environment has numpy and the standard library only.
"""

import csv
import json
from pathlib import Path

import numpy as np

from src.params import FPS, PIPELINE_ROOT, RUNS_DIR

ARTIFACTS_DIR = PIPELINE_ROOT / "artifacts"      # pytest --hardware output root
SEARCH_ROOTS = (RUNS_DIR, ARTIFACTS_DIR)
BUDGET_MS = 1000.0 / FPS


# =============================================================================
# Reading
# =============================================================================

class Table:
    """
    One CSV, column-oriented.

    columns:    header names in file order, stripped
    numeric(c): float64 array, NaN where the cell is blank or not a number
    text(c):    list of stripped strings
    """

    def __init__(self, path, skip: int = 0):
        self.path = Path(path)
        with open(self.path, newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError(f"{self.path.name}: empty file") from None
            rows = [r for r in reader if r]
        self.columns = [h.strip() for h in header]
        if len(rows) <= skip:
            raise ValueError(f"{self.path.name}: {len(rows)} rows, not more than skip={skip}")
        rows = rows[skip:]
        self._raw = {c: [r[i].strip() if i < len(r) else "" for r in rows]
                     for i, c in enumerate(self.columns)}
        self._num = {}

    def __len__(self) -> int:
        return len(next(iter(self._raw.values()))) if self._raw else 0

    def has(self, column: str) -> bool:
        return column in self._raw

    def text(self, column: str) -> list:
        return list(self._raw[column])

    def numeric(self, column: str) -> np.ndarray:
        if column not in self._num:
            out = np.full(len(self), np.nan)
            for i, cell in enumerate(self._raw[column]):
                try:
                    out[i] = float(cell)
                except ValueError:
                    pass
            self._num[column] = out
        return self._num[column]

    def has_values(self, column: str) -> bool:
        """True when the column exists and at least one cell is a number."""
        return self.has(column) and not np.all(np.isnan(self.numeric(column)))

    def first_with_values(self, *columns):
        """The first of columns that has numbers, or None."""
        return next((c for c in columns if self.has_values(c)), None)


def interval_ms(table: Table):
    """
    Per-frame loop or frame interval in ms, from the best column available:
    interval_ms (measured loop), dt_ms (capture), dt_s (Phase 3), then
    successive timestamp_ms. None when there's nothing to derive it from.
    """
    if table.has_values("interval_ms"):
        return table.numeric("interval_ms")
    if table.has_values("dt_ms"):
        return table.numeric("dt_ms")
    if table.has_values("dt_s"):
        return table.numeric("dt_s") * 1000.0
    if table.has_values("timestamp_ms"):
        ts = table.numeric("timestamp_ms")
        return np.concatenate(([np.nan], np.diff(ts)))
    return None


def find_csv(arg, names, roots=SEARCH_ROOTS) -> Path:
    """
    A CSV path.

    arg:   a CSV file, or a folder searched (recursively) for any of names,
           or None for the newest run under roots that has one
    names: acceptable file names, in order of preference
    """
    names = (names,) if isinstance(names, str) else tuple(names)
    if arg:
        p = Path(arg)
        if p.is_file():
            return p
        if p.is_dir():
            for name in names:
                if (p / name).is_file():
                    return p / name
            for name in names:
                found = sorted(p.rglob(name))
                if found:
                    return found[-1]
        raise FileNotFoundError(f"no {' or '.join(names)} in {p}")
    found = [f for root in roots for name in names for f in Path(root).rglob(name)]
    if not found:
        raise FileNotFoundError(f"no {' or '.join(names)} under {', '.join(str(r) for r in roots)}")
    return max(found, key=lambda f: f.stat().st_mtime)


def read_manifest(path, key: str) -> list:
    """
    [(value, run path)] from a two-column CSV, <key>,run: one recorded run
    per measured condition (a position, a distance). Runs resolve against
    the manifest's folder.
    """
    path = Path(path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {key, "run"} <= {k.strip() for k in rows[0]}:
        raise ValueError(f"{path.name}: needs columns {key},run")
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    return [(float(r[key]), path.parent / r["run"].strip()) for r in rows]


# =============================================================================
# Numbers
# =============================================================================

def stats(values) -> dict:
    """NaN-aware summary; every field None when there are no values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    keys = ("mean", "std", "min", "p50", "p95", "p99", "max")
    if v.size == 0:
        return {"n": 0, **{k: None for k in keys}}
    return {
        "n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
        "min": float(v.min()), "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
        "max": float(v.max()),
    }


def runs_of(labels) -> list:
    """
    Consecutive runs of equal labels as (label, start_index, length),
    in order. Works on any sequence of hashable values.
    """
    out = []
    for i, lb in enumerate(labels):
        if out and out[-1][0] == lb:
            out[-1][2] += 1
        else:
            out.append([lb, i, 1])
    return [tuple(r) for r in out]


# =============================================================================
# Output
# =============================================================================

def pyplot():
    """matplotlib.pyplot on the headless backend, or None when not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def write_json(path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_jsonable))
    return path


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def fmt(v, digits: int = 2) -> str:
    """A number for a printed table; blank for None or NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.{digits}f}"
