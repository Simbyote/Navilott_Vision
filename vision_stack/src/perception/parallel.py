"""Phase 2's color and obstacle branches on a side thread, so perception uses two cores.

Purpose:
    Geometry, color and obstacle read only the cropped ROIs, never each
    other's output, so color and obstacle can run on a side thread while
    the main thread runs geometry and what needs it. The results are the
    same either way; whether it's faster is measured, not assumed
    (docs/vision_pipeline/phase2_perception.md, "Parallel perception").

Main package:
    The side thread (side_pool(), named p2-side) and the run's choice of
    mode (enabled(): --parallel / --serial, else params.PARALLEL_PERCEPTION).

Flow:
    1. A linker's --parallel or --serial sets the run's mode (apply()).
    2. Per frame, the caller submits color and obstacle to the side thread
       (run_side()), runs geometry itself, then waits for the side's results.
"""
import time
from concurrent.futures import ThreadPoolExecutor

from src.params import PARALLEL_PERCEPTION

SIDE_THREAD_NAME = "p2-side"

_pool: ThreadPoolExecutor | None = None
_default: bool | None = None


def set_default(on: bool | None) -> None:
    """This run's choice for every caller that doesn't say; None goes back to params.PARALLEL_PERCEPTION."""
    global _default
    _default = on


def enabled(explicit: bool | None = None) -> bool:
    """explicit when given, else this run's choice, else params.PARALLEL_PERCEPTION."""
    if explicit is not None:
        return explicit
    return PARALLEL_PERCEPTION if _default is None else _default


def add_arguments(ap) -> None:
    """--parallel / --serial on an argparse parser; neither leaves params.PARALLEL_PERCEPTION."""
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--parallel", dest="parallel", action="store_true", default=None,
                   help="Phase 2's color and obstacle on a side thread (perception/parallel.py)")
    g.add_argument("--serial", dest="parallel", action="store_false",
                   help="every Phase 2 stage on the main thread")


def apply(args) -> bool:
    """Make the parsed --parallel / --serial this run's choice; returns whether the run is parallel."""
    set_default(getattr(args, "parallel", None))
    return enabled()


def _name_thread() -> None:
    try:
        from src.os_threads import name_os_thread
        name_os_thread(SIDE_THREAD_NAME)
    except Exception:               # naming is for the diagnostics only; never a reason to fail
        pass


def side_pool() -> ThreadPoolExecutor:
    """The one side thread every Phase 2 caller shares, started on first use."""
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=SIDE_THREAD_NAME, initializer=_name_thread)
    return _pool


def run_side(fns: list[tuple[str, callable]]) -> tuple[dict, dict]:
    """
    Run each (name, fn) in order on the calling thread.

    Outputs:
        ({name: fn()}, {name: wall ms}).
    """
    results, ms = {}, {}
    for name, fn in fns:
        t0 = time.perf_counter()
        results[name] = fn()
        ms[name] = (time.perf_counter() - t0) * 1000.0
    return results, ms
