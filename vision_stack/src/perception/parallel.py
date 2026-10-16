"""Phase 2's side branches on a second thread, so perception uses two cores.

Purpose:
    Phase 2 ran every stage on the main thread, one after another: the main
    thread sat near 100% of one core while the others were half idle
    (2026-10-14 diag run: main 70% mean, 99.7% peak; the Pi 43%). Three
    stages read only the cropped ROIs and nothing of each other's: geometry
    (~20.6 ms on the course), color (~5.2) and obstacle (~5.9). With
    PARALLEL_PERCEPTION on, color and obstacle run on one side thread while
    the main thread runs geometry, then the lane offset and the stop line
    (which need geometry's output), and only then waits for the side thread.

    It's a thread, not a process: the heavy work is inside OpenCV calls,
    which release the GIL, so the two run on two cores without copying a
    frame. The Python between those calls still takes turns on the GIL,
    which is why the gain is measured, not assumed: stage timing reports
    branch_wait, the main thread's time spent waiting for the side thread
    (0 when the side finished first), and the side thread's own times.

    The results are the same either way: the branches are pure functions of
    the ROIs and the config (test_parallel holds them equal frame by frame).

Main package:
    enabled(explicit) -> whether a call runs parallel: explicit when given,
        else set_default()'s choice, else params.PARALLEL_PERCEPTION.
    set_default(on) -> a run's choice (the linkers' --parallel / --serial).
    add_arguments(ap) / apply(args) -> those two flags on a command line.
    side_pool() -> the shared single-thread executor (named p2-side, so
        diagnostics' per-thread CPU shows it), created on first use.
    run_side(fns) -> (results, ms): run named callables in order, timing
        each; what the side thread executes.
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
        from src.diagnostics.threads import name_os_thread
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
