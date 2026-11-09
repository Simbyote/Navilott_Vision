"""Phase 2's color and obstacle branches on a side thread, so perception uses two cores.

Purpose:
    Color and obstacle never read geometry's output, so they can run beside
    it. The results are the same either way; whether it's faster is measured
    (docs/vision_pipeline/phase2_perception.md, "Parallel perception").

Main package:
    side_pool(): the shared side thread; enabled(): this run's mode.
"""
import time
from concurrent.futures import ThreadPoolExecutor

from src.params import PARALLEL_PERCEPTION

SIDE_THREAD_NAME = "p2-side"

_pool: ThreadPoolExecutor | None = None
_default: bool | None = None


def set_default(on: bool | None) -> None:
    """None goes back to params.PARALLEL_PERCEPTION."""
    global _default
    _default = on


def enabled(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    return PARALLEL_PERCEPTION if _default is None else _default


def add_arguments(ap) -> None:
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--parallel", dest="parallel", action="store_true", default=None,
                   help="Phase 2's color and obstacle on a side thread (perception/parallel.py)")
    g.add_argument("--serial", dest="parallel", action="store_false",
                   help="every Phase 2 stage on the main thread")


def apply(args) -> bool:
    """Make --parallel / --serial this run's choice; returns whether it's parallel."""
    set_default(getattr(args, "parallel", None))
    return enabled()


def _name_thread() -> None:
    try:
        from src.os_threads import name_os_thread
        name_os_thread(SIDE_THREAD_NAME)
    except Exception:               # naming is cosmetic
        pass


def side_pool() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=SIDE_THREAD_NAME, initializer=_name_thread)
    return _pool


def run_side(fns: list[tuple[str, callable]]) -> tuple[dict, dict]:
    """Each (name, fn) in order on the calling thread: ({name: result}, {name: wall ms})."""
    results, ms = {}, {}
    for name, fn in fns:
        t0 = time.perf_counter()
        results[name] = fn()
        ms[name] = (time.perf_counter() - t0) * 1000.0
    return results, ms
