"""OS-level thread controls: a thread's kernel name, and real-time priority.

Purpose:
    Every thread shows as "python3" in top and /proc unless it names itself,
    and the sensor hub runs at real-time priority so it wakes on time. The
    drivers, the pipeline and the tools all use these.

Main package:
    name_os_thread() and set_realtime(); neither raises.
"""
import ctypes
import ctypes.util
import os
import sys
import threading

OS_NAME_MAX = 15            # the kernel's thread name limit (16 bytes with the NUL)
PIGPIO_CALLBACK = "pigpio-cb"


def _libc():
    if not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        fn = libc.pthread_setname_np
    except (OSError, AttributeError):
        return None
    fn.argtypes = (ctypes.c_ulong, ctypes.c_char_p)
    fn.restype = ctypes.c_int
    return fn


_SETNAME = _libc()


def name_os_thread(name: str, thread: threading.Thread | None = None) -> bool:
    """False off Linux or for a thread not running. thread=None names the caller."""
    thread = thread or threading.current_thread()
    if _SETNAME is None or thread.ident is None or not thread.is_alive():
        return False
    try:
        return _SETNAME(thread.ident, name.encode()[:OS_NAME_MAX]) == 0
    except (ctypes.ArgumentError, OSError):
        return False


def name_pigpio_threads() -> int:
    """Name each pigpio connection's callback thread, where the encoders count; returns how many."""
    return sum(name_os_thread(PIGPIO_CALLBACK, t) for t in threading.enumerate()
               if type(t).__name__ == "_callback_thread")


def set_realtime(priority: int) -> str:
    """
    SCHED_FIFO for the calling thread only: it runs the moment it wakes. Needs root or an
    rtprio limit (setup.mk rt-limits). "" if taken, else why not.
    """
    if not hasattr(os, "sched_setscheduler") or not hasattr(os, "SCHED_FIFO"):
        return "not supported here"
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
    except PermissionError as exc:
        return f"not permitted: {exc.strerror or exc} (rtprio limit {_rtprio_limit()})"
    except (OSError, ValueError) as exc:
        return f"refused: {exc}"
    return ""


def _rtprio_limit() -> str:
    try:
        import resource
        return str(resource.getrlimit(resource.RLIMIT_RTPRIO)[0])
    except (ImportError, AttributeError, OSError, ValueError):
        return "?"
