"""OS-level thread controls: a thread's kernel name, and real-time priority.

Purpose:
    The robot's threads (the frame loop, the sensor hub, pigpio's callbacks,
    the recorder) all show as "python3" in top and /proc unless they name
    themselves, and the sensor hub runs at real-time priority so it wakes on
    time. The drivers, the pipeline and the tools all need these, so they
    live at the bottom with params and utils, and import nothing from src.

Main package:
    name_os_thread(), name_pigpio_threads(): kernel names, cosmetic, never
    raise. set_realtime(): SCHED_FIFO for the calling thread, or why not.
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
    """
    Set a thread's kernel name, as top, ps and /proc show it.

    Inputs:
        name: Cut to OS_NAME_MAX bytes.
        thread: A started thread; None is the calling thread. Thread.ident is
            the pthread handle on Linux, so any started thread can be named.
    Outputs:
        True if the kernel took it; False off Linux, or for a thread not
        started or already finished. Never raises: naming is cosmetic.
    """
    thread = thread or threading.current_thread()
    if _SETNAME is None or thread.ident is None or not thread.is_alive():
        return False
    try:
        return _SETNAME(thread.ident, name.encode()[:OS_NAME_MAX]) == 0
    except (ctypes.ArgumentError, OSError):
        return False


def name_pigpio_threads() -> int:
    """
    Name every running pigpio callback thread PIGPIO_CALLBACK.

    Each pigpio.pi() connection starts one (class _callback_thread) to run
    the pin callbacks: the encoders count in it. Called after a connection
    is made (drive.py); safe to call again.

    Outputs:
        How many were named.
    """
    return sum(name_os_thread(PIGPIO_CALLBACK, t) for t in threading.enumerate()
               if type(t).__name__ == "_callback_thread")


def set_realtime(priority: int) -> str:
    """
    Put the calling thread on SCHED_FIFO at priority: it then runs the
    moment it wakes, ahead of every normal thread, until it blocks again.

    On Linux, sched_setscheduler() with pid 0 changes only the calling
    thread, not the process. It needs CAP_SYS_NICE (root) or an rtprio
    limit at least as high (setup.mk rt-limits).

    Outputs:
        "" if the kernel took it; otherwise why not ("not permitted: ...",
        "not supported here"). Never raises: a thread that can't get the
        priority runs as before.
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
    """The process's soft RLIMIT_RTPRIO, as text; "?" where it can't be read."""
    try:
        import resource
        return str(resource.getrlimit(resource.RLIMIT_RTPRIO)[0])
    except (ImportError, AttributeError, OSError, ValueError):
        return "?"
