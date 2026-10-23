"""Pad timing: how long each element of a GStreamer pipeline holds each frame, by buffer probes.

Purpose:
    GStreamer's latency tracer stalls libcamerasrc before PLAYING on the
    robot's Pi (2026-10-14). This measures the same from inside the
    pipeline: a probe on each element's pads stamps each frame in and out,
    with nothing injected into the stream. Stdlib and PyGObject only, so the
    system python3 can run it when the venv can't import gi.

Main package:
    Its stdout, after the run: "[pad-timing] KIND ELEMENT TS_MS MS" rows,
    KIND element (sink pad in to src pad out), pipeline (source to sink) or
    camera (sensor timestamp to the source's pad), then EOS, ERROR or NOGI.

Flow:
    1. python3 pad_timing.py "PIPELINE" (one that ends itself: num-buffers).
    2. Probes on every pad; run to EOS or an error.
    3. Print the rows.
"""
import sys
import time

TAG = "[pad-timing]"
SKIP_FACTORIES = ("capsfilter",)     # pass-through: ~0 ms, only noise in the table


def _out(*fields) -> None:
    print("\t".join([TAG, *map(str, fields)]), flush=True)


def _elements(pipe, Gst) -> list:
    it, found = pipe.iterate_elements(), []
    while True:
        r, el = it.next()
        if r != Gst.IteratorResult.OK:
            return found
        found.append(el)


def main(pipeline: str) -> int:
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except (ImportError, ValueError) as exc:
        _out("NOGI", exc)
        return 3
    Gst.init(None)
    try:
        pipe = Gst.parse_launch(pipeline)
    except Exception as exc:                # GLib.Error: a pipeline that doesn't parse
        _out("ERROR", f"parse_launch: {exc}")
        return 1

    t0 = time.monotonic_ns()
    rows = []                       # (kind, element, t_ns, ms), printed after the run
    entered = {}                    # (element, pts) -> ns, at its sink pad
    left_source = {}                # pts -> ns, at the source's src pad
    source = []
    OK = Gst.PadProbeReturn.OK

    def on_in(pad, info, name):
        b = info.get_buffer()
        if b is not None:
            entered[(name, b.pts)] = time.monotonic_ns()
        return OK

    def on_out(pad, info, name):
        now, b = time.monotonic_ns(), info.get_buffer()
        t = entered.pop((name, b.pts), None) if b is not None else None
        if t is not None:
            rows.append(("element", name, t, (now - t) / 1e6))
        return OK

    def on_source(pad, info, name):
        now, b = time.monotonic_ns(), info.get_buffer()
        if b is None:
            return OK
        left_source[b.pts] = now
        clock = pipe.get_clock()
        if clock is not None and b.pts != Gst.CLOCK_TIME_NONE:
            running = clock.get_time() - pipe.get_base_time()
            rows.append(("camera", name, now, (running - b.pts) / 1e6))
        return OK

    def on_sink(pad, info, name):
        now, b = time.monotonic_ns(), info.get_buffer()
        t = left_source.pop(b.pts, None) if b is not None else None
        if t is not None and source:
            rows.append(("pipeline", f"{source[0]} -> {name}", t, (now - t) / 1e6))
        return OK

    for el in _elements(pipe, Gst):
        factory = el.get_factory()
        if factory is not None and factory.get_name() in SKIP_FACTORIES:
            continue
        name, sink, src = el.get_name(), el.get_static_pad("sink"), el.get_static_pad("src")
        if sink is None and src is not None:
            source.append(name)
            src.add_probe(Gst.PadProbeType.BUFFER, on_source, name)
        elif src is None and sink is not None:
            sink.add_probe(Gst.PadProbeType.BUFFER, on_sink, name)
        elif src is not None and sink is not None:
            sink.add_probe(Gst.PadProbeType.BUFFER, on_in, name)
            src.add_probe(Gst.PadProbeType.BUFFER, on_out, name)

    bus = pipe.get_bus()
    pipe.set_state(Gst.State.PLAYING)
    msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.EOS | Gst.MessageType.ERROR)
    pipe.set_state(Gst.State.NULL)
    for kind, name, t, ms in rows:
        _out(kind, name, f"{(t - t0) / 1e6:.3f}", f"{ms:.4f}")
    if msg is not None and msg.type == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        _out("ERROR", f"{err.message} ({dbg})" if dbg else err.message)
        return 1
    _out("EOS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))