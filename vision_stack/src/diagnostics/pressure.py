"""System pressure: what the system waited on around a late frame.

Purpose:
    A frame loop that suddenly takes half a second, its own stages timed as
    normal, was held by something outside it: the CPU, the SD card or
    memory. The kernel counts exactly that (Pressure Stall Information,
    /proc/pressure), and /proc/meminfo says what was still waiting to be
    written. Each long gap gets one row with the system's side of it (the
    2026-10-20 figure eights had 584 ms and 1,080 ms frames no one could
    explain). Without PSI on (psi=1, setup.mk psi), the timings and meminfo alone.

Main package:
    StallLog: stalls.csv, one row per gap of stall_ms or more between frames.

Flow, per frame:
    1. At the frame's arrival, read the PSI files (raw, unparsed: cheap).
    2. A gap of stall_ms or more (not after the run's first frame): one row
       with each resource's stall over it, meminfo and the loop's own timings.
    3. Keep this read as the next frame's "before".
"""
import csv
import os
from pathlib import Path

PROC = Path("/proc")
PSI_RESOURCES = ("cpu", "io", "memory")
# A gap this long between two frames' arrivals is a stall: three camera
# periods, past anything the loop does on its own (p99 between frames ~70 ms
# driving, 2026-10-20) and well under the motor watchdog's 500 ms
STALL_MS = 150.0
STALL_FIELDS = ("frame_id", "t", "gap_ms", "work_ms", "capture_ms", "outside_ms",
                "cpu_some_ms", "io_some_ms", "io_full_ms", "memory_some_ms", "memory_full_ms",
                "dirty_kb", "writeback_kb", "available_mb", "imu_reads", "imu_read_ms")


class Pressure:
    """
    The PSI files, opened once and re-read with pread (no open or seek per read).

    Inputs:
        root: /proc, or a fake tree in tests.

    on: True when every resource's file opened and reads.
    """
    def __init__(self, root: Path | str = PROC) -> None:
        self._fds: dict[str, int] = {}
        for r in PSI_RESOURCES:
            try:
                fd = os.open(Path(root) / "pressure" / r, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.pread(fd, 512, 0)              # opened but unreadable (PSI built in, switched off)
            except OSError:
                os.close(fd)
                continue
            self._fds[r] = fd
        self.on = len(self._fds) == len(PSI_RESOURCES)

    def read(self) -> dict[str, bytes]:
        """Each resource's file as it reads now, unparsed; {} when PSI is off."""
        out = {}
        for r, fd in self._fds.items():
            try:
                out[r] = os.pread(fd, 512, 0)
            except OSError:
                pass
        return out

    @staticmethod
    def totals(raw: dict[str, bytes]) -> dict[str, int]:
        """
        {"cpu_some": us, "io_some": us, "io_full": us, ...}: each line's
        total= from read()'s text ("some avg10=0.00 avg60=0.00 avg300=0.00 total=1234").
        """
        out = {}
        for r, text in raw.items():
            for line in text.decode(errors="replace").splitlines():
                kind, _, rest = line.partition(" ")
                for field in rest.split():
                    if field.startswith("total="):
                        try:
                            out[f"{r}_{kind}"] = int(field[6:])
                        except ValueError:
                            pass
        return out

    def close(self) -> None:
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds = {}


def meminfo(root: Path | str = PROC) -> dict[str, float | None]:
    """Dirty and Writeback (kB, written data not yet on the card) and MemAvailable (MB); None when unreadable."""
    want = {"Dirty": "dirty_kb", "Writeback": "writeback_kb", "MemAvailable": "available_mb"}
    out: dict[str, float | None] = dict.fromkeys(want.values())
    try:
        text = (Path(root) / "meminfo").read_text()
    except OSError:
        return out
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in want:
            try:
                kb = int(rest.split()[0])
            except (IndexError, ValueError):
                continue
            out[want[key]] = round(kb / 1024.0, 1) if key == "MemAvailable" else kb
    return out


class StallLog:
    """
    stalls.csv: one row per frame that arrived stall_ms or more after the
    one before, with what the system stalled on in between.

    Inputs:
        out_dir: The run folder; stalls.csv is created on the first stall.
        stall_ms: The gap that counts.
        pressure: A Pressure (tests pass one over a fake tree); None opens /proc's.
        proc: Where meminfo is read from.

    count: Stalls written. psi_on: whether PSI was readable.
    """
    def __init__(self, out_dir: str | Path, stall_ms: float = STALL_MS, pressure: Pressure | None = None,
                 proc: Path | str = PROC) -> None:
        self.path = Path(out_dir) / "stalls.csv"
        self.stall_ms = stall_ms
        self._p = pressure if pressure is not None else Pressure(proc)
        self._proc = proc
        self.psi_on = self._p.on
        self.count = 0
        self._file = self._writer = None
        self._t = self._raw = None
        self._work_ms = 0.0
        self._frames = 0

    def frame(self, frame_id, t_s: float, capture_ms: float, imu_reads=None, imu_read_ms=None) -> bool:
        """
        A frame arrived at t_s (seconds, the run's clock), its read having
        waited capture_ms. True if the gap before it was a stall (a row was written).
        """
        raw = self._p.read()
        stalled = False
        self._frames += 1
        if self._t is not None and self._frames > 2:          # the gap after the first frame is its start-up
            gap_ms = (t_s - self._t) * 1000.0
            if gap_ms >= self.stall_ms:
                self._write(frame_id, t_s, gap_ms, capture_ms, raw, imu_reads, imu_read_ms)
                stalled = True
        self._t, self._raw = t_s, raw
        return stalled

    def worked(self, ms: float) -> None:
        """The loop's own timed work on the frame just arrived (Phases 2-3, navigation); the next gap's work_ms."""
        self._work_ms = ms

    def _write(self, frame_id, t_s, gap_ms, capture_ms, raw, imu_reads, imu_read_ms) -> None:
        before, after = Pressure.totals(self._raw or {}), Pressure.totals(raw)
        stall = {k: round((after[k] - before[k]) / 1000.0, 1) for k in after if k in before}
        row = {"frame_id": frame_id, "t": round(t_s, 3), "gap_ms": round(gap_ms, 1),
               "work_ms": round(self._work_ms, 1), "capture_ms": round(capture_ms, 1),
               "outside_ms": round(gap_ms - self._work_ms - capture_ms, 1),
               **{f"{k}_ms": stall.get(k) for k in ("cpu_some", "io_some", "io_full", "memory_some", "memory_full")},
               **meminfo(self._proc), "imu_reads": imu_reads, "imu_read_ms": imu_read_ms}
        if self._writer is None:
            self._file = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._file, STALL_FIELDS)
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()                      # a stall row is worth having even if the run dies
        self.count += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = self._writer = None
        self._p.close()
