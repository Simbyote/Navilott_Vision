#!/usr/bin/env python3
"""Label recorded frames for the labelled replay (src/analysis/evaluate.py), a keypress at a time.

Purpose:
    evaluate.py scores the detectors against labels.csv: ranges of frames and
    what was really in view. Typing those ranges by hand from frame numbers is
    slow and easy to get wrong; here you step through the frames and set what
    is in view, and every frame you step over takes the current label, so a
    long approach is one keypress per change.

Main package:
    Labeller: the state, the keys and labels.csv's ranges, apart from the
    window (tested without a display). main() opens the window, or --template
    writes a labels.csv to fill in by hand.
"""
import argparse
import csv
import sys
from pathlib import Path

from src.analysis.evaluate import LABEL_FIELDS, NONE, find_frames, frames_in, read_labels

FIELDS = LABEL_FIELDS[2:]               # what a frame's label holds
NUMERIC = {"L": "line_cm", "O": "obstacle_cm", "F": "offset_cm", "R": "range_cm"}
SIGN_CYCLE = (None, "yes", "no")
LIGHT_CYCLE = (None, "off", "red", "yellow", "green")

_CLI_HELP = """\
Step through a run's frames and say what's in view; labels.csv is written beside the frames.

    make label RUN=runs/nav_<time>                       the window (needs a display)
    make label RUN=runs/nav_<time> ARGS="--conditions room,glare,dim"
    python3 -m src.scripts.label_frames runs/nav_<time> --template   a file to fill in by hand

Keys (the label shown top left is what each frame you step over takes):
    d / a        next / previous frame       D / A   10 on / back
    s            stop sign: yes, no, unset   l       light: off, red, yellow, green, unset
    digits . -   type a number, then:        L  stop line cm   O  obstacle cm
                                             F  offset cm (+ right)   R  range to sign or light, cm
                 with nothing typed, L and O toggle "none" (not there) and unset
    digits, g    jump to that frame id, labelling nothing on the way (g alone: the first frame)
    digits, f    fill: this frame through that frame id take the current label
    c            next condition (--conditions)       p  label this frame without moving
    e            erase this frame's label            w  write labels.csv    q  write and quit
"""


class Labeller:
    """Frame ids in order; labels[fid] is that frame's label (a dict of FIELDS, unset ones absent)."""
    def __init__(self, fids: list[int], conditions: list[str] | None = None, labels: dict | None = None) -> None:
        self.fids = list(fids)
        self.conditions = [None] + list(conditions or [])
        self.labels: dict[int, dict] = dict(labels or {})
        self.state: dict = {}
        self.pos = 0
        self.typed = ""
        self.dirty = False

    @property
    def fid(self) -> int:
        return self.fids[self.pos]

    def _set(self, field, value) -> None:
        if value is None:
            self.state.pop(field, None)
        else:
            self.state[field] = value

    def _cycle(self, field, options) -> None:
        cur = self.state.get(field)
        self._set(field, options[(options.index(cur) + 1) % len(options)] if cur in options else options[1])

    def paint(self) -> None:
        if self.state:
            self.labels[self.fid] = dict(self.state)
        else:
            self.labels.pop(self.fid, None)
        self.dirty = True

    def step(self, n: int) -> None:
        """Forward paints every frame passed (this one included); back only moves."""
        if n > 0:
            for _ in range(n):
                self.paint()
                if self.pos == len(self.fids) - 1:
                    break
                self.pos += 1
        else:
            self.pos = max(0, self.pos + n)

    def jump(self, typed: str) -> None:
        """To the first frame at or after that id, labelling nothing on the way; nothing typed, the first frame."""
        try:
            target = int(float(typed)) if typed else self.fids[0]
        except ValueError:
            return
        self.pos = next((i for i, f in enumerate(self.fids) if f >= target), len(self.fids) - 1)

    def fill(self, typed: str) -> None:
        """This frame through the one with that id take the current label; ends there."""
        try:
            target = int(float(typed))
        except ValueError:
            return
        while True:
            self.paint()
            if self.pos == len(self.fids) - 1 or self.fids[self.pos + 1] > target:
                break
            self.pos += 1

    def key(self, k: str) -> str | None:
        """One keypress; 'write' or 'quit' when the caller should act."""
        if k in "0123456789.-":
            self.typed += k
            return None
        if k in NUMERIC:
            field = NUMERIC[k]
            if self.typed:
                try:
                    self._set(field, float(self.typed))
                except ValueError:
                    pass
            elif field in ("line_cm", "obstacle_cm"):
                self._set(field, NONE if self.state.get(field) != NONE else None)
            else:
                self._set(field, None)
            self.typed = ""
            return None
        if k == "g":
            self.jump(self.typed)
            self.typed = ""
            return None
        if k == "f" and self.typed:
            self.fill(self.typed)
            self.typed = ""
            return None
        self.typed = ""
        if k == "d":
            self.step(1)
        elif k == "D":
            self.step(10)
        elif k == "a":
            self.step(-1)
        elif k == "A":
            self.step(-10)
        elif k == "s":
            self._cycle("sign", SIGN_CYCLE)
        elif k == "l":
            self._cycle("light", LIGHT_CYCLE)
        elif k == "c":
            cur = self.state.get("condition")
            i = self.conditions.index(cur) if cur in self.conditions else 0
            self._set("condition", self.conditions[(i + 1) % len(self.conditions)])
        elif k == "p":
            self.paint()
        elif k == "e":
            self.labels.pop(self.fid, None)
            self.dirty = True
        elif k == "w":
            return "write"
        elif k == "q":
            return "quit"
        return None

    def rows(self) -> list[dict]:
        """labels.csv's rows: runs of consecutive frames with the same label."""
        out, cur = [], None
        for fid in self.fids:
            lb = self.labels.get(fid)
            if cur is not None and lb == cur["label"] and lb is not None:
                cur["last"] = fid
                continue
            if cur is not None:
                out.append(cur)
            cur = None if lb is None else {"first": fid, "last": fid, "label": lb}
        if cur is not None:
            out.append(cur)
        return [{"first": r["first"], "last": r["last"], **{f: _cell(r["label"].get(f)) for f in FIELDS}} for r in out]


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def write_labels(path, rows: list[dict]) -> Path:
    path = Path(path)
    with open(path, "w", newline="") as f:
        f.write("# labels for src/analysis/evaluate.py: first,last are frame ids, inclusive; blank = not scored\n")
        w = csv.DictWriter(f, LABEL_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def load_labels(path, fids: list[int]) -> dict:
    """An existing labels.csv back into per-frame labels, to carry on where it was left."""
    out = {}
    for lb in read_labels(path):
        label = {}
        for f in FIELDS:
            if f not in lb:
                continue
            v = lb[f]
            label[f] = ("yes" if v else "no") if f == "sign" else v
        for fid in fids:
            if lb["first"] <= fid <= lb["last"]:
                out[fid] = dict(label)
    return out


def template(path, fids: list[int]) -> Path:
    return write_labels(path, [{"first": fids[0], "last": fids[-1], **{f: "" for f in FIELDS}}])


def _overlay(img, lab: Labeller, total: int):
    import cv2
    def put(text, y, color=(255, 255, 255)):
        cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    state = ", ".join(f"{k}={_cell(v)}" for k, v in lab.state.items()) or "nothing set"
    here = lab.labels.get(lab.fid)
    put(f"frame {lab.fid}  ({lab.pos + 1}/{total})   typed: {lab.typed}", 16)
    put(f"label: {state}", 34, (120, 255, 120))
    put("this frame: " + (", ".join(f"{k}={_cell(v)}" for k, v in here.items()) if here else "unlabelled"), 52,
        (255, 220, 120))
    return img


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m src.scripts.label_frames", description=_CLI_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a run folder (frames/ inside) or a folder of frames")
    ap.add_argument("--conditions", default="", help="comma-separated tags that c steps through")
    ap.add_argument("--template", action="store_true", help="write a labels.csv to fill in by hand, no window")
    ap.add_argument("--scale", type=float, default=2.0, help="window zoom (default 2)")
    args = ap.parse_args(argv)
    run = Path(args.run)
    try:
        frames = find_frames(run)
        items = frames_in(frames)
    except FileNotFoundError as e:
        print(f"label: {e}")
        return 2
    fids = [f for f, _ in items]
    out = run / "labels.csv"
    if args.template:
        if out.exists():
            print(f"label: {out} exists; not overwritten")
            return 2
        print(f"wrote {template(out, fids)}: frames {fids[0]}-{fids[-1]}")
        return 0
    import cv2
    try:
        existing = load_labels(out, fids) if out.exists() else {}
    except ValueError as e:
        print(f"label: {e}")
        return 2
    lab = Labeller(fids, [c for c in args.conditions.split(",") if c], existing)
    paths = dict(items)
    win = "label frames"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            img = cv2.imread(str(paths[lab.fid]))
            if img is None:
                lab.step(1)
                continue
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
            cv2.imshow(win, _overlay(img, lab, len(fids)))
            k = cv2.waitKey(0) & 0xFF
            act = lab.key(chr(k)) if k < 128 else None
            if act in ("write", "quit"):
                write_labels(out, lab.rows())
                lab.dirty = False
                print(f"wrote {out}: {len(lab.rows())} ranges, {len(lab.labels)} frames labelled")
            if act == "quit":
                break
    finally:
        if lab.dirty:
            write_labels(out, lab.rows())
            print(f"wrote {out}")
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
