# Documentation Standard

> How code is documented in Navilott_Vision. Adapted from Real Python's
> *Documenting Python Code*, trimmed to the parts that earn their lines.

---

## Principles

1. **Each layer answers a different question.** File docstring: *where does this fit?* Function docstring: *what is the contract?* Comment: *why is this line like this?* If two layers say the same thing, delete one.
2. **Types live in signatures, not prose.** Every public function has type hints. Docstrings describe behavior, not types.
3. **Documentation should shrink when code is refactored, not drift.** Don't list what a module contains or how a function works step by step. Those details go stale.
4. **When unsure, cut.** An outdated docstring is worse than a missing one.

---

## File Docstrings

Every module opens with a docstring that explains its role in the system.

| Section      | Answers                                                                 |
|--------------|-------------------------------------------------------------------------|
| Summary      | One line: what this module does for the pipeline.                       |
| Purpose      | Why it exists and which problem it owns. 2–4 sentences.                   |
| Main package | The primary data structure the module produces and what it represents.  |
| Flow         | The ordered stages the module performs, at pipeline level.              |

Rules:

- The **main package** is named explicitly (e.g., `LaneOffsetResult`). A module built around no single output type is a signal to split or merge it.
- The **Flow** matches the order of the code and stays high-level: one line per stage, with no details that belong in a function docstring.
- Do **not** list every class or function. The editor already provides that index.
- Target: 8–20 lines.

```python
"""Lateral lane offset estimation from a binary lane mask.

Purpose:
    Converts detected lane edges into the robot's lateral position within
    the lane. This is the primary steering input to PD control.

Main package:
    LaneOffsetResult: normalized lateral offset in [-1, 1] (negative = left
    of center), a detection mode, and a confidence score. Mode "none" means
    no usable lane was found this frame.

Flow:
    1. Crop the mask to the lookahead ROI.
    2. Locate left/right lane boundaries per scan row.
    3. Estimate lane center, falling back to single-edge mode if one is missing.
    4. Normalize the offset against image width and package the result.
"""
```

---

## Function Docstrings

### Full form: public functions and pipeline stages

```
Summary line.

Purpose:  (optional; only if the summary line isn't enough)
Inputs:   How each parameter changes behavior. Units and ranges, not types.
Outputs:  What the returned value represents.
Side effects: (only if any) Hardware, I/O, shared state, mutation.
```

- **Inputs** describe *effect*: "Larger values widen the search band and tolerate curves but admit more noise." They do not describe type: "int, the band width."
- **Always state units and ranges** where they're ambiguous: px vs. normalized, BGR vs. HSV, degrees vs. radians, 0–255 vs. bool mask.
- Parameters that are self-explanatory from name + type (`frame`, `timestamp`) can be skipped.
- **Side effects** are required when the function touches the camera, UART/GPIO, pigpio, files, or mutates an argument. Omit the line for pure functions.

```python
def estimate_offset(mask: np.ndarray, scan_rows: int = 8,
                    min_edge_px: int = 12) -> LaneOffsetResult:
    """Estimate lateral lane offset from a binary lane mask.

    Inputs:
        mask: 0/255 single-channel mask, already ROI-cropped.
        scan_rows: Rows sampled from the bottom up. More rows smooth the
            estimate on curves but weight far-field noise more heavily.
        min_edge_px: Minimum run length to count as a lane edge. Raise it
            if glare or tape seams cause false edges.

    Outputs:
        LaneOffsetResult for this frame. Mode "single" means only one
        edge was found and the center was inferred from nominal lane width.
    """
```

### Short form: private helpers and trivial functions

A single summary line, or nothing if the name says it all.

```python
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _row_edges(row: np.ndarray) -> tuple[int, int] | None:
    """Leftmost and rightmost lit pixel in a mask row, or None if empty."""
```

**Rule of thumb:** if the docstring would be longer than the function body and the function is private, use the short form.

---

## Classes and Dataclasses

- Class docstring: one-line purpose, plus any **invariants** (e.g., "confidence is 0 whenever mode is 'none'").
- Dataclass fields: document only fields whose meaning, units, or sign convention isn't obvious. Use a trailing comment on the field.
- Do not list methods or attributes wholesale.

```python
@dataclass(frozen=True)
class LaneOffsetResult:
    """Per-frame lane position estimate. confidence == 0 when mode == "none"."""
    offset: float        # normalized [-1, 1], negative = robot left of center
    mode: str            # "dual" | "single" | "none"
    confidence: float
```

---

## Inline Comments

Comments explain **why**, never **what**. Priority targets:

1. **Tuned constants.** Where did the value come from and what is it sensitive to?
2. **Non-obvious OpenCV/NumPy behavior.** Format quirks, in-place ops, axis order.
3. **Workarounds.** Hardware or library behavior being worked around.

```python
# Good: captures provenance and sensitivity
YELLOW_HSV_LO = (18, 80, 90)   # tuned on course frames 2026-09; V floor drops ~20 under shade

# Good: flags a non-obvious format
frame = frame[:, :, ::-1]      # videoconvert emits BGR; model expects RGB

# Bad: restates the code
count += 1                      # increment count
```

Delete:

- Comments that restate the line below them.
- Section-banner comments inside short functions (`# ---- Step 2 ----`). If a function needs banners, it needs splitting.
- Commented-out code. Git has it.
- Change-history comments (`# fixed 9/12`). Git has it.

---

## Tests

pytest prints the test name on failure, so **the name is the docstring**. It states the behavior under test: `test_failed_read_returns_none_and_costs_no_frame_id`, not `test_read_2`.

- **Module docstring:** the module under test, and what each mode (`--software`, `--hardware`) covers. No Main package or Flow; a test file produces nothing downstream.
- **Test functions:** no docstring. If the name can't carry the behavior, rename the test or split it.
- **Fakes and helpers:** one line saying what they stand in for or produce.
- **Comments** answer only:
  1. *Why the contract matters*: what breaks downstream if this test fails.
  2. *Why the setup is shaped this way*: a scripted sequence, a filter, an input range chosen to expose the behavior.
  3. *Where an expected value comes from*: known answers (`# 0.299 * 255`) and every tolerance or threshold.

```python
# Good: says what depends on the contract
def test_timestamps_never_decrease(monkeypatch):
    # Downstream dt math assumes this; a wall clock would go negative on NTP steps

# Good: explains an expected value
assert abs(int(gray.mean()) - 76) <= 1           # 0.299 * 255, ±1 for rounding

# Bad: restates the name
def test_read_before_open_raises():
    # checks that read() before open() raises
```

---

## Checklist (per file)

- [ ] Module docstring has Summary, Purpose, Main package, Flow; no symbol inventory
- [ ] Public functions: type hints + Inputs/Outputs describing effect, units, ranges
- [ ] Side effects stated where hardware/I/O/mutation occurs
- [ ] Private helpers: one line or none
- [ ] Every magic number has a provenance comment or is a named constant
- [ ] No what-comments, banners, dead code, or history comments

**Test files** replace the first two items with:

- [ ] Module docstring names the target module and what each mode covers
- [ ] Test names state the behavior; no test docstrings
- [ ] Fakes/helpers have a one-line docstring
- [ ] Every tolerance, threshold and known-answer value has its source in a comment
