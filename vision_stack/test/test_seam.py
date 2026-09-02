"""
test_seam.py

Regression harness for the Phase 2 <-> Phase 3 seam.

Every case below FAILS on the pre-fix code and PASSES after. Run it before
touching gains: if S1 does not return ~0.0 you are calibrating an artefact.

    python3 test_seam.py
"""
import sys
sys.path.insert(0, "vision_stack/src")

from contracts import Phase2Snapshot, SensorSample, LaneEstimate
from estimation import Phase3Processor, Phase3Config
from lane_offset import compute_lane_offset
from feature_fusion import DetectionObject

W = 480
CFG = Phase3Config(ema_alpha=0.35, vote_window=3, min_confidence_lane=0.30,
                   min_confidence_traffic=1.1, min_confidence_sign=1.1,
                   px_per_meter=(W / 2) / 0.35, lane_half_width_px=W / 2,
                   deadreck_max_frames=10, ema_reset_after_frames=4)

_FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    if not cond:
        _FAILS.append(name)
    print(f"  [{tag}] {name}{('  -> ' + detail) if detail else ''}")


def lane_det(x, conf, y=300.0):
    """A lane_boundary DetectionObject in lane-ROI pixel coordinates."""
    return DetectionObject(type="lane_boundary", position={"x": float(x), "y": y},
                           confidence=conf, timestamp=0,
                           bounding_box=(int(x), int(y), 4, 40),
                           label_detail="lane_boundary")


def fuse_order(dets):
    """feature_fusion forwards all lane candidates sorted by confidence DESC."""
    return sorted(dets, key=lambda d: d.confidence, reverse=True)


def run(frames, cfg=CFG, ts0=1000, step=55):
    """Drive the full seam. frames = list of detection lists."""
    proc = Phase3Processor(cfg)
    out = []
    for i, raw in enumerate(frames):
        dets = fuse_order(raw)
        lr = compute_lane_offset(dets, W, i, ts0 + i * step, conf_threshold=0.30)
        snap = Phase2Snapshot(detections=[], lane=lr.to_lane_estimate(),
                              frame_id=i, timestamp_ms=ts0 + i * step)
        out.append((lr, proc.process(snap, SensorSample())))
    return out


print("=" * 78)
print("S1  Centred lane, two boundaries at +/-120px, confidence ranking alternates")
print("    Truth: 0.0000 every frame.  Pre-fix Phase 3 gave -0.1750 .. +0.0304")
print("=" * 78)
frames = [[lane_det(120, 0.62 if i % 2 == 0 else 0.55, 300),
           lane_det(360, 0.58 if i % 2 == 0 else 0.66, 300)] for i in range(8)]
res = run(frames)
worst = max(abs(p.lane_offset_norm) for _, p in res)
for i, (lr, p) in enumerate(res):
    print(f"   f={i}  P2={lr.offset:+.4f} {lr.mode:<13} "
          f"P3_norm={p.lane_offset_norm:+.4f} valid={p.lane_offset_valid} age={p.lane_offset_age}")
check("S1 P3 offset stays centred", worst < 0.01, f"worst |offset_norm| = {worst:.4f}")
check("S1 P2 and P3 agree", all(abs(lr.offset - p.lane_offset_norm) < 0.01 for lr, p in res))
check("S1 no spurious holds", all(p.lane_offset_valid for _, p in res))

print()
print("=" * 78)
print("S2  Sign convention: lane centre RIGHT of image centre => robot LEFT => offset > 0")
print("=" * 78)
cases = [
    ("centred            (160,320)", [lane_det(160, .6), lane_det(320, .6)], 0.0),
    ("lane right of ctr  (200,440)", [lane_det(200, .6), lane_det(440, .6)], +1.0),
    ("lane left of ctr   ( 40,280)", [lane_det(40, .6), lane_det(280, .6)], -1.0),
]
for label, dets, want_sign in cases:
    lr = compute_lane_offset(fuse_order(dets), W, 0, 0)
    got = 0.0 if abs(lr.offset) < 1e-6 else (1.0 if lr.offset > 0 else -1.0)
    print(f"   {label}: offset={lr.offset:+.4f} mode={lr.mode}")
    check(f"S2 sign {label.split('(')[0].strip()}", got == want_sign)

print()
print("=" * 78)
print("S3  All four modes share one polarity (pre-fix: 3 of 4 were inverted)")
print("=" * 78)
# Robot right of lane centre in every construction => offset must be NEGATIVE.
mode_cases = {
    "two_boundary  ": [lane_det(40, .6), lane_det(280, .6)],
    "left_only     ": [lane_det(40, .6)],
    "width_rejected": [lane_det(40, .6), lane_det(120, .6)],   # 80px < 150 min
}
for mode, dets in mode_cases.items():
    lr = compute_lane_offset(fuse_order(dets), W, 0, 0)
    print(f"   {mode}: offset={lr.offset:+.4f} (mode={lr.mode})")
    check(f"S3 {mode.strip()} is negative", lr.offset < 0)

print()
print("=" * 78)
print("S4  Wall-side start. Lanes at 200/420 (w=220); wall seam at x=15, conf 0.70")
print("    Truth: robot is LEFT of lane centre => offset must be POSITIVE")
print("    Pre-fix Phase 3 gave -0.3281 m (near full scale, wrong direction)")
print("=" * 78)
no_wall = compute_lane_offset(fuse_order([lane_det(200, .61), lane_det(420, .58)]), W, 0, 0)
wall = compute_lane_offset(
    fuse_order([lane_det(200, .61), lane_det(420, .58), lane_det(15, .70, 345)]), W, 0, 0)
print(f"   without wall: offset={no_wall.offset:+.4f} mode={no_wall.mode}")
print(f"   with wall   : offset={wall.offset:+.4f} mode={wall.mode}")
check("S4 without wall: robot reads LEFT of centre", no_wall.offset > 0)
check("S4 with wall: sign survives the wall", wall.offset > 0,
      f"got {wall.offset:+.4f}")
check("S4 wall is not chosen as an anchor",
      wall.left_x == 200.0, f"left anchor = {wall.left_x}")
check("S4 wall presence does not change the estimate",
      abs(wall.offset - no_wall.offset) < 1e-6)

print()
print("=" * 78)
print("S5  Post-intersection. 3 frames lane, 7 frames blind, then reacquire at mid=280")
print("=" * 78)
seq = ([[lane_det(120, .62), lane_det(360, .58)]] * 3
       + [[]] * 7
       + [[lane_det(160, .60), lane_det(400, .63)]] * 8)
res = run(seq)
for i, (lr, p) in enumerate(res):
    mark = ""
    if i == 3:
        mark = "  <- vision lost"
    if i == 10:
        mark = "  <- reacquired"
    print(f"   f={i:>2} P2={lr.offset:+.4f} {lr.mode:<13} P3={p.lane_offset_norm:+.4f} "
          f"valid={str(p.lane_offset_valid):<5} age={p.lane_offset_age:>2} "
          f"stale={str(p.lane_offset_stale):<5} src={p.heading_source}{mark}")
blind = res[3:10]
check("S5 blind frames report valid=False", all(not p.lane_offset_valid for _, p in blind))
check("S5 age increments during the gap", [p.lane_offset_age for _, p in blind] == list(range(1, 8)))
check("S5 no stale flag during a 7-frame gap (below the 10-frame limit)",
      not any(p.lane_offset_stale for _, p in blind))
reacq = res[10][1].lane_offset_norm
truth = res[10][0].offset
check("S5 reacquisition frame adopts truth, not a stale blend",
      abs(reacq - truth) < 0.01, f"P3={reacq:+.4f} vs P2={truth:+.4f}")
post = [p.lane_offset_norm for _, p in res[10:]]
overshoot = max(abs(v) for v in post) - abs(truth)
check("S5 no overshoot past the true value", overshoot < 0.01,
      f"max |post| = {max(abs(v) for v in post):.4f}, truth = {abs(truth):.4f}")

print()
print("=" * 78)
print("S6  Dead-reckoning actually expires (pre-fix: deadreck_max_frames was inert)")
print("=" * 78)
proc = Phase3Processor(CFG)
lr = compute_lane_offset(fuse_order([lane_det(200, .8), lane_det(440, .8)]), W, 0, 1000)
proc.process(Phase2Snapshot([], lr.to_lane_estimate(), 0, 1000), SensorSample())
ages, stales = [], []
for i in range(1, 15):
    p = proc.process(Phase2Snapshot([], LaneEstimate.empty(i, 1000 + i * 55), i, 1000 + i * 55),
                     SensorSample())
    ages.append(p.lane_offset_age)
    stales.append(p.lane_offset_stale)
print(f"   ages   : {ages}")
print(f"   stale  : {[int(s) for s in stales]}")
check("S6 age keeps counting past the limit", ages == list(range(1, 15)))
check("S6 stale asserted exactly past deadreck_max_frames",
      stales[:10] == [False] * 10 and all(stales[10:]))

print()
print("=" * 78)
print("S7  Rate gate rejects a detection swap but state is not poisoned")
print("=" * 78)
proc = Phase3Processor(CFG)
seq = [(+0.10, 1000), (+0.12, 1055), (-0.95, 1110), (+0.14, 1165), (+0.15, 1220)]
outs = []
for i, (o, ts) in enumerate(seq):
    est = LaneEstimate(offset_norm=o, mode="two_boundary", confidence=0.6, valid=True,
                       boundary_count=2, lane_width_px=200.0, frame_id=i, timestamp_ms=ts)
    p = proc.process(Phase2Snapshot([], est, i, ts), SensorSample())
    outs.append(p)
    print(f"   f={i} in={o:+.3f} -> out={p.lane_offset_norm:+.4f} valid={p.lane_offset_valid}")
check("S7 the -0.95 outlier is rejected", outs[2].lane_offset_valid is False)
check("S7 the frame after the outlier is accepted (no poisoned reference)",
      outs[3].lane_offset_valid is True)
check("S7 output never follows the outlier", all(p.lane_offset_norm > 0 for p in outs))

print()
print("=" * 78)
if _FAILS:
    print(f"FAILED: {len(_FAILS)} check(s)")
    for f in _FAILS:
        print(f"   - {f}")
    sys.exit(1)
print("All seam checks passed.")