"""Actuation regression: the soft-start must never command a wheel to reverse."""
import sys; sys.path.insert(0, "vision_stack/src")
from pd_control import PDController, PDConfig

FPS_DT = 0.055
fails = []

def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail else ''}")
    if not cond: fails.append(name)

print("=" * 78)
print("A1  Wall-side start, saturated bad offset (-0.94 norm). 8 frames of ramp.")
print("    Pre-fix: 4 consecutive frames of counter-rotating wheels.")
print("=" * 78)
pd = PDController(PDConfig(kp=0.40, kd=0.05, base_speed=0.45, ramp_seconds=0.75))
rev = 0
for i in range(8):
    L, R = pd.update(-0.94, True, i * FPS_DT)
    if L < 0 or R < 0: rev += 1
    print(f"   f={i} t={i*FPS_DT:.3f}s  L={L:+.4f}  R={R:+.4f}"
          f"{'   <- WHEEL REVERSES' if (L < 0 or R < 0) else ''}")
check("A1 no wheel ever reverses during the ramp", rev == 0, f"{rev} reversing frames")
check("A1 robot still turns (correction is non-zero)",
      abs(pd.update(-0.94, True, 8 * FPS_DT)[0] - pd.update(-0.94, True, 8 * FPS_DT)[1]) > 0.01)

print()
print("=" * 78)
print("A2  Centred start: no derivative kick on frame 0")
print("=" * 78)
pd = PDController(PDConfig())
L0, R0 = pd.update(0.0, True, 0.0)
print(f"   f=0  L={L0:+.4f} R={R0:+.4f}")
check("A2 frame 0 commands no turn when centred", abs(L0 - R0) < 1e-9)

print()
print("=" * 78)
print("A3  Post-intersection: correction decays during a hold, does not persist")
print("=" * 78)
pd = PDController(PDConfig())
for i in range(20):                                  # settle at full ramp
    pd.update(0.30, True, i * FPS_DT)
L, R = pd.update(0.30, True, 20 * FPS_DT)
live = R - L
print(f"   live measurement   : differential = {live:+.4f}")
diffs = []
for i in range(21, 33):                              # 12 blind frames
    L, R = pd.update(0.30, False, i * FPS_DT)        # frozen offset, valid=False
    diffs.append(R - L)
print(f"   blind f+1..f+12    : {' '.join(f'{d:+.3f}' for d in diffs)}")
check("A3 correction decays while invalid", abs(diffs[-1]) < abs(live) * 0.35,
      f"{abs(diffs[-1]):.4f} vs live {abs(live):.4f}")
check("A3 correction never grows while invalid",
      all(abs(diffs[i+1]) <= abs(diffs[i]) + 1e-9 for i in range(len(diffs)-1)))

print()
print("=" * 78)
print("A4  Stop then resume: derivative does not carry a stale error")
print("=" * 78)
pd = PDController(PDConfig())
for i in range(20):
    pd.update(+0.80, True, i * FPS_DT)               # large sustained error
pd.stop()
L, R = pd.update(0.0, True, 21 * FPS_DT)             # resume, now centred
print(f"   first frame after resume: L={L:+.4f} R={R:+.4f}")
check("A4 no differential kick from the pre-stop error", abs(L - R) < 1e-9)
check("A4 soft start re-arms after stop", abs(L) < 1e-9 and abs(R) < 1e-9)

print()
print("=" * 78)
if fails:
    print(f"FAILED: {len(fails)}")
    for f in fails: print(f"   - {f}")
    sys.exit(1)
print("All actuation checks passed.")