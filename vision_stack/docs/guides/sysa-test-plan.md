# SYS-A: Battery Monitor Chain — Test Plan and Runbook

Revised from the March 2026 SYS-A in `docs/product-spec-subsystems.md` to match the as-built system. Due Wed 2026-10-21 (1–3 page PDF, one test, real data).

## What changed since March (the report must use the as-built values)

| Item | March spec | As built | Source |
|------|-----------|----------|--------|
| Critical cutoff | 9.0 V | **9.9 V** (9.0 V is "empty" for percentage only) | `Power.VOLTAGE_CRITICAL` |
| Display | LCD "LOW BATTERY" | **TM1637 4-digit**: `rdy ` / `Lo-b` / `Stop` / `Err ` | `battery_check.py` |
| Detection | plain threshold | EMA (alpha 0.3) then **3 consecutive samples** to change state | `battery_state.py` |
| Recovery | clears at 11.1 V | WARNING clears at **10.8 V** (hysteresis 0.3 V); **CRITICAL latches** | `battery_state.py` |
| Sampling | 1 Hz | 1 Hz (unchanged) | `start_monitoring(1.0)` |
| Stop path | "alert navigation" | `on_critical` callback / `should_stop()`. **Navigation does not call either yet** (only `battery_check.py` uses `Power`) | grep of `vision_stack/src` |

Predicted from the code: a 12 V to 9 V step needs about 6 samples (EMA lag plus 3-sample confirm), so CRITICAL latches ~5–6 s after the step. The March step 6 requires 2 s. Run the test first and let the data decide; see "Decision to make" below.

## Header (for the report)

**Test SYS-A — Battery Monitor Chain: Accuracy, Threshold Response, and Motor Cutoff**
**Subsystem:** Power + Interface (divider, ADS1115, `Power` monitor, motor cutoff)
**Written / Tested by:** Ana Carlos
**Equipment:** bench supply 0–15 V (current limit 2 A), multimeter (record model), complete bot with wheels off the ground, laptop over SSH, `sysa_power_chain.py` + `analyze_sysa.py`
**Description:** A bench supply replaces the 3S pack and is stepped through the battery range. The test checks that the 30k/10k divider + ADS1115 report voltage within ±2% of a multimeter, that the monitor raises WARNING at 10.5 V, ignores a brief sag, holds WARNING through the hysteresis band, clears at 10.8 V, latches CRITICAL at 9.9 V, and that the wheels actually stop (encoder counts) with the right display code.

## Procedure (fill the Actual column from `summary.json` and your notes)

| # | Procedure | Expected result | Actual | Comments |
|---|-----------|----------------|--------|----------|
| 1 | Supply off. Set 10.2 V, current limit 2 A, connect to the battery lead. Boot, run `python3 -m src.diagnostics.battery_check`. | Prints LOW; display `Lo-b` (preflight rejects a low pack) | | Separate quick run |
| 2 | Set 12.6 V. Reboot, start `sysa_power_chain`. Note preflight line. Type `g` (wheels spin). | Preflight OK, display `rdy `, preflight within ±2% of multimeter | | |
| 3 | Hold 12.6, 12.0, 11.1, 10.8 V, 20 s each. At each: `s <V>` when you change the supply, then `m <multimeter V>` once stable. | ADC mean within ±2% of multimeter at every point; state OK | | Accuracy, healthy range |
| 4 | At 11.1 V: `n sag-begin`, drop supply to 10.0 V for ~2 s, back to 11.1 V, `n sag-end`. | No WARNING; display stays `rdy `; wheels keep spinning | | Motor-start sag rejection |
| 5 | Step to 10.4 V, hold 30 s, `m` reading. | WARNING event; display `Lo-b`; wheels keep spinning | | Record latency |
| 6 | Raise to 10.6 V (above 10.5, below 10.8), hold 30 s. | Stays WARNING (no flicker) | | Hysteresis |
| 7 | Raise to 11.1 V, hold 20 s. | WARNING clears; display `rdy ` | | Recovery |
| 8 | Step 12.0 V then to 9.0 V as fast as the supply allows. | CRITICAL within 2 s of the crossing; wheels stop; display `Stop` | | **Predicted fail (~5 s)** |
| 9 | Raise to 11.1 V, hold 20 s. | Stays CRITICAL; wheels stay stopped | | Latch |
| 10 | Set 9.5 V and 9.0 V, `m` at each. | ADC within ±2% (ADC range check at low end) | | Accuracy, low end |
| 11 | `q`. Run `sudo shutdown -h now` before touching the supply. | Log written; clean shutdown | | **Never hard-cut power** (trixie-feedback#99) |

Optional if a scope is free: probe the divider midpoint (ADS1115 A0) with wheels spinning at the supply's 11.1 V; capture ripple and a motor-start sag. It justifies the EMA and debounce in the report.

## Run order and commands

On the Mac (new file only; the Pi already has `battery.py`, `battery_state.py`, `battery_check.py` from the 10/06 rebase):

```bash
scp docs/sysa-test/sysa_power_chain.py anac@<jarvis-ip>:~/Navilott_Vision/vision_stack/src/diagnostics/
```

On the Pi (wheels off the ground):

```bash
cd ~/Navilott_Vision/vision_stack && source ~/.venv/navilott/bin/activate
sudo pigpiod 2>/dev/null; python3 -m src.diagnostics.sysa_power_chain --out sysa_run1.csv
```

Copy the CSV back, then:

```bash
python3 docs/sysa-test/analyze_sysa.py sysa_run1.csv --outdir docs/sysa-test/results
```

Write down by hand during the run: multimeter model, what the display showed at each step (the log cannot see it), bench supply settling time, and the date.

## Decision to make after the run (Ana's call, not mine)

If step 8 fails at ~5 s, either:
- **Relax the requirement** to something justified (a gradual discharge moves the voltage far less than 0.1 V in 6 s, so the delay only matters for a sudden collapse; check the slope in your P-1 battery data if you have it), or
- **Shorten the response** (fewer confirm samples or a faster EMA for the critical path) and re-run steps 4 and 8, since both trade against sag rejection.

A failing test with data and explanation scores the same as a pass under the rubric, so report whichever you find.

## Wiring gap to raise with Ignacio/Mike

`Power.should_stop()` and `on_critical` are not referenced by navigation in `vision_stack/src`. This test proves the monitor and the motor driver respond, using the harness's spin loop as the stand-in. The end-to-end claim ("alert navigation to stop") is not demonstrated until `src.main` honours it.

## Schedule

- Thu 10/15–Fri 10/16: copy harness to the Pi, dry run with the battery pack (no supply) to check the log and encoders
- Sat 10/17–Sun 10/18: bench run (lab, supply + multimeter)
- Mon 10/19: analyze, write report
- Tue 10/20: proofread, PDF; Wed 10/21: submit
