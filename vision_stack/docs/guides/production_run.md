# Production Run

> Run the course: route on the screen, start button, drive, time on the display. Nothing is recorded.

`src/main.py` is what the robot runs on the course. Every camera frame goes through `Pipeline.step()` (`src/pipeline.py`: Phases 1–3, then Navigation) and the command it returns drives the motors. Nothing is drawn, logged per frame or written to disk. To see *why* the robot did something, run the same chain with `navigation_linker` (`navigation_linker.md`), its instrumented twin; `test_pipeline` holds the two to the same commands frame by frame.

**Code:** `src/main.py`, `src/pipeline.py` · **Route:** `vision_stack/route.json` (`navigation_contract.md`, "The route") · **Tests:** `src/tests/test_main.py`

---

## 1. Running it

```
sudo pigpiod                       # once per boot
python3 -m src.main                # route.json, 300 s cap
python3 -m src.main --route my_route.json
python3 -m src.main --max-run-s 120
```

1. The terminal prints the route: the maneuver count, each maneuver (turns marked `TBD: driven straight`), and the finish. A bad route file stops it here with `route error: ...` (exit 2), before anything opens.
2. It opens the motors (stopped), the display, the IMU and encoders, and the camera. Anything missing stops it with `hardware error: ...` (exit 2) and releases what did open.
3. The display shows **`St N`**, the route's maneuver count, until the start button is pressed.
4. Countdown 5–4–3–2–1, then GO: the clock starts and the robot drives. The display shows the elapsed **MM:SS**.

---

## 2. How it ends

The motors are halted first every time: a short brake for 0.3 s (`HALT_BRAKE_S`), then standby. Then the camera and sensors are released and the terminal prints one line, e.g. `finished at step 3 after 00:42 (840 frames)`.

| Ending | Why | Display afterwards |
| --- | --- | --- |
| **finished** | The route's finish: the lane running out after the last maneuver (`edge`), or the finish stop line (`stop_line`) | The final time |
| **ended early** | The lane stayed lost before the route was done: a safety stop | Alternates **`E  N`** (N = the step reached) and the time, every 2 s, until Ctrl-C |
| **run time cap** | `--max-run-s` (300 s) passed: the end was never seen | Alternates **`E  t`** and the time until Ctrl-C |
| **Ctrl-C** | You stopped it | The time so far |
| **error** | An exception (e.g. the camera died); the traceback is printed | **`Err `** |

The last screen stays up after the program exits (the display isn't blanked).

---

## 3. What it uses

- **Tuning:** `MEASURED` and `MEASURED_ESTIMATION` from `src/config.py`, with the gyro bias `navigation_linker` defaults to (`MANEUVER.gyro_bias_dps`), so a production run and a default `navigation_linker --camera` run drive alike.
- **Route:** `config.ROUTE_PATH` (`vision_stack/route.json`) unless `--route` is given.
- **Sensors:** IMU and both encoders, on `SensorHub`'s 100 Hz thread, one sample per frame.

---

## 4. Known limits

- **Turns** are TBD: every intersection is crossed straight, whatever the route says.
- **Intersection count:** every stop line passing under the view is the next step. A missed or false line shifts the route; replay the course with `navigation_linker` and check `step` in `nav.csv`.
- **End of course:** the lane lost for ~1.35 s ends the run (`END_STALE_MS`). Glare that long ends it early, shown as `E  N`.
- Stop sign and traffic light gates: see `navigation_linker.md`, "Known limits".
