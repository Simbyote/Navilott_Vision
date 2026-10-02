# Linker Code Maps

> Each linker's code drawn as diagrams, with the line numbers behind every box: who builds what, and what one frame does.

The linkers are written against **roles**, not hardware. `run()` takes a `source`, `sensors`, a `motor` and a `navigator` and only calls methods on them; `cli()` decides what fills each role (the camera or a replay, the real motors or `NoMotors`). Reading a linker is easier once you split it the same way:

1. **`cli()`, the construction space.** Parse the flags, open the hardware (`linker_io.open_rig()`), build the tuning, call `run()`.
2. **`run()`, the loop space.** Setup, then one iteration per frame, then `finally` (motors stop first), then the report and video.
3. **The chain, one layer down.** `run_chain()` (Phase 2), `processor.process()` (Phase 3), `navigator.update()` (Navigation), `SensorHub` (sensors).

Line numbers are as of 2026-10-02 (`main` after PR #21). If they drift, search for the quoted code.

**Contents**

- [The roles every linker uses](#the-roles-every-linker-uses)
- [The shared loop shape](#the-shared-loop-shape)
- [How the sensors reach a frame](#how-the-sensors-reach-a-frame)
- [phase2_linker](#phase2_linker)
- [phase3_linker](#phase3_linker)
- [maneuver_linker](#maneuver_linker)
- [navigation_linker](#navigation_linker)
- [intersection_linker](#intersection_linker)
- [Side by side](#side-by-side)

---

## The roles every linker uses

| Role | Must provide | Robot (`--camera`) | Replay (`--video` / `--frames`) | Tests |
| --- | --- | --- | --- | --- |
| `source` | `.read()` → `(frame, frame_id, timestamp_ms)`, `(None, None, None)` on a dropped frame, `None` at the end; `.close()`; `.fps` | `live_view.CameraFrameSource` over `capture.camera.CameraSource` | `VideoFrameSource` / `DirectoryFrameSource` | a class over synthetic scenes |
| `sensors` | `.read()` → `(SensorSample, SensorBatch)`; `.sample()`; `.stop()` | `sensing.Sensors(imu=True, encoders=True)` | `None` | a fake returning set samples |
| `motor` | `.drive(left, right)`, `.brake()`, `.stop()` | `drive.MotorController(pigpio.pi())` | `linker_io.NoMotors` | a fake that logs calls |
| `navigator` | `.update(packet)` → `Command`, `.reset()`; optional `.record`, `.finished`, `.outcome` | `navigation.Navigation(gyro_bias_dps, route)` | same | same |
| `system` | `wait_for_start()`, `run_countdown()`, `update_display()`, `show_final_time()`, `cleanup()` | `peripherals.system.System` (button, TM1637) | `None` | `None` or a fake |
| `clock` | returns seconds | `time.perf_counter` | same | `FakeClock` |

`linker_io.open_rig()` (`src/linker_io.py`) fills `source`, `sensors`, `motor` and `system` for the three driving linkers.

```mermaid
flowchart LR
    subgraph CLI["cli(): construction"]
        F["flags"] --> OR["linker_io.open_rig()"]
        F --> TUNE["config = MEASURED<br/>p3_config = replace(MEASURED_ESTIMATION, ...)"]
    end
    OR -->|"camera?"| C1["CameraFrameSource + Sensors"]
    OR -->|"replay?"| C2["VideoFrameSource / DirectoryFrameSource, sensors = None"]
    OR -->|"motors?"| M1["MotorController(pigpio.pi())"]
    OR -->|"no motors"| M2["NoMotors()"]
    OR -->|"button?"| S1["System()"]
    C1 & C2 & M1 & M2 & S1 --> RUN["run(source, sensors, motor, ..., system)"]
    TUNE --> RUN
```

---

## The shared loop shape

`phase3_linker`, `maneuver_linker` and `navigation_linker` all have this shape. `phase2_linker`'s loop lives in `live_view.run()` with the same structure, and `intersection_linker` reuses `navigation_linker`'s.

```mermaid
flowchart TB
    SETUP["Setup<br/>out_dir, CSV writers, recorder, stats,<br/>processor = None"] --> START["Start<br/>button + countdown (if system)<br/>sensors.read() once: empty the buffer"]
    START --> LOOP{"stop condition?<br/>limit, cap, done, finished"}
    LOOP -->|no| READ["item = source.read()"]
    READ -->|None| ENDS["ended_by = source"]
    READ -->|"frame is None"| DROP["drops += 1, continue"] --> LOOP
    READ -->|frame| WORK["sensors -> chain -> decide -> act -> record"]
    WORK --> LOOP
    LOOP -->|yes| FIN
    ENDS --> FIN
    FIN["finally<br/>motor.stop() FIRST<br/>recorder.close(), CSVs, sensors.stop(), source.close()"] --> REPORT["report.json, summary.txt<br/>render the video from the recording"]
```

`try / except KeyboardInterrupt / except Exception / finally` wraps the loop, so Ctrl-C or a crash still stops the motors and leaves complete CSVs.

---

## How the sensors reach a frame

Every linker that drives calls `sensors.read()` or `sensors.sample()` once per frame. That one call sits on top of a background thread (`src/peripherals/sensing.py`):

```mermaid
sequenceDiagram
    participant TH as SensorHub thread
    participant IMU as IMUReader.read() (imu.py, I2C)
    participant ENC as EncoderReader.counts() (drive.py, pigpio)
    participant BUF as hub buffer
    participant FL as linker loop
    Note over TH: started by Sensors() -> SensorHub.open() -> start()
    loop every 10 ms (100 Hz), tick()
        TH->>IMU: read()
        IMU-->>TH: raw gyro Z, accel Y
        TH->>ENC: counts()
        ENC-->>TH: left, right (cumulative)
        TH->>BUF: SensorReading(t, IMU_YAW_SIGN x yaw, accel, left, right)
    end
    FL->>BUF: Sensors.read() -> SensorHub.drain()
    Note over BUF: adds one fresh encoder reading,<br/>takes every reading since the last frame
    BUF-->>FL: SensorBatch
    Note over FL: SensorSample.from_batch(batch)<br/>mean yaw, peak lateral accel,<br/>left / right counts per second
```

`Sensors.read()` returns `(sample, batch)`. Most linkers keep only `sample` (`read()[0]` or `sample()`). `maneuver_linker` also uses `batch`, for its cumulative counts.

---

## phase2_linker

**Purpose:** Phases 1–2 with overlays and per-stage video; no Phase 3, no sensors, no motors. Its loop is `live_view.run()`. `phase2_linker` only supplies *what to do with each frame* (`process`), so `live_view` never imports the pipeline.

### Who calls what

```mermaid
flowchart TB
    MAIN["python3 -m src.phase2_linker<br/>phase2_linker.py:172"] --> LCLI["live_view.cli(run_live_view)<br/>live_view.py:487"]
    LCLI -->|"opens the source :551"| SRC["CameraFrameSource / VideoFrameSource / DirectoryFrameSource"]
    LCLI -->|"runner(source, ...) :566"| RLV["run_live_view(source, config=MEASURED)<br/>phase2_linker.py:131"]
    RLV -->|"builds the closure :164"| PROC["process(frame, id, ts)<br/>= run_chain(frame, id, ts, config, trace=True)"]
    RLV -->|":167"| LRUN["live_view.run(source, process, lane_config)<br/>live_view.py:385"]
    LRUN -->|"each frame :443"| PROC
    PROC --> RC["run_chain()<br/>phase2_linker.py:66"]
    RC --> STAGES["preprocess_frame :99 -> crop_rois :102<br/>-> run_geometry_stage :105 -> run_color_stage :110<br/>-> compute_lane_offset :113 -> compute_stop_line_distance :116<br/>-> fuse_detections :120 -> package_phase2 :123"]
    LRUN -->|":457"| VIEWS["views: LaneView + --views<br/>extract / observe / render"]
```

### One frame

```mermaid
sequenceDiagram
    participant L as live_view.run()
    participant S as source
    participant P as process (closure)
    participant RC as run_chain()
    participant V as views
    L->>S: read()
    S-->>L: (frame, frame_id, timestamp_ms)
    L->>P: process(frame, id, ts)
    P->>RC: run_chain(frame, id, ts, config, trace=True)
    Note over RC: 8 debug stages, each timed by lap()
    RC-->>L: ChainResult (every stage's output + timings_ms)
    L->>L: stats.update, stage_log.write -> stages.csv
    loop each view
        L->>V: extract(chain, frame), observe(data)
        L->>V: render(data) -> ViewWriter (video + csv)
    end
    L->>L: window.show(shots)
```

| Variable | What it is |
| --- | --- |
| `process` | A closure: `config` and `trace` baked in, so `live_view.run()` only passes `(frame, id, ts)` |
| `chain` | `ChainResult`: `.pre`, `.roi`, `.geometry`, `.offset`, `.stop_line`, `.fusion`, `.phase2`, plus each stage's debug dict and `timings_ms` |
| `views` | `debug_lane.LaneView` always, plus `--views stop,traffic,stopline,...`: each turns a `ChainResult` into a picture and a CSV row |

---

## phase3_linker

**Purpose:** Phases 1–3 with optional sensors (`--imu`, `--encoders`); records `p3.csv` and the Phase 3 video. No motors.

### Who calls what

```mermaid
flowchart TB
    CLI["cli()<br/>phase3_linker.py:493"] -->|"opens the source itself"| SRC["CameraFrameSource / Video / Directory"]
    CLI -->|":558"| P3C["p3_config = replace(MEASURED_ESTIMATION,<br/>gyro_bias_dps, cm_per_px)"]
    CLI -->|":571"| RUN["run(source, config, p3_config, use_imu, ...)<br/>:377"]
    RUN -->|":424"| SEN["Sensors(use_imu, use_encoders)"]
    RUN -->|"first frame :444"| MP["make_processor() :365<br/>-> TracedPhase3Processor"]
    RUN -->|"each frame :446"| RPC["run_phase3_chain() :122"]
    RPC -->|":148"| RC["phase2_linker.run_chain()"]
    RPC -->|":150"| PP["processor.process(phase2, sample)"]
    RUN -->|":449"| VIEW["Phase3View: extract / render -> p3_debug.avi"]
    RUN -->|":458, :460"| LOGS["CsvLog -> p3.csv<br/>EventTracker -> terminal"]
```

### One frame

```mermaid
sequenceDiagram
    participant R as run()
    participant S as source
    participant SN as sensors
    participant RPC as run_phase3_chain()
    participant P2 as run_chain()
    participant P3 as TracedPhase3Processor
    R->>S: read()
    S-->>R: (frame, fid, ts)
    R->>SN: sample()
    SN-->>R: SensorSample or None
    Note over R: first frame only: processor = make_processor(...)
    R->>RPC: (frame, fid, ts, processor, sample, config, capture_ms)
    RPC->>P2: run_chain(frame, fid, ts, config)
    P2-->>RPC: ChainResult
    RPC->>P3: process(chain.phase2, sample)
    P3-->>RPC: (EstimationPacket, p3_debug)
    RPC-->>R: Phase3Result(chain, packet, p3_debug, timings_ms)
    R->>R: view.render -> p3_debug.avi, log.write -> p3.csv, events -> terminal
```

| Variable | What it is |
| --- | --- |
| `processor` | `TracedPhase3Processor`, built on the first frame (the cm scale needs the lane ROI width from the frame size). Kept for the run: it holds the EMA, votes and hold counters |
| `res` | `Phase3Result`: `.chain` (Phase 2), `.packet` (`EstimationPacket`), `.p3_debug` (Phase 3's log), `.timings_ms` |
| `events` | `EventTracker`: prints only *changes* (lane status, drive state, stop sign, stop line) |
| `stats` | `Phase3Stats`: timing and counts for `summary.txt` |

---

## maneuver_linker

**Purpose:** a scripted drive trial (settle, yaw-sign spins, a straight leg, a 180° turn, back). `maneuver.Maneuver` decides every command from the **sensors only**; vision runs every frame but only records. **The motors get their command before vision runs**, so the control loop never waits on the camera chain.

### Who calls what

```mermaid
flowchart TB
    CLI["cli()<br/>maneuver_linker.py:371"] -->|"--render DIR: only re-render, exit"| RR["render_run()"]
    CLI -->|":428"| P3C["p3_config = replace(MEASURED_ESTIMATION,<br/>gyro_bias_dps = cfg.gyro_bias_dps)"]
    CLI -->|":430"| OR["open_rig(camera=True, motors, button)"]
    CLI -->|":444"| RUN["run(source, sensors, motor, cfg, ...)<br/>:117"]
    RUN -->|":167"| MAN["machine = Maneuver(cfg, hold)<br/>src/maneuver.py"]
    RUN -->|":169"| PROC["TracedPhase3Processor(p3_config)"]
    RUN -->|":212"| STEP["machine.step(Tick) -> Command"]
    RUN -->|":214-217"| MOT["motor.brake() / drive()"]
    RUN -->|":221"| RPC["run_phase3_chain() (records only)"]
    RUN -->|":229"| REC["recorder.put(fid, frame, _record(res, machine))"]
    RUN -->|":278"| VID["render_run() -> maneuver.avi"]
```

### One frame

```mermaid
sequenceDiagram
    participant R as run()
    participant S as source
    participant SN as sensors
    participant M as Maneuver
    participant MO as motor
    participant RPC as run_phase3_chain()
    R->>S: read()
    S-->>R: (frame, fid, ts)
    R->>SN: read()
    SN-->>R: (sample, batch)
    Note over R: --hold: if holding and button / Enter, machine.resume()
    Note over R: Tick(t, dt, yaw from sample, cumulative counts from batch, cps from sample)
    R->>M: step(tick)
    M-->>R: Command (from the sensors only)
    R->>MO: brake() or drive(left, right)
    Note over R: control is done, vision runs after
    R->>RPC: (frame, fid, ts, processor, sample, ...)
    RPC-->>R: Phase3Result (recorded, never steers)
    R->>R: p3.csv, maneuver.csv, recorder.put, lane re-acquired after the turn?
```

| Variable | What it is |
| --- | --- |
| `machine` | `Maneuver`: the trial's state machine (SETTLE, PULSE_LEFT / RIGHT, FORWARD, TURN, ...). `.step(tick)` → `Command`; `.record` is this frame's row; `.done` ends the loop |
| `tick` | `Tick` (`maneuver.py:91`): one frame's sensors in the trial's terms; `left_count` / `right_count` come from `batch` (cumulative), cps and yaw from `sample` |
| `resume` | `Resume`: `--hold`'s "continue" (button or Enter), polled without blocking |
| `reacquire` | When the lane came back (two boundaries on vision) after the turn, for the report |

---

## navigation_linker

**Purpose:** the course run (`main.py`) with a flight recorder: the whole chain decides and drives, and every frame is recorded.

### Who calls what

```mermaid
flowchart TB
    CLI["cli()<br/>navigation_linker.py"] -->|"--render DIR: only re-render, exit"| RR["render_run()"]
    CLI -->|":426"| RT["route = load_route(--route)<br/>(bad file: exit 2, nothing opened)"]
    CLI -->|":431"| P3C["p3_config = replace(MEASURED_ESTIMATION, ...)"]
    CLI -->|":434"| OR["open_rig(camera / video / frames,<br/>motors, button)"]
    CLI -->|":447"| CD["countdown() if --camera and no button"]
    CLI -->|":450"| RUN["run(source, sensors, motor,<br/>Navigation(gyro_bias_dps, route), ...)<br/>:174"]
    RUN -->|":264"| SEN["sensors.read()[0]"]
    RUN -->|":266"| MP["make_processor() (first frame)"]
    RUN -->|":267"| RPC["run_phase3_chain()"]
    RUN -->|":271"| NAV["enforce(navigator.update(pkt))"]
    RUN -->|":276-279"| MOT["motor.brake() / drive()"]
    RUN -->|":285-305"| REC["row n -> nav.csv, p3.csv, recorder"]
    RUN -->|":308"| SW["stop_when(n) (intersection_linker)"]
    RUN -->|":345"| VID["render_run() -> nav.avi"]
```

### One frame

```mermaid
sequenceDiagram
    participant R as run()
    participant S as source
    participant SN as sensors
    participant RPC as run_phase3_chain()
    participant N as Navigation
    participant MO as motor
    participant W as stop_when
    R->>S: read()
    S-->>R: (frame, fid, ts)
    R->>SN: read()
    SN-->>R: (sample, batch), keep sample
    R->>RPC: (frame, fid, ts, processor, sample, ...)
    RPC-->>R: Phase3Result, pkt = res.packet
    R->>N: update(pkt)
    Note over N: tracker, then StopSign, TrafficLight,<br/>Intersection, EndOfCourse, else lane keeping
    N-->>R: Command (+ navigator.record)
    Note over R: enforce(): BRAKE if the command breaks the contract
    R->>MO: brake() or drive(left, right)
    R->>R: row n -> nav.csv, p3.csv, recorder.put, display
    R->>W: stop_when(n)
    W-->>R: None, or a reason that ends the run
    Note over R: navigator.finished also ends it
```

| Variable | What it is |
| --- | --- |
| `processor` | As in `phase3_linker`: `TracedPhase3Processor`, built on frame 1 |
| `res`, `pkt` | `Phase3Result`, and its `EstimationPacket` |
| `cmd`, `problems` | `enforce()`'s result: the command to drive, and why it was braked if it broke the contract |
| `rec` | `navigator.record`: which rule decided and its details (stage, heading, turn_end, step) |
| `n` | One `nav.csv` row: timings + `rec` + packet fields + the command |
| `event` | Printed only when the reason changes (`-> crossing`, `-> turning`) |
| `stop_when` | A hook: called with `n` after the motors have the command; a string ends the run with that reason |

---

## intersection_linker

**Purpose:** `navigation_linker.run()` with a one-intersection route and a watcher that ends the run once the lane is held after the crossing, then judges PASS or CHECK. It has no loop of its own: it reuses `navigation_linker`'s.

### Who calls what

```mermaid
flowchart TB
    CLI["cli()<br/>intersection_linker.py:189"] --> FOR{"for each maneuver<br/>(one, or all three with --camera)"}
    FOR -->|":233"| OR["open_rig(...) fresh for each sequence"]
    FOR -->|":239"| GO["button, or Enter with --no-button"]
    FOR -->|":242"| RS["run_sequence(maneuver, source, sensors, motor, ...)<br/>:155"]
    RS -->|":177"| SW["watch = SequenceWatch(gyro_bias, settle_s)"]
    RS -->|":178"| NAV["nav = Navigation(gyro_bias, route = Route((maneuver,)))"]
    RS -->|":179"| NR["navigation_linker.run(..., nav, ..., stop_when = watch)"]
    NR -.->|"every frame's row n"| SW
    RS -->|":183"| JD["judge(findings) -> PASS / CHECK"]
    RS --> SJ["sequence.json"]
    FOR --> SUM["summary.txt, report.json<br/>exit 0 all PASS, 1 any CHECK"]
```

### What SequenceWatch does with each row

`SequenceWatch.__call__(n)` (`intersection_linker.py:84`) is the `stop_when` hook. It never touches the robot; it only reads `nav.csv` rows as they're written.

```mermaid
flowchart TB
    N["row n from navigation_linker.run()"] --> DT["dt from n['t']"]
    DT --> STEP["steps = max(steps, n['step'])<br/>(intersections the route counted)"]
    STEP --> ST{"rule == intersection<br/>and a stage?"}
    ST -->|yes| T1["started = True<br/>stage_s[stage] += dt"]
    ST -->|no| H
    T1 --> H{"started?"}
    H -->|yes| HD["heading += (yaw_rate - bias) x dt"]
    H -->|no| TE
    HD --> TE{"first turn_end?"}
    TE -->|yes| TE1["remember turn_end and the heading then"]
    TE -->|no| LK
    TE1 --> LK{"started and rule == lane_keeping?"}
    LK -->|no| RET1["return None"]
    LK -->|yes| SET["crossed = True<br/>settled += dt if lane on vision, else 0"]
    SET --> DONE{"settled >= 1 s?"}
    DONE -->|yes| SD["return 'sequence done'<br/>(navigation_linker ends the run)"]
    DONE -->|no| RET2["return None"]
```

`judge()` (`:116`) then checks the findings: exactly 1 intersection, a turn that ended on the gyro target, ended by "sequence done", heading within 20° of −90 / +90 / 0, no contract breaks.

---

## Side by side

| | phase2_linker | phase3_linker | maneuver_linker | navigation_linker | intersection_linker |
| --- | --- | --- | --- | --- | --- |
| Loop lives in | `live_view.run()` | `run()` | `run()` | `run()` | `navigation_linker.run()` |
| Opens hardware with | `live_view.cli()` | its own `cli()` | `open_rig()` | `open_rig()` | `open_rig()` per sequence |
| Sensors | none | `Sensors(--imu, --encoders)` | `Sensors(imu, encoders)` | `Sensors(imu, encoders)` with `--camera` | same |
| Decides commands | nothing | nothing | `Maneuver.step(Tick)` (sensors only) | `Navigation.update(packet)` | `Navigation` with a one-step route |
| Vision's role | the subject | the subject | recorded only | drives | drives |
| Ends on | source end, `q`, limit | source end, limit | `machine.done`, abort | cap, source end, `navigator.finished`, `stop_when` | `SequenceWatch` ("sequence done"), cap |
| Writes | stages.csv, per-view videos | p3.csv, p3_debug.avi | maneuver.csv, p3.csv, maneuver.avi | nav.csv, p3.csv, nav.avi | a navigation run folder per maneuver + sequence.json |
