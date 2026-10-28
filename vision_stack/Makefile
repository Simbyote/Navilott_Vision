# Navilott: every command in one place. Run from vision_stack/:
#
#     make                       the list of commands
#     make run                   the course (start button, time on the display)
#     make nav-dry               the whole chain on the bench, motors off
#     make phase2 VIEWS=traffic  watch perception, with the traffic-light view
#
# Options are make variables: make <target> NAME=value. The ones most used:
#     VIDEO=clip.mp4 / FRAMES=dir   replay instead of the camera (linkers)
#     RUN=runs/<folder>             which run an analysis reads (default: the newest)
#     ARGS="..."                    anything else, passed straight to the command
# docs/README.md says where each part of the system is described.
#
# Session start and the Pi's install live in ../session.mk and ../setup.mk; this
# file runs the robot's code. `make session TIME="YYYY-MM-DD HH:MM"` here is
# session.mk's session (clock, pigpiod, venv shell), and `pigpiod` is its
# pigpiod-start, so each step is written once. Commands run on the venv
# setup.mk creates (VENV_DIR, the same default as there), so it needn't be
# activated.

SESSION_MK := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))../session.mk)
VENV_DIR   ?= $(HOME)/.venv/navilott
PY         ?= $(if $(wildcard $(VENV_DIR)/bin/python),$(VENV_DIR)/bin/python,python3)

# --- options (make <target> NAME=value) ---------------------------------------
# Comments sit above each line: one after the value would become part of it.
# course plan for run / navigate
ROUTE     ?= route.json
# cap a run's length (s); empty = the command's default
MAX_RUN_S ?=
# replay this video instead of the camera
VIDEO     ?=
# replay this frame folder instead of the camera
FRAMES    ?=
# phase2 extra views: lanegeo,stop,traffic,stopline
VIEWS     ?=
# intersection: straight, left, right or all
MANEUVER  ?= all
# run folder an analysis reads; empty = the newest
RUN       ?=
# pi-load: the navigation run the recording watched
NAV       ?=
# compare: the older run
BASE      ?=
# compare: the newer run
NEW       ?=
# soak length
MINUTES   ?= 15
# ab-parallel: runs in all (alternating serial, parallel) and each one's seconds
RUNS      ?= 4
SECONDS   ?= 60
# frames per hardware test
HW_FRAMES ?= 100
# calib-lamps / sweep-lamps: "red=runs/lamp_red green=runs/nav_X:0-86 ..."
LAMPS     ?=
# session: the clock, "YYYY-MM-DD HH:MM" (the Pi has no clock of its own); make hands it,
# like every NAME=value given here, to session.mk
TIME      ?=
# monitor-attach: watch the newest process matching this
MATCH     ?= src.
# camera controls for this run, over params.CAMERA_CONTROLS: CAM="awb-mode=daylight saturation=1.5"
CAM       ?=
# sweep-camera: settings to try, split by |; empty = the backlight set
CAM_TRY   ?=
# extra arguments for the command
ARGS      ?=

# A linker's frame source: the camera unless VIDEO or FRAMES is given
SOURCE  = $(if $(VIDEO),--video $(VIDEO),$(if $(FRAMES),--frames $(FRAMES),--camera))
RUNCAP  = $(if $(strip $(MAX_RUN_S)),--max-run-s $(strip $(MAX_RUN_S)))
# The camera needs pigpiod (the encoders, motors, button); a replay doesn't
HW      = $(if $(VIDEO)$(FRAMES),,pigpiod)
# CAM as --camera-control flags, for the live camera only
CAMFLAGS = $(if $(VIDEO)$(FRAMES),,$(foreach c,$(CAM),--camera-control $(c)))
NEWEST  = $(shell ls -dt runs/$(1)*/ 2>/dev/null | head -1)
RENDER_RUN = $(or $(strip $(RUN)),$(call NEWEST,nav_))

.DEFAULT_GOAL := help
.PHONY: help session pigpiod run phase2 phase3 maneuver maneuver-dry navigate nav-dry intersection \
        intersection-dry render routines camera-controls sweep-camera diag-nav diag-run monitor-attach capture-anatomy frame-meta i2c-trace sched-latency pi-load nav-report timeline timing \
        jitter camera-meta capture-report sched-report loop-budget light-report load-compare fig8-report motor-health obstacle-map gates sign-check stability hw-report soak-report routine-report compare test test-hw soak lint calib-camera calib-stopline calib-lane calib-odometry compare-parallel ab-parallel \
        calib-ground calib-lamps sweep-lamps code-cards newest clean

##@ Start here
help: ## this list
	@awk 'BEGIN {FS = ":.*## "} \
	     /^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} \
	     /^[a-z0-9-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Options: make <target> NAME=value, e.g. VIDEO=clip.mp4  RUN=runs/nav_...  ARGS=\"--scale 2\""
	@echo "Python: $(PY)"

session: ## every boot on the Pi: set the clock, start pigpiod, open a venv shell (TIME="YYYY-MM-DD HH:MM")
	@$(MAKE) --no-print-directory -f $(SESSION_MK) session

pigpiod: ## start the GPIO daemon if it isn't running (motors, encoders, button, display)
	@$(MAKE) --no-print-directory -f $(SESSION_MK) pigpiod-start

##@ Race day
run: $(HW) ## the course: route, St N, button, countdown, time on the display; nothing recorded
	$(PY) -m src.main --route $(strip $(ROUTE)) $(RUNCAP) $(ARGS)

##@ Linkers: the chain up to one point, recorded
phase2: ## Phases 1-2 live with video (VIDEO= / FRAMES= to replay, VIEWS=lanegeo,stop,traffic,stopline,obstacle)
	$(PY) -m src.phase2_linker $(SOURCE) $(if $(strip $(VIEWS)),--views $(strip $(VIEWS))) $(CAMFLAGS) $(ARGS)

phase3: $(HW) ## Phases 1-3 with the IMU and encoders: is the packet right across frames?
	$(PY) -m src.phase3_linker $(SOURCE) $(if $(VIDEO)$(FRAMES),,--imu --encoders) $(CAMFLAGS) $(ARGS)

maneuver: $(HW) ## drive trial: yaw-sign spins, a leg, a 180 turn, back (motors on)
	$(PY) -m src.maneuver_linker $(RUNCAP) $(CAMFLAGS) $(ARGS)

maneuver-dry: $(HW) ## the drive trial with the motors off, started from the console
	$(PY) -m src.maneuver_linker --no-motors --no-button $(RUNCAP) $(CAMFLAGS) $(ARGS)

navigate: $(HW) ## the whole chain driving the route, recorded (motors on, start button)
	$(PY) -m src.navigation_linker $(SOURCE) --route $(strip $(ROUTE)) $(RUNCAP) $(CAMFLAGS) $(ARGS)

nav-dry: $(HW) ## the whole chain on the bench: motors off, console start (or a replay with VIDEO=)
	$(PY) -m src.navigation_linker $(SOURCE) $(if $(VIDEO)$(FRAMES),,--no-motors --no-button) \
	    --route $(strip $(ROUTE)) $(RUNCAP) $(CAMFLAGS) $(ARGS)

intersection: $(HW) ## one intersection per run, judged PASS/CHECK (MANEUVER=straight|left|right|all)
	$(PY) -m src.intersection_linker $(SOURCE) $(strip $(MANEUVER)) $(RUNCAP) $(CAMFLAGS) $(ARGS)

intersection-dry: $(HW) ## the intersection sequences with the motors off, Enter starts each
	$(PY) -m src.intersection_linker $(SOURCE) $(strip $(MANEUVER)) \
	    $(if $(VIDEO)$(FRAMES),,--no-motors --no-button) $(RUNCAP) $(CAMFLAGS) $(ARGS)

render: ## render an earlier navigation run's video (RUN=runs/nav_..., default the newest)
	@test -n "$(RENDER_RUN)" || (echo "no runs/nav_* folder: make render RUN=<run folder>"; exit 2)
	$(PY) -m src.navigation_linker --render $(RENDER_RUN) $(ARGS)

##@ Camera: what the camera's own controls do to the picture
camera-controls: ## every camera control, its values, and what params.CAMERA_CONTROLS sets
	$(PY) -m src.scripts.sweep_camera --list

sweep-camera: ## try camera settings one after another on the live view and score sign, light, lane (CAM_TRY="a=1|b=2 c=3")
	$(PY) -m src.scripts.sweep_camera $(if $(strip $(CAM_TRY)),--try "$(strip $(CAM_TRY))") $(ARGS)

##@ Routines: prompted hardware tests, judged PASS / FAIL
routines: ## list the routines; run one with make routine-<name> (ARGS="--trials 10 --notes ...")
	$(PY) -m src.routines --list

# make routine-<name>: one routine, by its name
routine-%:
	$(PY) -m src.routines $* $(ARGS)

##@ Diagnostics: how the run uses the Pi
diag-nav: $(HW) ## record threads, cores, heat while nav-dry runs; then: make pi-load
	$(PY) -m src.diagnostics.monitor -- $(PY) -m src.navigation_linker --camera --no-motors --no-button \
	    --route $(strip $(ROUTE)) $(RUNCAP) $(CAMFLAGS) $(ARGS)

diag-run: $(HW) ## record the course run itself (motors on, start button); then: make pi-load
	$(PY) -m src.diagnostics.monitor -- $(PY) -m src.main --route $(strip $(ROUTE)) $(RUNCAP) $(ARGS)

monitor-attach: ## record a process already running (MATCH=navigation_linker); Ctrl-C to stop
	$(PY) -m src.diagnostics.monitor --match $(strip $(MATCH)) $(ARGS)

capture-anatomy: ## the camera path once: media graph, ISP caps, ms per element, VideoCore (camera free)
	$(PY) -m src.diagnostics.capture_anatomy $(CAMFLAGS) $(ARGS)

frame-meta: ## every frame's exposure, gains, sensor-to-Python time and traffic label (camera free; Picamera2)
	$(PY) -m src.diagnostics.frame_meta $(CAMFLAGS) $(ARGS)

i2c-trace: ## every I2C transfer from the kernel for 10 s (sudo; start a run first): bus load, IMU timing, errors
	sudo $(PY) -m src.diagnostics.i2c_trace $(ARGS)

sched-latency: ## how late Linux wakes a thread vs the 50 ms / 10 ms deadlines (sudo; ARGS="--label idle", then loaded)
	sudo $(PY) -m src.diagnostics.sched_latency $(ARGS)

##@ Analysis: graphs and findings from a run; RUN= picks the run, default the newest
pi-load: ## threads, cores, heat, memory from a diag_ recording (NAV=runs/nav_... to judge only the run)
	$(PY) -m src.analysis.pi_load $(strip $(RUN)) $(if $(strip $(NAV)),--run $(strip $(NAV))) $(ARGS)

nav-report: ## a navigation run: rules, each intersection and the 2 s after, wheels, latency
	$(PY) -m src.analysis.nav_run $(strip $(RUN)) $(ARGS)

timeline: ## how long each rule, stage and state lasted, and what changed into what
	$(PY) -m src.analysis.state_timeline $(strip $(RUN)) $(ARGS)

timing: ## where each frame's time goes, against the 50 ms budget
	$(PY) -m src.analysis.stage_timing $(strip $(RUN)) $(ARGS)

compare-parallel: ## serial vs parallel perception on the same frames (ARGS="--frames runs/nav_<time>/frames" or "--camera")
	$(PY) -m src.analysis.parallel_compare $(ARGS)

ab-parallel: $(HW) ## serial vs parallel over whole dry runs, alternated (RUNS=4 SECONDS=60; ARGS=--motors to drive)
	$(PY) -m src.analysis.parallel_ab --runs $(RUNS) --seconds $(SECONDS) $(ARGS)

jitter: ## frame-interval tails, over-budget streaks, periodic spikes
	$(PY) -m src.analysis.jitter $(strip $(RUN)) $(ARGS)

camera-meta: ## what auto exposure and white balance chose per frame: exposure, gain, colour, lux (RUN=runs/meta_...)
	$(PY) -m src.analysis.camera_meta $(strip $(RUN)) $(ARGS)

capture-report: ## a capture-anatomy run as graphs: frame age at the sink, ARM time, each step per frame (RUN=...)
	$(PY) -m src.analysis.capture_report $(strip $(RUN)) $(ARGS)

sched-report: ## a sched-latency run as graphs: spread, deadlines, worst vs another run (RUN=... BASE=<idle run>)
	$(PY) -m src.analysis.sched_report $(strip $(RUN)) $(if $(strip $(BASE)),--base $(strip $(BASE))) $(ARGS)

loop-budget: ## where a navigation run's frame goes: capture, phases, outside the timers, by lane mode
	$(PY) -m src.analysis.loop_budget $(strip $(RUN)) $(ARGS)

light-report: ## the traffic light frame by frame: each reading, the vote, when it braked (render the run first)
	$(PY) -m src.analysis.light_report $(strip $(RUN)) $(ARGS)

load-compare: ## two diag_ recordings before and after a change: threads, cores, system (BASE=before NEW=after)
	@test -n "$(strip $(BASE))" -a -n "$(strip $(NEW))" || (echo "usage: make load-compare BASE=<diag before> NEW=<diag after>"; exit 2)
	$(PY) -m src.analysis.load_compare $(strip $(BASE)) $(strip $(NEW)) $(ARGS)

fig8-report: ## a figure-eight run's turns and laps as graphs (RUN=runs/routine_figure-eight_..., BASE= an earlier one)
	$(PY) -m src.analysis.fig8_report $(strip $(RUN)) $(if $(strip $(BASE)),--base $(strip $(BASE))) $(ARGS)

motor-health: ## each wheel's speed per unit duty across a run: a motor failing, a wheel driven but not turning
	$(PY) -m src.analysis.motor_health $(strip $(RUN)) $(ARGS)

obstacle-map: ## several obstacle-check runs as one object-by-gap grid, misses with their gate (RUN=<folder of runs>)
	$(PY) -m src.analysis.obstacle_map $(strip $(RUN)) $(ARGS)

gates: ## which detector gate refuses the most candidates
	$(PY) -m src.analysis.gate_rejections $(strip $(RUN)) $(ARGS)

sign-check: ## why a stop sign in view isn't detected: each gate, a larger red blob hiding it (ARGS="--frames runs/nav_<time>/frames")
	$(PY) -m src.analysis.sign_check $(ARGS)

stability: ## offset noise and lane-mode flicker on a still scene (record parked and centered)
	$(PY) -m src.analysis.stability $(strip $(RUN)) $(ARGS)

hw-report: ## every analysis a pytest --hardware run feeds, figures next to each CSV (RUN=artifacts/<time>)
	$(PY) -m src.analysis.hw_report $(strip $(RUN)) $(ARGS)

soak-report: ## heat, throttling, memory trend and loop time of a soak (RUN=artifacts/<time>)
	$(PY) -m src.analysis.soak $(strip $(RUN)) $(ARGS)

routine-report: ## a routine run as figures: criteria, its own plot, every trial; nav_run per attempt (RUN=runs/routine_...)
	$(PY) -m src.analysis.routine_report $(strip $(RUN)) $(ARGS)

compare: ## every number that moved between two runs (BASE=... NEW=...)
	@test -n "$(strip $(BASE))" -a -n "$(strip $(NEW))" || (echo "usage: make compare BASE=<older run> NEW=<newer run>"; exit 2)
	$(PY) -m src.analysis.compare_runs $(strip $(BASE)) $(strip $(NEW)) $(ARGS)

##@ Tests
test: ## the software tests: any machine, no hardware
	$(PY) -m pytest -q $(ARGS)

test-hw: $(HW) ## hardware characterization on the Pi (HW_FRAMES=600 for steadier numbers)
	$(PY) -m pytest --hardware --frames=$(strip $(HW_FRAMES)) $(ARGS)

soak: ## the soak test for MINUTES (default 15), recorded with diagnostics; then: make soak-report
	$(PY) -m src.diagnostics.monitor -- $(PY) -m pytest --hardware --soak-minutes=$(strip $(MINUTES)) \
	    src/tests/test_soak.py $(ARGS)

lint: ## pyflakes over src/
	$(PY) -m pyflakes src

##@ Calibration
calib-camera: ## lens: capture checkerboard frames, solve, verify
	$(PY) -m src.scripts.calibrate_camera $(ARGS)

calib-stopline: ## stop-line distance in cm from tape marks, one at a time
	$(PY) -m src.scripts.calibrate_stop_line $(ARGS)

calib-lane: ## where a centered robot sees its lane, and its width in px: robot_x_px and the lane widths for config.py
	$(PY) -m src.scripts.calibrate_lane $(CAMFLAGS) $(ARGS)

calib-odometry: ## encoder cm per count: push a measured distance (ARGS="--cm 150" or "--drive 3")
	$(PY) -m src.scripts.calibrate_odometry $(ARGS)

calib-ground: ## floor homography from a checkerboard (ARGS="--square-cm 2.46 --origin-x-cm -9.8")
	$(PY) -m src.scripts.calibrate_ground $(ARGS)

calib-lamps: ## traffic-light HSV bands (LAMPS="red=runs/lamp_red ... off=runs/lamps_off"; ARGS=--write to save)
	@test -n "$(strip $(LAMPS))" || (echo 'usage: make calib-lamps LAMPS="red=runs/lamp_red yellow=... green=..." [ARGS=--write]'; exit 2)
	$(PY) -m src.scripts.calibrate_lamps $(strip $(LAMPS)) $(ARGS)

sweep-lamps: ## sweep HSV floors + blob gates on labelled frames (LAMPS="green=runs/nav_X:0-86 red=runs/nav_X:108-200"; ARGS=--write)
	@test -n "$(strip $(LAMPS))" || (echo 'usage: make sweep-lamps LAMPS="green=runs/nav_X:0-86 yellow=runs/nav_X:87-107 red=runs/nav_X:108-200" [ARGS=--write]'; exit 2)
	$(PY) -m src.scripts.sweep_lamps $(strip $(LAMPS)) $(ARGS)

##@ Docs and housekeeping
code-cards: ## one structure card per source file, into runs/code_cards/
	$(PY) -m src.scripts.code_cards $(ARGS)

newest: ## the newest run folder of each kind
	@for k in nav_ intersection_ maneuver_ diag_ lamp_; do \
	    d=$$(ls -dt runs/$$k*/ 2>/dev/null | head -1); [ -n "$$d" ] && echo "  $$d"; done; \
	 d=$$(ls -dt artifacts/*/ 2>/dev/null | head -1); [ -n "$$d" ] && echo "  $$d (pytest --hardware)"; true

clean: ## remove Python caches (never touches runs/ or artifacts/)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
