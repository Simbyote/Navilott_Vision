"""
Interpreters: read what runs and hardware tests recorded, and turn it into
numbers, figures and verdicts. They never touch the camera or the pipeline;
collecting data is pytest's job (src/tests), interpreting it is this
package's. Each module runs on its own:

    python3 -m src.analysis.<module> [run folder or CSV]

    stage_timing      where each frame's time goes, against the frame budget
    jitter            frame-interval tails, over-budget streaks, periodic spikes
    stability         offset noise and lane-mode flicker on a still scene
    offset_accuracy   offset error at measured positions, verdict against +/-2 cm
    gate_rejections   which detector gate discards the most candidates
    state_timeline    Phase 3 state dwell times and transitions
    soak              heat, throttling, memory growth and slowdown over a long run
    nav_run           a navigation run: rules, lane keeping, each intersection and after, wheels, latency
    pi_load           a diagnostics recording: threads, cores, serial work, heat, clock, memory, slow frames
    detection_range   stop sign and traffic light detection rate by distance
    compare_runs      every number that changed between two runs

Shared reading, statistics and output helpers are in common.py. Their
software tests are in src/tests/test_<module>.py.
"""
