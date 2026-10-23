"""Debug views and the live-view harness: what each stage decided, drawn on the frames it decided it on.

Purpose:
    The robot has no monitor, so every decision worth checking is drawn
    and recorded, to review after the run. live_view runs the views on a
    whole-pipeline run; each debug_* module is one view; debug_maneuver and
    debug_navigation render the driving linkers' recordings after the run.
    How to read each view: docs/guides/phase2_linker.md, phase3_linker.md,
    navigation_linker.md and maneuver_linker.md.

Main package:
    The view interface (debug_video), and the per-view video and CSV each
    run writes.
"""
