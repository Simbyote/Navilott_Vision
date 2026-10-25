"""Debug views and the live-view harness: what each stage decided, drawn on its frames.

Purpose:
    The robot has no monitor, so every decision worth checking is drawn and
    recorded for review. live_view runs the views on a whole-pipeline run;
    debug_maneuver and debug_navigation render the driving linkers'
    recordings afterwards. Reading the views: docs/vision_pipeline/,
    "Seeing inside it" (Phase 2) and "Running it" (Phase 3).

Main package:
    The view interface (debug_video), and each view's video and CSV.
"""
