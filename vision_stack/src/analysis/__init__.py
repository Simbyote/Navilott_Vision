"""Interpreters: turn what runs and hardware tests recorded into numbers, figures and verdicts.

Purpose:
    None touch the camera or the pipeline, so they run on the Pi or on a
    laptop with a copied run folder: python3 -m src.analysis.<module> [run or
    CSV]. What each reads and answers: docs/guides/analysis.md.

Main package:
    One module per question; common.py holds what they share.
"""
