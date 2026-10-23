"""Interpreters: read what runs and hardware tests recorded, and turn it into numbers, figures and verdicts.

Purpose:
    Collecting data is the linkers' and pytest's job; interpreting it is
    this package's. None of the interpreters touch the camera or the
    pipeline, so they run on the Pi or on a laptop with a copied run folder.
    Each runs on its own (python3 -m src.analysis.<module> [run folder or
    CSV]); what each reads, writes and answers: docs/guides/analysis.md.

Main package:
    One module per question; common.py holds the reading, statistics and
    output they share.
"""
