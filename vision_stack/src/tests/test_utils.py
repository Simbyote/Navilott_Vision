"""
test_utils.py  --  src/utils.py

--software  Known-answer tests for the shared helpers. No camera.
"""
import pytest

from src.utils import clamp


@pytest.mark.software
@pytest.mark.parametrize("v, lo, hi, want", [(5, 0, 10, 5), (-3, 0, 10, 0), (99, 0, 10, 10), (0, 0, 10, 0), (10, 0, 10, 10)])
def test_clamp(v, lo, hi, want):
    assert clamp(v, lo, hi) == want