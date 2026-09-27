"""Unit tests for the model-loading progress pure helpers (_fmt_load / _load_pct).

Run:  .venv/Scripts/python tests/test_load.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline  # noqa: E402


def test_fmt_load_downloading():
    # VRAM ~ 0 -> the download/disk-read phase (the first run only)
    s = cz_pipeline._fmt_load("Z-Image base", 12.0, 0.0)
    assert "12s" in s and "downloading" in s.lower()


def test_fmt_load_in_vram():
    # VRAM > the threshold -> it displays the GB loaded
    s = cz_pipeline._fmt_load("Z-Image base", 45.0, 3.2)
    assert "45s" in s and "3.2 GB" in s


def test_load_pct_monotone_and_capped():
    target = 14.0
    # download: a small time-based bar, capped at 0.12
    assert cz_pipeline._load_pct(0.0, 0.0, target) == 0.0
    assert cz_pipeline._load_pct(6000.0, 0.0, target) == 0.12
    # in VRAM: the VRAM/target fraction, monotonic, capped at 0.95
    p_half = cz_pipeline._load_pct(30.0, 7.0, target)
    assert abs(p_half - 0.5) < 1e-6
    p_full = cz_pipeline._load_pct(60.0, 20.0, target)
    assert p_full == 0.95
    # a progression that grows with the VRAM
    assert cz_pipeline._load_pct(10.0, 2.0, target) < cz_pipeline._load_pct(20.0, 5.0, target)


def test_load_pct_target_guard():
    # an invalid target never divides by < 1
    assert cz_pipeline._load_pct(10.0, 5.0, 0.0) <= 0.95


def test_load_monitor_returns_and_raises():
    # enabled or not, _load_monitor must return fn's result
    assert cz_pipeline._load_monitor("noop", lambda: 123) == 123
    raised = False
    try:
        cz_pipeline._load_monitor("boom", lambda: (_ for _ in ()).throw(ValueError("x")))
    except ValueError:
        raised = True
    assert raised, "_load_monitor doit relever l'exception de fn"


if __name__ == "__main__":
    for fn in (test_fmt_load_downloading, test_fmt_load_in_vram,
               test_load_pct_monotone_and_capped, test_load_pct_target_guard,
               test_load_monitor_returns_and_raises):
        fn()
        print(f"OK {fn.__name__}")
    print("All load tests passed.")
