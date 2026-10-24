"""Monotonic, high-resolution timing utilities.

All latency measurement in this project MUST go through this module so that
"time.perf_counter_ns" is the single source of truth (immune to wall-clock
NTP adjustments, DST, and manual clock changes in the VM).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

PLATFORM_RESOLUTION_NS = time.get_clock_info("perf_counter").resolution


def now_ns() -> int:
    """Monotonic, high-resolution clock reading in nanoseconds.

    Uses ``time.perf_counter_ns`` which is guaranteed monotonic
    (non-decreasing) and is distinct from wall-clock time.
    """
    return time.perf_counter_ns()


class Timer:
    """Context manager measuring elapsed time with the monotonic clock."""

    def __init__(self) -> None:
        self._start: Optional[int] = None
        self._elapsed_ns: int = 0

    def __enter__(self) -> "Timer":
        self._start = now_ns()
        return self

    def __exit__(self, *exc) -> None:
        if self._start is not None:
            self._elapsed_ns += now_ns() - self._start
            self._start = None

    def elapsed_ns(self) -> int:
        if self._start is not None:
            return now_ns() - self._start
        return self._elapsed_ns

    def elapsed_ms(self) -> float:
        return elapsed_ms_from_ns(self.elapsed_ns())


def elapsed_ms_from_ns(ns: int) -> float:
    """Convert nanoseconds to milliseconds with sub-ms precision."""
    return ns / 1_000_000.0
