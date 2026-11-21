import time

from adaptive_qa.common import Timer, now_ns


def test_nanosecond_clock_is_monotonic():
    a = now_ns()
    b = now_ns()
    assert a <= b


def test_timer_measures_elapsed_time():
    with Timer() as t:
        time.sleep(0.02)
    ms = t.elapsed_ms()
    # wall clock sanity, far above noise, using the same monotonic clock
    assert 15.0 < ms < 1000.0
    assert t.elapsed_ns() > 0
