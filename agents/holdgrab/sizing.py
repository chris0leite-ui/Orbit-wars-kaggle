"""Lanchester-linear ship sizing.

Orbit Wars combat is the Lanchester *linear* law: when a fleet meets a
garrison the difference survives, so:

  * to CAPTURE a garrison G:                send  G + 1  (strict win).
  * to CAPTURE AND HOLD against a follow-on F that lands while we hold:
        we survive iff (our survivors) + (production we accrue before F lands)
        > F, i.e. ships > G + max(0, F - production * hold_ticks).
        So send  G + ceil(net_follow_on) + 1.

There is no square-law concentration bonus (linear law), so we size to
exactly what each capture needs and spread the surplus across more captures
rather than massing — the lever is allocation + timing, not raw mass.
"""

from __future__ import annotations

import math


def ships_to_capture(garrison_at_arrival) -> int:
    """Minimum fleet that strictly wins against ``garrison_at_arrival``."""
    return int(garrison_at_arrival) + 1


def net_follow_on(follow_on, production, hold_ticks) -> float:
    """Follow-on force NOT covered by the production the planet makes for us
    between capture and the follow-on's arrival."""
    covered = float(production) * float(max(0, int(hold_ticks)))
    return max(0.0, float(follow_on) - covered)


def ships_to_capture_and_hold(garrison_at_arrival, follow_on, production, hold_ticks) -> int:
    """Fleet that captures ``garrison_at_arrival`` and survives ``follow_on``
    after accounting for ``production * hold_ticks`` of post-capture growth."""
    net = net_follow_on(follow_on, production, hold_ticks)
    return int(garrison_at_arrival) + int(math.ceil(net)) + 1
