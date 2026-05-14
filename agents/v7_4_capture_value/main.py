"""v7.4 — drop-one chooser with `composite_capture_value` head.

The depth-2 and min-regret experiments (v7_2, v7_3) failed because
forward-sim was biased toward passive play: any opp model in the
rollout punishes our launches as "ships in flight = ships not at home,"
so the chooser preferred "don't launch" candidates.

v7_4 keeps v7_0's drop-one action space + K=10 rollout but replaces
`delta_us_minus_them` with `lib.value_heads.composite_capture_value`,
which credits in-flight fleets that will successfully capture and
penalises fleets that will bounce or escape OOB/sun. Two structural
fixes:

  - **Capture reward:** for each of our fleets at the rollout terminal,
    predict (via WorldModel ray-cast) whether it'll land successfully.
    If yes, credit the captured planet's production × remaining
    episode steps × `CAPTURE_REWARD_WEIGHT` (0.05). Directly rewards
    "go conquer the right planets."

  - **Waste penalty:** for each of our fleets predicted to bounce or
    OOB / sun, subtract `WASTE_PENALTY_WEIGHT × ships` (0.5). Directly
    penalises "don't waste ships by failing to conquer."

Action space is unchanged from v7_0 (drop-one of v7_0+H11 incumbent);
opp model is v3.5.1 mirror (same as v7_0). The ONLY delta is the
value-head swap, so any A/B lift is attributable to the new value
function.
"""

from __future__ import annotations

from lib.v7_search import choose
from lib.value_heads import composite_capture_value


def agent(obs, configuration=None):
    return choose(
        obs, configuration,
        enumerator_mode="drop_one",
        K=10,
        wallclock_ms=700.0,
        value_fn=composite_capture_value,
    )
