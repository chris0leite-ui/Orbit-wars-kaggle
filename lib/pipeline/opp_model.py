"""Stage 4 — Opponent model.

TurnContext → OppModelResult (opp arrivals + augmented WorldModel).

The reference implementation wraps `predict_opp_multi_launch` (greedy ROI,
action-independent) and merges the projected arrivals into the model's
ledger via `merge_ledgers` + `simulate_planet_timeline`. This matches
`mpc._model_with_opp_projection` bit-exact.

Alternative implementations (mirror-analytical, mixture-of-types) will
land in Phase D and are registered alongside `opp_greedy_roi`.
"""

from __future__ import annotations

from agents.baseline.chooser_trajectory import merge_ledgers
from lib.joint_solver.opp_projection import predict_opp_multi_launch
from lib.pipeline.types import OppModelResult, TurnContext
from lib.world_model import WorldModel, simulate_planet_timeline


def opp_greedy_roi(ctx: TurnContext) -> OppModelResult:
    """Reference Stage-4 implementation.

    Calls `predict_opp_multi_launch` for opp's projected schedule, then
    merges those arrivals into a fresh WorldModel that downstream stages
    consume. Mirror of `lib.joint_solver.mpc._model_with_opp_projection`.
    """
    try:
        opp_arrivals = predict_opp_multi_launch(
            ctx.world, int(ctx.me), int(ctx.num_seats),
        )
    except Exception:
        return OppModelResult(opp_arrivals=[], augmented_model=ctx.model)

    if not opp_arrivals:
        return OppModelResult(opp_arrivals=[], augmented_model=ctx.model)

    new_ledger = merge_ledgers(ctx.model.ledger, opp_arrivals)
    new_timelines = dict(ctx.model.timelines)
    for pid in {pid for (pid, _e, _o, _s) in opp_arrivals}:
        planet = ctx.world.planets_by_id.get(int(pid))
        if planet is None:
            continue
        new_timelines[int(pid)] = simulate_planet_timeline(
            planet, new_ledger.get(int(pid), []), ctx.model.horizon,
        )
    augmented = WorldModel(
        ledger=new_ledger, timelines=new_timelines, horizon=ctx.model.horizon,
    )
    return OppModelResult(opp_arrivals=opp_arrivals, augmented_model=augmented)
