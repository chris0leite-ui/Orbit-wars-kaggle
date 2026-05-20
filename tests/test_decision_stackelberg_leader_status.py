"""Bug #8 — Stackelberg empty/failed conflation.

`predict_opp_response_to_my_portfolio` returns an empty list for both
"opp has no planets" (legitimate "opp does nothing") AND
"underlying LP raised" (real failure). Stackelberg-leader's caller at
`decision_stackelberg_leader.py:128-139` treats them the same: both
fall back to the greedy ROI opponent and increment a single
`n_opp_fallback` counter.

Fix: extend `predict_opp_response_to_my_portfolio` with an optional
`return_status=True` mode that returns `(arrivals, status)` where
status ∈ {"ok", "empty", "failed"}. Caller updates three distinct
counters; debugability lifts (we can tell whether the fallback is
hiding a real bug or just an empty-opp scenario).

Pin (Rule 38): two scenarios that pre-fix produce identical "empty
list" → post-fix produce distinct statuses.
"""

from __future__ import annotations

from lib.intent import Planet, World
from lib.pipeline.types import TurnContext
from lib.world_model import WorldModel


def _make_ctx(planets, step_now=0, me=0, num_seats=2):
    world = World(my_id=me, planets_by_id={p.id: p for p in planets},
                  omega=0.0, comet_ids=frozenset(), step=step_now,
                  obs_raw={})
    model = WorldModel(ledger={}, timelines={}, horizon=100)
    return TurnContext(
        obs_d={"player": me}, configuration=None, me=me, num_seats=num_seats,
        step_now=step_now, omega=0.0, planets=planets, fleets=[],
        my_planets=[p for p in planets if p.owner == me],
        other_planets=[p for p in planets if p.owner != me],
        world=world, model=model,
    )


def test_status_empty_when_opp_has_no_planets():
    """When opp has zero planets, status must be 'empty' (legitimate
    no-action), distinguishable from a real failure."""
    from lib.pipeline.opp_mirror_analytical import (
        predict_opp_response_to_my_portfolio,
    )
    # Only my planet; no opp.
    planets = [
        Planet(id=0, owner=0, x=20.0, y=20.0, radius=5.0, ships=20,
               production=2),
    ]
    ctx = _make_ctx(planets)
    arrivals, status = predict_opp_response_to_my_portfolio(
        ctx, my_portfolio=[], return_status=True,
    )
    assert arrivals == [], "expected empty arrivals for no-opp scenario"
    assert status == "empty", (
        f"expected status='empty' when opp has no planets, got {status!r}"
    )


def test_status_failed_when_underlying_lp_raises(monkeypatch):
    """When the inner LP raises, status must be 'failed' — distinguishable
    from the empty scenario."""
    from lib.pipeline import opp_mirror_analytical as mod
    # Two planets, opp HAS a planet — so we get past the empty check.
    planets = [
        Planet(id=0, owner=0, x=20.0, y=20.0, radius=5.0, ships=20,
               production=2),
        Planet(id=1, owner=1, x=80.0, y=20.0, radius=5.0, ships=20,
               production=2),
    ]
    ctx = _make_ctx(planets)

    # Force the inner LP to raise.
    def boom(*a, **k):
        raise RuntimeError("synthetic LP failure")
    monkeypatch.setattr(mod, "solve_outcome_aware", boom)

    arrivals, status = mod.predict_opp_response_to_my_portfolio(
        ctx, my_portfolio=[], return_status=True,
    )
    assert arrivals == [], "expected empty arrivals on LP failure"
    assert status == "failed", (
        f"expected status='failed' when inner LP raises, got {status!r}"
    )


def test_backward_compat_list_return_when_return_status_false():
    """`return_status=False` (default) must keep the legacy list-only
    return type so existing callers don't have to change."""
    from lib.pipeline.opp_mirror_analytical import (
        predict_opp_response_to_my_portfolio,
    )
    planets = [
        Planet(id=0, owner=0, x=20.0, y=20.0, radius=5.0, ships=20,
               production=2),
    ]
    ctx = _make_ctx(planets)
    arrivals = predict_opp_response_to_my_portfolio(ctx, my_portfolio=[])
    # Default mode: just a list, no tuple.
    assert isinstance(arrivals, list)
