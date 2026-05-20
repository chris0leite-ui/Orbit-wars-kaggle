"""Stage 6 — closed-form leaf evaluator for the analytical pipeline.

Given (my arrivals, opp arrivals, current world + in-flight ledger),
compute a scalar leaf value: `Σ_p [prod_stream_me(p) − α · prod_stream_opp(p)]
  − β · Σ_c n_c`.

This is bit-exact: uses the same `simulate_planet_timeline` /
`resolve_arrivals` primitives the env validates against (and the
existing `outcome_table.enumerate_outcomes` cross-checks).

Used by Phase D's depth-2 maximin and any other game-theoretic
decision rule operating over portfolios. NO rollouts; pure
closed-form per the PI's analytical-native directive.
"""

from __future__ import annotations

from collections.abc import Iterable

from lib.world_model import simulate_planet_timeline

from lib.pipeline.types import TurnContext


def _prod_stream_from_timeline(timeline: dict, planet_production: int) -> dict[int, int]:
    """Reconstruct per-owner production stream from a timeline.

    `simulate_planet_timeline` returns owner_at[t]/ships_at[t] but not
    prod_stream. Per-tick semantics (from `simulate_planet_timeline`
    source): at each tick t, production accrues to owner_at[t-1]
    (the owner going INTO tick t, before that tick's arrivals resolve).
    """
    horizon = int(timeline["horizon"])
    owner_at = timeline["owner_at"]
    prod: dict[int, int] = {}
    for t in range(1, horizon + 1):
        owner_prev = owner_at.get(t - 1)
        if owner_prev is None or int(owner_prev) < 0:
            continue
        prod[int(owner_prev)] = prod.get(int(owner_prev), 0) + int(planet_production)
    return prod


def leaf_value_for_portfolios(
    my_arrivals: list[tuple[int, int, int, int]],
    opp_arrivals: Iterable[tuple[int, int, int, int]],
    ctx: TurnContext,
    *,
    horizon_truncate: int | None = None,
    alpha_opp_penalty: float = 1.0,
    ship_cost: float = 1.0,
    discount_gamma: float | None = None,
) -> float:
    """Closed-form scalar leaf value for one (my_portfolio, opp_portfolio) pair.

    `my_arrivals` and `opp_arrivals` are lists of
      `(target_pid, eta_relative_from_step_now, owner, ships)`
    representing the projected arrivals from each side.

    `horizon_truncate`: if set, truncate the per-planet simulation at this
    many ticks from step_now (matches Phase D's discounted-leaf option;
    None means use the model's existing horizon).

    `alpha_opp_penalty`: weight on opp's production stream (zero-sum=1.0).
    `ship_cost`: per-ship cost on my arrivals (matches `lp_outcome.SHIP_COST`).
    `discount_gamma`: if not None, apply γ^t per-tick discount to production
      streams. Defaults to no discount (matches existing LP behavior).

    Returns the scalar leaf value:
        Σ_p [me_stream(p) − α · opp_stream(p)] − β · my_total_ships
    """
    me = int(ctx.me)
    horizon = ctx.model.horizon
    if horizon_truncate is not None:
        horizon = min(int(horizon_truncate), int(horizon))

    # Merge arrivals per planet: in-flight (from ledger) + my + opp.
    arrivals_by_planet: dict[int, list[tuple[int, int, int]]] = {}
    for pid, in_flight in ctx.model.ledger.items():
        arrivals_by_planet.setdefault(int(pid), []).extend(
            (int(e), int(o), int(s)) for (e, o, s) in in_flight
        )
    for (pid, eta_rel, owner, ships) in my_arrivals:
        if int(ships) <= 0 or int(eta_rel) <= 0:
            continue
        arrivals_by_planet.setdefault(int(pid), []).append(
            (int(eta_rel), int(owner), int(ships))
        )
    for (pid, eta_rel, owner, ships) in opp_arrivals:
        if int(ships) <= 0 or int(eta_rel) <= 0:
            continue
        arrivals_by_planet.setdefault(int(pid), []).append(
            (int(eta_rel), int(owner), int(ships))
        )

    total = 0.0

    # Per-planet contribution: planets touched by any arrival use the
    # full timeline simulation; planets with no arrivals stay with their
    # initial owner forever.
    touched = set(arrivals_by_planet.keys())
    for pid, arrivals_p in arrivals_by_planet.items():
        planet = ctx.world.planets_by_id.get(int(pid))
        if planet is None:
            continue
        timeline = simulate_planet_timeline(planet, arrivals_p, horizon)
        if discount_gamma is None:
            prod = _prod_stream_from_timeline(timeline, int(planet.production))
        else:
            prod = _discounted_prod_stream(
                timeline, int(planet.production), float(discount_gamma),
            )
        for owner, p in prod.items():
            if int(owner) == me:
                total += float(p)
            elif int(owner) >= 0:
                total -= float(alpha_opp_penalty) * float(p)

    # Untouched planets: deterministic constant contribution.
    for pid, planet in ctx.world.planets_by_id.items():
        if int(pid) in touched:
            continue
        if int(planet.owner) < 0:
            continue
        if discount_gamma is None:
            stream = int(horizon) * int(planet.production)
        else:
            # Σ_{t=1}^{H} γ^t * prod
            g = float(discount_gamma)
            if g >= 1.0 or g <= 0.0:
                stream = int(horizon) * int(planet.production)
            else:
                stream = int(planet.production) * (g * (1 - g ** horizon) / (1 - g))
        if int(planet.owner) == me:
            total += float(stream)
        else:
            total -= float(alpha_opp_penalty) * float(stream)

    # Ship cost on my arrivals.
    my_total_ships = sum(int(s) for (_p, _e, _o, s) in my_arrivals if int(s) > 0)
    total -= float(ship_cost) * float(my_total_ships)

    return float(total)


def _discounted_prod_stream(timeline: dict, planet_production: int,
                            gamma: float) -> dict[int, float]:
    """Per-tick γ-discounted production stream."""
    horizon = int(timeline["horizon"])
    owner_at = timeline["owner_at"]
    prod: dict[int, float] = {}
    discount = 1.0
    for t in range(1, horizon + 1):
        discount *= float(gamma)
        owner_prev = owner_at.get(t - 1)
        if owner_prev is None or int(owner_prev) < 0:
            continue
        prod[int(owner_prev)] = prod.get(int(owner_prev), 0.0) + discount * int(planet_production)
    return prod


def column_to_arrival(col, step_now: int) -> tuple[int, int, int, int]:
    """Convert a Column into the `(pid, eta_rel, owner, ships)` shape this
    module consumes. Returns eta_rel measured from step_now."""
    return (
        int(col.tgt_id),
        int(col.eta),   # column.eta is already from step_now (proposer convention)
        int(col.owner),
        int(col.ships),
    )
