"""Value heads for v9 super-version lookahead rollouts.

A "value head" turns the terminal Snapshot of a K-step rollout into a
single scalar score. The lookahead chooser uses the score to rank
candidate actions. Today we use `delta_us_minus_them` (our minus
their total ships). For v9 we add `inflight_value` which extends the
K-step horizon by an extra 30 turns of static-substrate look-ahead
to credit in-flight fleets that will arrive AFTER the rollout
horizon.

Motivation: the receding-horizon pathology (documented in
`audit/2026-05-12-v4-planner-receding-horizon-pathology.md`).
At K=10 the rollout's leaf state may still have ships in flight
toward a planet they'll capture at eta=11–15. The naive ship-delta
head rates these ships as pure COST (ships not yet on a planet). A
candidate that does NOT launch them keeps them at home and scores
higher — even though the launching candidate would actually win
the capture. This biases lookahead toward "wait" / "don't fire".

`inflight_value` fixes this by building a `WorldModel` over the
terminal Snapshot and reading predicted ownership at terminal+30.
Fleets that arrive within that extended horizon and flip ownership
in our favour contribute their target's production × time-to-hold
to the score.

`composite_capture_value` (2026-05-14) extends this with two more
terms motivated by the PI observation that depth-2 search was biased
toward passive play and we were wasting ships:
- **Capture bonus:** for each of our in-flight fleets, predict whether
  it will land successfully and, if so, credit the captured planet's
  production × remaining episode steps. Directly rewards "go conquer
  the right planets."
- **Waste penalty:** subtract a fraction of ship counts for fleets
  that won't capture — either because they're targeting nothing
  (OOB / sun trajectory) or because they'll bounce off a stronger
  defender. Directly penalises "don't waste ships by sending too few
  or by not conquering."
"""

from __future__ import annotations

from typing import Any

from lib.fast_sim import ship_totals
from lib.intent import World
from lib.scoring import pv_horizon
# Single-line imports below: the submission bundler's per-line
# import-stripping regex would leak continuation lines from a parenthesised
# multi-line import as indented orphans (IndentationError at runtime).
# Friction tag: `bundler-modular-agent-namespace-access-breaks-bundle`
# documented in agents/baseline/main.py.
from lib.world_model import DEFAULT_HORIZON, WorldModel, comet_remaining_lifetime, fleet_target_planet
from lib.game.interpreter import CENTER, SUN_RADIUS, point_to_segment_distance


# Phase 2 audit established AUC ≈ oracle at K=50. K=10 + 30 extra of
# static substrate ≈ K=40 effective; close enough.
INFLIGHT_EXTRA_HORIZON: int = 30

# Patchable value-head selector used by agents/baseline/value.select_favor_fn.
# 0 = favor (default v15 baseline)
# 1 = composite_capture_value
# 2 = projected_rank_diff (production-compounding unified head)
# No type annotation — scripts/ab_variants.py regex-patches `NAME = number`
# lines and does not match annotated assignments. Falls back to env var
# BASELINE_VALUE_HEAD when VALUE_HEAD_CHOICE == 0.
VALUE_HEAD_CHOICE = 0

# How much weight to give the in-flight production credit relative to
# ship-delta. 0.5 chosen so a captured 3-production planet (worth
# ~3*30=90 production-points) approximately balances 90 ships of
# delta. Calibration knob.
INFLIGHT_WEIGHT: float = 0.5


def delta_us_minus_them_obs(obs: Any, my_id: int) -> float:
    """Plain `(our ships) − (their ships)` from a Snapshot's primary
    observation. Phase 2 validated this at AUC ≈ oracle for K=50.

    Renamed from `delta_us_minus_them` to avoid bundle-shadow collision
    with the identically-named `lib.fast_sim.delta_us_minus_them(snap, ...)`.
    The fast_sim version takes a Snapshot; this one takes an obs.
    Same logic, different first-arg type.

    `obs` is `snap.state[my_id].observation` (a `Struct`). Sums
    planet garrisons + in-flight fleet ship counts for owned planets/
    fleets; subtracts each other seat's total.
    """
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
    ours = 0.0
    theirs = 0.0
    for p in planets:
        owner = int(p[1])
        if owner == my_id:
            ours += float(p[5])
        elif owner >= 0:
            theirs += float(p[5])
    for f in fleets:
        owner = int(f[1])
        if owner == my_id:
            ours += float(f[6])
        elif owner >= 0:
            theirs += float(f[6])
    return ours - theirs


def inflight_value(
    obs: Any, my_id: int,
    *, extra_horizon: int = INFLIGHT_EXTRA_HORIZON,
    weight: float = INFLIGHT_WEIGHT,
) -> float:
    """Composite scoring head: `delta_us_minus_them + weight × inflight_credit`.

    The credit term reads the predicted owner of each planet at
    `step + extra_horizon` from the terminal Snapshot's WorldModel
    (which integrates in-flight fleets). For planets that flip TO
    us within the extended horizon, the credit is `production`.
    Sum across all such planets, weight by `weight`.

    The default weight=0.5 is the calibration knob the v9_inflight
    A/B is gated on. Phase 2 said AUC≈oracle at K=50; this head
    effectively extends K from 10 to ~40 via the static substrate
    while keeping the same rollout cost.

    Empty world (no planets) → returns the base ship-delta only
    (which is 0).
    """
    base = delta_us_minus_them_obs(obs, my_id)
    # Build World from the terminal observation. fast_sim's Snapshot
    # uses Struct, so World.from_obs accepts it.
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return base

    # Build a per-planet timeline that integrates current in-flight
    # fleets out to `extra_horizon`. WorldModel.simulate_planet_timeline
    # is O(horizon) per planet and ~1ms total at horizon=30 for a
    # typical board (see audit/2026-05-12-fast-sim-bench.md).
    model = WorldModel.from_world(world, horizon=extra_horizon)

    bonus = 0.0
    for p in world.planets_by_id.values():
        if p.owner == my_id:
            # Already ours; no in-flight credit needed (counted in base).
            continue
        pred_owner = model.owner_at(p.id, extra_horizon)
        if pred_owner == my_id:
            # We'll own it within extra_horizon → credit the production.
            bonus += float(p.production)
    return base + weight * bonus


# ---------------------------------------------------------------------------
# composite_capture_value — anti-waste + capture-aware (v7.4)
# ---------------------------------------------------------------------------


# Coefficients tuned so the three terms are comparable in scale on a
# typical mid-game board (ship-delta in the ~10-50 range, capture bonus
# ~0.05 × 3 × 300 = 45 per high-value capture, waste penalty ~0.5 × ships).
CAPTURE_REWARD_WEIGHT: float = 0.05
WASTE_PENALTY_WEIGHT: float = 0.5
EPISODE_STEPS_TOTAL: int = 500

# Discount factor for the per-planet production-PV term. Matches
# `agents/baseline/value.favor`'s default gamma so composite's
# ownership-credit scales consistently with favor across the 2P
# composite / 4P A2-favor split in `favor_hybrid`.
PRODUCTION_PV_GAMMA: float = 0.99

# Diagnostic toggle for the production-PV term in `composite_capture_value`.
# Bug #15 fix v2 (2026-05-18 PM) shipped this term default-ON; subsequent
# A/B vs the pre-fix bundle settled at 39.6% (n=96, Wlo=0.304, FAIL).
# Bug #14 option 5 (smart reactive defense in candidate rollouts) was the
# hypothesised cure; it ALSO failed at 39.6% — the hypothesis is fully
# falsified. The convergent failure means the PV term itself over-credits
# captures: chooser was calibrated WITHOUT PV; adding ~100 units per
# captured planet at leaf uniformly inflates candidate scores → over-
# emission → drained sources → losses. Disabling PV restores the chooser's
# pre-#15 calibration (~50% vs bundle). Cost: sanity oracle (`test_oracle
# _sanity_trivial_capture`) reverts to xfail — that property is real but
# the cost-benefit tilts to "disable and revisit with chooser-gate
# recalibration in a future session". See
# audit/2026-05-18-postmortem-bug-15-v2-and-bug-14-option-5.md and
# knowledge-base/thoughts/2026-05-18-PV-term-recalibration-debt.md.
# Default OFF as of 2026-05-18 PM session wrap. Set
# `COMPOSITE_PRODUCTION_PV=1` to re-enable for A/Bs.
import os as _os
_COMPOSITE_PV_ENABLED = _os.environ.get("COMPOSITE_PRODUCTION_PV", "0") != "0"


def composite_capture_value(
    obs: Any, my_id: int,
    *,
    horizon: int = DEFAULT_HORIZON,
    capture_weight: float = CAPTURE_REWARD_WEIGHT,
    waste_weight: float = WASTE_PENALTY_WEIGHT,
) -> float:
    """Ship-delta + production-PV + per-fleet waste penalty.

    Base = `(my_ships − opp_ships) + (my_prod − opp_prod) × pv`. The PV
    term values planet ownership beyond the leaf horizon so captures
    register at the leaf even after the capturing fleet has arrived
    (without it, ship counts net out symmetrically and equal-production
    captures score Δ = 0 vs idle). This is bug #15's fix: a leaf state
    where we just captured opp's last planet now scores higher than
    the do-nothing baseline.

    For each of OUR in-flight fleets, the per-fleet WASTE PENALTY fires
    when the launch is structurally lost:
    - no planet on the trajectory (OOB);
    - trajectory crosses the sun (engine kills the fleet mid-flight);
    - target is a comet that expires before arrival;
    - predicted owner at ETA is NOT us (we bounce off a stronger
      defender, or multi-arrival combat goes the other way).

    There is NO per-fleet capture-credit term. The bug #15 fix v1
    (2026-05-18 AM) added a counterfactual per-fleet capture credit
    on top of the PV term, but the A/B ablation (n=64) showed the two
    terms double-credit the same capture and the chooser systematically
    over-emits (winrate 40.6% with both halves on, 46.9% with PV only,
    vs 50% baseline). v2 (this version) keeps PV-only — the PV term
    already credits the capture at the leaf via planet ownership.

    Set `COMPOSITE_PRODUCTION_PV=0` to disable the PV term for a clean
    A/B revert to pre-2026-05-18 behaviour (sanity oracle fails when
    PV is off, but the chooser's calibration matches the pre-bug-#15
    state used by submission `52754310`).
    """
    base = delta_us_minus_them_obs(obs, my_id)
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return base
    step_now = int(world.step)

    # Per-planet production-PV term. Mirrors `favor()`'s
    # `(my_prod - opp_prod) * pv_horizon`. Without this term the base
    # ship-delta is invariant to a capture of an equal-production
    # planet (both owners produce at the same rate over the rollout,
    # so net ships cancel out), which means a candidate that captures
    # opp's planet during the rollout scores Δ ≈ 0 vs idle even though
    # we won the planet's future production. Bug #15 root cause is two
    # things together: (a) the per-fleet credit was broken by a
    # chicken-and-egg in WorldModel prediction (see below), AND (b)
    # base lacked any term that values ownership beyond the leaf
    # horizon — so even with the per-fleet fix, post-arrival captures
    # (eta < rollout horizon) would still not register. Sanity oracle
    # `tests/test_planner_oracles.py::test_oracle_sanity_trivial_capture`
    # surfaced (b); the bug catalog at audit/2026-05-18-bug-catalog.md
    # documents (a). 2026-05-18 fix.
    if _COMPOSITE_PV_ENABLED:
        pv = pv_horizon(
            step_now, 0,
            gamma=PRODUCTION_PV_GAMMA,
            t_total=EPISODE_STEPS_TOTAL,
        )
        my_prod = 0.0
        opp_prod = 0.0
        for p in world.planets_by_id.values():
            owner = int(p.owner)
            if owner == my_id:
                my_prod += float(p.production)
            elif owner >= 0:
                opp_prod += float(p.production)
        base += (my_prod - opp_prod) * pv

    raw = world.obs_raw
    fleets_raw = (
        raw.get("fleets", []) if isinstance(raw, dict)
        else getattr(raw, "fleets", [])
    )
    if not fleets_raw:
        return base

    # Reuse the kaggle namedtuple so `fleet_target_planet` gets the same
    # type it expects.
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet  # noqa: E402
    fleets = [Fleet(*f) for f in fleets_raw]
    planets_list = list(world.planets_by_id.values())
    # Thread omega through to fleet_target_planet so orbiting-target
    # attribution works (bug #11 fix, 2026-05-18).
    omega = float(
        raw.get("angular_velocity", 0.0) if isinstance(raw, dict)
        else getattr(raw, "angular_velocity", 0.0) or 0.0
    )

    # Pre-pass: compute each of OUR fleets' target/eta so we can scope
    # the WorldModel build to the actual look-ahead needed. The full
    # DEFAULT_HORIZON (=30) is overkill when our longest fleet eta is
    # 10 — WorldModel.from_world is O(horizon × planets), so scaling
    # horizon to max_eta cuts the dominant cost roughly in half on
    # short-range turns. 2026-05-17 timing-fix item #2.
    fleet_targets: list[tuple[Fleet, float, object | None, int]] = []
    max_eta = 0
    for f in fleets:
        if int(f.owner) != my_id:
            continue
        ships = float(f.ships)
        target, eta = fleet_target_planet(f, planets_list, omega)
        eta_int = int(eta) if eta is not None else 0
        fleet_targets.append((f, ships, target, eta_int))
        if target is not None and eta_int > max_eta:
            max_eta = eta_int

    if not fleet_targets:
        return base

    effective_horizon = max(1, min(horizon, max_eta + 1))
    model = WorldModel.from_world(world, horizon=effective_horizon)

    delta = 0.0
    for f, ships, target, eta in fleet_targets:
        if target is None:
            # No planet on our trajectory — destined for OOB or sun.
            delta -= waste_weight * ships
            continue
        # Sun-crossing gate: `fleet_target_planet` ray-casts to the first
        # planet on the angle, ignoring the sun. If the fleet's chord
        # (current pos → target pos) passes within SUN_RADIUS of the
        # sun, the engine kills the fleet at the crossing tick
        # (orbit_wars.py:607: `point_to_segment_distance((CENTER, CENTER),
        # old_pos, new_pos) < SUN_RADIUS`). Without this gate, composite
        # silently credits captures the fleet never gets to make. Origin:
        # PI live observation 2026-05-17 PM ("large fleet into the sun").
        fleet_pos = (float(f.x), float(f.y))
        target_pos = (float(target.x), float(target.y))
        if point_to_segment_distance(
            (CENTER, CENTER), fleet_pos, target_pos,
        ) < SUN_RADIUS:
            delta -= waste_weight * ships
            continue
        # Comet-lifetime gate: WorldModel's simulate_planet_timeline
        # assumes planets persist for the full horizon and is unaware
        # that comets exit the board after `path_index` reaches the
        # end of the path (engine: orbit_wars.py:528-561). A fleet
        # aimed at a comet that expires before arrival hits empty space
        # — it never enters combat, never captures, never bounces. The
        # pred_owner check below would say "we'll own it at eta" for a
        # comet that's actually GONE by then. Pre-check matches the
        # engine's truth. Mirrors lib/missions/snipe.py:404-420 (H15)
        # and PI direction 2026-05-17: "use comets only if really
        # worth the risk and short lifetime".
        if int(target.id) in world.comet_ids:
            comet_life = comet_remaining_lifetime(int(target.id), world)
            if comet_life is None or comet_life <= eta:
                delta -= waste_weight * ships
                continue
        # Predict ownership at ETA. WorldModel includes THIS fleet in
        # its ledger, so the prediction reflects "world if we let this
        # fleet land." If pred_owner is NOT us at eta, the launch is
        # structurally lost (we bounce off a stronger defender, or
        # multi-arrival combat goes the other way) — apply waste
        # penalty. Otherwise the launch is constructive (causes the
        # capture OR over-reinforces a planet we'd hold anyway); either
        # way we do NOT add a per-fleet capture credit — the PV term in
        # the base already values the resulting ownership at the leaf.
        # See bug #15 fix v2 rationale in the docstring above.
        pred_owner = model.owner_at(target.id, eta)
        if pred_owner != my_id:
            delta -= waste_weight * ships

    return base + delta


# ---------------------------------------------------------------------------
# projected_rank_diff — production-compounding unified value head
# ---------------------------------------------------------------------------
#
# Working backward from Kaggle's evaluation (TrueSkill on ordinal rank by
# total ships at T=500), the cheap sufficient statistic for a seat's
# final score is:
#
#     ProjectedTotal_i = ships_now_i
#                      + in_flight_capture/waste_credit_i
#                      + PROJECTION_LAMBDA × Σ_p (P_p × turns_remaining)
#                        for planets p owned by seat i at the leaf state
#
# V(s) = ProjectedTotal_us − max_{j != us} ProjectedTotal_j.
#
# Generalises composite_capture_value's "P × turns_remaining" credit from
# in-flight fleets only to ALL planets at the leaf, and replaces the 2P
# `delta_us_minus_them` aggregation with `max` over opponents — which is
# what TrueSkill ordinal ranking measures against.
#
# Compounding pressure: a P=5 planet owned at step 100 contributes
# 0.05 × 5 × 400 = 100 to your projection — orders of magnitude bigger
# than ship-balance differentials. Pressure to launch (rather than hoard)
# is built into the math, not a tuned passivity-penalty term.


PROJECTION_LAMBDA: float = 0.05  # same scale as CAPTURE_REWARD_WEIGHT


def _per_seat_in_flight_credit(
    obs: Any,
    num_seats: int,
    *,
    capture_weight: float = CAPTURE_REWARD_WEIGHT,
    waste_weight: float = WASTE_PENALTY_WEIGHT,
    horizon: int = DEFAULT_HORIZON,
) -> dict:
    """Per-seat in-flight capture / waste credit.

    Generalises composite_capture_value's per-fleet logic: instead of
    attributing the credit to one seat (`my_id`), attributes each fleet's
    predicted fate to its OWN seat's bucket. WorldModel built once.
    """
    credits = {i: 0.0 for i in range(num_seats)}
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return credits
    raw = world.obs_raw
    fleets_raw = (
        raw.get("fleets", []) if isinstance(raw, dict)
        else getattr(raw, "fleets", [])
    )
    if not fleets_raw:
        return credits

    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet  # noqa: E402
    fleets = [Fleet(*f) for f in fleets_raw]
    planets_list = list(world.planets_by_id.values())
    model = WorldModel.from_world(world, horizon=horizon)
    step_now = int(world.step)

    for f in fleets:
        owner = int(f.owner)
        if owner < 0 or owner >= num_seats:
            continue
        ships = float(f.ships)
        target, eta = fleet_target_planet(f, planets_list)
        if target is None:
            credits[owner] -= waste_weight * ships
            continue
        pred_owner = model.owner_at(target.id, eta)
        pred_ships = model.ships_at(target.id, eta) or 0.0
        if pred_owner == owner:
            continue  # over-reinforcement — neutral
        if ships > pred_ships:
            time_remaining = max(0, EPISODE_STEPS_TOTAL - step_now - eta)
            credits[owner] += capture_weight * float(target.production) * float(time_remaining)
        else:
            credits[owner] -= waste_weight * ships
    return credits


def projected_rank_diff(
    obs: Any,
    my_id: int,
    num_seats: int = 2,
    *,
    capture_weight: float = CAPTURE_REWARD_WEIGHT,
    waste_weight: float = WASTE_PENALTY_WEIGHT,
    projection_lambda: float = PROJECTION_LAMBDA,
    horizon: int = DEFAULT_HORIZON,
) -> float:
    """Production-compounding unified value head.

    V(s) = ProjectedTotal_us − max_{j != us} ProjectedTotal_j
    where ProjectedTotal_i  = ships_i(now)
                            + in_flight_credit_i
                            + λ · Σ_p P_p · (T − step) for p owned by i at leaf.

    `max` matches TrueSkill ordinal ranking — the next opponent above us
    is what we're racing. In 4P, opponents fighting each other shrinks
    `max` for free (high-risk shots that move us past the leader pay off
    even when the bottom opp does well in absolute terms).

    Linear time-remaining (no γ-discount) per the PV-off finding
    (live A/B 81.2% on submission 52784853). T=500 is a hard horizon;
    exponential decay double-counts.

    Compounding emerges from `P_p × (T − step)`: a P=3 capture at step 100
    is worth ≈ 60 ship-units (0.05 × 3 × 400); at step 400, ≈ 15. Early
    captures are super-linear in elapsed-game-time; the chooser will
    prefer launching to hoarding without a passivity penalty.
    """
    if isinstance(obs, dict):
        planets = obs.get("planets", []) or []
        fleets = obs.get("fleets", []) or []
        step = int(obs.get("step", 0))
    else:
        planets = getattr(obs, "planets", []) or []
        fleets = getattr(obs, "fleets", []) or []
        step = int(getattr(obs, "step", 0))
    rem = max(0, EPISODE_STEPS_TOTAL - step)

    ships_per = {i: 0.0 for i in range(num_seats)}
    proj_per = {i: 0.0 for i in range(num_seats)}
    for p in planets:
        owner = int(p[1])
        if owner < 0 or owner >= num_seats:
            continue
        ships_per[owner] += float(p[5])
        proj_per[owner] += float(p[6]) * rem
    for f in fleets:
        owner = int(f[1])
        if owner < 0 or owner >= num_seats:
            continue
        ships_per[owner] += float(f[6])

    credits = _per_seat_in_flight_credit(
        obs, num_seats,
        capture_weight=capture_weight, waste_weight=waste_weight,
        horizon=horizon,
    )

    totals = {
        i: ships_per[i] + credits[i] + projection_lambda * proj_per[i]
        for i in range(num_seats)
    }
    my_total = totals[my_id]
    if num_seats <= 1:
        return my_total
    opp_total = max(v for k, v in totals.items() if k != my_id)
    return my_total - opp_total
