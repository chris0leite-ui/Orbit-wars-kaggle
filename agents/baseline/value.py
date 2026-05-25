"""Leaf value function: F1 + F2 favor with PV-discounted production.

F1 = my_ships - opp_ships_agg          (in-flight + on-planet)
F2 = (my_prod - opp_prod_agg) * pv     (pv = pv_horizon discount)

PV-discount keeps F2 on a comparable scale to F1; without it the future-
production term over-weights captures by ~100x in late game and the
chooser stops valuing ship preservation. opp aggregation is max-of-opps
in 2P (unchanged from baseline) and weighted-sum-of-opps in 4P
(weakest opp 1.5x).

A2 (4P weakness exploitation) derives from
romantamrazov/orbit-star-wars-lb-max-1224 (peak LB μ=1224, +109 above
our v15 ceiling).

  - 4P: 1.5x bias on the WEAKEST opponent's contribution; other opps
    unweighted. Biases leaf valuation toward states that further
    weaken (or eliminate) them.
  - Elimination bonus: +55 when weakest's strength (ships + 15*prod)
    <= 110 AND my_strength >= 0.9 * weakest's (only fire when WE can
    finish — no elim-then-die bias). 4P only.

History — 2P bias was tested and rolled back: a uniform 1.25x
multiplier on the single opp regressed h2h vs v15 in 2P (25/64,
39.1%, Wlo=0.281, Whi=0.513 INCONCLUSIVE) because v15 is well-tuned
and biasing the chooser toward attacks degrades its calibration.
The "weakness exploitation" thesis is 4P-specific (per-weakest, not
uniform); the 2P path is unchanged from the original baseline.

Opt-in alternative head: `BASELINE_VALUE_HEAD=composite` switches the
chooser to `lib.value_heads.composite_capture_value` (waste +
capture-aware per-fleet credit). 2P-only — composite does not
distinguish opp identity in 4P. Default remains `favor` with A2.
"""

from __future__ import annotations

import math
import os

from lib.scoring import pv_horizon

EPISODE_STEPS = 500
DEFAULT_GAMMA = 0.99

ELIMINATION_BONUS = 55.0
WEAK_ENEMY_THRESHOLD = 110.0
WEAKEST_ENEMY_MULT_4P = 1.5
ELIMINATION_GATE_RATIO = 0.9
STRENGTH_PROD_WEIGHT = 15.0

# Spatial leaf params (favor_hybrid_spatial only).
# Idle-trajectory audit 2026-05-17 on submission 52754310 (mu=1271.8)
# showed 43.8% of our ship-turns were on planets >50 units from any
# non-our planet. Spatial term rewards positioning ships near
# capturable targets so the chooser naturally drains rear/isolated
# garrisons forward.
SPATIAL_WEIGHT = float(os.environ.get("BASELINE_SPATIAL_WEIGHT", "0.5"))
SPATIAL_DECAY = float(os.environ.get("BASELINE_SPATIAL_DECAY", "30.0"))

# Phase D — model enrichment (favor_strategic; PI 2026-05-25 evening).
# Default OFF — only activated when BASELINE_VALUE_HEAD=strategic.
# See plan /root/.claude/plans/go-1-compressed-hummingbird.md Phase D.
HOLD_HORIZON = float(os.environ.get("BASELINE_HOLD_HORIZON", "20"))
FORWARD_REACH_WEIGHT = float(os.environ.get("BASELINE_FORWARD_REACH_WEIGHT", "0.5"))
FORWARD_REACH_HORIZON = float(os.environ.get("BASELINE_FORWARD_REACH_HORIZON", "15"))
# Term C — finishing pressure (Phase E, PI 2026-05-25 evening).
# Continuous bonus that grows as the weakest opp's strength approaches 0.
# 0 disables Term C; default keeps parity with strategic-no-C.
FINISH_BONUS = float(os.environ.get("BASELINE_FINISH_BONUS", "50"))
FINISH_THRESHOLD = float(os.environ.get("BASELINE_FINISH_THRESHOLD", "200"))


def _read(obs, attr, default):
    if hasattr(obs, attr):
        return getattr(obs, attr)
    return obs.get(attr, default) if isinstance(obs, dict) else default


def favor(obs, me: int, num_seats: int = 2, gamma: float = DEFAULT_GAMMA) -> float:
    planets = _read(obs, "planets", []) or []
    fleets = _read(obs, "fleets", []) or []
    step = int(_read(obs, "step", 0))

    ships_by_owner: dict[int, float] = {}
    prod_by_owner: dict[int, float] = {}
    for p in planets:
        owner = int(p[1])
        if owner < 0:
            continue
        ships_by_owner[owner] = ships_by_owner.get(owner, 0.0) + float(p[5])
        prod_by_owner[owner] = prod_by_owner.get(owner, 0.0) + float(p[6])
    for f in fleets:
        owner = int(f[1])
        if owner < 0:
            continue
        ships_by_owner[owner] = ships_by_owner.get(owner, 0.0) + float(f[6])

    my_ships = ships_by_owner.get(me, 0.0)
    my_prod = prod_by_owner.get(me, 0.0)

    opps = sorted(
        o for o in (set(ships_by_owner) | set(prod_by_owner))
        if o != me and o >= 0
    )

    elim_bonus = 0.0
    if num_seats <= 2 or len(opps) < 2:
        # 2P (or degenerate <=1 opp survives): UNCHANGED from baseline —
        # max-of-opps, no bias, no bonus. The 2P uniform bias was tested
        # and rolled back (regresses vs v15).
        opp_ships = max((ships_by_owner.get(o, 0.0) for o in opps), default=0.0)
        opp_prod = max((prod_by_owner.get(o, 0.0) for o in opps), default=0.0)
    else:
        # 4P: weighted sum (weakest 1.5x) + elim bonus when we can finish.
        opp_strengths = {
            o: ships_by_owner.get(o, 0.0)
               + prod_by_owner.get(o, 0.0) * STRENGTH_PROD_WEIGHT
            for o in opps
        }
        weakest = min(opps, key=lambda o: opp_strengths[o])
        weakest_str = opp_strengths[weakest]
        opp_ships = sum(
            ships_by_owner.get(o, 0.0)
            * (WEAKEST_ENEMY_MULT_4P if o == weakest else 1.0)
            for o in opps
        )
        opp_prod = sum(
            prod_by_owner.get(o, 0.0)
            * (WEAKEST_ENEMY_MULT_4P if o == weakest else 1.0)
            for o in opps
        )
        my_strength = my_ships + my_prod * STRENGTH_PROD_WEIGHT
        if (weakest_str <= WEAK_ENEMY_THRESHOLD
                and my_strength >= ELIMINATION_GATE_RATIO * weakest_str):
            elim_bonus = ELIMINATION_BONUS

    pv = pv_horizon(step, 0, gamma=gamma, t_total=EPISODE_STEPS)
    return (my_ships - opp_ships) + (my_prod - opp_prod) * pv + elim_bonus


def favor_composite(obs, me: int, num_seats: int = 2,
                    gamma: float = DEFAULT_GAMMA) -> float:
    """`composite_capture_value` adapted to the (obs, me, num_seats, gamma)
    signature `chooser` expects. `gamma` is intentionally ignored —
    composite uses linear time-remaining weighting instead of γ-discount.
    `num_seats` is ignored — composite doesn't differentiate opps.

    Prior live evidence (iter_v1 sub 52661990, 2026-05-14):
    composite head on the v7_0 chooser → ladder μ 1034.7 (vs v15 1108.4).
    Wire only as an opt-in A/B; do NOT default this on. The clean
    baseline value is `favor` (with A2 4P-weakness exploitation).
    """
    from lib.value_heads import composite_capture_value
    return composite_capture_value(obs, me)


def _positional_ship_value(obs, me: int) -> float:
    """Sum over my ships (on-planet + in-flight) of
    1.0 / (1.0 + d_min / SPATIAL_DECAY), where d_min = distance to
    nearest non-our planet. Value ranges 0..1 per ship:
    1.0 when adjacent (d=0), 0.5 at d=SPATIAL_DECAY, ~0.2 at d=120.

    Returns 0.0 if no non-our planet remains (degenerate end-state).
    """
    planets = _read(obs, "planets", []) or []
    fleets = _read(obs, "fleets", []) or []
    non_our = [(float(p[2]), float(p[3])) for p in planets if int(p[1]) != me]
    if not non_our:
        return 0.0
    total = 0.0
    for p in planets:
        if int(p[1]) != me:
            continue
        x, y = float(p[2]), float(p[3])
        d_min = min(math.hypot(x - tx, y - ty) for tx, ty in non_our)
        weight = 1.0 / (1.0 + d_min / SPATIAL_DECAY)
        total += float(p[5]) * weight
    for f in fleets:
        if int(f[1]) != me:
            continue
        x, y = float(f[2]), float(f[3])
        d_min = min(math.hypot(x - tx, y - ty) for tx, ty in non_our)
        weight = 1.0 / (1.0 + d_min / SPATIAL_DECAY)
        total += float(f[6]) * weight
    return total


def favor_hybrid_spatial(obs, me: int, num_seats: int = 2,
                         gamma: float = DEFAULT_GAMMA) -> float:
    """favor_hybrid + positional pull toward non-our planets (2P only).

    Layered on top of the validated hybrid head (composite in 2P,
    A2-favor in 4P). The spatial term is applied ONLY in 2P games —
    in 4P, the A2 weakness-exploitation already biases toward the
    weakest opp's positions, and the bv33jlzwj A/B (3/32 first-place,
    max=1503ms) showed spatial regresses 4P substantially. 2P-only
    keeps the validated A2-4P path identical to favor_hybrid.

    The spatial term is purely additive — when SPATIAL_WEIGHT=0 or
    num_seats > 2 it equals favor_hybrid exactly.
    """
    base = favor_hybrid(obs, me, num_seats, gamma)
    if SPATIAL_WEIGHT == 0.0 or num_seats > 2:
        return base
    return base + SPATIAL_WEIGHT * _positional_ship_value(obs, me)


def favor_hybrid(obs, me: int, num_seats: int = 2,
                 gamma: float = DEFAULT_GAMMA) -> float:
    """2P uses composite (waste-aware, validated by audit-workflow A/B:
    93.8% vs v9_scavenge, 67.2% vs v15). 4P uses `favor` with A2
    4P-weakness exploitation. Domains are disjoint by construction —
    composite has no 4P opp aggregation (`composite-value-head-2p-only.md`
    flag), and A2's per-weakest multiplier + elim bonus only fire when
    num_seats > 2.
    """
    if num_seats <= 2:
        return favor_composite(obs, me, num_seats, gamma)
    return favor(obs, me, num_seats, gamma)


def favor_strategic(obs, me: int, num_seats: int = 2,
                    gamma: float = DEFAULT_GAMMA) -> float:
    """Model enrichment (Phase D, PI 2026-05-25 evening, Rule 40).

    On top of the F1+F2+elim base from `favor`, adds two terms that
    encode information the static-snapshot leaf was missing:

    Term A — Per-planet hold discount. For each of my owned planets P,
        scale its contribution to F2 (production stream) by
        `min(1, threat_eta / HOLD_HORIZON)`. Planets the enemy can reach
        soon contribute less. Naturally penalises capture-then-lose
        ("softening") because the captured planet's threat_eta is small.
    Term B — Forward-reach. For each of my owned planets P, sum the
        production of enemy planets reachable from P via straight-line
        flight within `FORWARD_REACH_HORIZON` turns. Adds
        `FORWARD_REACH_WEIGHT * sum_P reach(P)` to favor. Naturally
        rewards frontier captures and penalises back-yard captures —
        no "direction bonus" or "frontier bonus" needed.

    Term C — Finishing pressure (Phase E, 2026-05-25). Continuous bonus
        that grows as the weakest opp's strength approaches 0. Adds
        `FINISH_BONUS * max(0, 1 - weakest_str / FINISH_THRESHOLD)` to
        favor. Generalises the 4P-only discrete ELIMINATION_BONUS to
        2P AND ramps earlier (before opp is at the eliminable threshold).
        Naturally rewards finishing weak opponents instead of nibbling.

    Default knobs (HOLD_HORIZON=20, FORWARD_REACH_WEIGHT=0.5,
    FORWARD_REACH_HORIZON=15, FINISH_BONUS=50, FINISH_THRESHOLD=200)
    shape the curves; tune via env vars.

    Parity: with HOLD_HORIZON=0 AND FORWARD_REACH_WEIGHT=0 AND
    FINISH_BONUS=0, this reduces to `favor` exactly.
    """
    planets = _read(obs, "planets", []) or []
    fleets = _read(obs, "fleets", []) or []
    step = int(_read(obs, "step", 0))

    ships_by_owner: dict[int, float] = {}
    prod_by_owner: dict[int, float] = {}
    for p in planets:
        owner = int(p[1])
        if owner < 0:
            continue
        ships_by_owner[owner] = ships_by_owner.get(owner, 0.0) + float(p[5])
        prod_by_owner[owner] = prod_by_owner.get(owner, 0.0) + float(p[6])
    for f in fleets:
        owner = int(f[1])
        if owner < 0:
            continue
        ships_by_owner[owner] = ships_by_owner.get(owner, 0.0) + float(f[6])

    my_ships = ships_by_owner.get(me, 0.0)

    opps = sorted(
        o for o in (set(ships_by_owner) | set(prod_by_owner))
        if o != me and o >= 0
    )

    # Opp aggregation: same as `favor` (2P max, 4P weighted-sum-with-weakest-1.5x).
    elim_eligible = False
    weakest_str = 0.0
    if num_seats <= 2 or len(opps) < 2:
        opp_ships = max((ships_by_owner.get(o, 0.0) for o in opps), default=0.0)
        opp_prod = max((prod_by_owner.get(o, 0.0) for o in opps), default=0.0)
    else:
        opp_strengths = {
            o: ships_by_owner.get(o, 0.0)
               + prod_by_owner.get(o, 0.0) * STRENGTH_PROD_WEIGHT
            for o in opps
        }
        weakest = min(opps, key=lambda o: opp_strengths[o])
        weakest_str = opp_strengths[weakest]
        opp_ships = sum(
            ships_by_owner.get(o, 0.0)
            * (WEAKEST_ENEMY_MULT_4P if o == weakest else 1.0)
            for o in opps
        )
        opp_prod = sum(
            prod_by_owner.get(o, 0.0)
            * (WEAKEST_ENEMY_MULT_4P if o == weakest else 1.0)
            for o in opps
        )
        elim_eligible = True

    # Term A — hold-discount on MY planets. Build WorldModel for threat ETA.
    # HOLD_HORIZON=0 disables Term A (every planet gets hold_score=1.0 = no discount).
    if HOLD_HORIZON <= 0:
        my_prod_discounted = prod_by_owner.get(me, 0.0)
    else:
        my_prod_discounted = 0.0
        try:
            from lib.intent import World
            from lib.world_model import WorldModel
            world = World.from_obs(obs)
            wm = WorldModel.from_world(world)
            for p in planets:
                if int(p[1]) != me:
                    continue
                pid = int(p[0])
                threat_eta = wm.time_to_enemy_threat(pid, me, world, arrival_eta=0)
                if threat_eta is None:
                    hold_score = 1.0
                else:
                    hold_score = min(1.0, float(threat_eta) / HOLD_HORIZON)
                my_prod_discounted += float(p[6]) * hold_score
        except Exception:
            # If WorldModel construction fails (e.g. malformed mid-rollout obs),
            # fall back to undiscounted my_prod so we never zero-out the leaf.
            my_prod_discounted = prod_by_owner.get(me, 0.0)

    pv = pv_horizon(step, 0, gamma=gamma, t_total=EPISODE_STEPS)
    score = (my_ships - opp_ships) + (my_prod_discounted - opp_prod) * pv

    # Elimination bonus — uses discounted my_prod for the strength check.
    if elim_eligible:
        my_strength = my_ships + my_prod_discounted * STRENGTH_PROD_WEIGHT
        if (weakest_str <= WEAK_ENEMY_THRESHOLD
                and my_strength >= ELIMINATION_GATE_RATIO * weakest_str):
            score += ELIMINATION_BONUS

    # Term B — forward-reach bonus.
    if FORWARD_REACH_WEIGHT > 0.0:
        from lib.fleet import speed as fleet_speed
        my_planets_list = [p for p in planets if int(p[1]) == me]
        enemy_planets_list = [
            p for p in planets
            if int(p[1]) >= 0 and int(p[1]) != me
        ]
        if my_planets_list and enemy_planets_list:
            # Approximate launch speed via mean garrison; the chooser's actual
            # launches use the source's ship count, but this is a reach proxy
            # not an admissibility test.
            ships_proxy = max(
                2.0,
                sum(float(p[5]) for p in my_planets_list) / len(my_planets_list),
            )
            v = fleet_speed(int(ships_proxy))
            if v > 0.0:
                reach_sum = 0.0
                for mp in my_planets_list:
                    mx, my_p = float(mp[2]), float(mp[3])
                    for ep in enemy_planets_list:
                        ex, ey = float(ep[2]), float(ep[3])
                        dist = math.hypot(ex - mx, ey - my_p)
                        eta = dist / v
                        if eta <= FORWARD_REACH_HORIZON:
                            reach_sum += float(ep[6])
                score += FORWARD_REACH_WEIGHT * reach_sum

    # Term C — finishing pressure. Continuous bonus that grows as the
    # weakest opp's strength approaches 0. Applies in BOTH 2P and 4P
    # (generalises the 4P-only discrete ELIMINATION_BONUS). Disabled when
    # FINISH_BONUS=0 (preserves Phase-D-only parity).
    if opps and FINISH_BONUS > 0:
        opp_strengths_c = {
            o: ships_by_owner.get(o, 0.0)
               + prod_by_owner.get(o, 0.0) * STRENGTH_PROD_WEIGHT
            for o in opps
        }
        target_str = min(opp_strengths_c.values())
        finishing_pressure = max(0.0, 1.0 - target_str / FINISH_THRESHOLD)
        score += FINISH_BONUS * finishing_pressure

    return score


def select_favor_fn():
    """Pick the leaf value function.

    Env var `BASELINE_VALUE_HEAD`:
      - unset / anything else -> `favor` (default, v15 baseline + A2 4P).
      - "composite"           -> `favor_composite` (2P waste-aware,
                                  composite_capture_value head).
      - "hybrid"              -> `favor_hybrid` (composite in 2P,
                                  A2-favor in 4P).

    The chooser uses the same function for both `build_idle_baseline` and
    `score_action` so the Δ stays well-defined.

    Wave-incentive layering (baseline_wave v2 2026-05-24 PM):
      - BASELINE_STOCKPILE_PENALTY=1    subtracts ε·Σ excess² over our
                                        planets (gentle drainage floor).

    v1 also wrapped `inflight_hhi_bonus` here; removed in v2 because the
    inflight-at-leaf HHI doesn't measure combat-rule-1 stacking (the
    fleet has usually arrived by leaf time). The replacement signal is
    `_coord_bonus` in chooser_trajectory.py which counts arrival-cohort
    concentration during the rollout — the right granularity for waves.
    `inflight_hhi_bonus` stays in lib/value_heads.py as research code
    + unit-test coverage; BASELINE_HHI_BONUS env var is no-op now.

    Both layers default OFF so the orbitfix-peak path is byte-identical.
    """
    choice = os.environ.get("BASELINE_VALUE_HEAD", "").strip().lower()
    if choice == "composite":
        base = favor_composite
    elif choice == "hybrid":
        base = favor_hybrid
    elif choice == "hybrid_spatial":
        base = favor_hybrid_spatial
    elif choice == "strategic":
        base = favor_strategic
    else:
        base = favor

    stk_on = os.environ.get("BASELINE_STOCKPILE_PENALTY", "0").strip() == "1"
    if not stk_on:
        return base

    eps = float(os.environ.get("BASELINE_STOCKPILE_EPS", "0.001"))
    target = float(os.environ.get("BASELINE_STOCKPILE_TARGET", "50"))
    # v5: turn gate. Pre-gate (step < turn_gate), the penalty is silenced so
    # early-game solo expansion isn't starved — that was the v3.1 bug.
    # Post-gate, mid/late-game stockpiles face the drainage pressure that
    # forces the chooser to fire its 100+ ship rear planets into waves.
    turn_gate = int(os.environ.get("BASELINE_STOCKPILE_TURN_GATE", "0"))

    from lib.value_heads import stockpile_pressure_penalty

    # NOTE: parameter names must match `num_seats` / `gamma` because callers
    # in chooser.py and chooser_trajectory.py pass `gamma=` as a keyword.
    # Using `gamma_` here caused a silent TypeError on every leaf eval which
    # the rollout swallowed, zeroing every candidate's Δ — agent emitted no
    # fleets at all even with δ → 0. (Found 2026-05-24 by ablation showing
    # δ=1e-6 still fully suppresses emissions.)
    def _wave_wrapped(obs, me, num_seats=2, gamma=DEFAULT_GAMMA):
        v = base(obs, me, num_seats, gamma)
        if int(obs.get("step", 0)) >= turn_gate:
            v -= stockpile_pressure_penalty(obs, me, eps=eps, target=target)
        return v

    return _wave_wrapped
