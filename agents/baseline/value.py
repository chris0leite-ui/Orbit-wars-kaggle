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

# Attack-pull leaf params (favor_hybrid_attack_pull only).
# Spatial sibling that pulls ships toward ENEMY planets specifically
# (not neutral). Neutrals are diffuse early-game; using only enemy
# positions gives the chooser a sharper "press forward" signal from
# turn 1. The existing BASELINE_NEUTRAL_BONUS handles neutral
# expansion separately, so no double-counting.
ATTACK_PULL_WEIGHT = float(os.environ.get("BASELINE_ATTACK_PULL_WEIGHT", "0.5"))
ATTACK_PULL_DECAY = float(os.environ.get("BASELINE_ATTACK_PULL_DECAY", "30.0"))


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


def _attack_pull_ship_value(obs, me: int) -> float:
    """Sum over my ships (on-planet + in-flight) of weight by closeness
    to the nearest ENEMY (non-our, non-neutral) planet.

    Weight per ship: 1.0 / (1.0 + d_enemy_min / ATTACK_PULL_DECAY).
      d=0   → 1.0   (on an enemy planet — captured)
      d=30  → 0.5   (forward-deployed; ATTACK_PULL_DECAY default)
      d=80  → ~0.27 (mid-board)
      d=120 → ~0.20 (rear)

    Sibling to `_positional_ship_value` but enemy-only. The
    non_our-based positional term gives weak signal early-game when
    most planets are neutral; this term gives a clean "move toward
    the opponent" signal active from turn 1. Sums on-planet ships
    AND in-flight fleets so a fleet en route to enemy territory
    accrues value as it approaches.

    Returns 0.0 in degenerate end-state (no enemy planets left —
    likely already winning).
    """
    planets = _read(obs, "planets", []) or []
    fleets = _read(obs, "fleets", []) or []
    enemy = [(float(p[2]), float(p[3])) for p in planets
             if int(p[1]) >= 0 and int(p[1]) != me]
    if not enemy:
        return 0.0
    total = 0.0
    for p in planets:
        if int(p[1]) != me:
            continue
        x, y = float(p[2]), float(p[3])
        d_min = min(math.hypot(x - ex, y - ey) for ex, ey in enemy)
        total += float(p[5]) * (1.0 / (1.0 + d_min / ATTACK_PULL_DECAY))
    for f in fleets:
        if int(f[1]) != me:
            continue
        x, y = float(f[2]), float(f[3])
        d_min = min(math.hypot(x - ex, y - ey) for ex, ey in enemy)
        total += float(f[6]) * (1.0 / (1.0 + d_min / ATTACK_PULL_DECAY))
    return total


def favor_attack_pull(obs, me: int, num_seats: int = 2,
                      gamma: float = DEFAULT_GAMMA) -> float:
    """favor (cheap positionless baseline) + attack-pull term (2P only).

    The cheap variant of the attack-pull design. Layers the positional
    pull onto the canonical `favor` head — does NOT go through the
    expensive composite_capture_value path (which runs predict_fleet_fate
    per fleet at every leaf call, adding ~600ms per turn when
    COMPOSITE_FLEET_SURVIVAL_CHECK=1).

    Wallclock vs `favor`: ~10-30 ms per turn extra (single _attack_pull
    loop per leaf call × ~60 leaf calls).

    4P fallback: returns favor unchanged. The A2 weakness-exploitation
    inside `favor` already biases toward weakest opp position in 4P;
    adding positional bias on top historically regresses 4P per the
    favor_hybrid_spatial audit.
    """
    base = favor(obs, me, num_seats, gamma)
    if ATTACK_PULL_WEIGHT == 0.0 or num_seats > 2:
        return base
    return base + ATTACK_PULL_WEIGHT * _attack_pull_ship_value(obs, me)


def favor_hybrid_attack_pull(obs, me: int, num_seats: int = 2,
                             gamma: float = DEFAULT_GAMMA) -> float:
    """favor_hybrid + attack-pull term (2P only). Heavy variant.

    Layers attack_pull on top of favor_hybrid (which calls
    favor_composite in 2P — the expensive waste-aware leaf with
    per-fleet predict_fleet_fate). Per-turn cost roughly 3-5x the
    cheap `favor_attack_pull` variant; use only if the composite
    head's per-fleet survival check is desired AND has been verified
    to fit inside the wallclock budget at the agent's K/N settings.

    PI 2026-05-23 smoke: enabling this on orbitfix_kt_p23 (Phase 2
    adaptive K + Phase 3 survival check) blew p95 from 790ms → 1931ms
    and lost vs v7_0. Stay on `favor_attack_pull` unless explicitly
    tuned.

    Purely additive — when ATTACK_PULL_WEIGHT=0 or num_seats > 2 it
    equals favor_hybrid exactly.
    """
    base = favor_hybrid(obs, me, num_seats, gamma)
    if ATTACK_PULL_WEIGHT == 0.0 or num_seats > 2:
        return base
    return base + ATTACK_PULL_WEIGHT * _attack_pull_ship_value(obs, me)


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
    """
    choice = os.environ.get("BASELINE_VALUE_HEAD", "").strip().lower()
    if choice == "composite":
        return favor_composite
    if choice == "hybrid":
        return favor_hybrid
    if choice == "hybrid_spatial":
        return favor_hybrid_spatial
    if choice == "hybrid_attack_pull":
        return favor_hybrid_attack_pull
    if choice == "attack_pull":
        return favor_attack_pull
    return favor
