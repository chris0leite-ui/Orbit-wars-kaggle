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

# Phase F — F3 observability. favor_strategic's Term A may fall back to
# undiscounted my_prod if the leaf-time threat-ETA approximation raises
# (e.g. on a malformed mid-rollout obs). Public for inspection from
# harness scripts / tests.
_TERM_A_FALLBACK_COUNT = 0
_TERM_A_WARNED = False


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


MIN_FLEET_SIZE_LOCAL = 2  # mirrors agents/baseline/proposer.MIN_FLEET_SIZE


def _realistic_threat_eta(d_ships: float, d_x: float, d_y: float,
                          a_ships: float, a_x: float, a_y: float) -> float:
    """Straight-line ETA for an attacker at (a_x, a_y) with `a_ships`
    garrison to capture a defender at (d_x, d_y) with `d_ships`.

    Models the realistic launch: the attacker sends a capture-size fleet
    (`max(MIN_FLEET_SIZE, d_ships + 1)`). Speed is `fleet_speed(launch)`
    at that size. Returns `inf` if the attacker cannot afford a capture-
    size launch (no threat from that source).

    The speed depends on the LAUNCH size, not on the attacker's total
    garrison or on the defender's garrison directly. fleet_speed is
    monotone increasing in ship count (lib/fleet.py:20-35) — so the
    capture-size launch travels FASTER against a well-defended planet
    than against a near-neutral one (you have to send more, you fly
    quicker). This is the right physics for "how soon could opp reach
    me realistically?", replacing the prior MIN_FLEET_SIZE-floor
    approximation that under-estimated all threats.
    """
    from lib.fleet import speed as fleet_speed
    capture = max(MIN_FLEET_SIZE_LOCAL, int(d_ships) + 1)
    if int(a_ships) < capture:
        return float("inf")  # attacker can't field a capture-size launch
    v = fleet_speed(capture)
    if v <= 0.0:
        return float("inf")
    return math.hypot(a_x - d_x, a_y - d_y) / v


def _hold_discounted_prod(planets, self_owner: int, threats,
                          horizon: float) -> float:
    """Sum `prod * hold_score` over `self_owner`'s planets.

    `threats` is a list of (x, y, ships) tuples describing the attacker
    planets that could threaten `self_owner`'s holdings. The hold score
    for each owned planet is `min(1, min_eta / horizon)`, where
    `min_eta` is the smallest realistic threat ETA from any attacker.

    Returns the raw (undiscounted) prod sum when `horizon <= 0` or when
    there are no threats — i.e. degenerates to the un-discounted F2
    contribution exactly, which is what we want for parity.
    """
    if horizon <= 0 or not threats:
        return sum(float(p[6]) for p in planets if int(p[1]) == self_owner)
    total = 0.0
    for p in planets:
        if int(p[1]) != self_owner:
            continue
        p_x, p_y, p_ships = float(p[2]), float(p[3]), float(p[5])
        min_eta = float("inf")
        for a_x, a_y, a_ships in threats:
            eta = _realistic_threat_eta(p_ships, p_x, p_y, a_ships, a_x, a_y)
            if eta < min_eta:
                min_eta = eta
        if min_eta == float("inf"):
            hold_score = 1.0
        else:
            hold_score = min(1.0, min_eta / horizon)
        total += float(p[6]) * hold_score
    return total


def favor_strategic(obs, me: int, num_seats: int = 2,
                    gamma: float = DEFAULT_GAMMA) -> float:
    """Unified leaf for any-P (2P, 4P, ...). One code path, no mode split.

    Measures progress toward the FFA win condition `my_strength −
    max_o(opp_strength_o)` — same formula in 2P (trivially the one
    opponent) and 4P (the leader). The asymmetric Phase-F hold-discount
    on my_prod (the calibrated defensive "fear" gradient) is preserved
    in BOTH modes; max-of-opps controls the F2 scale across opp counts
    so the asymmetric form no longer needs a symmetric counterpart in
    4P. The previous 2P-asymmetric / 4P-symmetric mode split (commits
    523a221 → fcaf414) is collapsed.

    Term A — Asymmetric hold-discount on MY production. Each of my
        planets' contribution to F2 is scaled by
        `min(1, threat_eta / HOLD_HORIZON)`, using the realistic
        capture-size launch ETA from the nearest non-owner planet
        that can afford it. Opp prod is RAW max-of-opps.

    Term B — Capture-feasible forward-reach. For each of my owned
        planets P and each enemy planet T, credit `T.prod` to the
        reach sum iff (a) my launch from P is capture-size or larger
        (`launch ≥ capture`), AND (b) the launch arrives within
        `FORWARD_REACH_HORIZON` at the per-pair realistic launch
        speed. The capture-feasibility gate (added 2026-05-26 v2)
        prevents leaf inflation against strong-stockpile defenders
        where we could REACH a planet but couldn't TAKE it.

    Term C — Finishing pressure (continuous through elimination).
        For each EXPECTED opp seat (`num_seats - 1` of them), credit
        `FINISH_BONUS * max(0, 1 - opp_strength / FINISH_THRESHOLD)`.
        Eliminated seats (gone from obs) contribute the full bonus —
        no anti-elimination cliff. In 4P the total snowballs toward
        `(num_seats - 1) * FINISH_BONUS` as opps die.

    Knobs (all env-var configurable; defaults below when
    BASELINE_VALUE_HEAD=strategic):
        HOLD_HORIZON=20, FORWARD_REACH_WEIGHT=0.5,
        FORWARD_REACH_HORIZON=15, FINISH_BONUS=50, FINISH_THRESHOLD=200.

    BASELINE_HOLD_HORIZON=0 disables Term A.
    BASELINE_FORWARD_REACH_WEIGHT=0 disables Term B.
    BASELINE_FINISH_BONUS=0 disables Term C AND restores the discrete
    4P+ ELIMINATION_BONUS path (back-compat parity with `favor`).

    Approximations kept (with reasons):
      - Static planet positions at the leaf. The leaf is computed
        AFTER the rollout (positions already advanced K steps by
        fs_step). The further HOLD_HORIZON / FORWARD_REACH_HORIZON
        projection from leaf time uses straight-line distance,
        ignoring orbital motion. Exact for static planets; bounded
        error for orbiters. Plumbing `lib/orbit.predict_relative_cached`
        here is a separate experiment.

    Approximations removed (from earlier iterations):
      - Phase F's `v_opp = fleet_speed(2)` threat-speed floor.
      - Phase F's `ships ≥ 2` opp-planet hard filter.
      - Phase F's mean-garrison `ships_proxy` in Term B.
      - The 523a221 symmetric Term A on opp_prod in 4P (broke 2P calibration).
      - The fcaf414 4P-only symmetric branch (2P/4P mode split).

    F1/F2 aggregation: max-of-opps in both modes. `weakest_str` is
    still computed for the discrete 4P+ ELIMINATION_BONUS path that
    fires when FINISH_BONUS=0.

    Parity: with HOLD_HORIZON=0 AND FORWARD_REACH_WEIGHT=0 AND
    FINISH_BONUS=0, this returns the max-of-opps version of
    (my_ships - opp_ships) + (my_prod - opp_prod) * pv plus the
    discrete elim_bonus when applicable. Differs from `favor` in 4P
    by the F1/F2 aggregation rule (max-of-opps vs `favor`'s weighted-
    sum-with-1.5x-weakest); equals `favor` in 2P.
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

    # --- F1 aggregation: unified max-of-opps in both 2P and 4P.
    # Same formula for any num_seats. `weakest_str` is still computed
    # for the discrete elim_bonus path (back-compat with `favor` when
    # FINISH_BONUS <= 0). `elim_eligible` only fires in 4P+ since `favor`
    # only fires its discrete bonus there.
    weakest_str = 0.0
    elim_eligible = False
    if not opps:
        opp_ships = 0.0
        weakest = None
    else:
        opp_strengths = {
            o: ships_by_owner.get(o, 0.0)
               + prod_by_owner.get(o, 0.0) * STRENGTH_PROD_WEIGHT
            for o in opps
        }
        weakest = min(opps, key=lambda o: opp_strengths[o])
        weakest_str = opp_strengths[weakest]
        opp_ships = max(ships_by_owner.get(o, 0.0) for o in opps)
        elim_eligible = (num_seats >= 3 and len(opps) >= 2)

    # --- Term A: ASYMMETRIC unified hold-discount.
    # My production stream is scaled by hold_score per planet (the
    # calibrated Phase-F "fear" gradient). Opp production stays RAW;
    # max-of-opps controls F2 scale so the asymmetric form works in
    # both 2P and 4P without symmetric counterpart — symmetric Term A
    # broke 2P calibration (sub 523a221 panel 2/8) because the chooser
    # was tuned to the asymmetric leaf. The unified asymmetric form
    # preserves Phase F's 2P signal AND the post-fcaf414 4P max-of-opps
    # scale fix in a single code path.
    try:
        all_other_threats = [
            (float(p[2]), float(p[3]), float(p[5]))
            for p in planets
            if int(p[1]) >= 0 and int(p[1]) != me
        ]
        my_prod_discounted = _hold_discounted_prod(
            planets, me, all_other_threats, HOLD_HORIZON,
        )
        if not opps:
            opp_prod_discounted = 0.0
        else:
            opp_prod_discounted = max(
                prod_by_owner.get(o, 0.0) for o in opps
            )
    except (KeyError, IndexError, AttributeError, ValueError, TypeError, ZeroDivisionError):
        # F3 observability — narrow catch on malformed mid-rollout obs.
        global _TERM_A_FALLBACK_COUNT, _TERM_A_WARNED
        _TERM_A_FALLBACK_COUNT += 1
        if not _TERM_A_WARNED:
            import warnings
            warnings.warn(
                "favor_strategic Term A fell back to undiscounted my_prod "
                "(malformed mid-rollout obs?). Further fallbacks counted in "
                "value._TERM_A_FALLBACK_COUNT but not warned.",
                RuntimeWarning,
                stacklevel=2,
            )
            _TERM_A_WARNED = True
        my_prod_discounted = prod_by_owner.get(me, 0.0)
        if not opps:
            opp_prod_discounted = 0.0
        else:
            opp_prod_discounted = max(
                prod_by_owner.get(o, 0.0) for o in opps
            )

    pv = pv_horizon(step, 0, gamma=gamma, t_total=EPISODE_STEPS)
    score = (my_ships - opp_ships) + (my_prod_discounted - opp_prod_discounted) * pv

    # --- Discrete 4P ELIMINATION_BONUS (back-compat).
    # Suppressed when Term C is active (FINISH_BONUS > 0) — Term C IS the
    # continuous generalisation. Uses RAW my_prod for the strength gate
    # (F4: discount in the gate makes the gate harder to clear in exactly
    # the threatened-defender case the gate exists for).
    if elim_eligible and FINISH_BONUS <= 0:
        my_prod_raw = prod_by_owner.get(me, 0.0)
        my_strength = my_ships + my_prod_raw * STRENGTH_PROD_WEIGHT
        if (weakest_str <= WEAK_ENEMY_THRESHOLD
                and my_strength >= ELIMINATION_GATE_RATIO * weakest_str):
            score += ELIMINATION_BONUS

    # --- Term B: per-(src, tgt) realistic-launch forward reach.
    # Speed = fleet_speed(realistic capture-size launch from src to tgt),
    # bounded by src's actual garrison. A 2-ship source plays at 2-ship
    # speed; a 100-ship source attacking a strong defender plays at
    # capture-size speed. No global mean-garrison fudge.
    if FORWARD_REACH_WEIGHT > 0.0:
        from lib.fleet import speed as fleet_speed
        my_planets_list = [p for p in planets if int(p[1]) == me]
        enemy_planets_list = [
            p for p in planets
            if int(p[1]) >= 0 and int(p[1]) != me
        ]
        if my_planets_list and enemy_planets_list:
            reach_sum = 0.0
            for mp in my_planets_list:
                mx, my_p = float(mp[2]), float(mp[3])
                m_ships = float(mp[5])
                if int(m_ships) < MIN_FLEET_SIZE_LOCAL:
                    continue  # can't reach anywhere — no launchable garrison
                for ep in enemy_planets_list:
                    ex, ey = float(ep[2]), float(ep[3])
                    e_ships = float(ep[5])
                    capture = max(MIN_FLEET_SIZE_LOCAL, int(e_ships) + 1)
                    launch = min(int(m_ships), capture)
                    # Capture-feasibility gate: only credit reach when our
                    # launch can actually take the planet (launch ≥ capture).
                    # The prior version credited any reachable enemy planet
                    # even when our launch was undersized, inflating the leaf
                    # against strong-stockpile defenders (orbitfix-style).
                    if launch < capture:
                        continue
                    v = fleet_speed(launch)
                    if v <= 0.0:
                        continue
                    dist = math.hypot(ex - mx, ey - my_p)
                    if dist / v <= FORWARD_REACH_HORIZON:
                        reach_sum += float(ep[6])
            score += FORWARD_REACH_WEIGHT * reach_sum

    # --- Term C: finishing pressure, continuous through elimination.
    # Sum FINISH_BONUS * (1 - strength/threshold) over EXPECTED opp seats
    # (num_seats - 1). Opps present in obs scale by their current strength;
    # opp seats missing from obs (eliminated — no planets, no in-flight
    # ships) credit the full FINISH_BONUS. This removes the anti-cliff
    # the prior min-over-finishable_opps produced when the weakest opp
    # transitions from low-strength to gone.
    if FINISH_BONUS > 0 and FINISH_THRESHOLD > 0 and num_seats >= 2:
        expected_opps = max(0, int(num_seats) - 1)
        finishing_score = 0.0
        for o in opps:
            str_o = (ships_by_owner.get(o, 0.0)
                     + prod_by_owner.get(o, 0.0) * STRENGTH_PROD_WEIGHT)
            finishing_score += FINISH_BONUS * max(
                0.0, 1.0 - str_o / FINISH_THRESHOLD,
            )
        dead_count = max(0, expected_opps - len(opps))
        finishing_score += FINISH_BONUS * dead_count
        score += finishing_score

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
