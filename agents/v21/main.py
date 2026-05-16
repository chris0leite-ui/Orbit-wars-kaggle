"""v21 = v20 + target-quality prefilter (E1) + joint-commitment emit (A)
       + capture-and-hold check (E2). Replaces v20's independent-Δ emit
       with empirical-waste-driven filtering and joint scoring.

Empirical motivation (16 live replays, 8 v20 + 8 v15, 2026-05-16):
- 15% of launches target COMETS; 100% miss → ~20 ships/game wasted.
- 60–70% of CAPTUREs lost back within 50 turns (median hold = 8 turns).
- 43–53% of lost-backs are UNDEFENSIBLE (outnumbered locally at
  recapture in an R=30 game-unit neighborhood).
- v20's dogpile change regressed −12 μ vs v15 live despite local
  65.6% h2h — independent-Δ summation overcounts joint value.

Three patches stacked behind flags so each can be A/B'd in isolation:
  E1 (CAPTURE_HOLD_PREFILTER): in `_cheap_marginal_value`, multiply
     capture credit by `target_quality` = `comet_discount * force_factor`.
     Comet targets get 0.0 (empirical 100% miss). Targets in opp-favored
     neighborhoods (local force ratio < 0.35) get soft-discounted.
  A  (JOINT_RESCORE): emit loop becomes greedy joint-commitment.
     After each commit, re-score next K candidates against a fresh
     rollout that includes all committed moves. Stops when no
     remaining candidate has Δ > 0 vs the latest committed-baseline,
     wallclock is one rollout away, or 4-round cap is hit.
  E2 (CAPTURE_HOLD_CHECK): for each top candidate after joint-Δ,
     run one extra 40-step mini-rollout under a COUNTER-RECAPTURE-aware
     opp; multiply joint-Δ by hold_p (fraction of final 5 steps where
     we own the candidate's target). Drop candidates with score < ε.

Pipeline (when all flags True):
  cheap-rank with target_quality (E1) → banded dedup →
  validate top n_affordable (existing) → greedy joint-commit emit (A) →
  optional hold-check filter on the committed candidates (E2) → return.

Original v8_scavenge / v20 design notes preserved below.

---

v8_scavenge — fast-sim chooser with opp-trajectory baseline subtraction.

Pipeline:
  Δ = favor(leaf with my_action at wait_N + opp_traj replayed)
      − favor(leaf with me idle, same opp_traj replayed)

`opp_traj` is pre-computed once per turn via `_opp_policy` (lite_greedy
with bounce-check) on each non-me seat's observation. Both baseline and
every candidate replay this SAME trajectory — common random numbers —
so opp's expansion cancels in Δ and only my action's marginal value
remains. The wait-N candidate's value emerges from this evaluation:
long waits are correctly penalised when opp's expansion outpaces my
hoarding, and rewarded when waiting unlocks a high-value near target
that fire-now can't afford.

Why the prior strict-idle baseline failed: when opp_idle was assumed,
my "fire fast at far-prod-target" candidate scored +386 at horizon=30
because the baseline saw no opp captures. In reality (Felipe seed
1492346051), opp captures 4 planets in those 30 turns and the right
play is "wait 17 turns, accumulate 31 ships, fire at near prod-5
neutral" (Δ=+475 under the corrected baseline). With the opp_traj fix
the chooser picks the wait-then-fire candidate naturally.

For ORBITING wait-N candidates the aim must rotate BOTH src and tgt
forward by omega*wait_N (the planets co-rotate, so relative geometry
is preserved at fire time). Rotating only the target — the prior code —
gave wildly wrong angles and inflated eta, blocking the wait-N
candidate via the MAX_HORIZON check.
"""

from __future__ import annotations

import math
import os
import time

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from lib.aim import aim_orbiting
from lib.fast_sim import clone as fs_clone
from lib.fast_sim import from_obs as fs_from_obs
from lib.fast_sim import step as fs_step
from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.opp_model import lite_greedy_policy as _opp_policy
from lib.orbit import is_orbiting as _is_orbiting
from lib.orbit import predict_relative as _orbit_predict_relative
from lib.scoring import pv_horizon
from lib.world_model import WorldModel

# ---------------------------------------------------------------------------
# Tunable knobs
# ---------------------------------------------------------------------------

EPISODE_STEPS = 500
NUM_TARGETS_PER_SOURCE = 8       # K nearest non-owned planets per source
MIN_FLEET_SIZE = 2               # 1-ship fleets are slow + rarely useful

# Two-stage scoring (Iteration 1, 2026-05-16):
# Cheap pre-rank with WorldModel.owner_at/ships_at (~0.1ms each), then
# only run the expensive fast_sim K-step rollout on the top
# N_VALIDATE candidates. The pre-rank's known weakness is misattributing
# orbital captures (lib.world_model.fleet_target_planet uses straight
# ray-cast) — but we only need RANK to be approximately right; the
# fast_sim validation is ground truth for the actual outcome.
#
# N_VALIDATE bumped 30→60 after first try regressed to 65.6% (Wlo 0.534)
# vs the 75% (Wlo 0.579) single-stage baseline. The cheap-rank was
# dropping borderline candidates that fast_sim would have scored
# positive; widening the validate pool restores most of the lift.
# Pre-rank filter also relaxed: include cheap-zero candidates (potential
# reinforcement/scavenge fast_sim might value).
#
# Budget impact: pre-rank ~15ms + validate ~60×5ms = ~315ms per turn —
# still well under the 1000ms ceiling.
N_VALIDATE = 60

# Forward-sim horizon parameters
SIM_SETTLE_TURNS = 2             # extra idle turns after arrival to settle combat
MIN_HORIZON = 25                 # Every candidate's rollout runs at least
                                 # this many steps. Long enough for a typical
                                 # capture to be exposed to opp's reactive
                                 # counter-launch (which is the real test
                                 # of whether a captured planet "sticks").
MAX_HORIZON = 40                 # baseline cache depth.

# Wallclock safety. The env's actTimeout is 1000ms. Two-stage scoring
# brought max from 1494ms (single-stage) down to 1131ms (Iter 1 panel)
# — still occasional outliers because the deadline check fires BETWEEN
# candidates, and a single fast_sim K-step rollout in mid-late game with
# many in-flight fleets can take 200-300ms (per-step cost ~10ms instead
# of the docs-stated 0.12ms with few fleets). Worst case used to be:
# budget(600) + one-slow-candidate(~300) + overhead(~30) = ~930ms in
# theory, but panel showed 1131ms outliers — slower per-step cost in
# practice.
#
# Adaptive fix (Iteration 1.1, 2026-05-16): measure per-step cost ONCE
# at the start of agent(), use it to compute N_AFFORDABLE_VALIDATE.
# Effective cap = min(N_VALIDATE, N_AFFORDABLE). Bounds the worst
# case to ~(budget + 1 candidate worth) ≈ 700ms reliably.
WALLCLOCK_BUDGET_MS = 600.0

# Parity-test override: the bundle-parity gate sets this env var to
# effectively unbound the budget, so every candidate is scored and the
# agent becomes a pure function of `obs`. Otherwise mid-candidate-list
# deadline bails create source-vs-bundle action drift from CPU jitter
# alone — exactly what the parity gate is designed to catch as a real
# bundling defect, but timing isn't one. Pattern lifted from
# `lib/v7_search.py::_WALLCLOCK_ENV_VAR`.
_WALLCLOCK_ENV_VAR = "ORBIT_WARS_PARITY_WALLCLOCK_MS"


def _effective_wallclock_ms() -> float:
    override = os.environ.get(_WALLCLOCK_ENV_VAR)
    if not override:
        return WALLCLOCK_BUDGET_MS
    try:
        return float(override)
    except ValueError:
        return WALLCLOCK_BUDGET_MS

# Safety factor on the per-candidate cost estimate. fast_sim's per-step
# cost varies within a rollout (combat steps are slower than no-combat
# steps), so a one-shot measurement underestimates. 1.5× covers the
# variance.
_PER_CANDIDATE_SAFETY = 1.5
# Reserved for non-validate work (pre-rank, baseline build, emit).
_RESERVED_OVERHEAD_MS = 50.0

# ---------------------------------------------------------------------------
# v21 patch flags
# ---------------------------------------------------------------------------
# Each flag isolates one patch so a regression in any single patch can be
# diagnosed by toggling its flag while keeping the others on.
JOINT_RESCORE          = True   # Patch A — joint-commitment emit
CAPTURE_HOLD_PREFILTER = True   # Patch E1 — cheap comet/neighborhood discount
CAPTURE_HOLD_CHECK     = True   # Patch E2 — rollout-based hold probability

# E1 constants (replay-empirical, 2026-05-16):
# - All tracked comet shots in the v20/v15 replay sample MISSED, so the
#   comet credit is zero, not a soft discount. The cheap-rank can still
#   surface a comet candidate (zero × eta_credit = 0 → −10 floor cut).
# - At local force ratio < 0.35, lost-back rate spiked in the data.
#   Below the gate, capture credit is linearly attenuated; at 0 (deep
#   opp territory) the credit is 0; at the gate it's full.
NEIGHBORHOOD_RADIUS    = 30.0   # game-units; matches replay analyzer R
UNDEFENSIBLE_GATE      = 0.35   # local force ratio below which we discount

# A constants:
# Max greedy-commit rounds (hard safety cap on commit loop, in addition
# to the deadline / Δ≤0 termination conditions).
MAX_COMMIT_ROUNDS      = 4
# Per-round, how many of the next-best candidates to re-score against
# the new committed-baseline. Wallclock-bounded; 5 is enough to capture
# most plausible 2nd/3rd commits without burning rollouts on tail
# candidates that won't win the re-scoring.
RESCORE_TOP_K          = 5

# E2 constants:
# Top-N candidates that get the expensive hold-check rollout (cost ~150ms
# each in late-game; cap protects wallclock).
HOLD_CHECK_TOP_N       = 3
# Mini-rollout length for the hold-check. Long enough to see opp's
# realistic counter-recapture window (~15 turns past the typical
# capture eta of 20-25).
HOLD_CHECK_HORIZON     = 40
# Fraction of the last K_TAIL steps we must still own the target
# planet to count as a successful hold (so a single mid-rollout
# transient flip doesn't tank hold_p).
HOLD_CHECK_TAIL_STEPS  = 5
# joint-Δ × hold_p must exceed this threshold to survive E2.
HOLD_CHECK_MIN_SCORE   = 1.0
# Wallclock guard: if less than this many ms remain when we enter E2,
# skip the hold-check entirely and emit Patch A's output unfiltered.
HOLD_CHECK_DEADLINE_MS = 200.0


# ---------------------------------------------------------------------------
# Instrumentation counters (write-only side effect; safe for parity-gate).
# ---------------------------------------------------------------------------
# Populated each agent() call. Read externally by scripts/instrument_v21.py.
# These are NOT inputs to any agent decision — they only RECORD what
# happened, so the parity gate (bit-identical actions across two runs of
# the same obs) is unaffected.
_INSTRUMENT_COUNTERS = {
    "last_n_candidates": 0,
    "last_n_validated": 0,
    "last_n_committed": 0,
    "last_n_rescore_rounds": 0,
    "last_n_filtered_by_prefilter": 0,
    "last_n_filtered_by_hold_check": 0,
    "last_n_comet_targets_filtered": 0,
}


def _reset_counters():
    for k in _INSTRUMENT_COUNTERS:
        _INSTRUMENT_COUNTERS[k] = 0

# v12: there is no hard MAX_WAIT cap. With the full opp_trajectory
# replayed in baseline + every candidate (common random numbers),
# wait-N's value emerges from evaluation; long waits are correctly
# penalised when opp's expansion outpaces my hoarding, and rewarded
# when waiting unlocks a high-prod near target that fire-now can't
# afford. The only remaining cap on wait_N is structural:
# `wait_N + eta + SIM_SETTLE_TURNS ≤ MAX_HORIZON` — a computational
# horizon bound, not a behavioural restriction.


# ---------------------------------------------------------------------------
# Obs helpers
# ---------------------------------------------------------------------------


def _as_dict(obs):
    """Coerce an obs (Struct or dict) into a dict for consistent access."""
    if isinstance(obs, dict):
        return obs
    return {
        "player": getattr(obs, "player", 0),
        "step": getattr(obs, "step", 0),
        "planets": list(getattr(obs, "planets", []) or []),
        "fleets": list(getattr(obs, "fleets", []) or []),
        "comets": list(getattr(obs, "comets", []) or []),
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", []) or []),
        "angular_velocity": float(getattr(obs, "angular_velocity", 0.0)),
    }


def _num_seats(planets, fleets):
    """Detect 2P vs 4P from the obs."""
    max_owner = -1
    for p in planets:
        if int(p.owner) > max_owner:
            max_owner = int(p.owner)
    for f in fleets:
        if int(f.owner) > max_owner:
            max_owner = int(f.owner)
    return 4 if max_owner >= 2 else 2


# ---------------------------------------------------------------------------
# Geometry / timing primitives
# ---------------------------------------------------------------------------


def _aim_and_eta(src, tgt, ships, omega, wait_N=0):
    """Return (lead_aim_angle, integer_eta) for one candidate fleet.

    For ORBITING targets, `lib.aim.aim_orbiting` jointly solves the
    aim angle AND the arrival eta via fixed-point iteration.

    For wait-then-fire candidates (wait_N > 0): the fleet fires
    `wait_N` turns AFTER the current step. By then BOTH src and tgt
    have rotated by `omega * wait_N` around the center. We pre-rotate
    BOTH endpoints via `predict_relative` so aim_orbiting operates on
    the geometry that will hold at fire time (verified empirically on
    Felipe seed 1492346051: at step 4 with wait_N=17, shifting both
    matches the step-21 ground-truth aim to 0.001 rad). Rotating only
    the target — as the prior code did — gave a wildly wrong aim
    because it computed "from src at step 4 toward tgt at step 21,"
    which is not what the fleet would do when launched at step 21.

    Falls back to straight-aim + straight-eta for non-orbiting
    targets (static planets don't rotate; wait_N is a no-op for aim).
    """
    if _is_orbiting(list(tgt)):
        tgt_list = list(tgt)
        src_x, src_y = float(src.x), float(src.y)
        if wait_N > 0:
            # Rotate both endpoints to their position at fire time.
            # Co-rotating planets preserve their relative geometry,
            # so the angle returned by aim_orbiting from rotated_src
            # to rotated_tgt is the correct world-frame aim at step
            # (current_step + wait_N) — exactly what fast_sim will
            # use when it replays wait_N idle steps then fires.
            fx, fy = _orbit_predict_relative(tgt_list, omega, wait_N)
            tgt_list = list(tgt_list)
            tgt_list[2] = fx
            tgt_list[3] = fy
            src_x, src_y = _orbit_predict_relative(list(src), omega, wait_N)
        res = aim_orbiting(
            (src_x, src_y), src.radius, tgt_list, tgt.radius, ships, omega,
        )
        if res is not None:
            return float(res[0]), max(1, int(math.ceil(float(res[2]))))
    angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
    flight = max(
        0.0,
        math.hypot(src.x - tgt.x, src.y - tgt.y)
        - src.radius - tgt.radius - 0.1,
    )
    spd = fleet_speed(ships)
    if spd <= 0:
        return angle, 999
    return angle, int(math.ceil(flight / spd))


def _nearest_k(targets, src, k):
    return sorted(
        targets,
        key=lambda t: math.hypot(src.x - t.x, src.y - t.y),
    )[:k]


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def _capture_size(src, tgt, model, omega, me, world):
    """WorldModel-aware minimum capture size.

    For NON-MINE targets (capture): predicted defenders at eta + 1.
    For MINE targets (reinforce): the predicted SHORTFALL against the
      strongest incoming enemy fleet, i.e. enough ships to win the
      defense at the moment of conflict. If no incoming threat,
      returns 0 (no reinforce needed).

    One Newton-style iteration: initial size from current tgt garrison,
    use orbital-aware `_aim_and_eta` for arrival turn, query model for
    predicted defenders at that eta.
    """
    if int(tgt.owner) == me:
        # Reinforce: size to survive the predicted enemy threat. Uses
        # `time_to_enemy_threat` which covers both in-flight fleets AND
        # potential launches from stationary enemy planets at current
        # garrisons (Fix 2 of v9). Drop-in widening of the threat pool.
        enemy_eta = model.time_to_enemy_threat(int(tgt.id), me, world)
        if enemy_eta is None:
            return 0  # no near-term threat; reinforce unnecessary
        # Sum in-flight enemy ships landing at-or-before enemy_eta + 1.
        enemy_arrivals = model.ledger.get(int(tgt.id), [])
        enemy_ship_sum_inflight = sum(
            ships for (eta_arr, owner, ships) in enemy_arrivals
            if owner != me and eta_arr <= enemy_eta + 1
        )
        # If no in-flight threat (preemptive case), the threat is a
        # potential launch from a stationary enemy planet. Estimate the
        # threat magnitude as the nearest enemy planet's CURRENT garrison
        # (worst-case full send).
        enemy_potential = 0.0
        if enemy_ship_sum_inflight <= 0:
            tgt_x, tgt_y = float(tgt.x), float(tgt.y)
            best_enemy_ships = 0.0
            for p in world.planets_by_id.values():
                if int(p.owner) < 0 or int(p.owner) == me:
                    continue
                # Match the enemy's eta range — they could have launched
                # at most enemy_eta turns ago from any of their planets.
                if int(p.ships) > best_enemy_ships:
                    best_enemy_ships = float(p.ships)
            enemy_potential = best_enemy_ships
        enemy_strength = max(enemy_ship_sum_inflight, enemy_potential)
        # Predicted defender at enemy_eta (with production accrual).
        my_garrison_at_eta = float(tgt.ships) + float(tgt.production) * enemy_eta
        shortfall = enemy_strength - my_garrison_at_eta + 1
        return max(0, int(math.ceil(shortfall)))
    # Capture (non-mine target)
    initial = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
    _angle, eta = _aim_and_eta(src, tgt, initial, omega)
    pred = float(model.ships_at(int(tgt.id), eta) or 0.0)
    size = int(math.ceil(pred)) + 1
    return max(MIN_FLEET_SIZE, size)


def _enumerate_ship_counts_basic(src, tgt, model, omega, me, world):
    """Phase 1 ship-count set: capture/reinforce size, 2×, full budget.

    For reinforce (my own target), size 0 means no threat → skip.
    """
    cap = _capture_size(src, tgt, model, omega, me, world)
    budget = int(src.ships)
    if cap == 0:
        return []  # no threat; don't reinforce
    sizes = set()
    if MIN_FLEET_SIZE <= cap <= budget:
        sizes.add(cap)
    if 2 * cap <= budget:
        sizes.add(2 * cap)
    if budget >= MIN_FLEET_SIZE and budget > cap:
        sizes.add(budget)
    return sorted(sizes)


_WAIT_EXTRA_SURPLUS = (0, 5, 12)  # multi-wait grid: 3 variants per pair


def _wait_then_fire_candidate(src, tgt, model, omega, me):
    """Generate "wait N turns then fire" candidates for one (src, tgt)
    pair. Returns a list of (ships, wait_N, angle, eta) tuples — one
    per surplus target in `_WAIT_EXTRA_SURPLUS`. Empty list if no
    variant is applicable.

    v15: extended in two ways from the v10 mechanism:
    1. Multi-wait grid: instead of one wait_N (just enough to capture),
       generate variants with extra surplus = 0, 5, 12 — fleet sizes
       cap+surplus. Longer waits = more robust captures less prone to
       counter-recapture.
    2. Wait-N for feasible-now pairs too: previously skipped pairs
       where the source could fire now; now generates wait variants
       with extra surplus even for feasible pairs (the user's
       opening-game directive: "consider more actions, don't converge
       prematurely on something he could do").

    Common preconditions still hold:
    - tgt is not mine (reinforces are deadline-bound; can't wait)
    - src.production > 0 (otherwise can't accumulate)
    - wait_N ≥ 1 (wait_N=0 is covered by fire-now path)
    - wait_N + eta + SETTLE ≤ MAX_HORIZON (computational cap)

    For each surplus target s, compute:
    - target fleet = cap_after_wait + s (where cap_after_wait depends on wait_N)
    - wait_N = ceil((target_fleet - src.ships) / prod), bumped to 1 minimum
    - re-aim/eta at the post-wait arrival
    """
    if int(tgt.owner) == me:
        return []
    prod = int(src.production)
    if prod <= 0:
        return []

    # Initial estimate at fire-now eta (used to seed Newton iteration).
    initial = max(MIN_FLEET_SIZE, int(tgt.ships) + 1)
    _a0, eta0 = _aim_and_eta(src, tgt, initial, omega)
    pred_now = float(model.ships_at(int(tgt.id), eta0) or 0.0)
    cap_now = max(MIN_FLEET_SIZE, int(math.ceil(pred_now)) + 1)

    variants = []
    seen_wait_ships = set()  # dedup variants that collapse to the same (wait_N, ships)
    for extra_surplus in _WAIT_EXTRA_SURPLUS:
        # Target fleet size = capture-size + extra_surplus. Wait long
        # enough for src to accumulate that many ships from current
        # garrison + production.
        target_fleet = cap_now + extra_surplus
        shortfall = target_fleet - int(src.ships)
        if shortfall <= 0:
            # Feasible-now even with surplus → wait_N would be 0;
            # bump to 1 so this is distinct from fire-now.
            wait_N = 1
        else:
            wait_N = (shortfall + prod - 1) // prod  # ceil
        if wait_N < 1:
            continue

        # Newton step at post-wait arrival: a bigger fleet may arrive
        # faster (or slower), so cap_final may differ from cap_now.
        # Pass wait_N to _aim_and_eta so orbital lead accounts for
        # target rotation DURING the wait phase.
        ships_attempt = target_fleet
        angle, eta = _aim_and_eta(src, tgt, ships_attempt, omega, wait_N=wait_N)
        pred_at_arr = float(model.ships_at(int(tgt.id), wait_N + eta) or 0.0)
        cap_final = max(MIN_FLEET_SIZE, int(math.ceil(pred_at_arr)) + 1)
        # Final fleet honors extra surplus relative to the refined cap.
        final_fleet = cap_final + extra_surplus

        budget_at_wait = int(src.ships) + prod * wait_N
        if final_fleet > budget_at_wait:
            # Can't accumulate enough during this wait — clamp to the
            # budget and re-derive wait_N if needed.
            final_fleet = budget_at_wait

        if wait_N + eta + SIM_SETTLE_TURNS > MAX_HORIZON:
            continue

        key = (wait_N, final_fleet)
        if key in seen_wait_ships:
            continue
        seen_wait_ships.add(key)

        variants.append((final_fleet, wait_N, angle, eta))

    return variants


# ---------------------------------------------------------------------------
# v21 Patch E1 — cheap target-quality filter
# ---------------------------------------------------------------------------


def _local_force_factor(tgt, world, fleets, me, R=NEIGHBORHOOD_RADIUS):
    """Local force ratio = my_ships / (my_ships + opp_ships) within R of tgt.

    Sums planet garrisons and in-flight fleet ships within R game-units of
    tgt. The target planet itself is excluded (its own ships are not
    "defenders of the neighborhood"). Cheap approximation — full fleet
    destination forecasting is expensive and we already pay for it inside
    fast_sim downstream. This is for RANKING only.

    Returns float in [0, 1]. > 0.5 favors me; < 0.5 favors opp. If the
    neighborhood is empty, returns 1.0 (don't penalise opening grabs of
    empty zones — those are exactly where production lands first).
    """
    tx, ty = float(tgt.x), float(tgt.y)
    tgt_id = int(tgt.id)
    R2 = R * R
    my_ships = 0.0
    opp_ships = 0.0
    for p in world.planets_by_id.values():
        if int(p.id) == tgt_id:
            continue
        owner = int(p.owner)
        if owner < 0:
            continue
        dx = float(p.x) - tx
        dy = float(p.y) - ty
        if dx * dx + dy * dy > R2:
            continue
        if owner == me:
            my_ships += float(p.ships)
        else:
            opp_ships += float(p.ships)
    for f in fleets:
        owner = int(f.owner)
        if owner < 0:
            continue
        dx = float(f.x) - tx
        dy = float(f.y) - ty
        if dx * dx + dy * dy > R2:
            continue
        if owner == me:
            my_ships += float(f.ships)
        else:
            opp_ships += float(f.ships)
    total = my_ships + opp_ships
    if total <= 0.0:
        return 1.0
    return my_ships / total


def _target_quality(tgt, world, fleets, me):
    """Empirical-waste-driven target_quality in [0, 1].

    factor = comet_discount * min(1, local_force_ratio / UNDEFENSIBLE_GATE)

    - Comet targets: 0.0 (16/16 tracked comet shots MISSED in v20/v15
      replay sample, 2026-05-16). Multiplied through, this zeros the
      capture credit, dropping comet candidates below the -10 prerank
      floor cut.
    - Below UNDEFENSIBLE_GATE local force share, capture credit is
      linearly attenuated. At ratio 0 (deep opp territory) the factor is
      0; at the gate, factor is 1; above the gate, factor is capped at 1
      (capture in friendly neighborhoods isn't bonus-credited, just not
      penalised).
    """
    if not CAPTURE_HOLD_PREFILTER:
        return 1.0
    if int(tgt.id) in world.comet_ids:
        return 0.0
    ratio = _local_force_factor(tgt, world, fleets, me)
    return min(1.0, ratio / UNDEFENSIBLE_GATE)


# ---------------------------------------------------------------------------
# Cheap pre-rank (Stage 1 of two-stage scoring)
# ---------------------------------------------------------------------------


def _cheap_marginal_value(src, tgt, ships, eta, world, model, me,
                          wait_N=0, target_quality=1.0):
    """Approximate Δ value for ranking only — NOT the final score.

    Reads the BASELINE WorldModel (built once per turn) to predict
    pred_owner + pred_ships at our arrival eta.

    Three cases:
    - **CAPTURE** (pred_owner != me, ships > pred_ships): credit by
      capture_weight × production × time_remaining.
    - **BOUNCE** (pred_owner != me, ships ≤ pred_ships): penalty
      = −waste_weight × ships.
    - **REINFORCE** (pred_owner == me): if WorldModel.time_to_enemy_threat
      predicts an enemy could attack this planet within a relevant
      horizon (eta + 30), score as "value of preventing loss" =
      capture_weight × production × pv_horizon(threat_eta).
      Otherwise return 0 (no near-term threat → reinforce unnecessary).

    The reinforce branch is Fix 1 of the v9 iteration: previously
    returned 0 for all reinforce, which caused them to be cut from
    the validate stage by the adaptive wallclock cap in late game.
    Now reinforce candidates RANK competitive with captures so they
    reach fast_sim validation.

    Known weakness: `fleet_target_planet` does a non-orbital ray-cast,
    so for orbital captures the model's predicted state at our eta is
    off by 1-2 turns of orbital drift. This is acceptable for RANKING:
    relative ordering is mostly preserved. The fast_sim downstream is
    the ground truth for the FINAL decision.
    """
    # v10: arrival_step = wait_N + eta. For fire-now wait_N=0 → unchanged.
    arrival_step = wait_N + eta
    pred_owner = model.owner_at(int(tgt.id), arrival_step)
    pred_ships = float(model.ships_at(int(tgt.id), arrival_step) or 0.0)

    if pred_owner == me:
        # REINFORCE: score value of preventing loss of this planet.
        # (wait-N is not generated for reinforce targets, so this branch
        # only sees fire-now reinforce candidates.)
        t_to_threat = model.time_to_enemy_threat(int(tgt.id), me, world)
        if t_to_threat is None or t_to_threat > eta + 30:
            return 0.0  # no near-term threat → reinforce truly unnecessary
        # Loss-prevention credit: planet's pv-discounted production
        # stream from threat onward, scaled by capture_weight (0.05) to
        # match the offensive capture-credit scale. E1 factor applied:
        # reinforcing a doomed-area planet (low local force ratio) is
        # itself low-value defense.
        pv = pv_horizon(int(world.step), int(t_to_threat),
                        gamma=0.99, t_total=EPISODE_STEPS)
        return 0.05 * float(tgt.production) * float(pv) * target_quality

    if ships > pred_ships:
        # CAPTURE credit. PV-discounted production stream from arrival.
        # For wait-N: discount is γ^(wait_N+eta), matching the longer
        # delay before production begins. E1 factor applied: comet
        # targets get 0, isolated deep-opp captures get attenuated.
        pv = pv_horizon(int(world.step), int(arrival_step),
                        gamma=0.99, t_total=EPISODE_STEPS)
        return 0.05 * float(tgt.production) * float(pv) * target_quality
    # BOUNCE penalty (waste_weight=0.5). Bounces don't get the E1
    # multiplier — the penalty already accounts for the wasted ships.
    return -0.5 * float(ships)


# ---------------------------------------------------------------------------
# Favor (F1 + F2) — bootstrap's proven leaf scorer
# ---------------------------------------------------------------------------


def _favor(obs, me, num_seats=2):
    """F1 + F2 favor with PV-discount and 4P-aware opp aggregation.

    F1 = my_ships − opp_ships_agg (in-flight + planets).
    F2 = (my_prod − opp_prod_agg) × pv_horizon(step, 0, γ=0.99).

    PV-discount (Fix 3 of v9): linear `turns_remaining` over-weights
    far-future production. In late-game with opp prod-lead, F2 dominates
    F1 by 100× and the chooser stops valuing ship preservation. PV
    with γ=0.99 makes a unit production stream worth ~99 (vs 500), so
    F1 and F2 are on comparable scales.

    4P-aware opp aggregation (Fix 4 of v9): in 2P use max-of-opps
    (identical to "the only opp"); in 4P use SUM-of-opps so capturing
    from a weak opp gets 2× credit (my +prod AND their −prod),
    matching the credit for capturing from the leader. This corrects
    the systematic under-credit of non-leader captures that left v8
    passive in 4P.
    """
    planets = obs.planets if hasattr(obs, "planets") else obs.get("planets", [])
    fleets = obs.fleets if hasattr(obs, "fleets") else obs.get("fleets", [])
    step = obs.step if hasattr(obs, "step") else obs.get("step", 0)

    # Per-owner totals
    ships_by_owner = {}
    prod_by_owner = {}
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
    if num_seats <= 2:
        opp_ships = max(
            (v for k, v in ships_by_owner.items() if k != me),
            default=0.0,
        )
        opp_prod = max(
            (v for k, v in prod_by_owner.items() if k != me),
            default=0.0,
        )
    else:
        # 4P / 3P: sum across all opps. Capturing from any weakens the
        # collective; same credit as capturing from the leader.
        opp_ships = sum(v for k, v in ships_by_owner.items() if k != me)
        opp_prod = sum(v for k, v in prod_by_owner.items() if k != me)

    pv = pv_horizon(int(step), 0, gamma=0.99, t_total=EPISODE_STEPS)
    return (my_ships - opp_ships) + (my_prod - opp_prod) * pv


# ---------------------------------------------------------------------------
# Idle baseline + per-candidate score
# ---------------------------------------------------------------------------


def _opp_actions_for_snap(snap, me, num_seats):
    """Compute each non-me seat's lite_greedy action against the CURRENT
    snap. Used inline by both baseline and candidate rollouts so opp
    reacts to the evolving state (including my fleets and captures)
    rather than replaying a precomputed trajectory."""
    actions = [[] for _ in range(num_seats)]
    for opp_id in range(num_seats):
        if opp_id == me:
            continue
        try:
            actions[opp_id] = _opp_policy(snap.state[opp_id].observation) or []
        except Exception:
            actions[opp_id] = []
    return actions


def _opp_actions_for_snap_with_counter(snap, me, num_seats,
                                       counter_radius=NEIGHBORHOOD_RADIUS):
    """Patch E2 stronger opp: lite_greedy + counter-recapture.

    For each opp seat, after lite_greedy emits its offensive moves, we
    inspect any of MY in-flight fleets targeted at THAT opp's owned
    planets. If a nearby opp planet (within counter_radius) has surplus
    ≥ the incoming fleet, opp launches a defensive counter from that
    nearby planet toward its threatened planet. One counter per
    threatened planet per turn; sources are deduplicated so opp doesn't
    over-deploy from one planet.

    Used ONLY by `_capture_hold_check` (Patch E2). Keeps `lite_greedy_policy`
    in `lib/opp_model.py` unchanged so v7_X agents that share the symbol
    are not affected.
    """
    actions = [[] for _ in range(num_seats)]
    for opp_id in range(num_seats):
        if opp_id == me:
            continue
        try:
            obs = snap.state[opp_id].observation
            offensive = _opp_policy(obs) or []
            actions[opp_id] = list(offensive)
            # Collect planet/fleet info from opp's observation.
            obs_planets = obs.planets if hasattr(obs, "planets") else obs.get("planets", [])
            obs_fleets = obs.fleets if hasattr(obs, "fleets") else obs.get("fleets", [])
            # Opp's own planets, by id.
            opp_planets_by_id = {
                int(p[0]): p
                for p in obs_planets
                if int(p[1]) == opp_id
            }
            if not opp_planets_by_id:
                continue
            used_srcs_this_seat = {int(m[0]) for m in offensive}
            # MY in-flight fleets that could be targeting an opp planet.
            # We approximate "targeting opp planet X" by:
            #   for each my-fleet f, find the nearest opp-owned planet
            #   to f's current position within counter_radius — if any.
            # Counter-launch from the OPP planet nearest to the
            # threatened opp planet (within counter_radius) that has
            # surplus garrison.
            for f in obs_fleets:
                f_owner = int(f[1])
                if f_owner != me:
                    continue
                f_ships = float(f[6])
                fx = float(f[4]) if len(f) > 4 else 0.0
                fy = float(f[5]) if len(f) > 5 else 0.0
                # Nearest opp planet to fleet's current position.
                threatened_p = None
                best_d2 = (counter_radius * counter_radius)
                for p in opp_planets_by_id.values():
                    px, py = float(p[2]), float(p[3])
                    d2 = (fx - px) * (fx - px) + (fy - py) * (fy - py)
                    if d2 < best_d2:
                        best_d2 = d2
                        threatened_p = p
                if threatened_p is None:
                    continue
                # Find opp source with surplus >= f_ships, nearest to
                # the threatened planet.
                tpx, tpy = float(threatened_p[2]), float(threatened_p[3])
                best_src = None
                best_src_d2 = float("inf")
                for sp in opp_planets_by_id.values():
                    src_id = int(sp[0])
                    if src_id == int(threatened_p[0]):
                        continue  # can't reinforce self
                    if src_id in used_srcs_this_seat:
                        continue
                    surplus = float(sp[5]) - 1.0
                    if surplus < f_ships:
                        continue
                    spx, spy = float(sp[2]), float(sp[3])
                    d2 = (spx - tpx) * (spx - tpx) + (spy - tpy) * (spy - tpy)
                    if d2 < best_src_d2:
                        best_src_d2 = d2
                        best_src = sp
                if best_src is None:
                    continue
                # Aim from best_src toward threatened_p (straight aim is
                # fine — this is approximate counter-defense; the
                # opponent in reality would orbital-aim, but for E2's
                # hold-check we only need a "would opp defend?" signal).
                angle = math.atan2(tpy - float(best_src[3]),
                                   tpx - float(best_src[2]))
                ships_to_send = max(MIN_FLEET_SIZE, int(math.ceil(f_ships)) + 1)
                actions[opp_id].append(
                    [int(best_src[0]), float(angle), int(ships_to_send)]
                )
                used_srcs_this_seat.add(int(best_src[0]))
        except Exception:
            # Any error in the counter logic — fall back to bare
            # lite_greedy (already populated above). Never raise out
            # of the rollout.
            pass
    return actions


def _capture_hold_check(snap_base, me, num_seats, committed_moves,
                        candidate_move, horizon=HOLD_CHECK_HORIZON,
                        tail_steps=HOLD_CHECK_TAIL_STEPS):
    """Patch E2 — rollout-based hold-probability for one candidate.

    Runs a single horizon-step rollout under counter-recapture opp,
    with all committed_moves + candidate_move firing at their wait_N.
    Returns hold_p in [0, 1]: fraction of the final `tail_steps` steps
    where we own the candidate's target planet.

    `candidate_move` is (src_id, angle, ships, wait_N, target_id).
    `committed_moves` is a list of (src_id, angle, ships, wait_N) (no target_id
    needed for the rollout itself).

    Cost: 1 fast_sim rollout of length `horizon` × n_seats. In late
    game ~150-200 ms; called per top candidate, so capped to
    HOLD_CHECK_TOP_N = 3 invocations.
    """
    src_id, angle, ships, wait_N, target_id = candidate_move
    all_moves = list(committed_moves) + [(src_id, angle, ships, wait_N)]
    # Pre-bucket as in _rollout_with_moves but record per-step ownership
    # of target_id over the tail.
    moves_at_step = {}
    for mv in all_moves:
        sid, ang, sh, wN = mv[0], mv[1], mv[2], (mv[3] if len(mv) > 3 else 0)
        moves_at_step.setdefault(int(wN), []).append(
            [int(sid), float(ang), int(sh)]
        )
    snap = fs_clone(snap_base)
    tail_owns = []
    last_steps_start = max(0, horizon - tail_steps)
    for step_i in range(horizon):
        if snap.fake_env.done:
            # Record final ownership for remaining tail steps.
            obs = snap.state[me].observation
            obs_planets = obs.planets if hasattr(obs, "planets") else obs.get("planets", [])
            owned = any(
                int(p[0]) == int(target_id) and int(p[1]) == me
                for p in obs_planets
            )
            remaining = horizon - step_i
            for _ in range(min(remaining, tail_steps)):
                tail_owns.append(owned)
            break
        actions = _opp_actions_for_snap_with_counter(snap, me, num_seats)
        my_moves = moves_at_step.get(step_i)
        if my_moves:
            actions[me] = list(my_moves)
        snap = fs_step(snap, actions, in_place=True)
        if step_i >= last_steps_start:
            obs = snap.state[me].observation
            obs_planets = obs.planets if hasattr(obs, "planets") else obs.get("planets", [])
            owned = any(
                int(p[0]) == int(target_id) and int(p[1]) == me
                for p in obs_planets
            )
            tail_owns.append(owned)
    if not tail_owns:
        return 0.0
    return sum(1.0 for o in tail_owns if o) / float(len(tail_owns))


def _build_idle_baseline(snap_base, me, num_seats, max_horizon):
    """Pre-compute favor at every horizon 0..max_horizon under me-idle.

    Opp acts REACTIVELY at each step via `_opp_policy` against the
    evolving snap. Lost CRN cancellation (vs precomputed opp_traj),
    gained: opp counter-attacks against my captures in candidate
    rollouts emerge naturally, so F2 over-credit on fragile captures
    is corrected at the leaf.
    """
    snap = fs_clone(snap_base)
    out = [_favor(snap.state[me].observation, me, num_seats)]
    for _step_i in range(max_horizon):
        if snap.fake_env.done:
            out.append(out[-1])
            continue
        actions = _opp_actions_for_snap(snap, me, num_seats)
        # me slot stays [] (idle baseline)
        snap = fs_step(snap, actions, in_place=True)
        out.append(_favor(snap.state[me].observation, me, num_seats))
    return out


def _rollout_with_moves(snap_base, me, num_seats, moves, horizon):
    """v21 helper for Patch A — fast_sim rollout that fires a SET of my
    moves at their respective wait_N, with reactive opp at every step.
    Returns the final leaf-favor.

    `moves` is a list of dicts (or 4-tuples) with keys/positions
    `(src_id, angle, ships, wait_N)`. Multiple moves may share wait_N
    (fired simultaneously). The rollout runs for `horizon` steps; if
    `wait_N >= horizon` for some move, that move never fires (the
    caller is responsible for choosing a horizon that covers all
    committed wait_Ns).
    """
    # Pre-bucket moves by their wait_N for O(1) lookup per step.
    moves_at_step = {}
    for mv in moves:
        if isinstance(mv, dict):
            sid, ang, sh, wN = mv["src_id"], mv["angle"], mv["ships"], mv.get("wait_N", 0)
        else:
            sid, ang, sh, wN = mv[0], mv[1], mv[2], (mv[3] if len(mv) > 3 else 0)
        moves_at_step.setdefault(int(wN), []).append(
            [int(sid), float(ang), int(sh)]
        )
    snap = fs_clone(snap_base)
    for step_i in range(horizon):
        if snap.fake_env.done:
            break
        actions = _opp_actions_for_snap(snap, me, num_seats)
        my_moves = moves_at_step.get(step_i)
        if my_moves:
            actions[me] = list(my_moves)
        snap = fs_step(snap, actions, in_place=True)
    return _favor(snap.state[me].observation, me, num_seats)


def _score_action(snap_base, me, num_seats, src_id, angle, ships,
                  horizon, baseline_favors, wait_N=0):
    """Δ favor at horizon = leaf(me_action @ wait_N + reactive opp) − baseline.

    Opp acts reactively at each step (lite_greedy on opp's evolving obs),
    so my captured planets DO trigger opp counter-launches in the rollout
    — which collapses F2's over-credit on fragile (low-garrison) captures.
    """
    snap = fs_clone(snap_base)
    for step_i in range(horizon):
        if snap.fake_env.done:
            break
        actions = _opp_actions_for_snap(snap, me, num_seats)
        if step_i == int(wait_N):
            actions[me] = [[int(src_id), float(angle), int(ships)]]
        # else: actions[me] stays []
        snap = fs_step(snap, actions, in_place=True)
    leaf_favor = _favor(snap.state[me].observation, me, num_seats)
    return leaf_favor - baseline_favors[horizon]


# ---------------------------------------------------------------------------
# Public agent
# ---------------------------------------------------------------------------


def agent(obs, configuration=None):
    _reset_counters()
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    raw_planets = obs_d.get("planets", []) or []
    raw_fleets = obs_d.get("fleets", []) or []
    if not raw_planets:
        return []

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if int(p.owner) == me]
    other_planets = [p for p in planets if int(p.owner) != me]
    if not my_planets or not other_planets:
        return []

    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0))
    num_seats = _num_seats(planets, fleets)

    # Identify threatened MY planets via WorldModel.time_to_enemy_threat,
    # which considers BOTH (a) in-flight enemy fleets AND (b) potential
    # launches from stationary enemy planets at current garrison sizes.
    # The previous version used `incoming_enemy_eta` (only in-flight)
    # and missed preemptive threats from large enemy garrisons that
    # could launch any turn — the dominant failure mode in the Naoism
    # 2P loss (turn 70-95 attrition). Fix 2 of v9 iteration.
    threatened_mine = [
        p for p in my_planets
        if model.time_to_enemy_threat(int(p.id), me, world) is not None
    ]
    # Target pool = capture targets + defensive reinforce targets
    target_pool = other_planets + threatened_mine

    # Patch E1: precompute target_quality per target planet (one O(planets +
    # fleets) pass per target, cached for the inner enumeration loops).
    # Comet ids zero out their entry; everything else gets a local-force
    # factor in [0, 1]. Looked up once per (src, tgt) candidate below.
    target_quality_by_id = {
        int(t.id): _target_quality(t, world, fleets, me)
        for t in target_pool
    }

    # Build the fast_sim snapshot once per turn (~1 ms).
    snap_base = fs_from_obs(obs, num_seats=num_seats)

    # Probe fast_sim per-step cost for THIS board state — used to bound
    # how many candidates we can afford to validate inside the wallclock
    # budget. Per-step cost varies with the number of in-flight fleets
    # (mid-late game can be 5-15× the empty-board cost). One step + a
    # clock measurement, ~1-3ms total.
    t_probe = time.perf_counter()
    probe_snap = fs_clone(snap_base)
    probe_snap = fs_step(probe_snap, [[] for _ in range(num_seats)],
                         in_place=True)
    per_step_ms = max(0.05, (time.perf_counter() - t_probe) * 1000.0)
    # Expected per-candidate cost: K steps × per_step_ms × safety
    # factor. Use the AVERAGE expected K (MIN_HORIZON + a few) as the
    # estimate; outliers get caught by the post-loop deadline guard.
    avg_K = (MIN_HORIZON + MAX_HORIZON) / 2.0
    per_cand_ms = per_step_ms * avg_K * _PER_CANDIDATE_SAFETY
    # How many candidates fit inside the budget AFTER reserving overhead?
    wallclock_ms = _effective_wallclock_ms()
    budget_for_validate = wallclock_ms - _RESERVED_OVERHEAD_MS
    n_affordable = max(8, int(budget_for_validate / per_cand_ms))

    # v13: reactive opp inside every rollout. lite_greedy is recomputed
    # against the evolving snap at each step, so my captured planets
    # trigger opp counter-launches in the rollout — collapsing F2's
    # over-credit on fragile (low-garrison) captures. Drops the
    # precomputed opp_traj and common-random-numbers cancellation;
    # accepts more Δ variance for realistic counter-attacks.
    baseline_favors = _build_idle_baseline(
        snap_base, me, num_seats, MAX_HORIZON,
    )

    # ---------------------------------------------------------------
    # Stage 1: cheap pre-rank via analytic marginal_value (~0.1ms each).
    # Enumerate every (src, tgt, ships) candidate, rank by approximate Δ.
    # Also append ONE "wait-then-fire" candidate per (src, tgt) where
    # capture is INFEASIBLE-NOW but feasible inside MAX_HORIZON.
    # ---------------------------------------------------------------
    # Prerank tuple: (cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N).
    # wait_N=0 means fire-now; wait_N>0 means idle-then-fire.
    prerank = []
    n_enumerated = 0
    n_filtered_by_prefilter = 0
    n_comet_targets_filtered = 0
    for src in my_planets:
        if int(src.ships) < MIN_FLEET_SIZE:
            continue
        for tgt in _nearest_k(target_pool, src, NUM_TARGETS_PER_SOURCE):
            if int(tgt.id) == int(src.id):
                continue
            tq = target_quality_by_id.get(int(tgt.id), 1.0)
            tgt_is_comet = int(tgt.id) in world.comet_ids
            # Fire-now candidates.
            for ships in _enumerate_ship_counts_basic(src, tgt, model, omega, me, world):
                if ships < MIN_FLEET_SIZE or ships > int(src.ships):
                    continue
                n_enumerated += 1
                angle, eta = _aim_and_eta(src, tgt, ships, omega)
                horizon = max(eta + SIM_SETTLE_TURNS, MIN_HORIZON)
                if horizon >= len(baseline_favors):
                    horizon = len(baseline_favors) - 1
                cheap = _cheap_marginal_value(
                    src, tgt, ships, eta, world, model, me, wait_N=0,
                    target_quality=tq,
                )
                if cheap > -10.0:
                    prerank.append(
                        (cheap, src, tgt, ships, angle, eta, horizon, 0)
                    )
                else:
                    n_filtered_by_prefilter += 1
                    if tgt_is_comet:
                        n_comet_targets_filtered += 1
            # v15 wait-then-fire candidates (multi-wait grid per pair,
            # including feasible-now pairs — see _wait_then_fire_candidate).
            for w_ships, w_wait_N, w_angle, w_eta in _wait_then_fire_candidate(
                src, tgt, model, omega, me,
            ):
                w_horizon = max(w_wait_N + w_eta + SIM_SETTLE_TURNS, MIN_HORIZON)
                if w_horizon >= len(baseline_favors):
                    continue
                n_enumerated += 1
                w_cheap = _cheap_marginal_value(
                    src, tgt, w_ships, w_eta, world, model, me,
                    wait_N=w_wait_N, target_quality=tq,
                )
                if w_cheap > -10.0:
                    prerank.append(
                        (w_cheap, src, tgt, w_ships, w_angle, w_eta,
                         w_horizon, w_wait_N)
                    )
                else:
                    n_filtered_by_prefilter += 1
                    if tgt_is_comet:
                        n_comet_targets_filtered += 1

    _INSTRUMENT_COUNTERS["last_n_candidates"] = n_enumerated
    _INSTRUMENT_COUNTERS["last_n_filtered_by_prefilter"] = n_filtered_by_prefilter
    _INSTRUMENT_COUNTERS["last_n_comet_targets_filtered"] = n_comet_targets_filtered

    if not prerank:
        return []

    # Stage 2: per-(src, tgt, wait_band) deduplication. v15 (option 3):
    # buckets wait_N into bands so multiple wait variants per pair
    # survive into validation. The previous per-(src, tgt) dedup
    # collapsed every wait variant to wait_min via cheap-Δ ranking
    # (cheap-Δ is strictly decreasing in wait_N for the same target),
    # so the chooser never validated "wait longer for a more robust
    # capture against the same target". With banded dedup, the cheap
    # rank still picks the BEST within each band, but the rollout
    # validator gets to compare fire-now vs short-wait vs long-wait.
    #
    # Bands (chosen to match _WAIT_EXTRA_SURPLUS = (0, 5, 12)):
    #   band 0: wait_N == 0  (fire-now)
    #   band 1: 1..7        (short wait — extra_surplus≈5 territory)
    #   band 2: >= 8        (long wait — extra_surplus≈12 territory)
    def _wait_band(w):
        if w == 0:
            return 0
        return 1 if w <= 7 else 2

    best_per_pair = {}  # (src_id, tgt_id, wait_band) → entry
    for entry in prerank:
        cheap, src, tgt, _ships, _angle, _eta, _horizon, wait_N = entry
        key = (int(src.id), int(tgt.id), _wait_band(int(wait_N)))
        prev = best_per_pair.get(key)
        if prev is None or cheap > prev[0]:
            best_per_pair[key] = entry
    deduped = list(best_per_pair.values())

    # Stage 3: validate the top candidates via fast_sim K-step rollout.
    deduped.sort(key=lambda e: -e[0])
    effective_cap = min(N_VALIDATE, n_affordable)
    top = deduped[:effective_cap]

    t_deadline = time.perf_counter() + wallclock_ms / 1000.0
    candidates = []  # validated (delta, src, tgt, ships, angle, wait_N)
    for _cheap, src, tgt, ships, angle, _eta, horizon, wait_N in top:
        if time.perf_counter() > t_deadline:
            break
        delta = _score_action(
            snap_base, me, num_seats,
            int(src.id), angle, ships,
            horizon, baseline_favors, wait_N=wait_N,
        )
        if delta > 0:
            candidates.append((delta, src, tgt, ships, angle, wait_N))

    if not candidates:
        _INSTRUMENT_COUNTERS["last_n_validated"] = 0
        return []

    candidates.sort(key=lambda c: -c[0])
    _INSTRUMENT_COUNTERS["last_n_validated"] = len(candidates)

    # ------------------------------------------------------------------
    # Patch A — joint-commitment emit with greedy re-scoring.
    # ------------------------------------------------------------------
    # Commit one candidate at a time. After each commit, rebuild the
    # baseline as a fresh rollout that fires ALL committed moves at
    # their wait_N. Re-score the next RESCORE_TOP_K candidates against
    # the new baseline (joint_Δ); drop those <= 0. Pick the new top.
    # Stop on: empty remaining pool, no positive joint_Δ, deadline
    # within one rollout, or MAX_COMMIT_ROUNDS cap.
    #
    # When JOINT_RESCORE=False the loop degenerates to v20-classic emit:
    # commit every positive-Δ candidate (respecting max-1-per-source)
    # without re-scoring.

    # `pool` entries: (joint_delta, src, tgt, ships, angle, wait_N)
    pool = list(candidates)
    used_srcs = set()
    committed = []  # list of (joint_delta, src, tgt, ships, angle, wait_N)
    n_rescore_rounds = 0

    def _filter_pool(p, used):
        return [c for c in p if int(c[1].id) not in used]

    pool = _filter_pool(pool, used_srcs)

    while pool:
        if len(committed) >= MAX_COMMIT_ROUNDS + 1:
            break
        best = pool[0]
        if best[0] <= 0.0:
            break
        committed.append(best)
        used_srcs.add(int(best[1].id))
        pool = _filter_pool(pool[1:], used_srcs)

        if not JOINT_RESCORE or not pool:
            # v20-classic path: keep walking candidates in original Δ
            # order, committing each unique-source positive candidate.
            for c in pool:
                if c[0] <= 0.0:
                    break
                committed.append(c)
                used_srcs.add(int(c[1].id))
            break

        # Wallclock guard: rebuilding the committed baseline + re-scoring
        # K candidates is ~ (K+1) rollouts. If less time than that
        # remains, stop committing and emit what we have.
        rounds_remaining_ms = (t_deadline - time.perf_counter()) * 1000.0
        if rounds_remaining_ms < (RESCORE_TOP_K + 1) * per_cand_ms:
            break

        # Build the new committed-baseline. Horizon = max of committed
        # horizons + a small settle window, capped at MAX_HORIZON.
        max_committed_horizon = MAX_HORIZON  # use full horizon
        committed_moves_for_rollout = [
            (int(c[1].id), float(c[4]), int(c[3]), int(c[5]))
            for c in committed
        ]
        leaf_committed = _rollout_with_moves(
            snap_base, me, num_seats,
            committed_moves_for_rollout, max_committed_horizon,
        )

        # Re-score next RESCORE_TOP_K candidates against new baseline.
        # joint_Δ for cand = leaf(committed + cand) − leaf(committed).
        next_top = pool[:RESCORE_TOP_K]
        rescored = []
        for c in next_top:
            cand_moves = committed_moves_for_rollout + [
                (int(c[1].id), float(c[4]), int(c[3]), int(c[5]))
            ]
            cand_horizon = max_committed_horizon
            leaf_with = _rollout_with_moves(
                snap_base, me, num_seats, cand_moves, cand_horizon,
            )
            joint_delta = leaf_with - leaf_committed
            rescored.append((joint_delta, c[1], c[2], c[3], c[4], c[5]))
        # Replace re-scored entries in pool, re-sort.
        pool = rescored + pool[RESCORE_TOP_K:]
        pool.sort(key=lambda c: -c[0])
        n_rescore_rounds += 1

    _INSTRUMENT_COUNTERS["last_n_rescore_rounds"] = n_rescore_rounds
    _INSTRUMENT_COUNTERS["last_n_committed"] = len(committed)

    # ------------------------------------------------------------------
    # Patch E2 — hold-check post-filter on committed candidates.
    # ------------------------------------------------------------------
    # For the top HOLD_CHECK_TOP_N committed candidates (by joint_Δ),
    # run a separate mini-rollout under counter-recapture opp; multiply
    # joint_Δ by hold_p; drop any candidate whose product is below
    # HOLD_CHECK_MIN_SCORE.
    n_filtered_by_hold_check = 0
    if CAPTURE_HOLD_CHECK and committed:
        deadline_ms = (t_deadline - time.perf_counter()) * 1000.0
        if deadline_ms >= HOLD_CHECK_DEADLINE_MS:
            # Sort committed by joint_Δ desc; only check the top N.
            committed.sort(key=lambda c: -c[0])
            survivors = []
            for idx, c in enumerate(committed):
                if idx >= HOLD_CHECK_TOP_N:
                    survivors.append(c)
                    continue
                # Run hold-check for this candidate, given the OTHER
                # committed candidates are in flight too.
                others = [committed[j] for j in range(len(committed)) if j != idx]
                committed_moves_excl = [
                    (int(o[1].id), float(o[4]), int(o[3]), int(o[5]))
                    for o in others
                ]
                cand_move = (
                    int(c[1].id), float(c[4]), int(c[3]), int(c[5]),
                    int(c[2].id),
                )
                hold_p = _capture_hold_check(
                    snap_base, me, num_seats,
                    committed_moves_excl, cand_move,
                )
                score = c[0] * hold_p
                if score >= HOLD_CHECK_MIN_SCORE:
                    survivors.append(c)
                else:
                    n_filtered_by_hold_check += 1
            committed = survivors
    _INSTRUMENT_COUNTERS["last_n_filtered_by_hold_check"] = n_filtered_by_hold_check

    # ------------------------------------------------------------------
    # Emit. Max 1 per source (already enforced by used_srcs above).
    # wait_N == 0 → fire now; wait_N > 0 → reserve source, no emit.
    # ------------------------------------------------------------------
    moves = []
    for _delta, src, _tgt, ships, angle, wait_N in committed:
        if wait_N == 0:
            moves.append([int(src.id), float(angle), int(ships)])
        # else: wait-N → reserve src, emit nothing this turn.
    return moves
