"""Phase 5C: outcome-table-aware LP for post-opening turns.

The Phase 4 LP scored each candidate column with a scalar value (W1/W2
lower bounds in isolation). This makes the LP unable to express the
GLOBAL property "if I commit ships to offense here, am I leaving home
undefended?" — the root cause of the steps 70-100 collapse documented
in audit/2026-05-20-phase5b-root-cause-analysis.md.

This module replaces per-candidate scoring with per-planet-subset
scoring. For each planet `p` with candidate arrivals targeting it,
we enumerate all 2^k subsets via Phase 1's outcome_table, compute the
production stream per owner for each subset, and let the LP pick
EXACTLY ONE subset per planet via auxiliary binary variables y_{p,S}.

The objective rewards `prod_stream_me(p, S) - α · prod_stream_opp(p, S)`,
so DEFENSE EMERGES FROM THE MATH: a planet under heavy opp threat has
large `prod_stream_opp(empty)` (opp captures it and produces); firing
defenders shifts the subset choice to one where `prod_stream_opp` is
smaller, raising objective value. No separate defensive-value
multiplier or W2 mid-bound hack.

Per-source ship budget over time is preserved from Phase 4 (still a
linear constraint on x variables).

MILP via scipy.optimize.milp (HiGHS); greedy fallback if unavailable
or infeasible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    _MILP_AVAILABLE = True
except ImportError:
    _MILP_AVAILABLE = False
    milp = None  # type: ignore[assignment]
    LinearConstraint = None  # type: ignore[assignment]
    Bounds = None  # type: ignore[assignment]

from lib.joint_solver.columns import Column
from lib.joint_solver.outcome_table import MAX_ENUMERATION_BITS, Arrival, OutcomeRow, enumerate_outcomes
from lib.joint_solver.predicate import (
    is_winning_state,
    is_winning_state_if_lost,
    is_winning_state_if_owned,
    remaining_turns,
)
from lib.mirror import detect_num_players


# ---------------------------------------------------------------------------
# Constants (initial values; tunable via introspect)
# ---------------------------------------------------------------------------

# Game horizon. PRINCIPLED VALUE: 500. The Orbit Wars episode ends at step 500
# (per the env: `lib/fast_sim.DEFAULT_CONFIG['episodeSteps']`); production
# accrued through that tick is real game-end ship count. Earlier values
# (T_END=200) were arbitrary clipping to keep "forecasts speculative" —
# the right answer is to model the actual game.
T_END = 500
# Weight on opp production in the objective.
# PRINCIPLED VALUE: 1.0. The game's win condition is `my_ships - opp_ships`
# at T_END. Opp's accumulated production directly subtracts from our margin.
# Anything less than 1.0 is arbitrary.
ALPHA_OPP_PENALTY = 1.0
# Ship cost coefficient in the objective. Phase 5G (2026-05-20): bumped
# from 0.01 to 1.0 after the critique diagnosis: enumerate_ship_counts
# in agents/baseline/proposer.py emits 3 ship-count variants per (src,
# tgt) — [capture_size, 2×capture_size, full_budget] — all with
# IDENTICAL cheap_delta (per-candidate value doesn't depend on ship
# count for a successful capture) AND identical outcome_table value
# (same prod_stream regardless of ships). At SHIP_COST=0.01, the
# penalty (42 ships × 0.01 = 0.42) is dwarfed by value (~180), and
# the LP picks the LARGEST variant by tie-break. Result: each source
# drains in ONE fire then is idle for many turns. SHIP_COST=1.0
# makes the per-ship penalty meaningful: 42-ship variant costs 42 vs
# 9-ship variant costs 9, breaking the tie toward efficient launches.
SHIP_COST = 1.0
MAX_CONTESTERS_PER_PLANET = MAX_ENUMERATION_BITS  # 2^6 = 64 subsets per planet
TIME_LIMIT_SECONDS = 0.3          # MILP wallclock cap
# PRINCIPLED VALUE: 0. The LP's objective already penalizes losing a planet
# (opp's prod_stream gets credit when ownership flips); over-draining a
# source is naturally bad math, no need for an arbitrary reservation.
DEFENDER_GUARD = 0

# Phase 4 Step 2 (lighthouse plan): source-aware ship cost. A ship sent
# from a planet under enemy threat carries DEFENSIVE VALUE at the source
# — its removal costs more than 1 unit per ship. The LP previously
# priced ships uniformly at `SHIP_COST * col.ships`, blind to the
# defensive cost of stripping a threatened source. Threatened sources
# now pay `SHIP_COST_THREAT_MULT * SHIP_COST * col.ships`. Per Rule 40,
# this prices something the LP currently fails to see — not a cap or
# threshold. Initial value 2.0 is conservative (lighthouse plan suggested
# 3-5); calibrate via introspect. Setting to 1.0 reverts to uniform
# pricing (no-op).
#
# "Threatened" includes BOTH (a) in-flight enemy fleets inbound to the
# source AND (b) potential launches from close opp planets (PI directive
# 2026-05-21: "indirect fleet over close opponent planets"). The latter
# is the same metric `WorldModel.time_to_enemy_threat` computes; we
# filter it by `SHIP_COST_THREAT_ETA_THRESHOLD` so a DISTANT potential
# launch (e.g., opp planet across the board, eta=50) doesn't fire the
# multiplier — the defensive value of source ships is sharply elevated
# only when the threat is IMMINENT, not when it's purely geometric.
SHIP_COST_THREAT_MULT = 2.0
# Cutoff (turns) below which a `time_to_enemy_threat` reading counts as
# "imminent." Empirically: cross-board flights at fleet_speed ~3 take
# ~50 turns; half-board ~25; so 30 is a "close half of the map" cutoff.
# Tune via introspect if the multiplier under-fires (raise) or
# over-fires (lower).
SHIP_COST_THREAT_ETA_THRESHOLD = 30

# Phase 4 Step 1 (lighthouse plan): endgame predicate term in the objective.
# When a subset captures a planet whose acquisition tips
# `is_winning_state` from False→True, award +LAMBDA_ENDGAME. When a subset
# loses an own planet whose loss flips us out of winning state, apply
# −LAMBDA_ENDGAME. Per-(planet, subset) bonus — the joint state is
# approximated as the SUM of per-planet contributions; the predicate is
# monotone in ownership, so this approximation is conservative (no false
# positives by construction). Calibration knob: typical prod_stream
# magnitudes are in the hundreds (prod=2 × horizon=200 ≈ 400); λ=1000
# makes predicate flips dominate marginal production differences. Tune
# down to 100 if introspect shows the bonus over-firing.
LAMBDA_ENDGAME = 1000.0


# Phase α (composed-noodling-riddle plan): smooth-ΔW endgame replacement.
# The step `_endgame_bonus` returns ±LAMBDA_ENDGAME or 0, dominating the
# objective and crowding out finer-grained value signals (topology
# features fired but did not move LP argmax — see audit/2026-05-23/
# phase-beta-result-and-next.md). Replacement: λ_W · ΔW(p, S), where
# ΔW is the per-(planet, subset) contribution to
# `winning_margin = prod_advantage × remaining_turns − opp_pool`.
# Smooth across captures, signed (rewards both captures and prevented
# losses), magnitude-proportional to planet importance.
#
# Calibration default (from audit/2026-05-23/calibrate_W_results.json,
# r(W, focal_reward)=0.545 — marginal; the Plan agent's pressure-test
# recommended starting conservative): LAMBDA_W = 0.3. Per-planet ΔW
# magnitudes empirically run 600-5000 for a meaningful neutral/opp
# capture, so λ_W=0.3 puts the smooth contribution at ~200-1500 per
# planet — comparable to topology lambdas (50-300) and prod_stream
# (200-800), so it influences argmax without dominating.
LAMBDA_W_DEFAULT = 0.3


def _smooth_delta_w_enabled() -> bool:
    """Read env at call time (lazy) — Phase β taught us setdefault loses
    races against the inlined bundle's module-init order."""
    return _os.environ.get("LP_SMOOTH_DELTA_W", "0") == "1"


def _lambda_w() -> float:
    raw = _os.environ.get("LP_LAMBDA_W", "")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return LAMBDA_W_DEFAULT


# ---------------------------------------------------------------------------
# Phase ζ.v2 — Hold-aware prod_stream (root-cause fix for analytical's
# scale-invariant leaf value).
#
# The LP's `prod_stream_me = production × (HORIZON − arrival)` is
# scale-invariant in ship count for a successful capture, so under
# SHIP_COST > 0 the LP picks the smallest viable variant — chronic
# undercommitment in the opening (orbitfix fires 10, we fire 6).
#
# Fix: inject a SYNTHETIC opp-counter arrival into each per-planet
# `fixed_arrivals` list. The existing `_simulate_one` per-tick
# `resolve_arrivals` then naturally distinguishes subsets by their
# residual garrison: bigger ships_fired → bigger residual → survives
# opp counter → keeps accruing prod_stream. No coefficient hack;
# the math becomes correct.
#
# Default OFF; opt-in via `LP_HOLD_AWARE=1`.
def _hold_aware_enabled() -> bool:
    """Read env at call time (lazy) — same pattern as smooth_delta_w
    and topology features. Default OFF."""
    return _os.environ.get("LP_HOLD_AWARE", "0") == "1"


def _predict_opp_counter(planet_id: int, world, my_id: int
                         ) -> tuple[int | None, int]:
    """Predict the opp counter-fire to `planet_id` if we capture it.

    Returns `(counter_eta, counter_ships)`:
      - `counter_eta`: ticks-from-now until earliest opp arrival, or
        `None` if no opp source can plausibly counter (no opps, all
        out-of-range, etc.).
      - `counter_ships`: ship count of the opp source that achieves
        `counter_eta`. 0 when `counter_eta is None`.

    Walks `world.planets_by_id` once. The closest opp source by
    flight time (using `fleet_speed(opp.ships)`) is the assumed
    counter; bigger opp garrisons fly faster, naturally weighting
    toward the strongest threat.

    Tie-break by ship count descending (pick the strongest source
    among same-ETA sources — conservative worst-case for our hold).
    """
    target = world.planets_by_id.get(int(planet_id))
    if target is None:
        return None, 0
    tx, ty = float(target.x), float(target.y)

    best_eta: int | None = None
    best_ships: int = 0
    for p in world.planets_by_id.values():
        if int(p.id) == int(planet_id):
            continue
        if int(p.owner) == int(my_id) or int(p.owner) == -1:
            continue
        if int(p.ships) <= 0:
            continue
        from lib.fleet import speed as _fleet_speed
        dx = tx - float(p.x)
        dy = ty - float(p.y)
        dist = (dx * dx + dy * dy) ** 0.5
        v = _fleet_speed(int(p.ships))
        if v <= 0:
            continue
        eta = int(-(-dist // v))  # ceil(dist/v) without math import
        if best_eta is None or eta < best_eta or (
            eta == best_eta and int(p.ships) > best_ships
        ):
            best_eta = eta
            best_ships = int(p.ships)
    return best_eta, best_ships


# Level 1 — per-planet topology features (PI directive 2026-05-21: "we
# need joint optimization that considers topology"). Three closed-form
# per-planet bonuses added to the leaf value, awarded when row.owner_T
# == my_id (so the LP only credits planets we'd own post-subset).
# Computed ONCE per turn from `lib.geo.sense.sense_state(world, model)`;
# cached per planet, looked up per (planet, subset).
#
# Each prefactor scaled so the topology contribution is a meaningful
# fraction of `prod_stream_me` (typical magnitude ~200-800 per planet
# at T_END=500) — not dominant. Calibration knobs: bump LAMBDA_REACH up
# if introspect shows the LP still preferring isolated captures; bump
# LAMBDA_FRONT up if it captures frontier planets it can't hold.
#
# Each feature is gated by an env var (default ON when the bundle sets
# LP_TOPOLOGY_FEATURES=1; disabled cleanly by setting the corresponding
# LP_*_BONUS=0). LP_TOPOLOGY_FEATURES=0 disables all three for clean
# pre-fix / post-fix A/B comparison.
import os as _os


# Lazy evaluation — read env on every call so bundled-agent setdefault
# (which runs AFTER the lib code is inlined) takes effect. Module-level
# cached form locked these to False in the bundle because the inlined
# lp_outcome.py section evaluated them BEFORE the agent's
# `os.environ.setdefault("LP_TOPOLOGY_FEATURES", "1")` ran.
def _topology_features_enabled() -> bool:
    return _os.environ.get("LP_TOPOLOGY_FEATURES", "0") == "1"


def _reach_bonus_enabled() -> bool:
    return _topology_features_enabled() and (
        _os.environ.get("LP_REACH_BONUS", "1") == "1"
    )


def _defense_bonus_enabled() -> bool:
    return _topology_features_enabled() and (
        _os.environ.get("LP_DEFENSE_BONUS", "1") == "1"
    )


def _front_penalty_enabled() -> bool:
    return _topology_features_enabled() and (
        _os.environ.get("LP_FRONT_PENALTY", "1") == "1"
    )

# Term 1: reachability_bonus(p) = Σ_{q ∈ neutrals reachable from p} prod(q)/(1+eta(p→q))
LAMBDA_REACH = 50.0
REACH_HORIZON = 30        # ticks; neutrals farther than this contribute 0

# Term 2: mutual_defense_bonus(p) = count of my OTHER planets within DEFENSE_HORIZON
LAMBDA_DEFENSE = 10.0
DEFENSE_HORIZON = 12

# Term 3: recapture_risk(p) = prod(p)/hold_time_after_capture when opp can counter
LAMBDA_FRONT = 30.0
RECAPTURE_HORIZON = 25    # ticks; threats farther than this contribute 0


# ---------------------------------------------------------------------------
# Opening tempo bias — Phase ε option 3 of
# /root/.claude/plans/do-it-thoroughly-consider-tingly-fox.md
#
# Mimics ladder-leader sub 52882014's heuristic (`BASELINE_NEUTRAL_BONUS=2.0`
# × `EARLY_EXTRA=1.5 for step<50`) in the LP's vocabulary: at step <
# OPENING_TEMPO_HORIZON, multiplicatively boost the my-prod contribution
# from subsets that capture currently-neutral planets for us.
#
# Why this and not "scale all me_prod": (a) we want to bias TOWARD
# action vs inaction (neutral captures are pure new value, vs already-
# owned planets which we'd accrue regardless), (b) the leader's bonus
# was specifically on the neutral target tier, not blanket scaling.
#
# Closed-form, no extra trajectory work. Gated OFF by default; opt in
# via LP_OPENING_TEMPO=1 for the small A/B before flipping.
OPENING_TEMPO_HORIZON = 50
OPENING_TEMPO_FACTOR = 1.5


def _opening_tempo_enabled() -> bool:
    """Re-read env var on every call so test fixtures and runtime
    monkeypatches take effect without an importlib.reload."""
    return _os.environ.get("LP_OPENING_TEMPO", "0") == "1"


# ---------------------------------------------------------------------------
# Pending-aware source budget — Phase ζ.1 fix (do-it-thoroughly plan).
#
# Bug surfaced by the seed-42 step-1..7 LP introspect: the LP's
# source-budget constraint reads `src.ships` directly from `world`, but
# `commit_persistent` may have already committed wait_N>0 fires from
# previous turns whose ships are reserved for an upcoming decant. When
# the LP picks a new wait_N=0 fire from the same source on the SAME
# turn a decant resolves, both fires command from the source, the env
# caps total emission, and one fire dies silently.
#
# Fix: deduct pending-fire ships from each source-budget RHS at the
# constraint level. For a pending fire with `fire_step == step_now + d`,
# its ships are unavailable for any LP fire with wait_N >= d (the LP
# fire would compete with the decanted fire at step step_now+d).
#
# Opt-in via LP_PENDING_AWARE_BUDGET=1 (default OFF for clean A/B).
# Flip default to ON after A/B clears.
def _pending_aware_budget_enabled() -> bool:
    return _os.environ.get("LP_PENDING_AWARE_BUDGET", "0") == "1"


def _fetch_pending_fires(my_id: int) -> list:
    """Return the list of ScheduledFires from the pending-schedule
    singleton for `my_id`. Lazy import to keep this module independent
    of the pipeline layer in unit tests."""
    try:
        from lib.pipeline.pending_schedule import get_default_pending
        return list(get_default_pending().get_pending(int(my_id)))
    except Exception:
        return []


def _pending_ships_consumed_by(sid: int, step_now: int, u: int,
                               pending_fires: list) -> int:
    """Sum ships from pending fires from source `sid` whose decant time
    (in ticks from now) is <= `u`. These fires reserve ships that the
    LP cannot also emit at any wait_N <= u from the same source.

    A fire with `fire_step == step_now + d` consumes ships at the d'th
    tick from now. Past-due fires (d < 0; prune_past should have caught
    them) are clamped to d=0 defensively.
    """
    total = 0
    for f in pending_fires:
        if int(getattr(f, "src_id", -1)) != int(sid):
            continue
        d = max(0, int(getattr(f, "fire_step", 0)) - int(step_now))
        if d <= int(u):
            total += int(getattr(f, "ships", 0))
    return total


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OutcomeAwareResult:
    """Output of solve_outcome_aware."""
    moves: list                          # [src_id, angle, ships] for wait_N==0 fires
    fired_columns: list[Column]          # all selected columns (any wait_N)
    objective: float                     # achieved objective value
    status: str                          # solver status string
    n_x_vars: int                        # number of candidate (x) variables
    n_y_vars: int                        # number of subset (y) variables
    n_constraints: int                   # number of constraint rows
    per_planet_chosen: dict[int, tuple[int, ...]] = field(default_factory=dict)
    per_planet_value: dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lp_outcome_source_inventory(columns: list[Column], world, *, my_id: int
                      ) -> dict[int, tuple[int, int]]:
    """For each our-source-id present in columns, return (initial_ships, production)."""
    out: dict[int, tuple[int, int]] = {}
    for col in columns:
        if int(col.owner) != int(my_id):
            continue
        sid = int(col.src_id)
        if sid in out:
            continue
        p = world.planets_by_id.get(sid)
        if p is None:
            continue
        out[sid] = (int(p.ships), int(p.production))
    return out


def _build_per_planet_arrivals(
    active_columns: list[Column],
    world,
    model,
    *,
    my_id: int,
    step_now: int,
    threat_cache: dict[int, tuple[int | None, int]] | None = None,
    opp_id: int | None = None,
) -> dict[int, tuple[list[Arrival], list[Arrival]]]:
    """For each planet that has ≥1 candidate column targeting it, build the
    (fixed_arrivals, candidate_arrivals) pair for outcome_table.

    Phase 5D: fixed arrivals come ONLY from `model.ledger[p.id]`. The
    upstream `_model_with_opp_projection` already merges opp's projected
    counter-launches into the ledger; reading them a second time from a
    separate `opp_arrivals` parameter (as Phase 5C did) double-counted
    every opp arrival.

    Candidate arrivals = active columns with tgt_id == p.id.

    If a planet has more than MAX_CONTESTERS_PER_PLANET candidates, keep
    the top ones by per-candidate value (in Column.value) as a pre-filter.
    Excess candidates are dropped from the LP entirely.
    """
    # Group columns by target.
    by_tgt: dict[int, list[Column]] = {}
    for col in active_columns:
        by_tgt.setdefault(int(col.tgt_id), []).append(col)

    # Bug #6: build the keep-set of column ids referenced as parents
    # by ANY column. The per-planet pre-filter would otherwise drop a
    # low-value parent and the linkage constraint at L495-518 would
    # force-zero every compound that pointed at it — silent action-space
    # shrinkage. Compute across all active columns (parents may live in
    # a different by_tgt group than their compounds).
    parent_keepset: set[int] = {
        int(getattr(c, "parent_column_id"))
        for c in active_columns
        if getattr(c, "parent_column_id", None) is not None
    }

    out: dict[int, tuple[list[Arrival], list[Arrival]]] = {}
    for tgt_pid, cols in by_tgt.items():
        # Cap at MAX_CONTESTERS_PER_PLANET via per-candidate value. The
        # keep-set is force-kept regardless of value rank.
        if len(cols) > MAX_CONTESTERS_PER_PLANET:
            forced = [c for c in cols if int(c.column_id) in parent_keepset]
            optional = [c for c in cols if int(c.column_id) not in parent_keepset]
            # Bug #7: secondary sort keys so prerank_passthrough's
            # uniform value=1.0 doesn't yield non-deterministic survivors.
            # Prefer higher ships, earlier launches (smaller wait_N),
            # smaller column_id (id-ascending for stable identity tie-break).
            optional.sort(
                key=lambda c: (
                    float(c.value),
                    int(c.ships),
                    -int(c.wait_N),
                    -int(c.column_id),
                ),
                reverse=True,
            )
            budget = max(0, MAX_CONTESTERS_PER_PLANET - len(forced))
            cols = forced + optional[:budget]

        # Fixed arrivals — ONLY from model.ledger (opp projections live there).
        fixed: list[Arrival] = []
        for eta_arr, owner, ships in model.ledger.get(int(tgt_pid), []):
            if int(ships) <= 0:
                continue
            fixed.append(Arrival(
                eta=int(eta_arr), owner=int(owner), ships=int(ships),
                column_id=None,
            ))

        # Phase ζ.v2 hold-aware injection: append a SYNTHETIC opp-counter
        # arrival to fixed_arrivals so the per-tick `_simulate_one` +
        # `resolve_arrivals` distinguishes subsets by post-capture
        # residual garrison. Bigger ship_fired → bigger residual →
        # survives opp counter → keeps accruing prod_stream → LP picks
        # bigger fires naturally.
        #
        # Skip if (a) hold-aware off (threat_cache=None), (b) no opp_id
        # (4P), (c) target is already mine (no capture, no counter),
        # (d) no plausible opp threat, (e) model.ledger already has an
        # opp arrival for this target (avoid double-count with
        # opp_greedy_roi's projection).
        if threat_cache is not None and opp_id is not None:
            tgt_planet = world.planets_by_id.get(int(tgt_pid))
            if (tgt_planet is not None
                    and int(tgt_planet.owner) != int(my_id)):
                ledger_has_opp = any(
                    int(o) == int(opp_id) for (_e, o, _s)
                    in model.ledger.get(int(tgt_pid), [])
                )
                if not ledger_has_opp:
                    counter = threat_cache.get(int(tgt_pid))
                    if counter is not None:
                        c_eta, c_ships = counter
                        if c_eta is not None and int(c_ships) > 0:
                            fixed.append(Arrival(
                                eta=int(c_eta), owner=int(opp_id),
                                ships=int(c_ships), column_id=None,
                            ))

        # Candidate arrivals from our columns. Total arrival tick from the
        # planner's NOW perspective = wait_N + eta (flight time).
        cands: list[Arrival] = []
        for col in cols:
            total_eta = int(col.wait_N) + int(col.eta)
            cands.append(Arrival(
                eta=total_eta, owner=int(my_id), ships=int(col.ships),
                column_id=int(col.column_id),
            ))
        out[int(tgt_pid)] = (fixed, cands)
    return out


def _derive_opp_id_2p(world, my_id: int) -> int | None:
    """For a 2P game, return the unique non-me seat. None for 4P (or if
    seat count can't be inferred).

    Phase 4 MVP: endgame predicate is 2P-only. 4P falls through to no
    bonus — `is_winning_state` aggregates per-opp and the right
    formulation across 3 opps is out of scope.
    """
    try:
        num_players = int(detect_num_players(world.planets_by_id.values()))
    except Exception:
        return None
    if num_players != 2:
        return None
    return 1 - int(my_id)


def _ship_cost(col: Column, world, model, my_id: int) -> float:
    """Source-aware per-column ship cost (Phase 4 Step 2).

    Returns `SHIP_COST * col.ships` for a rear source (no enemy threat
    within `SHIP_COST_THREAT_ETA_THRESHOLD` turns). Returns
    `SHIP_COST_THREAT_MULT * SHIP_COST * col.ships` when an in-flight
    enemy fleet OR a potential launch from a close opp planet would
    reach the source within the threshold.

    "Close" here means `time_to_enemy_threat <= SHIP_COST_THREAT_ETA_THRESHOLD`.
    Per PI 2026-05-21: indirect fleets (potential launches from CLOSE
    opp planets) count; distant opp planets do not. The threshold gives
    the modeling-clean rear-vs-threatened distinction the LP needs.

    `arrival_eta=0` on the threat call: we're asking "is THIS source
    under threat right now" — not "post-arrival hold at a target." The
    source IS at its current position; rotation prediction doesn't apply.

    **Compound-column special case** (review of commit 16c9be7): a
    Phase-F2 compound column's `src_id` points to an opp-owned planet
    that we hypothetically capture mid-horizon. The ship-cost concept
    (defensive value of source ships) doesn't apply — those "source
    ships" are FUTURE production after our capture, not ours-now. Also,
    `time_to_enemy_threat` against an opp-owned src would find sibling
    opp planets nearby and spuriously fire the multiplier, inflating
    compound costs ~2x and starving the LP of useful action space.
    Early-return base for compound columns.

    Returns base cost when `model` is None or the lookup fails.
    """
    base = float(SHIP_COST) * float(col.ships)
    if getattr(col, "parent_column_id", None) is not None:
        # Compound column — see docstring. Ship cost = base (no
        # source-aware multiplier; the source ships don't exist yet).
        return base
    if model is None:
        return base
    try:
        threat_eta = model.time_to_enemy_threat(
            int(col.src_id), int(my_id), world,
        )
    except Exception:
        return base
    if threat_eta is None or int(threat_eta) > int(SHIP_COST_THREAT_ETA_THRESHOLD):
        return base
    return float(SHIP_COST_THREAT_MULT) * base


def _endgame_bonus_step(planet_id: int, row: OutcomeRow, world,
                        my_id: int, opp_id: int | None,
                        currently_winning: bool) -> float:
    """Per-(planet, subset) endgame predicate bonus / penalty — STEP form.

    `currently_winning` is `is_winning_state(world, my_id, opp_id)`
    pre-computed once per LP solve (cheap; passed in to avoid O(planets)
    recomputation per subset).

    Cases:
      + λ_endgame  if currently NOT mine, row.owner_T == me, AND owning
                   this planet would satisfy is_winning_state_if_owned.
      + λ_endgame  if currently mine, row.owner_T == me, AND we are
                   currently winning (defending in winning state).
      − λ_endgame  if currently mine, row.owner_T != me, AND losing
                   this planet would flip us out of winning state.
      0            otherwise (no opp_id ⇒ 4P, predicate unavailable).

    Exception safety: predicate calls are wrapped — any error from
    `is_winning_state_if_owned` / `is_winning_state_if_lost` falls
    through to 0.
    """
    if opp_id is None:
        return 0.0
    current = world.planets_by_id.get(int(planet_id))
    if current is None:
        return 0.0
    cur_owner = int(current.owner)
    pred_owner = int(row.owner_T)
    me = int(my_id)
    opp = int(opp_id)

    try:
        if cur_owner != me:
            if pred_owner == me:
                if is_winning_state_if_owned(world, me, opp, {int(planet_id)}):
                    return LAMBDA_ENDGAME
            return 0.0
        if pred_owner == me:
            return LAMBDA_ENDGAME if currently_winning else 0.0
        if is_winning_state_if_lost(world, me, opp, {int(planet_id)}):
            return 0.0
        if currently_winning:
            return -LAMBDA_ENDGAME
        return 0.0
    except Exception:
        return 0.0


def _endgame_bonus_smooth(planet_id: int, row: OutcomeRow, world,
                          my_id: int, opp_id: int | None) -> float:
    """Phase α — smooth-ΔW endgame bonus.

    Returns `λ_W · ΔW(p, S)` where ΔW is the change in
    `winning_margin = prod_advantage × remaining_turns − opp_pool`
    attributable to this (planet, subset) ownership transition.

    Closed-form: only the per-planet ownership change contributes;
    other planets' terms cancel. The decomposition:

      Δprod_advantage =
         +prod   if (cur != me, pred == me)  [we capture, gain prod]
         +prod   if (cur == opp, pred == me) [opp also loses prod]
         −prod   if (cur == me,  pred != me) [we lose, lose prod]
         −prod   if (cur == me,  pred == opp)[opp also gains prod]

      Δopp_pool =
         −(ships + prod·rem)  if (cur == opp, pred == me)
                              [opp loses garrison + future prod stream]
         +prod·rem            if (cur == me,  pred == opp)
                              [opp gains future prod stream]
         (we don't model the ship transfer in opp_pool on our loss —
         matches `is_winning_state_if_lost`'s conservative approach.)

      ΔW = Δprod_advantage × rem − Δopp_pool

    Signed: positive = captures that strengthen winning margin; negative
    = subsets that weaken it (e.g. we lose a planet to opp). The
    objective rewards positive and penalizes negative — defense emerges
    proportional to planet importance.
    """
    if opp_id is None:
        return 0.0
    current = world.planets_by_id.get(int(planet_id))
    if current is None:
        return 0.0
    cur_owner = int(current.owner)
    pred_owner = int(row.owner_T)
    me = int(my_id)
    opp = int(opp_id)

    if cur_owner == me and pred_owner == me:
        return 0.0  # no transition
    if cur_owner != me and pred_owner != me:
        return 0.0  # opp/neutral → opp/neutral; we don't care

    prod = int(current.production)
    ships = int(current.ships)
    rem = int(remaining_turns(world))

    d_adv = 0
    d_op = 0

    if cur_owner != me and pred_owner == me:
        # We capture.
        d_adv += prod
        if cur_owner == opp:
            d_adv += prod         # opp loses prod
            d_op -= ships + prod * rem  # opp loses garrison + future prod
    elif cur_owner == me and pred_owner != me:
        # We lose.
        d_adv -= prod
        if pred_owner == opp:
            d_adv -= prod
            d_op += prod * rem    # opp gains future prod

    delta_w = d_adv * rem - d_op
    return _lambda_w() * float(delta_w)


def _endgame_bonus(planet_id: int, row: OutcomeRow, world,
                   my_id: int, opp_id: int | None,
                   currently_winning: bool) -> float:
    """Dispatch to step or smooth form based on `LP_SMOOTH_DELTA_W` env.

    Step form (default): Phase 4 Step 1 ±LAMBDA_ENDGAME indicator.
    Smooth form (opt-in): Phase α λ_W·ΔW signed, magnitude-proportional.
    The two forms are independent code paths so the A/B isolates the
    objective change cleanly.
    """
    if _smooth_delta_w_enabled():
        return _endgame_bonus_smooth(planet_id, row, world, my_id, opp_id)
    return _endgame_bonus_step(planet_id, row, world, my_id, opp_id,
                               currently_winning)


def _per_planet_topology_score(planet_id: int, world, model, sense,
                               my_id: int) -> float:
    """Closed-form per-planet topology score (Level 1).

    Sums three per-planet topology contributions; each can be
    individually disabled via env var. Computed ONCE per planet per
    turn from the pre-computed `sense_state` snapshot; subsets look up
    the cached value via `_topology_bonus` and apply it conditionally
    on `row.owner_T == my_id`.

    Returns 0.0 if all three features disabled, if `sense` is None, or
    if the planet is not in `world.planets_by_id`.

    Exception safety: any failure in the underlying primitives falls
    through to 0.0 — the LP cost-vector loop must never raise.
    """
    if sense is None:
        return 0.0
    p = world.planets_by_id.get(int(planet_id))
    if p is None:
        return 0.0

    score = 0.0
    try:
        if _reach_bonus_enabled():
            from lib.geo.sense import _planet_eta as _sense_eta
            reach = 0.0
            for n_pid, cluster_idx in sense.voronoi.items():
                if cluster_idx < 0:  # CONTESTED neutral
                    continue
                n = world.planets_by_id.get(int(n_pid))
                if n is None or n.id == int(planet_id):
                    continue
                eta = _sense_eta(p, n)
                if eta > REACH_HORIZON:
                    continue
                score += LAMBDA_REACH * float(n.production) / (1.0 + float(eta))
    except Exception:
        pass

    try:
        if _defense_bonus_enabled():
            from lib.geo.sense import _planet_eta as _sense_eta
            nearby = 0
            for own_pid in sense.pid_to_cluster.keys():
                if int(own_pid) == int(planet_id):
                    continue
                op = world.planets_by_id.get(int(own_pid))
                if op is None:
                    continue
                eta = _sense_eta(p, op)
                if eta <= DEFENSE_HORIZON:
                    nearby += 1
            score += LAMBDA_DEFENSE * float(nearby)
    except Exception:
        pass

    try:
        if _front_penalty_enabled() and model is not None:
            threat = model.time_to_enemy_threat(int(planet_id), int(my_id), world)
            if threat is not None and int(threat) <= RECAPTURE_HORIZON:
                hold = max(1, int(threat))
                score -= LAMBDA_FRONT * float(p.production) / float(hold)
    except Exception:
        pass

    return float(score)


def _solve_via_dual_decomp(
    active: list[Column],
    per_planet_tables: dict[int, dict[tuple[int, ...], OutcomeRow]],
    inv: dict[int, tuple[int, int]],
    world,
    model,
    *,
    my_id: int,
    opp_id: int | None,
    currently_winning: bool,
    topology_scores: dict[int, float] | None,
    alpha_opp_penalty: float,
    ship_cost: float,
    discount_gamma: float | None,
    step_now: int,
    time_limit_seconds: float,
) -> "OutcomeAwareResult":
    """ITEM 5 dispatcher → `lib.joint_solver.dual_decomp.solve_dual_decomp_inner`.

    Builds the per-(planet, subset) base_value_fn closure that captures
    V_p + endgame_bonus + topology_bonus + opening_tempo_bonus (same
    objective the MILP path sums), then calls the dual-decomp inner
    solver. Returns the same OutcomeAwareResult shape as MILP so
    callers don't need to know which path ran.
    """
    from lib.joint_solver.dual_decomp import solve_dual_decomp_inner

    use_discounted_value = (
        discount_gamma is not None and 0.0 < float(discount_gamma) < 1.0
    )

    def base_value_fn(pid: int, row: OutcomeRow) -> float:
        v = _value_for_outcome(row, my_id, alpha_opp_penalty,
                               use_discounted_value)
        v += _endgame_bonus(pid, row, world, my_id, opp_id,
                            currently_winning)
        v += _topology_bonus(pid, row, my_id, topology_scores)
        v += _opening_tempo_bonus(pid, row, world, my_id, step_now)
        return float(v)

    chosen, per_planet_value, status = solve_dual_decomp_inner(
        per_planet_tables, active, inv,
        my_id=int(my_id),
        base_value_fn=base_value_fn,
        ship_cost=float(ship_cost),
        time_limit_seconds=float(time_limit_seconds),
    )

    # Build OutcomeAwareResult.
    fired_col_ids: set[int] = set()
    for subset in chosen.values():
        fired_col_ids.update(int(c) for c in subset)
    col_by_id = {int(c.column_id): c for c in active}
    fired = [col_by_id[cid] for cid in fired_col_ids if cid in col_by_id]
    moves = [
        [int(c.src_id), float(c.angle), int(c.ships)]
        for c in fired if int(c.wait_N) == 0
    ]
    objective = float(sum(per_planet_value.values())) - float(ship_cost) * sum(
        int(c.ships) for c in fired
    )
    return OutcomeAwareResult(
        moves=moves, fired_columns=fired, objective=objective,
        status=status,
        n_x_vars=len(active),
        n_y_vars=sum(len(t) for t in per_planet_tables.values()),
        n_constraints=0,
        per_planet_chosen=chosen,
        per_planet_value=per_planet_value,
    )


def _topology_bonus(planet_id: int, row: OutcomeRow, my_id: int,
                    topology_scores: dict[int, float] | None) -> float:
    """Per-(planet, subset) topology bonus lookup.

    Returns the pre-computed score for `planet_id` IFF this subset
    predicts we'd own it post-capture (row.owner_T == my_id). Otherwise
    0.0 — the LP doesn't credit topology of planets we don't end up
    owning.

    The bonus only fires when `topology_scores` was populated upstream
    (LP_TOPOLOGY_FEATURES=1 and at least one LP_*_BONUS=1). Returns
    0.0 otherwise — no behaviour change vs pre-Level-1.
    """
    if topology_scores is None:
        return 0.0
    if int(row.owner_T) != int(my_id):
        return 0.0
    return float(topology_scores.get(int(planet_id), 0.0))


def _opening_tempo_bonus(planet_id: int, row: OutcomeRow, world,
                          my_id: int, step_now: int) -> float:
    """Phase ε opening-tempo bias: extra credit for capturing a
    currently-neutral planet during the opening.

    Returns 0.0 when:
    - The feature is disabled (LP_OPENING_TEMPO != "1") — every callsite
      then gets pure no-op behaviour, no objective change.
    - step_now >= OPENING_TEMPO_HORIZON — we're past the opening.
    - The subset doesn't end with us owning the planet (owner_T != my_id).
    - The planet's CURRENT owner isn't -1 (neutral) — we only boost
      captures of neutrals, not contests over already-owned planets.

    When active, returns `(OPENING_TEMPO_FACTOR - 1) × prod_stream_me_for_subset`.
    Added on top of the existing `_value_for_outcome` prod-stream term,
    this produces an effective multiplier of OPENING_TEMPO_FACTOR on
    the my-prod component for the qualifying subset — exactly the
    leader's `EARLY_EXTRA` scaling applied selectively to neutral captures.
    """
    if not _opening_tempo_enabled():
        return 0.0
    if int(step_now) >= OPENING_TEMPO_HORIZON:
        return 0.0
    if int(row.owner_T) != int(my_id):
        return 0.0
    p = world.planets_by_id.get(int(planet_id))
    if p is None or int(p.owner) != -1:
        return 0.0
    my_prod = float(row.prod_stream.get(int(my_id), 0))
    return (OPENING_TEMPO_FACTOR - 1.0) * my_prod


def _value_for_outcome(row: OutcomeRow, my_id: int,
                       alpha_opp_penalty: float,
                       discounted: bool = False) -> float:
    """Subset value: prod_stream_me − α · prod_stream_opp.

    Sums opp production across ALL non-me, non-neutral owners (4P-aware).

    `discounted`: when True, reads `row.prod_stream_discounted` (the
    γ-weighted per-tick production accrual) instead of the integer
    `prod_stream`. Caller must have constructed the row via
    `enumerate_outcomes(..., discount_gamma=γ)` so the discounted dict
    is populated.
    """
    if discounted:
        me_prod = float(row.prod_stream_discounted.get(int(my_id), 0.0))
        opp_prod = float(sum(
            v for owner, v in row.prod_stream_discounted.items()
            if int(owner) >= 0 and int(owner) != int(my_id)
        ))
        return me_prod - float(alpha_opp_penalty) * opp_prod
    me_prod = float(row.prod_stream.get(int(my_id), 0))
    opp_prod = float(sum(
        v for owner, v in row.prod_stream.items()
        if int(owner) != int(my_id) and int(owner) >= 0
    ))
    return me_prod - float(alpha_opp_penalty) * opp_prod


# ---------------------------------------------------------------------------
# Greedy fallback
# ---------------------------------------------------------------------------


def _lp_outcome_greedy_fallback(
    active_columns: list[Column],
    per_planet_tables: dict[int, dict[tuple[int, ...], OutcomeRow]],
    world,
    *,
    my_id: int,
    alpha_opp_penalty: float,
    discounted: bool = False,
    opp_id: int | None = None,
    currently_winning: bool = False,
    topology_scores: dict[int, float] | None = None,
) -> OutcomeAwareResult:
    """Pure-Python greedy fallback when MILP is unavailable / infeasible.

    For each planet, pick the subset with the highest
    (prod_stream_me − α · prod_stream_opp + endgame_bonus) value. Then
    check global source budget feasibility; if a launch would over-spend,
    drop it (greedy, lowest-value-marginal first).
    """
    step_now = int(getattr(world, "step", 0) or 0)
    inv = _lp_outcome_source_inventory(active_columns, world, my_id=int(my_id))

    # Per planet: choose best subset.
    chosen: dict[int, tuple[int, ...]] = {}
    per_planet_value: dict[int, float] = {}
    for pid, table in per_planet_tables.items():
        empty_row = table[()]
        best_subset = ()
        best_value = (
            _value_for_outcome(empty_row, my_id, alpha_opp_penalty, discounted)
            + _endgame_bonus(pid, empty_row, world, my_id, opp_id, currently_winning)
            + _topology_bonus(pid, empty_row, my_id, topology_scores)
            + _opening_tempo_bonus(pid, empty_row, world, my_id, step_now)
        )
        for subset, row in table.items():
            v = (
                _value_for_outcome(row, my_id, alpha_opp_penalty, discounted)
                + _endgame_bonus(pid, row, world, my_id, opp_id, currently_winning)
                + _topology_bonus(pid, row, my_id, topology_scores)
                + _opening_tempo_bonus(pid, row, world, my_id, step_now)
            )
            if v > best_value:
                best_value = v
                best_subset = subset
        chosen[pid] = best_subset
        per_planet_value[pid] = best_value

    # Collect fired column_ids from chosen subsets.
    fired_ids = {cid for s in chosen.values() for cid in s}
    by_col_id = {int(c.column_id): c for c in active_columns}

    # Check source budget; greedy drop launches if over-spent.
    # Pending-aware: when LP_PENDING_AWARE_BUDGET=1, subtract already-
    # committed (wait_N>0 from prior turns) ships from the source budget.
    emitted_per_src_fire: dict[tuple[int, int], int] = {}
    fired: list[Column] = []
    drop_order = sorted(
        (by_col_id[cid] for cid in fired_ids),
        key=lambda c: float(c.value),
    )
    pending_fires_g = (
        _fetch_pending_fires(int(my_id))
        if _pending_aware_budget_enabled() else []
    )
    for col in drop_order:
        sid = int(col.src_id)
        initial, prod = inv.get(sid, (0, 0))
        wait_N = int(col.wait_N)
        used = sum(v for (s, w), v in emitted_per_src_fire.items()
                   if s == sid and w <= wait_N)
        pending_used = _pending_ships_consumed_by(
            sid, step_now, wait_N, pending_fires_g,
        )
        if used + int(col.ships) > (
            initial + prod * max(0, wait_N) - DEFENDER_GUARD - pending_used
        ):
            # Drop this column from the chosen subset (replace with empty).
            for pid, s in chosen.items():
                if int(col.column_id) in s:
                    chosen[pid] = tuple(c for c in s if c != int(col.column_id))
            continue
        emitted_per_src_fire[(sid, wait_N)] = (
            emitted_per_src_fire.get((sid, wait_N), 0) + int(col.ships)
        )
        fired.append(col)

    # Compute final objective.
    obj = sum(
        _value_for_outcome(per_planet_tables[pid][s], my_id, alpha_opp_penalty,
                           discounted)
        + _endgame_bonus(pid, per_planet_tables[pid][s], world, my_id, opp_id,
                         currently_winning)
        + _topology_bonus(pid, per_planet_tables[pid][s], my_id, topology_scores)
        + _opening_tempo_bonus(pid, per_planet_tables[pid][s], world, my_id,
                               step_now)
        for pid, s in chosen.items()
    )

    moves = [
        [int(c.src_id), float(c.angle), int(c.ships)]
        for c in fired if int(c.wait_N) == 0
    ]
    return OutcomeAwareResult(
        moves=moves, fired_columns=fired, objective=float(obj),
        status="greedy_fallback",
        n_x_vars=len(active_columns),
        n_y_vars=sum(len(t) for t in per_planet_tables.values()),
        n_constraints=0,
        per_planet_chosen=chosen,
        per_planet_value=per_planet_value,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def solve_outcome_aware(
    columns: list[Column],
    world,
    model,
    *,
    my_id: int,
    t_end: int = T_END,
    alpha_opp_penalty: float = ALPHA_OPP_PENALTY,
    ship_cost: float = SHIP_COST,
    time_limit_seconds: float = TIME_LIMIT_SECONDS,
    discount_gamma: float | None = None,
) -> OutcomeAwareResult:
    """Solve the outcome-aware LP for post-opening turns.

    Inputs:
      columns: list of Column with pre-computed per-candidate value (used
        only as a pre-filter when a planet has > MAX_CONTESTERS_PER_PLANET
        candidates).
      world, model: World and WorldModel snapshots. `model.ledger` is
        EXPECTED to already include opp projections (the upstream
        `_model_with_opp_projection` merges them in); the Phase 5C
        separate `opp_arrivals` parameter was removed in Phase 5D after
        the double-count audit.
      my_id: our seat.

    Returns OutcomeAwareResult with the chosen moves (wait_N==0 fires)
    plus diagnostics.
    """
    if not columns:
        return OutcomeAwareResult(
            moves=[], fired_columns=[], objective=0.0,
            status="empty_columns",
            n_x_vars=0, n_y_vars=0, n_constraints=0,
        )

    step_now = int(getattr(world, "step", 0) or 0)

    # Phase 4: derive opp_id for the endgame predicate term. 2P-only;
    # 4P games get opp_id=None ⇒ no bonus (additive zero, no behaviour
    # change vs pre-Phase-4).
    opp_id = _derive_opp_id_2p(world, int(my_id))
    if opp_id is not None:
        try:
            currently_winning = bool(is_winning_state(world, int(my_id), int(opp_id)))
        except Exception:
            currently_winning = False
    else:
        currently_winning = False

    # Level 1 topology features: compute per-planet topology scores
    # ONCE per turn (Frank-Wolfe linearization — scored against current
    # board, not post-LP state). Empty dict when LP_TOPOLOGY_FEATURES=0,
    # which short-circuits the bonus to 0.0 in all 5 call sites.
    #
    # 4P gate (PHASE β POST-MORTEM, 2026-05-22 4P A/B 0/16): topology
    # features were tuned on 2P scenarios and the recapture_risk
    # penalty + mutual_defense_bonus make the agent over-defensive in
    # 4P (rank-2-always pattern). Smooth ΔW (Phase α) already gates
    # itself to 2P via `_derive_opp_id_2p`; topology now does the same.
    # `LP_TOPOLOGY_4P=1` re-enables for the 4P-recalibration session
    # planned for next cycle.
    topology_scores: dict[int, float] | None = None
    if _topology_features_enabled():
        is_2p = (opp_id is not None)
        topology_4p_enabled = _os.environ.get("LP_TOPOLOGY_4P", "0") == "1"
        if is_2p or topology_4p_enabled:
            try:
                from lib.geo.sense import sense_state as _sense_state
                sense = _sense_state(world, model)
                topology_scores = {}
                for pid in world.planets_by_id:
                    topology_scores[int(pid)] = _per_planet_topology_score(
                        int(pid), world, model, sense, my_id=int(my_id),
                    )
            except Exception:
                topology_scores = None

    # Phase ζ.v2: hold-aware threat cache. Precompute per-planet
    # `(counter_eta, counter_ships)` once per turn. Threaded into
    # `_build_per_planet_arrivals` which appends a synthetic opp-counter
    # arrival to each target's fixed_arrivals when the gate is on.
    # 2P-only; 4P returns None (no synthetic injection).
    threat_cache: dict[int, tuple[int | None, int]] | None = None
    if _hold_aware_enabled() and opp_id is not None:
        try:
            threat_cache = {}
            for pid in world.planets_by_id:
                threat_cache[int(pid)] = _predict_opp_counter(
                    int(pid), world, my_id=int(my_id),
                )
        except Exception:
            threat_cache = None

    # Filter to our positive-value columns with a valid source.
    # EXCEPTION: compound columns (parent_column_id != None) are
    # Phase F2a production-feedback fires from planets we'd capture
    # mid-horizon. Their src isn't yet in inv (it's opp-owned now).
    # They're feasible only if their parent capture fires; that's
    # enforced via a linkage constraint added below, not by inv.
    inv = _lp_outcome_source_inventory(columns, world, my_id=int(my_id))
    active: list[Column] = []
    for col in columns:
        if int(col.owner) != int(my_id):
            continue
        is_compound = getattr(col, "parent_column_id", None) is not None
        if not is_compound and int(col.src_id) not in inv:
            continue
        if float(col.value) <= 0.0:
            continue
        active.append(col)
    if not active:
        return OutcomeAwareResult(
            moves=[], fired_columns=[], objective=0.0,
            status="no_positive_columns",
            n_x_vars=0, n_y_vars=0, n_constraints=0,
        )

    # Build per-planet arrival sets and outcome tables.
    per_planet_arrivals = _build_per_planet_arrivals(
        active, world, model,
        my_id=int(my_id), step_now=step_now,
        threat_cache=threat_cache, opp_id=opp_id,
    )
    per_planet_tables: dict[int, dict[tuple[int, ...], OutcomeRow]] = {}
    for tgt_pid, (fixed, cands) in per_planet_arrivals.items():
        planet = world.planets_by_id.get(int(tgt_pid))
        if planet is None:
            continue
        try:
            table = enumerate_outcomes(
                initial_owner=int(planet.owner),
                initial_ships=float(int(planet.ships)),
                production=int(planet.production),
                horizon=int(t_end),
                fixed_arrivals=fixed,
                candidate_arrivals=cands,
                discount_gamma=discount_gamma,
            )
        except ValueError:
            continue  # too many candidates after pre-filter; shouldn't happen
        per_planet_tables[int(tgt_pid)] = table

    # The set of column_ids that actually made it into per-planet enumeration
    # (the pre-filter may have dropped some when k > MAX_CONTESTERS_PER_PLANET).
    enumerated_col_ids: set[int] = set()
    for table in per_planet_tables.values():
        for subset in table:
            enumerated_col_ids.update(subset)
    # Drop active columns whose column_id wasn't enumerated.
    active = [c for c in active if int(c.column_id) in enumerated_col_ids
              or int(c.tgt_id) not in per_planet_tables]
    # Actually, only columns whose target was enumerated count. If column.tgt
    # isn't in per_planet_tables, it was pruned. Drop them.
    active = [c for c in active if int(c.column_id) in enumerated_col_ids]
    if not active:
        return OutcomeAwareResult(
            moves=[], fired_columns=[], objective=0.0,
            status="no_active_after_prefilter",
            n_x_vars=0, n_y_vars=0, n_constraints=0,
        )

    if not _MILP_AVAILABLE:
        return _lp_outcome_greedy_fallback(
            active, per_planet_tables, world,
            my_id=int(my_id), alpha_opp_penalty=float(alpha_opp_penalty),
            discounted=(discount_gamma is not None
                        and 0.0 < float(discount_gamma) < 1.0),
            opp_id=opp_id, currently_winning=currently_winning,
            topology_scores=topology_scores,
        )

    # ITEM 5 — Lagrangian dual decomposition route.
    # LP_SOLVER=dual selects the closed-form per-source water-fill +
    # per-target subset enum inner. Same per_planet_tables / active /
    # inv / value-fn inputs as the MILP path; different solver.
    # Falls back to MILP on any exception (safety).
    if _os.environ.get("LP_SOLVER", "milp").strip().lower() == "dual":
        try:
            return _solve_via_dual_decomp(
                active, per_planet_tables, inv, world, model,
                my_id=int(my_id), opp_id=opp_id,
                currently_winning=currently_winning,
                topology_scores=topology_scores,
                alpha_opp_penalty=float(alpha_opp_penalty),
                ship_cost=float(ship_cost),
                discount_gamma=discount_gamma,
                step_now=int(step_now),
                time_limit_seconds=float(time_limit_seconds),
            )
        except Exception as exc:
            # On failure, fall through to MILP and tag the status.
            import traceback
            _dual_fallback_status = (
                f"dual_decomp:fallback_to_milp:{type(exc).__name__}"
            )
            # Continue to MILP build below.

    import numpy as np

    # ---- Build MILP -------------------------------------------------------
    n_x = len(active)
    col_id_to_x_idx: dict[int, int] = {int(c.column_id): j for j, c in enumerate(active)}

    # y variables: list per planet.
    y_index: dict[tuple[int, tuple[int, ...]], int] = {}
    y_planet_subsets: dict[int, list[tuple[int, ...]]] = {}
    for pid, table in per_planet_tables.items():
        subsets = list(table.keys())
        y_planet_subsets[pid] = subsets
        for s in subsets:
            y_index[(pid, s)] = n_x + len(y_index)
    n_y = len(y_index)
    n_total = n_x + n_y

    # Cost vector. milp minimizes c^T·x, so negate values.
    # Phase 4 Step 2: source-aware ship cost — threatened sources pay more.
    # Honour the `ship_cost` kwarg (legacy uniform override) by scaling the
    # base; the threat multiplier is independent.
    c_vec = np.zeros(n_total, dtype=float)
    base_scale = float(ship_cost) / float(SHIP_COST) if SHIP_COST != 0 else 1.0
    for j, col in enumerate(active):
        c_vec[j] = _ship_cost(col, world, model, my_id) * base_scale
    use_discounted_value = (
        discount_gamma is not None and 0.0 < float(discount_gamma) < 1.0
    )
    for (pid, s), y_idx in y_index.items():
        row = per_planet_tables[pid][s]
        value = _value_for_outcome(row, my_id, alpha_opp_penalty,
                                   use_discounted_value)
        value += _endgame_bonus(pid, row, world, my_id, opp_id,
                                currently_winning)
        value += _topology_bonus(pid, row, my_id, topology_scores)
        value += _opening_tempo_bonus(pid, row, world, my_id, step_now)
        c_vec[y_idx] = -float(value)  # negate so milp picks high-value subsets

    A_eq_rows: list[list[float]] = []
    b_eq: list[float] = []
    A_ub_rows: list[list[float]] = []
    b_ub: list[float] = []

    # (1) Subset uniqueness per planet: Σ_S y_{p,S} = 1.
    for pid, subsets in y_planet_subsets.items():
        row = [0.0] * n_total
        for s in subsets:
            row[y_index[(pid, s)]] = 1.0
        A_eq_rows.append(row)
        b_eq.append(1.0)

    # (2) Candidate↔subset linkage: x_c - Σ_{S∋c} y_{p(c),S} = 0.
    for j, col in enumerate(active):
        pid = int(col.tgt_id)
        row = [0.0] * n_total
        row[j] = 1.0
        for s in y_planet_subsets.get(pid, []):
            if int(col.column_id) in s:
                row[y_index[(pid, s)]] = -1.0
        A_eq_rows.append(row)
        b_eq.append(0.0)

    # (3) Source budget over time. Phase F2a: skip compound columns —
    # their ships come from a planet we'd capture mid-horizon, not from
    # any source in `inv`. Compound columns are gated via the linkage
    # constraint (4) below; the captured planet's post-capture
    # production is implicit in the column's ship-count construction.
    src_ids = sorted({int(c.src_id) for c in active
                      if getattr(c, "parent_column_id", None) is None})
    fire_times = sorted({int(c.wait_N) for c in active})
    # Pending-aware budget (LP_PENDING_AWARE_BUDGET=1): pull pending fires
    # ONCE so each (sid, u) constraint can deduct already-committed ships
    # from the RHS.
    pending_fires = (
        _fetch_pending_fires(int(my_id))
        if _pending_aware_budget_enabled() else []
    )
    for sid in src_ids:
        if sid not in inv:
            continue
        initial, prod = inv[sid]
        for u in fire_times:
            row = [0.0] * n_total
            any_in_row = False
            for j, col in enumerate(active):
                if getattr(col, "parent_column_id", None) is not None:
                    continue  # compound col not in src-budget
                if int(col.src_id) == sid and int(col.wait_N) <= u:
                    row[j] = float(col.ships)
                    any_in_row = True
            if not any_in_row:
                continue
            pending_used = _pending_ships_consumed_by(
                sid, step_now, u, pending_fires,
            )
            A_ub_rows.append(row)
            b_ub.append(float(
                initial + prod * max(0, u) - DEFENDER_GUARD - pending_used
            ))

    # (4) Phase F2a linkage: x_compound <= x_parent_capture.
    # Encoded as A_ub row `+1 * x_compound − 1 * x_parent <= 0`.
    # parent_column_id may reference a column that got dropped at the
    # per-planet-MILP-prefilter step (lp_outcome.py:381). In that case
    # we instead pin x_compound = 0 (force the row to b_ub=0 with only
    # the +1 term — equivalent to x_compound <= 0).
    col_id_to_idx: dict[int, int] = {
        int(col.column_id): j for j, col in enumerate(active)
    }
    for j, col in enumerate(active):
        pid_parent = getattr(col, "parent_column_id", None)
        if pid_parent is None:
            continue
        row = [0.0] * n_total
        row[j] = 1.0
        parent_idx = col_id_to_idx.get(int(pid_parent))
        if parent_idx is None:
            # Parent dropped; force this compound col to 0.
            A_ub_rows.append(row)
            b_ub.append(0.0)
        else:
            row[parent_idx] = -1.0
            A_ub_rows.append(row)
            b_ub.append(0.0)

    # Compose constraints.
    constraints_list = []
    if A_eq_rows:
        A_eq = np.array(A_eq_rows, dtype=float)
        b_eq_arr = np.array(b_eq, dtype=float)
        constraints_list.append(LinearConstraint(A_eq, lb=b_eq_arr, ub=b_eq_arr))
    if A_ub_rows:
        A_ub = np.array(A_ub_rows, dtype=float)
        b_ub_arr = np.array(b_ub, dtype=float)
        constraints_list.append(LinearConstraint(A_ub, ub=b_ub_arr))
    if not constraints_list:
        return _lp_outcome_greedy_fallback(
            active, per_planet_tables, world,
            my_id=int(my_id), alpha_opp_penalty=float(alpha_opp_penalty),
            discounted=(discount_gamma is not None
                        and 0.0 < float(discount_gamma) < 1.0),
            opp_id=opp_id, currently_winning=currently_winning,
            topology_scores=topology_scores,
        )

    bounds = Bounds(lb=np.zeros(n_total), ub=np.ones(n_total))
    integrality = np.ones(n_total, dtype=int)
    n_constraints = len(A_eq_rows) + len(A_ub_rows)

    try:
        res = milp(c=c_vec, constraints=constraints_list,
                   integrality=integrality, bounds=bounds,
                   options={"time_limit": float(time_limit_seconds)})
    except Exception:
        return _lp_outcome_greedy_fallback(
            active, per_planet_tables, world,
            my_id=int(my_id), alpha_opp_penalty=float(alpha_opp_penalty),
            discounted=(discount_gamma is not None
                        and 0.0 < float(discount_gamma) < 1.0),
            opp_id=opp_id, currently_winning=currently_winning,
            topology_scores=topology_scores,
        )

    if res.x is None:
        return _lp_outcome_greedy_fallback(
            active, per_planet_tables, world,
            my_id=int(my_id), alpha_opp_penalty=float(alpha_opp_penalty),
            discounted=(discount_gamma is not None
                        and 0.0 < float(discount_gamma) < 1.0),
            opp_id=opp_id, currently_winning=currently_winning,
            topology_scores=topology_scores,
        )

    # Extract.
    fired: list[Column] = []
    moves: list = []
    for j, col in enumerate(active):
        if res.x[j] > 0.5:
            fired.append(col)
            if int(col.wait_N) == 0:
                moves.append([int(col.src_id), float(col.angle), int(col.ships)])
    per_planet_chosen: dict[int, tuple[int, ...]] = {}
    per_planet_value: dict[int, float] = {}
    for (pid, s), y_idx in y_index.items():
        if res.x[y_idx] > 0.5:
            per_planet_chosen[pid] = s
            row = per_planet_tables[pid][s]
            per_planet_value[pid] = (
                _value_for_outcome(
                    row, my_id, alpha_opp_penalty, use_discounted_value,
                )
                + _endgame_bonus(pid, row, world, my_id, opp_id,
                                 currently_winning)
                + _topology_bonus(pid, row, my_id, topology_scores)
                + _opening_tempo_bonus(pid, row, world, my_id, step_now)
            )

    return OutcomeAwareResult(
        moves=moves, fired_columns=fired, objective=float(-res.fun),
        status=str(getattr(res, "message", "milp_ok")),
        n_x_vars=int(n_x), n_y_vars=int(n_y), n_constraints=int(n_constraints),
        per_planet_chosen=per_planet_chosen,
        per_planet_value=per_planet_value,
    )
