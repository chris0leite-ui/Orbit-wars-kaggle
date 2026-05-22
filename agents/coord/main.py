"""coord — multi-source bundle coordinator (Day 1 scaffold).

Replaces minimal's per-source-then-joint-pair flow with bundle-as-primitive
+ Lagrangian-priced clearing. Day 1 deliverable: data types, attack-bundle
enumeration, unit tests. The agent entry is a placeholder until later days
wire cheap-filter / Tier-2 / clearing / emit.

Design reference: `/root/.claude/plans/eventual-skipping-breeze.md`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations
import os
import time

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# Single-line imports for bundle-safety. The bundler's regex strips line 1
# of a multi-line `from X import (...)` but leaves the continuation lines
# as orphan indented text → IndentationError. agents/coord/_minimal_inline
# is a local copy of minimal's helpers so the bundler can inline it via
# the standard intra-package submodule pattern.
from agents.coord._minimal_inline import EPISODE_STEPS
from agents.coord._minimal_inline import GAMMA
from agents.coord._minimal_inline import MIN_FLEET_SIZE
from agents.coord._minimal_inline import MIN_HORIZON
from agents.coord._minimal_inline import MAX_HORIZON
from agents.coord._minimal_inline import SIM_SETTLE_TURNS
from agents.coord._minimal_inline import CHEAP_REJECT_THRESHOLD
from agents.coord._minimal_inline import NUM_TARGETS_PER_SOURCE
from agents.coord._minimal_inline import WALLCLOCK_BUDGET_MS
from agents.coord._minimal_inline import affordable_validate_cap
from agents.coord._minimal_inline import aim_and_eta
from agents.coord._minimal_inline import build_trajectory_baseline
from agents.coord._minimal_inline import cheap_marginal_value
from agents.coord._minimal_inline import enumerate_ship_counts
from agents.coord._minimal_inline import favor_hybrid
from agents.coord._minimal_inline import nearest_k
from agents.coord._minimal_inline import score_candidate_v4_joint
from agents.coord._minimal_inline import wait_then_fire_variants
from agents.coord._minimal_inline import _as_dict
from agents.coord._minimal_inline import _num_seats
from agents.coord._minimal_inline import _source_survives_launch
from agents.coord._minimal_inline import _target_holdable_after_capture
from agents.coord._minimal_inline import _target_cost_parity_ok
from agents.coord._endgame import bundle_delta_w_attack
from agents.coord._endgame import bundle_delta_w_defend
from agents.coord._endgame import opp_pool as endgame_opp_pool
from agents.coord._endgame import remaining_turns as endgame_remaining_turns
from agents.coord._endgame import EPISODE_STEPS as _ENDGAME_EPISODE_STEPS
from lib.fast_sim import from_obs as fs_from_obs
from lib.intent import World
from lib.scoring import pv_horizon
from lib.trajectory import predict_fleet_fate
from lib.world_model import WAVE_LOOKAHEAD, WorldModel, comet_remaining_lifetime


# Drift guard: `_endgame.py` and `_minimal_inline.py` both define
# EPISODE_STEPS. They must match — a divergence would make
# endgame_remaining_turns() use a different horizon than the leaf head's
# pv_horizon discounting, silently mis-scaling the bonus.
assert _ENDGAME_EPISODE_STEPS == EPISODE_STEPS, (
    f"EPISODE_STEPS drift: _endgame.py={_ENDGAME_EPISODE_STEPS} "
    f"vs _minimal_inline.py={EPISODE_STEPS}"
)


NEAREST_SOURCES_PER_TARGET = 5
# Multi-source coordination cap. Gate 3 (Day 10) probe at MAX_BUNDLE_SIZE=3
# showed 3-source bundles win on only 1.8% of turns (786 samples), and
# never unlock targets that 2-source can't reach. Compute savings of
# MAX_BUNDLE_SIZE=2 (smaller enumeration + tier-2 set) outweigh the
# 1.8% × +9.47 mean-lift EV. Revisit in v2 if 3-source patterns emerge
# more strongly against non-self opponents.
MAX_BUNDLE_SIZE = 2
ARRIVAL_WINDOW_SLACK = 2
DEFEND_LOOKAHEAD = 30

# Cheap-filter — opportunity cost penalty per committed ship (small;
# the leaf-Δ from favor_hybrid dominates the score, this just breaks
# ties against bundles that needlessly drain large garrisons for low gain).
CHEAP_OPPORTUNITY_COST = 0.01
# Top-K admitted to Tier-2. Calibrated from Day 4 probe — K=50 gave
# 89% attack rank-1 retention; K=75 widens to ~97% by admitting bundles
# whose cheap_score is within ~20% of the top-50 boundary. Tier-2 cost
# 75 × ~5ms = 375ms, still under the 600ms agent budget when paired
# with the existing safe_deadline pre-bail watchdog.
CHEAP_FILTER_TOP_K = 75

# Tier-2 wallclock budget (of the 600ms agent budget, reserve ~150ms for
# enumeration + cheap-filter + Lagrangian + emit). The actual pre-bail
# threshold is computed per-machine via `affordable_validate_cap`.
TIER2_BUDGET_MS = 450.0

# Lagrangian iteration parameters (Day 6).
# - LAGRANGIAN_MAX_ITERS: cap on subgradient iterations per turn.
# - LAGRANGIAN_ALPHA0:    initial step size for the subgradient update,
#                         decays as alpha0 / (1 + iter).
# - LAGRANGIAN_TARGET_UTIL: soft pressure to use this fraction of each
#                           source's ship budget. 0.8 = nudge toward
#                           80% utilisation. Avoids both idle hoarding
#                           and over-commitment to single bundles.
# - LAGRANGIAN_BUDGET_MS: total wallclock budget for the dual loop;
#                        well under 5ms in practice for K=75 bundles.
LAGRANGIAN_MAX_ITERS = 12
LAGRANGIAN_ALPHA0 = 0.01
LAGRANGIAN_TARGET_UTIL = 0.8
LAGRANGIAN_BUDGET_MS = 20.0

# Smooth-ΔW endgame bonus — closed-form `winning_margin` contribution added
# to each bundle's tier2_score before Lagrangian clearing. Per-bundle
# opponent-of-record: target's owner for ATTACK; largest-inbound-threat
# owner for DEFEND. 2P degenerates to source-branch formula; 4P uses (c1)
# per-bundle attribution (see plan §4P attribution).
#
# LAMBDA_W_DEFAULT = 0.002 chosen from
# `scripts/check_coord_endgame_calibration.py` over seeds {0,1} × 30
# early-mid-game turns: median |tier2_score| = 11.0, median |ΔW| = 1413.
# Anchor that puts median bonus at 30% of median |tier2| is λ_W ≈ 0.0023;
# rounded to 0.002 for cleaner runtime arithmetic.
#
# Env vars (truthy values: "1", "true", "yes", "on", case-insensitive):
#   COORD_DELTA_W       — master gate (default ON). Off → no bonus at all.
#   COORD_ATTACK_BONUS  — gate ATTACK branch only (default ON).
#   COORD_DEFEND_BONUS  — gate DEFEND branch only (default ON).
#   COORD_LAMBDA_W      — float override of LAMBDA_W_DEFAULT. Setting "0"
#                         scales the bonus to 0 (equivalent to disabling).
#   COORD_LEAF_FLOOR    — float; bundles with tier2_score < floor are
#                         dropped during Lagrangian primal even if the
#                         endgame bonus would push them above zero. Default
#                         0.0 = "only admit tactically-non-negative
#                         bundles". Set to a large negative (e.g. -1e9) to
#                         disable the floor and let the bonus rescue
#                         tactically-losing bundles.
LAMBDA_W_DEFAULT = 0.002
LEAF_FLOOR_DEFAULT = 2.0
# Lagrangian break threshold on reduced_score. The original loop breaks
# when reduced_score <= 0 (only admit positive-net-value bundles). Setting
# COORD_REDUCED_FLOOR=-1e9 admits any bundle (test the hypothesis that
# per-bundle leaf-Δ is too pessimistic for ensemble emission — single
# bundles look catastrophic when scored in isolation but many-bundle
# sets can succeed via spread-the-defense).
REDUCED_FLOOR_DEFAULT = 2.0
# Demand-spread mixing (Option 3 from 2026-05-22 design):
# Per-opp defensive capacity × per-bundle attention demand → mixing_weight ∈
# [0,1]. composite = w·tier2 + (1-w)·cheap_score + endgame. When our total
# demand exceeds opp's capacity, mixing_weight drops below 1, shifting
# bundle scores toward the undefended cheap_score (less pessimistic) so
# the Lagrangian's `reduced > 0` gate stops blocking the whole ensemble.
DEMAND_SPREAD_ENABLE_ENV = "COORD_DEMAND_SPREAD"
OPP_CAPACITY_FACTOR_DEFAULT = 1.0  # × opp.ships → defensive throughput estimate
OPP_CAPACITY_FACTOR_ENV = "COORD_OPP_CAPACITY_FACTOR"
DEMAND_REACH_WINDOW = 12  # turns; opp source counts as responder if it can
                          # reach our target within this window (closed-form via aim_and_eta)
LAMBDA_W_ENV = "COORD_LAMBDA_W"
DELTA_W_ENABLE_ENV = "COORD_DELTA_W"
ATTACK_BONUS_ENV = "COORD_ATTACK_BONUS"
DEFEND_BONUS_ENV = "COORD_DEFEND_BONUS"
LEAF_FLOOR_ENV = "COORD_LEAF_FLOOR"
REDUCED_FLOOR_ENV = "COORD_REDUCED_FLOOR"

_TRUTHY_ENV = {"1", "true", "yes", "on"}


def _env_truthy(name: str, default: str = "1") -> bool:
    """Tolerant env-var boolean: '1', 'true', 'yes', 'on' (case-insensitive)
    count as true; anything else (including missing) uses `default`."""
    return os.environ.get(name, default).strip().lower() in _TRUTHY_ENV


def _lambda_w() -> float:
    raw = os.environ.get(LAMBDA_W_ENV, "")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return LAMBDA_W_DEFAULT


def _leaf_floor() -> float:
    raw = os.environ.get(LEAF_FLOOR_ENV, "")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return LEAF_FLOOR_DEFAULT


def _reduced_floor() -> float:
    raw = os.environ.get(REDUCED_FLOOR_ENV, "")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return REDUCED_FLOOR_DEFAULT


def _delta_w_enabled() -> bool:
    return _env_truthy(DELTA_W_ENABLE_ENV)


def _attack_bonus_enabled() -> bool:
    """Gate for the ATTACK branch of the endgame bonus. Default ON; set to
    "0" to test DEFEND-only attribution (mirrors COORD_DEFEND_BONUS)."""
    return _env_truthy(ATTACK_BONUS_ENV)


def _defend_bonus_enabled() -> bool:
    """Gate for the DEFEND branch of the endgame bonus. Default on; set to
    "0" to test ATTACK-only attribution (used to isolate whether DEFEND's
    opp-independent magnitude is over-weighting defense).
    """
    return _env_truthy(DEFEND_BONUS_ENV)


def _demand_spread_enabled() -> bool:
    """Gate for demand-spread mixing. Default ON; set COORD_DEMAND_SPREAD=0
    to revert to pure-tier2 scoring (mixing_weight stays at 1.0 always)."""
    return _env_truthy(DEMAND_SPREAD_ENABLE_ENV)


def _opp_capacity_factor() -> float:
    raw = os.environ.get(OPP_CAPACITY_FACTOR_ENV, "")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return OPP_CAPACITY_FACTOR_DEFAULT


class BundleKind(Enum):
    ATTACK = "attack"
    DEFEND = "defend"
    RECAPTURE = "recap"


@dataclass(frozen=True)
class Leg:
    src_id: int
    ships: int
    angle: float
    wait_N: int
    eta: int

    @property
    def arrival_step(self) -> int:
        return self.wait_N + self.eta


@dataclass(frozen=True)
class Bundle:
    target_id: int
    arrival_step: int
    legs: tuple[Leg, ...]
    kind: BundleKind
    cheap_score: float = 0.0
    # tier2_score is the leaf-Δ from `score_candidate_v4_joint` — the
    # tactical signal (ships gained/lost over a ~25-turn rollout). It does
    # NOT include the strategic endgame bonus; that lives in
    # `endgame_bonus` so the Lagrangian can apply a leaf-floor guard
    # against tactically-losing bundles that the bonus would otherwise
    # rescue. The composite is read via `_composite_score(b)`.
    tier2_score: float = 0.0
    # λ_W × ΔW (closed-form winning_margin contribution; see _endgame.py).
    # Computed by `_bundle_endgame_bonus`; zero when env var
    # `COORD_DELTA_W=0` or when the bundle's kind has its own gate off.
    endgame_bonus: float = 0.0
    # Demand-spread mixing weight ∈ [0, 1]:
    #   1.0 = use full tier2_score (opp fully defends this bundle)
    #   0.0 = use cheap_score (opp ignores; undefended Δ-favor)
    # Populated by `_compute_mixing_weights` in `tier2_score_bundles`
    # when `COORD_DEMAND_SPREAD=1` (default). Default 1.0 = backward-
    # compat with code paths that construct Bundles without setting it.
    mixing_weight: float = 1.0


def _admissible_fire_now(src, tgt, angle: float, ships: int, world,
                         wait_N: int = 0) -> bool:
    """Trajectory admissibility — fleet must reach target (not sun/oob/
    intercepted) and the target (if a comet) must still exist at arrival.

    H44 fix (Day 12, 2026-05-22): wait_N>0 legs are checked via
    predict_fleet_fate's wait_N parameter (lib/trajectory.py:81) which
    pre-rotates source position and planet positions before ray-casting.
    The previous bypass left ~65% of live in-flight deaths uncaught
    (H44 audit, btjeK 2026-05-20; fix in
    claude/extract-physics-trajectory-Vjaz9 commit c6a0c80).
    """
    fate = predict_fleet_fate(
        src, tgt, float(angle), int(ships), world, wait_N=int(wait_N),
    )
    if fate.outcome != "target":
        return False
    if int(tgt.id) in world.comet_ids:
        life = comet_remaining_lifetime(int(tgt.id), world)
        if life is None or life <= int(fate.step):
            return False
    return True


def _legs_for_pair(src, tgt, world, model, me: int, omega: float,
                   baseline_len: int) -> list[Leg]:
    """Build all candidate legs for one (src, tgt) — ship-count variants
    + wait-grid variants, with per-leg filters applied. Reuses minimal's
    helpers verbatim.
    """
    legs: list[Leg] = []

    # Fire-now ship-count variants.
    for ships in enumerate_ship_counts(src, tgt, model, omega, me, world):
        if ships < MIN_FLEET_SIZE or ships > int(src.ships):
            continue
        angle, eta = aim_and_eta(src, tgt, ships, omega, world=world)
        horizon = max(eta + SIM_SETTLE_TURNS, 1)
        if horizon >= baseline_len:
            continue
        cheap = cheap_marginal_value(
            src, tgt, ships, eta, world, model, me, wait_N=0,
        )
        if cheap <= CHEAP_REJECT_THRESHOLD:
            continue
        if not _admissible_fire_now(src, tgt, angle, ships, world):
            continue
        if not _source_survives_launch(src, int(ships), 0, world, model, me):
            continue
        if not _target_holdable_after_capture(
            src, tgt, int(ships), 0, int(eta), world, model, me,
        ):
            continue
        if not _target_cost_parity_ok(
            src, tgt, int(ships), 0, int(eta), world, model, me,
        ):
            continue
        legs.append(Leg(
            src_id=int(src.id), ships=int(ships),
            angle=float(angle), wait_N=0, eta=int(eta),
        ))

    # Wait-grid variants. H44 fix (Day 12, 2026-05-22): admissibility
    # check now runs for wait_N>0 too via _admissible_fire_now's wait_N
    # parameter. The earlier bypass left ~65% of live in-flight deaths
    # uncaught (H44 audit + claude/extract-physics-trajectory-Vjaz9
    # c6a0c80).
    for w_ships, w_wait, w_angle, w_eta in wait_then_fire_variants(
        src, tgt, model, omega, me, world=world,
    ):
        w_horizon = max(w_wait + w_eta + SIM_SETTLE_TURNS, 1)
        if w_horizon >= baseline_len:
            continue
        w_cheap = cheap_marginal_value(
            src, tgt, w_ships, w_eta, world, model, me, wait_N=w_wait,
        )
        if w_cheap <= CHEAP_REJECT_THRESHOLD:
            continue
        if not _admissible_fire_now(
            src, tgt, w_angle, w_ships, world, wait_N=w_wait,
        ):
            continue
        if not _source_survives_launch(
            src, int(w_ships), int(w_wait), world, model, me,
        ):
            continue
        if not _target_holdable_after_capture(
            src, tgt, int(w_ships), int(w_wait), int(w_eta),
            world, model, me,
        ):
            continue
        if not _target_cost_parity_ok(
            src, tgt, int(w_ships), int(w_wait), int(w_eta),
            world, model, me,
        ):
            continue
        legs.append(Leg(
            src_id=int(src.id), ships=int(w_ships),
            angle=float(w_angle), wait_N=int(w_wait), eta=int(w_eta),
        ))

    return legs


def _cluster_arrival_windows(legs: list[Leg],
                             slack: int = ARRIVAL_WINDOW_SLACK) -> list[list[Leg]]:
    """For each leg as anchor, return the set of legs whose arrival_step is
    in [anchor.arrival_step, anchor.arrival_step + slack].

    Over-enumerates (one window per anchor) so no admissible bundle is
    missed; dedup happens at the bundle-emission step via frozenset of leg
    identities.
    """
    if not legs:
        return []
    legs_sorted = sorted(legs, key=lambda L: L.arrival_step)
    windows: list[list[Leg]] = []
    for i, anchor in enumerate(legs_sorted):
        upper = anchor.arrival_step + slack
        window: list[Leg] = []
        for j in range(i, len(legs_sorted)):
            if legs_sorted[j].arrival_step > upper:
                break
            window.append(legs_sorted[j])
        if window:
            windows.append(window)
    return windows


def _emit_subsets(window: list[Leg], max_size: int) -> list[tuple[Leg, ...]]:
    """All subsets of `window` of size 1..max_size with NO source repeats."""
    out: list[tuple[Leg, ...]] = []
    for r in range(1, max_size + 1):
        for subset in combinations(window, r):
            seen_srcs = set()
            ok = True
            for leg in subset:
                if leg.src_id in seen_srcs:
                    ok = False
                    break
                seen_srcs.add(leg.src_id)
            if ok:
                out.append(subset)
    return out


def enumerate_attack_bundles(my_planets, target_pool, world, model,
                             me: int, omega: float,
                             baseline_len: int = MAX_HORIZON + 1,
                             max_bundle_size: int = MAX_BUNDLE_SIZE,
                             deadline: float | None = None,
                             ) -> list[Bundle]:
    """Enumerate multi-source ATTACK bundles for each non-own target.

    For each target × arrival-window × subset-of-nearest-N-reachable-sources
    (size 1..MAX_BUNDLE_SIZE):

    1. Build candidate legs via minimal's per-source-per-target machinery
       (ship-count + wait-grid + per-leg filters).
    2. Cluster legs by arrival_step into windows of width
       ARRIVAL_WINDOW_SLACK.
    3. Per window, enumerate subsets where no source repeats.
    4. Emit one Bundle per unique subset (frozenset-of-legs dedup).

    `deadline` is a `time.perf_counter()` value; when reached, the
    per-(src, tgt) loop short-circuits and the rest of the function
    runs with whatever legs were built. Returning fewer bundles is
    strictly better than the agent emitting no moves at all
    (timing probe 2026-05-22 showed 84% idle turns when this loop
    was unbounded — total p50 wallclock 722ms vs 600ms budget).

    cheap_score and tier2_score are zeroed; populated by later passes.
    """
    legs_by_target: dict[int, list[Leg]] = defaultdict(list)
    deadline_hit = False
    for tgt in target_pool:
        if deadline_hit:
            break
        if int(tgt.owner) == me:
            continue  # ATTACK targets non-own only
        # Limit to nearest-N reachable sources per target.
        sources = nearest_k(my_planets, tgt, NEAREST_SOURCES_PER_TARGET)
        for src in sources:
            if deadline is not None and time.perf_counter() > deadline:
                deadline_hit = True
                break
            if int(src.ships) < MIN_FLEET_SIZE:
                continue
            if int(src.id) == int(tgt.id):
                continue
            legs_by_target[int(tgt.id)].extend(
                _legs_for_pair(src, tgt, world, model, me, omega, baseline_len),
            )

    seen: set[tuple[int, frozenset]] = set()
    bundles: list[Bundle] = []
    for tgt_id, legs in legs_by_target.items():
        if not legs:
            continue
        windows = _cluster_arrival_windows(legs, slack=ARRIVAL_WINDOW_SLACK)
        for window in windows:
            for subset in _emit_subsets(window, max_bundle_size):
                key_legs = frozenset(
                    (L.src_id, L.ships, L.wait_N, L.eta) for L in subset
                )
                key = (tgt_id, key_legs)
                if key in seen:
                    continue
                seen.add(key)
                arrival_step = max(L.arrival_step for L in subset)
                bundles.append(Bundle(
                    target_id=tgt_id,
                    arrival_step=arrival_step,
                    legs=tuple(subset),
                    kind=BundleKind.ATTACK,
                ))
    return bundles


def enumerate_defend_bundles(my_planets, world, model, me: int, omega: float,
                             baseline_len: int = MAX_HORIZON + 1,
                             max_bundle_size: int = MAX_BUNDLE_SIZE,
                             deadline: float | None = None,
                             ) -> list[Bundle]:
    """Enumerate multi-source DEFEND bundles for each own planet under threat.

    `time_to_enemy_threat` (not `incoming_enemy_eta`) captures BOTH in-flight
    enemy fleets AND potential launches from nearby enemy planets — matches
    the threat surface that minimal's `threatened_mine` filter uses
    (agents/minimal/main.py L1353-1356). `incoming_enemy_eta` alone would
    miss preemptive defense.

    For each threatened own planet:
    1. Find peer source planets via nearest-N.
    2. Build candidate legs via Day-1 `_legs_for_pair` — reuses every
       per-leg filter including `_source_survives_launch` (so we don't
       drain a peer that's itself threatened).
    3. Hard-filter legs whose arrival_step >= enemy_eta (too slow).
    4. Enumerate subsets size 1..MAX_BUNDLE_SIZE with no source repeats.

    NO arrival-window clustering — same-owner reinforcements arriving on
    DIFFERENT turns add to garrison additively. The only constraint is
    `arrival_step < enemy_eta`, enforced per-leg.

    cheap_score and tier2_score are zeroed; populated by later passes.
    """
    bundles: list[Bundle] = []
    seen: set[tuple[int, frozenset]] = set()
    deadline_hit = False
    for own in my_planets:
        if deadline_hit:
            break
        enemy_eta = model.time_to_enemy_threat(int(own.id), me, world)
        if enemy_eta is None or enemy_eta > DEFEND_LOOKAHEAD:
            continue
        peers = nearest_k(my_planets, own, NEAREST_SOURCES_PER_TARGET)
        all_legs: list[Leg] = []
        for peer in peers:
            if deadline is not None and time.perf_counter() > deadline:
                deadline_hit = True
                break
            if int(peer.id) == int(own.id):
                continue
            if int(peer.ships) < MIN_FLEET_SIZE:
                continue
            pair_legs = _legs_for_pair(peer, own, world, model, me, omega,
                                       baseline_len)
            for leg in pair_legs:
                if leg.arrival_step >= enemy_eta:
                    continue
                all_legs.append(leg)
        if not all_legs:
            continue
        for subset in _emit_subsets(all_legs, max_bundle_size):
            key_legs = frozenset(
                (L.src_id, L.ships, L.wait_N, L.eta) for L in subset
            )
            key = (int(own.id), key_legs)
            if key in seen:
                continue
            seen.add(key)
            arrival_step = max(L.arrival_step for L in subset)
            bundles.append(Bundle(
                target_id=int(own.id),
                arrival_step=arrival_step,
                legs=tuple(subset),
                kind=BundleKind.DEFEND,
            ))
    return bundles


def enumerate_recapture_bundles(my_planets, target_pool, world, model,
                                me: int, omega: float,
                                baseline_len: int = MAX_HORIZON + 1
                                ) -> list[Bundle]:
    """v1 stub — recapture deferred to v2.

    "Recently lost" detection requires inter-turn state (the previous-turn
    owner of each planet), which the stateless agent doesn't track.
    `WorldModel.owner_at` returns FUTURE predictions, not past history.

    The high-value cheap targets that recapture would prioritise —
    currently enemy/neutral planets with low garrison and high production
    — are ALREADY enumerated by `enumerate_attack_bundles`. The
    kind=RECAPTURE label would be metadata-only; bundle composition would
    not change.

    PI-approved deferral 2026-05-22. Revisit in v2 if recapture-priority
    scoring turns out to add value, at which point a small module-level
    dict cache keyed by planet_id storing previous-turn-owner can be
    added with bounded blast radius.
    """
    return []


# ---------------------------------------------------------------------------
# Cheap-filter — synthesised-obs Δ-favor without rollout.
# ---------------------------------------------------------------------------

def _resolve_target_post_bundle(bundle: "Bundle", target_planet,
                                 pred_owner, pred_ships, me: int
                                 ) -> tuple[int, int]:
    """Apply combat-rule-1 between the bundle and the predicted target
    state at bundle.arrival_step. Returns (post_owner, post_ships).

    For ATTACK / RECAPTURE: if bundle force > predicted defender,
    capture; else defender absorbs and we lose the bundle's ships.

    For DEFEND: my reinforcement ADDS to garrison if predicted owner
    at arrival is still me. If we'd already lost the planet by arrival,
    treat as recapture (combat rule 1 vs the new owner's garrison).
    """
    bundle_force = sum(L.ships for L in bundle.legs)
    pred_owner_val = (
        pred_owner if pred_owner is not None else int(target_planet.owner)
    )
    pred_ships_val = int(
        pred_ships if pred_ships is not None else target_planet.ships
    )

    if bundle.kind == BundleKind.DEFEND:
        if pred_owner_val == me:
            return me, pred_ships_val + bundle_force
        # Already lost by arrival_step (rare given DEFEND deadline filter,
        # but possible if threat ETA is very tight) — handle as recapture.
        if bundle_force > pred_ships_val:
            return me, bundle_force - pred_ships_val
        return pred_owner_val, pred_ships_val - bundle_force

    # ATTACK / RECAPTURE
    if pred_owner_val == me:
        return me, pred_ships_val + bundle_force
    if bundle_force > pred_ships_val:
        return me, bundle_force - pred_ships_val
    return pred_owner_val, pred_ships_val - bundle_force


def _cheap_synth_step(bundle: "Bundle", world, model, me: int) -> int:
    """Choose the step at which to synthesise for the cheap-filter.

    For ATTACK / RECAPTURE: bundle.arrival_step (post-capture state).

    For DEFEND: max(arrival_step, enemy_eta + 1). The enemy's hit at
    `enemy_eta` resolves combat; synthesising at `enemy_eta + 1` lets
    `model.owner_at` show the post-combat ownership (likely enemy if no
    reinforcement) so `_resolve_target_post_bundle` correctly computes
    "did my reinforcement flip the outcome?". Synthesising at
    `arrival_step` instead misses this — at arrival_step the planet is
    still ours and the bundle is just garrison-padding, giving DEFEND
    an under-priced score relative to ATTACK.
    """
    if bundle.kind == BundleKind.DEFEND:
        enemy_eta = model.time_to_enemy_threat(
            int(bundle.target_id), me, world,
        )
        if enemy_eta is not None and enemy_eta >= 0:
            return max(int(bundle.arrival_step), int(enemy_eta) + 1)
    return int(bundle.arrival_step)


def _synthesise_obs_at_step(world, model, me: int, step_offset: int,
                             target_id: int | None = None,
                             target_override: tuple[int, int] | None = None,
                             source_subtractions: dict[int, int] | None = None,
                             ) -> dict:
    """Build a synthesised obs dict at `step_offset` from now.

    With both `target_override` and `source_subtractions` None, this is
    the IDLE projection — what the world would look like at step_offset
    if nobody acted.

    With them set, it reflects a bundle's effect: target planet uses
    `target_override` (owner, ships); each source planet has its idle
    projection reduced by `source_subtractions[src_id]` committed ships.
    """
    if source_subtractions is None:
        source_subtractions = {}

    planets_synth: list[tuple] = []
    for pid, p in world.planets_by_id.items():
        pred_owner = model.owner_at(pid, step_offset)
        pred_ships = model.ships_at(pid, step_offset)
        if pid == target_id and target_override is not None:
            owner = int(target_override[0])
            ships = max(0, int(target_override[1]))
        elif pid in source_subtractions:
            base_owner = (
                int(pred_owner) if pred_owner is not None else int(p.owner)
            )
            base_ships = (
                int(pred_ships) if pred_ships is not None else int(p.ships)
            )
            owner = base_owner
            ships = max(0, base_ships - source_subtractions[pid])
        else:
            owner = (
                int(pred_owner) if pred_owner is not None else int(p.owner)
            )
            ships = max(
                0, int(pred_ships) if pred_ships is not None else int(p.ships),
            )
        planets_synth.append((
            int(pid), owner, float(p.x), float(p.y), float(p.radius),
            int(ships), int(p.production),
        ))

    return {
        "player": int(me),
        "step": int(world.step) + step_offset,
        "planets": planets_synth,
        "fleets": [],
        "angular_velocity": float(getattr(world, "omega", 0.0)),
        "comet_planet_ids": list(getattr(world, "comet_ids", [])),
    }


def _synthesise_post_arrival_obs(bundle: "Bundle", world, model,
                                  me: int,
                                  step_offset: int | None = None) -> dict:
    """Synthesised obs reflecting the bundle's effect at the cheap-filter
    synthesis step (default: `_cheap_synth_step(bundle, ...)`).

    Used by `_bundle_cheap_delta`; thin wrapper over
    `_synthesise_obs_at_step` that applies the bundle's combat resolution
    on the target and subtracts committed ships from sources.
    """
    if step_offset is None:
        step_offset = _cheap_synth_step(bundle, world, model, me)

    sources_committed: dict[int, int] = defaultdict(int)
    for leg in bundle.legs:
        sources_committed[int(leg.src_id)] += int(leg.ships)

    target_p = world.planets_by_id.get(int(bundle.target_id))
    if target_p is None:
        # Defensive: synthesise idle if target missing.
        return _synthesise_obs_at_step(world, model, me, step_offset)
    pred_owner = model.owner_at(int(bundle.target_id), step_offset)
    pred_ships = model.ships_at(int(bundle.target_id), step_offset)
    post_owner, post_ships = _resolve_target_post_bundle(
        bundle, target_p, pred_owner, pred_ships, me,
    )

    return _synthesise_obs_at_step(
        world, model, me, step_offset,
        target_id=int(bundle.target_id),
        target_override=(int(post_owner), int(post_ships)),
        source_subtractions=dict(sources_committed),
    )


def _production_balance_pv(planets: list, me: int, step: int,
                            horizon: int = MAX_HORIZON) -> float:
    """PV-discounted (my_prod − opp_prod) × pv_horizon over the rollout
    look-ahead. Mirrors `composite_capture_value`'s gated PV term so the
    cheap filter sees the future value of producing planets WITHOUT
    relying on the rollout (Tier-2 captures it via K-step sim; the
    cheap filter is rollout-free).

    The default composite head gates this term behind
    COMPOSITE_PRODUCTION_PV=1 (off in production). The cheap filter adds
    it explicitly because without it, capturing equal-production planets
    yields favor_hybrid Δ ≈ 0, indistinguishable from defender-erosion
    attacks.
    """
    my_prod = 0.0
    opp_prod = 0.0
    for p in planets:
        owner = int(p[1])
        prod = float(p[6])
        if owner == me:
            my_prod += prod
        elif owner >= 0:
            opp_prod += prod
    pv = pv_horizon(int(step), 0, gamma=GAMMA, t_total=EPISODE_STEPS)
    # Truncate the PV to the rollout horizon — Tier-2 sees ~MAX_HORIZON
    # turns of production; the cheap filter should approximate the same
    # truncated value, not the full 99-turn pv. This keeps cheap and
    # Tier-2 scores in similar magnitude ranges.
    truncated_pv = min(pv, float(horizon))
    return (my_prod - opp_prod) * truncated_pv


def _estimate_threat_strength(target_id: int, enemy_eta: int,
                               world, model, me: int) -> float:
    """Estimate the enemy ship force impacting `target_id` at `enemy_eta`,
    accounting for BOTH in-flight enemy fleets AND potential launches
    from nearby enemy planets. Mirrors minimal's `capture_size` own-target
    branch (agents/minimal/main.py L242-263) so the cheap-filter values
    defense by the same threat surface that triggered enumeration.
    """
    enemy_inflight = sum(
        sh
        for (eta_arr, owner, sh) in model.ledger.get(int(target_id), [])
        if owner != me and eta_arr <= int(enemy_eta) + WAVE_LOOKAHEAD
    )
    enemy_potential = 0.0
    if enemy_inflight <= 0:
        best_enemy_ships = 0.0
        best_enemy_prod = 0.0
        for p in world.planets_by_id.values():
            if int(p.owner) < 0 or int(p.owner) == me:
                continue
            if int(p.ships) > best_enemy_ships:
                best_enemy_ships = float(p.ships)
                best_enemy_prod = float(p.production)
        enemy_potential = best_enemy_ships + best_enemy_prod * float(enemy_eta)
    return float(max(enemy_inflight, enemy_potential))


def _defend_cheap_delta(bundle: "Bundle", world, model, me: int) -> float:
    """Cheap Δ for DEFEND bundles. Uses minimal's threat-strength
    formulation (in-flight + potential enemy launch) rather than the
    synthesised-obs approach used for ATTACK.

    Rationale: `WorldModel` only simulates in-flight fleets. For
    preemptive defense (no enemy fleet airborne yet, just an adjacent
    enemy planet), the model's idle projection still shows me owning
    the planet at enemy_eta + 1, so the synthesised-obs approach
    under-prices the bundle to ~0. The explicit threat-strength
    calculation matches what triggered the bundle's enumeration in the
    first place (`time_to_enemy_threat`) and credits the bundle with
    PV-discounted production preservation IF it covers the shortfall.

    Value model (for the cheap-filter only — Tier-2 does the precise
    leaf-Δ via a real rollout):
      - shortfall = enemy_strength − (target.ships + target.prod × eta)
      - if bundle covers shortfall: + target.production × truncated PV
      - if partial coverage: scaled fraction of full save value
      - minus epsilon × total ships committed
    """
    target_id = int(bundle.target_id)
    target_p = world.planets_by_id.get(target_id)
    if target_p is None:
        return -float("inf")

    enemy_eta = model.time_to_enemy_threat(target_id, me, world)
    if enemy_eta is None:
        return 0.0

    enemy_strength = _estimate_threat_strength(
        target_id, enemy_eta, world, model, me,
    )
    my_garrison_at_eta = (
        float(target_p.ships) + float(target_p.production) * float(enemy_eta)
    )
    shortfall = max(0.0, enemy_strength - my_garrison_at_eta + 1.0)

    bundle_force = float(sum(L.ships for L in bundle.legs))
    coverage = (
        1.0 if shortfall <= 0
        else min(1.0, bundle_force / shortfall)
    )

    pv_full = pv_horizon(
        int(world.step), int(enemy_eta),
        gamma=GAMMA, t_total=EPISODE_STEPS,
    )
    truncated_pv = min(pv_full, float(MAX_HORIZON))
    save_value = coverage * float(target_p.production) * truncated_pv

    return save_value - CHEAP_OPPORTUNITY_COST * bundle_force


def _attack_cheap_delta(bundle: "Bundle", world, model, me: int,
                        num_seats: int) -> float:
    """Cheap Δ for ATTACK / RECAPTURE bundles via synthesised-obs.

    Δ = (favor_with_bundle + prod_pv_with_bundle)
      − (favor_idle      + prod_pv_idle)
      − CHEAP_OPPORTUNITY_COST × total ships committed

    Both synthesis points use the same step (`bundle.arrival_step`) so
    time-passing effects cancel; the residual is the bundle's effect.
    Production-PV is added explicitly because favor_hybrid's PV term is
    gated off in production.
    """
    step_offset = _cheap_synth_step(bundle, world, model, me)

    idle_obs = _synthesise_obs_at_step(world, model, me, step_offset)
    idle_favor = favor_hybrid(idle_obs, me, num_seats, gamma=GAMMA)
    idle_pv = _production_balance_pv(
        idle_obs["planets"], me, int(idle_obs["step"]),
    )

    bundle_obs = _synthesise_post_arrival_obs(
        bundle, world, model, me, step_offset=step_offset,
    )
    bundle_favor = favor_hybrid(bundle_obs, me, num_seats, gamma=GAMMA)
    bundle_pv = _production_balance_pv(
        bundle_obs["planets"], me, int(bundle_obs["step"]),
    )

    total_ships = sum(L.ships for L in bundle.legs)
    return (
        (bundle_favor + bundle_pv) - (idle_favor + idle_pv)
        - CHEAP_OPPORTUNITY_COST * float(total_ships)
    )


def _bundle_cheap_delta(bundle: "Bundle", world, model, me: int,
                        num_seats: int,
                        current_favor: float | None = None) -> float:
    """Cheap Δ-favor dispatcher (rollout-free). DEFEND bundles use the
    explicit threat-strength formulation; ATTACK / RECAPTURE use the
    synthesised-obs formulation. The `current_favor` parameter is kept
    for backward compatibility with earlier signatures; both paths
    compare against idle-at-same-step internally.
    """
    if bundle.kind == BundleKind.DEFEND:
        return _defend_cheap_delta(bundle, world, model, me)
    return _attack_cheap_delta(bundle, world, model, me, num_seats)


def cheap_filter_bundles(bundles: list[Bundle], world, model, me: int,
                          num_seats: int, K: int = CHEAP_FILTER_TOP_K
                          ) -> list[Bundle]:
    """Score every bundle by `_bundle_cheap_delta`, return top-K by score.

    Returns bundles with `cheap_score` field populated. tier2_score
    remains zero until the Day-5 Tier-2 pass.
    """
    if not bundles:
        return []
    scored: list[Bundle] = []
    for b in bundles:
        score = _bundle_cheap_delta(b, world, model, me, num_seats)
        scored.append(replace(b, cheap_score=score))
    scored.sort(key=lambda b: -b.cheap_score)
    return scored[:K]


# ---------------------------------------------------------------------------
# Tier-2 scoring — bundle → leaf-Δ via score_candidate_v4_joint.
# ---------------------------------------------------------------------------

def _bundle_to_launches(bundle: "Bundle", planets_by_id: dict
                       ) -> list[tuple] | None:
    """Convert Bundle.legs to score_candidate_v4_joint's `launches` format:
    [(src_planet, tgt_planet, ships, angle, wait_N), ...].

    Returns None if any leg's source or the target isn't in
    `planets_by_id` (defensive — shouldn't happen for bundles built from
    current world state, but covers stale-bundle edge cases).
    """
    tgt = planets_by_id.get(int(bundle.target_id))
    if tgt is None:
        return None
    launches: list[tuple] = []
    for leg in bundle.legs:
        src = planets_by_id.get(int(leg.src_id))
        if src is None:
            return None
        launches.append(
            (src, tgt, int(leg.ships), float(leg.angle), int(leg.wait_N)),
        )
    return launches


def _strongest_opp(world, me: int, num_seats: int) -> int | None:
    """Opp with highest `opp_pool`. Used for neutral-target attribution
    in 4P (where the bundle's target is neutral, so no natural
    opponent-of-record — pick the strongest surviving opp)."""
    candidates = [o for o in range(int(num_seats)) if o != int(me)]
    if not candidates:
        return None
    return max(candidates, key=lambda o: endgame_opp_pool(world, o))


def _largest_threat_owner(target_id: int, model, me: int) -> int | None:
    """Owner of the largest inbound enemy fleet at target_id.

    `model.ledger[planet_id]` is a list of `(eta, owner, ships)` tuples
    (see `lib/world_model.build_arrival_ledger`). Filter to non-me
    non-neutral, pick the entry with max ships."""
    ledger = getattr(model, "ledger", None)
    if ledger is None:
        return None
    entries = ledger.get(int(target_id)) or []
    threats = [
        (int(owner), int(ships))
        for (_eta, owner, ships) in entries
        if int(owner) != int(me) and int(owner) >= 0
    ]
    if not threats:
        return None
    return max(threats, key=lambda t: t[1])[0]


def _bundle_endgame_bonus(bundle: Bundle, world, model, me: int,
                          num_seats: int) -> float:
    """λ_W × ΔW(bundle). Closed-form winning-margin contribution.

    Per-bundle opponent-of-record attribution:
      - ATTACK: opp = current owner of target (or strongest opp for
        neutral targets in 4P).
      - DEFEND: opp = owner of largest inbound enemy fleet at target.
      - 2P: opp degenerates to the unique non-me seat automatically.

    `model=None` short-circuits to 0.0 — diagnostic probes / unit tests
    that don't need the endgame term can skip building a WorldModel.
    """
    if model is None or not _delta_w_enabled():
        return 0.0
    rem = endgame_remaining_turns(world)
    if rem <= 0:
        return 0.0
    target = world.planets_by_id.get(int(bundle.target_id))
    if target is None:
        return 0.0

    if bundle.kind == BundleKind.ATTACK:
        if not _attack_bonus_enabled():
            return 0.0
        cur_owner = int(target.owner)
        if cur_owner == int(me):
            return 0.0
        if cur_owner >= 0:
            opp_id = cur_owner
        else:
            opp_id = _strongest_opp(world, me, num_seats)
            if opp_id is None:
                return 0.0
        dw = bundle_delta_w_attack(target, int(me), int(opp_id), int(rem))
        return _lambda_w() * float(dw)

    if bundle.kind == BundleKind.DEFEND:
        if not _defend_bonus_enabled():
            return 0.0
        # Fail-fast guard: DEFEND only makes sense for own targets. The
        # formula in bundle_delta_w_defend also returns 0 if target.owner
        # != me, but checking here saves a ledger lookup for the (currently
        # impossible) case where an enumeration bug emits a DEFEND on a
        # non-own target.
        if int(target.owner) != int(me):
            return 0.0
        opp_threat = _largest_threat_owner(bundle.target_id, model, me)
        if opp_threat is None:
            return 0.0
        dw = bundle_delta_w_defend(target, int(me), int(opp_threat), int(rem))
        return _lambda_w() * float(dw)

    # RECAPTURE stub remains stub-priced (current enumerate_recapture_bundles
    # returns []).
    return 0.0


def tier2_score_bundles(bundles: list[Bundle], snap_base, me: int,
                        num_seats: int, world, model=None,
                        wallclock_ms: float = TIER2_BUDGET_MS,
                        omega: float = 0.0,
                        ) -> list[Bundle]:
    """Score each bundle via `score_candidate_v4_joint`, populate
    `tier2_score` (leaf-Δ only) and `endgame_bonus` (λ_W·ΔW), return
    bundles sorted descending by composite (`tier2_score + endgame_bonus`).

    Bundles that fail Tier-2 admissibility (sun/oob/comet_expired/
    path_blocked) are dropped — no point passing them to the Lagrangian.

    The leaf-Δ and the bonus are kept on separate fields so the Lagrangian
    can apply a leaf-floor guard against bundles whose tactical verdict is
    negative but whose strategic bonus drags them above zero. See plan
    §"Fix 2 — Bound Lagrangian acceptance against the leaf-Δ floor".

    Budget enforcement mirrors minimal's `choose_trajectory` pattern:
    probe per-rollout cost via `affordable_validate_cap`, compute
    `safe_deadline = deadline − per_cand_ms`, pre-bail before starting
    any bundle whose Tier-2 call would push past wallclock_ms.
    """
    if not bundles:
        return []

    t_start = time.perf_counter()
    deadline = t_start + wallclock_ms / 1000.0

    _, per_cand_ms = affordable_validate_cap(
        snap_base, me, num_seats, MAX_HORIZON,
        max(50.0, wallclock_ms),
        MIN_HORIZON,
    )
    safe_deadline = deadline - (per_cand_ms / 1000.0)

    baseline_horizon = MAX_HORIZON
    baseline_favors = build_trajectory_baseline(
        snap_base, me, num_seats, baseline_horizon,
    )

    planets_by_id = {p.id: p for p in world.planets_by_id.values()}

    scored: list[Bundle] = []
    for b in bundles:
        if time.perf_counter() > safe_deadline:
            break
        launches = _bundle_to_launches(b, planets_by_id)
        if launches is None:
            continue
        horizon = max(
            MIN_HORIZON,
            min(int(b.arrival_step) + SIM_SETTLE_TURNS, MAX_HORIZON - 1),
        )
        try:
            t2_score, t2_status = score_candidate_v4_joint(
                snap_base, launches, me, num_seats, world,
                baseline_favors, horizon=horizon,
            )
        except Exception:
            continue
        if t2_status != "scored":
            continue
        endgame = _bundle_endgame_bonus(b, world, model, me, num_seats)
        scored.append(replace(
            b,
            tier2_score=float(t2_score),
            endgame_bonus=float(endgame),
        ))

    # Demand-spread mixing: when our total demand exceeds opp's defensive
    # capacity, mixing_weight drops below 1.0 → composite shifts toward
    # cheap_score (less pessimistic) → Lagrangian admits more bundles.
    if _demand_spread_enabled() and scored:
        weights = _compute_mixing_weights(
            scored, world, model, me, num_seats, omega,
        )
        scored = [
            replace(b, mixing_weight=weights.get(i, 1.0))
            for i, b in enumerate(scored)
        ]

    scored.sort(key=lambda b: -_composite_score(b))
    return scored


def _composite_score(bundle: "Bundle") -> float:
    """Demand-spread mixed leaf + strategic endgame bonus.

    Effective leaf = mixing_weight × tier2_score + (1-mixing_weight) × cheap_score.
    When mixing_weight=1.0 (default; current behavior when DEMAND_SPREAD off or
    no responders) → composite = tier2 + endgame_bonus (same as pre-Option-3).
    When mixing_weight<1.0 → composite shifts toward cheap_score, reflecting
    that opp can't defend everywhere when our total demand exceeds capacity.

    `tier2_score` stays leaf-only (the tactical signal); `_composite_score`
    is the strategic + tactical full read used by the Lagrangian.
    """
    w = float(bundle.mixing_weight)
    leaf = w * float(bundle.tier2_score) + (1.0 - w) * float(bundle.cheap_score)
    return leaf + float(bundle.endgame_bonus)


# ---------------------------------------------------------------------------
# Demand-spread mixing — per-opp defensive capacity × per-bundle attention
# demand → mixing_weight per bundle. Captures "opp can't defend everywhere"
# so the Lagrangian sees less pessimistic scores when ensembles overwhelm
# the defender. Option 3 from 2026-05-22 design.
# ---------------------------------------------------------------------------

def _opp_defensive_capacity(world, me: int) -> dict[int, float]:
    """For each opp source planet, defensive throughput estimate.

    Simplest model: capacity[s] = opp_capacity_factor × s.ships. The factor
    (env `COORD_OPP_CAPACITY_FACTOR`, default 1.0) lets us tune empirically
    if 1.0 over- or under-estimates opp's actual defense-vs-offense split.

    Neutrals (owner == -1) excluded — they don't defend.
    """
    factor = _opp_capacity_factor()
    cap: dict[int, float] = {}
    for p in world.planets_by_id.values():
        if int(p.owner) != int(me) and int(p.owner) >= 0:
            cap[int(p.id)] = factor * float(p.ships)
    return cap


def _bundle_attention(bundle: "Bundle", world, model, me: int,
                       opp_sources: list, omega: float) -> dict[int, float]:
    """Per-opp-source attention demand for this bundle.

    An opp source S counts as a "responder" if S can reach the bundle's
    target T within DEMAND_REACH_WINDOW turns. Demand magnitude = total
    ships in the bundle (the threat S must counter).

    Closed-form via `aim_and_eta` — no rollout. Returns dict keyed by
    opp source id, value = demand contribution.
    """
    tgt = world.planets_by_id.get(int(bundle.target_id))
    if tgt is None or not opp_sources:
        return {}
    total_ships = sum(int(L.ships) for L in bundle.legs)
    if total_ships <= 0:
        return {}
    arrival = int(bundle.arrival_step)
    deadline = arrival + DEMAND_REACH_WINDOW
    out: dict[int, float] = {}
    for opp in opp_sources:
        try:
            _, eta = aim_and_eta(opp, tgt, total_ships, omega, world=world)
        except Exception:
            continue
        if int(eta) <= deadline:
            out[int(opp.id)] = float(total_ships)
    return out


def _compute_mixing_weights(
    scored: list["Bundle"], world, model, me: int, num_seats: int,
    omega: float,
) -> dict[int, float]:
    """Per-bundle mixing_weight ∈ [0, 1].

    For each opp source S:
      capacity_fraction[S] = min(1.0, capacity[S] / max(1.0, total_demand[S]))
    For each bundle B:
      mixing_weight[B] = mean of capacity_fraction[S] over S in
                        responders(B), weighted uniformly. If B has no
                        responders → mixing_weight = 1.0 (no spread; opp
                        can't reach, so opp's defensive bandwidth doesn't
                        constrain the leaf head's pessimism).

    Returns dict keyed by bundle INDEX in `scored` (since Bundle is
    frozen and not hashable to itself — index gives a stable handle).
    """
    if not scored:
        return {}
    # Opp source planets (excluding neutrals).
    opp_sources = [
        p for p in world.planets_by_id.values()
        if int(p.owner) != int(me) and int(p.owner) >= 0
    ]
    if not opp_sources:
        # No opps → no spread possible (defensive capacity is irrelevant).
        return {i: 1.0 for i, _ in enumerate(scored)}

    capacity = _opp_defensive_capacity(world, me)
    # Per-bundle attention demand: list[dict[opp_id, demand]] indexed by bundle.
    attentions: list[dict[int, float]] = []
    for b in scored:
        attentions.append(
            _bundle_attention(b, world, model, me, opp_sources, omega)
        )
    # Total demand per opp source across all candidate bundles.
    total_demand: dict[int, float] = defaultdict(float)
    for att in attentions:
        for opp_id, d in att.items():
            total_demand[opp_id] += d
    # Capacity fraction per opp source.
    capacity_fraction: dict[int, float] = {}
    for opp_id, cap in capacity.items():
        demand = total_demand.get(opp_id, 0.0)
        capacity_fraction[opp_id] = min(1.0, cap / max(1.0, demand))
    # Per-bundle mixing weight: uniform mean over responders' capacity_fractions.
    weights: dict[int, float] = {}
    for i, att in enumerate(attentions):
        if not att:
            weights[i] = 1.0
            continue
        fractions = [capacity_fraction.get(opp_id, 1.0) for opp_id in att]
        weights[i] = sum(fractions) / float(len(fractions))
    return weights


# ---------------------------------------------------------------------------
# Lagrangian clearing — shadow-priced bundle selection.
# ---------------------------------------------------------------------------

_LAGRANGIAN_CYCLE_EPS = 1e-9


def _reduced_score(bundle: "Bundle", lam: dict[int, float]) -> float:
    """(tier2_score + endgame_bonus) − shadow-price-weighted ship cost.

    Uses the COMPOSITE score (leaf-Δ + strategic bonus). The leaf-floor
    guard against tactically-losing bundles is applied separately in
    `_greedy_primal` — this function alone is not the right place
    because the dual update reads reduced-score gradients.
    """
    cost = sum(lam.get(int(L.src_id), 0.0) * float(L.ships) for L in bundle.legs)
    return _composite_score(bundle) - cost


def _used_ships_per_source(chosen: list["Bundle"]) -> dict[int, int]:
    """Sum of leg.ships per source planet across the chosen bundle set."""
    used: dict[int, int] = defaultdict(int)
    for b in chosen:
        for L in b.legs:
            used[int(L.src_id)] += int(L.ships)
    return used


def _greedy_primal(scored: list["Bundle"], lam: dict[int, float],
                   leaf_floor: float | None = None,
                   reduced_floor: float | None = None) -> list["Bundle"]:
    """Greedy primal at the current shadow prices.

    Sort bundles by reduced_score descending (composite score − shadow
    cost), take in order subject to:
    - reduced_score > reduced_floor (default 0.0 = positive-net-value
      only; set COORD_REDUCED_FLOOR=-1e9 to admit negative-composite
      bundles for ensemble-style emission).
    - tier2_score >= leaf_floor (TACTICAL viability — prevents the
      strategic endgame_bonus from rescuing a bundle whose 25-turn
      rollout was net-negative). Default floor read from env via
      `_leaf_floor()`.
    - no two bundles share a source (one launch per planet per turn)
    - no two bundles share a target (one bundle per target).

    Always feasible by construction.
    """
    lf = _leaf_floor() if leaf_floor is None else float(leaf_floor)
    rf = _reduced_floor() if reduced_floor is None else float(reduced_floor)
    ordered = sorted(scored, key=lambda b: -_reduced_score(b, lam))
    chosen: list["Bundle"] = []
    used_src: set[int] = set()
    used_tgt: set[int] = set()
    for b in ordered:
        if _reduced_score(b, lam) <= rf:
            break
        if float(b.tier2_score) < lf:
            continue  # tactically losing — bonus is not allowed to rescue
        if any(int(L.src_id) in used_src for L in b.legs):
            continue
        if int(b.target_id) in used_tgt:
            continue
        chosen.append(b)
        for L in b.legs:
            used_src.add(int(L.src_id))
        used_tgt.add(int(b.target_id))
    return chosen


def lagrangian_clear(scored: list["Bundle"], my_planets,
                     wallclock_ms: float = LAGRANGIAN_BUDGET_MS
                     ) -> list["Bundle"]:
    """Subgradient dual ascent on per-source ship-budget constraints.

    At each iteration:
      1. Greedy primal at the current λ (always feasible).
      2. Update λ via subgradient: raise prices where sources are over-
         utilised (> 80% of budget), lower otherwise. Floored at 0.
      3. Track the BEST primal-value-ever across iterations — return
         that, not the final iteration's solution. This handles the
         integer-program duality gap and dual oscillation cleanly.

    Termination: 2-cycle detection on primal_value (epsilon equality —
    additive endgame bonus introduces FP noise that exact-`==` would
    miss) OR LAGRANGIAN_MAX_ITERS OR wallclock deadline.

    Returns the best feasible bundle set found.
    """
    if not scored:
        return []

    deadline = time.perf_counter() + wallclock_ms / 1000.0
    src_budget = {int(p.id): max(1, int(p.ships)) for p in my_planets}
    lam: dict[int, float] = {pid: 0.0 for pid in src_budget}

    best_value = float("-inf")
    best_feasible: list["Bundle"] = []
    prev_values: list[float] = []

    for it in range(LAGRANGIAN_MAX_ITERS):
        chosen = _greedy_primal(scored, lam)

        # Primal value uses the COMPOSITE score (the actual objective).
        primal_value = sum(_composite_score(b) for b in chosen)
        if primal_value > best_value:
            best_value = primal_value
            best_feasible = chosen

        # Subgradient: g_s = used_s − target_util * budget_s.
        used = _used_ships_per_source(chosen)
        step = LAGRANGIAN_ALPHA0 / float(1 + it)
        for pid, budget in src_budget.items():
            g = float(used.get(pid, 0)) - LAGRANGIAN_TARGET_UTIL * float(budget)
            lam[pid] = max(0.0, lam[pid] + step * g)

        # Termination: 2-cycle on primal value (epsilon-tolerant).
        prev_values.append(primal_value)
        if (len(prev_values) >= 3
                and abs(prev_values[-1] - prev_values[-3]) < _LAGRANGIAN_CYCLE_EPS):
            break
        if time.perf_counter() > deadline:
            break

    return best_feasible


# ---------------------------------------------------------------------------
# Emission — bundles → env-format moves.
# ---------------------------------------------------------------------------

def _bundle_fire_now_viable(bundle: "Bundle", fire_now_legs: list["Leg"],
                            world, model, me: int) -> bool:
    """Stranded-singleton safeguard for mixed-wait bundles.

    A bundle whose legs include BOTH wait_N==0 (fire-now) and wait_N>0
    (reserved) emits only the fire-now subset this turn. If the bundle's
    expected outcome depends on ALL legs arriving together, firing only
    a strict subset wastes ships. This check verifies the fire-now
    subset would have been viable as a stand-alone bundle.

    DEFEND: always viable — extra ships on an own planet can't be a
    net-negative even if they're insufficient to cover the threat.

    ATTACK / RECAPTURE: combined fire-now ships must satisfy
    `_target_holdable_after_capture` standalone.
    """
    if bundle.kind == BundleKind.DEFEND:
        return True
    tgt = world.planets_by_id.get(int(bundle.target_id))
    if tgt is None:
        return False
    if not fire_now_legs:
        return False
    fire_now_ships = sum(int(L.ships) for L in fire_now_legs)
    src_ref_id = min(int(L.src_id) for L in fire_now_legs)
    src_ref = world.planets_by_id.get(src_ref_id)
    if src_ref is None:
        return False
    eta = max(int(L.eta) for L in fire_now_legs)
    return _target_holdable_after_capture(
        src_ref, tgt, int(fire_now_ships), 0, int(eta), world, model, me,
    )


def emit_bundle_actions(selected: list["Bundle"], world, model,
                        me: int) -> list[list]:
    """Convert Lagrangian-selected bundles to env-format moves.

    Receding-horizon: only wait_N==0 legs emit this turn; wait_N>0 legs
    are conceptually reserved but produce no immediate action (next turn
    re-clears with fresh state).

    Cross-bundle source-deduplication: one launch per source per turn
    (matches minimal's emit convention).

    Stranded-singleton safeguard: bundles with mixed wait_N legs only
    emit their fire-now subset if it's viable standalone — see
    `_bundle_fire_now_viable`.
    """
    moves: list[list] = []
    used_srcs: set[int] = set()
    for bundle in selected:
        fire_now_legs = [L for L in bundle.legs if int(L.wait_N) == 0]
        if not fire_now_legs:
            continue
        if len(fire_now_legs) < len(bundle.legs):
            if not _bundle_fire_now_viable(
                bundle, fire_now_legs, world, model, me,
            ):
                continue
        for leg in fire_now_legs:
            sid = int(leg.src_id)
            if sid in used_srcs:
                continue
            used_srcs.add(sid)
            moves.append([sid, float(leg.angle), int(leg.ships)])
    return moves


# ---------------------------------------------------------------------------
# Agent entry — orchestrates the full pipeline.
# ---------------------------------------------------------------------------

def agent(obs, configuration=None) -> list[list]:
    """Multi-source bundle coordinator. Pipeline per turn:

      1. Parse obs → world, model, snap_base, target_pool.
      2. Enumerate attack + defend bundles.
      3. Cheap-filter to top-K by closed-form Δ-favor.
      4. Tier-2 score via score_candidate_v4_joint (budget-aware).
      5. Lagrangian clearing — pick bundles with shadow-priced selection.
      6. Emit fire-now legs as env-format moves.

    Returns [] if no own planets, no enemy/neutral planets, or no
    positive-value bundles after Tier-2 scoring.
    """
    t_start = time.perf_counter()

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

    snap_base = fs_from_obs(obs, num_seats=num_seats)

    # Env-var knobs for Gate 1 singleton-parity (Day 8) and ablations.
    # COORD_MAX_BUNDLE_SIZE: int override of MAX_BUNDLE_SIZE (=3 default).
    #   Set to 1 to force singleton-only — reduces coord to "minimal-shape"
    #   for the structural-correctness gate.
    # COORD_DISABLE_DEFEND: '1' to skip defense enumeration entirely.
    #   Mirrors Day 8 / Gate 1 protocol (defense off → attack-only path).
    max_bundle_size = int(os.environ.get(
        "COORD_MAX_BUNDLE_SIZE", str(MAX_BUNDLE_SIZE),
    ))
    disable_defend = os.environ.get("COORD_DISABLE_DEFEND", "0") == "1"

    # Enumerate budget: leave at least 250ms for cheap_filter + Tier-2 +
    # Lagrangian + emit + safety margin. Empirically (timing probe
    # 2026-05-22) enumerate at p50=607ms ate the whole budget when
    # unbounded, causing 84% idle turns. Deadline-bounded enumeration
    # returns partial results — strictly better than emitting nothing.
    enumerate_budget_ms = max(
        100.0,
        float(WALLCLOCK_BUDGET_MS) - 250.0
        - (time.perf_counter() - t_start) * 1000.0,
    )
    enumerate_deadline = t_start + enumerate_budget_ms / 1000.0

    attacks = enumerate_attack_bundles(
        my_planets, other_planets, world, model, me, omega,
        max_bundle_size=max_bundle_size,
        deadline=enumerate_deadline,
    )
    if disable_defend:
        defends = []
    else:
        defends = enumerate_defend_bundles(
            my_planets, world, model, me, omega,
            max_bundle_size=max_bundle_size,
            deadline=enumerate_deadline,
        )
    all_bundles = attacks + defends
    if not all_bundles:
        return []

    cheap = cheap_filter_bundles(
        all_bundles, world, model, me, num_seats, K=CHEAP_FILTER_TOP_K,
    )

    # Budget-aware Tier-2: pass remaining wallclock, reserving ~60ms for
    # Lagrangian + emit + outer-env overhead.
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    tier2_budget_ms = max(
        50.0, float(WALLCLOCK_BUDGET_MS) - elapsed_ms - 60.0,
    )
    scored = tier2_score_bundles(
        cheap, snap_base, me, num_seats, world, model,
        wallclock_ms=tier2_budget_ms,
        omega=omega,
    )

    selected = lagrangian_clear(scored, my_planets=my_planets)
    return emit_bundle_actions(selected, world, model, me)
