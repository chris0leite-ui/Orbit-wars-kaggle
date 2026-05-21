"""baseline — clean modular re-implementation of v15 (live champion μ=1115.5).

Pipeline (per turn):
  1. proposer.propose       enumerate fire-now + multi-wait grid, cheap-rank,
                            dedup by (src, tgt, wait_band).
  2. chooser.build_idle_baseline   precompute favor under (me-idle, opp-reactive).
  3. chooser.choose         validate top candidates with fast_sim K-step rollout,
                            emit greedy non-dogpile moves.

Knobs (env var overrides, all optional):
  BASELINE_GAMMA              PV-discount γ for favor() and cheap-rank.   default 0.99
  BASELINE_WALLCLOCK_MS       per-turn validate budget (env actTimeout=1000).
                                                                          default 600
  ORBIT_WARS_PARITY_WALLCLOCK_MS    bundle-parity override (very large
                                    value disables mid-loop deadline bail
                                    so the agent is a pure function of obs).
"""

from __future__ import annotations

import math
import os

# Production default: hybrid value head (composite in 2P, A2-favor in 4P).
# `setdefault` lets local A/B drivers (fast.py) override via env var without
# patching source, while submission-bundle / Kaggle-runner sees hybrid out
# of the box. See agents/baseline/value.select_favor_fn for the dispatch.
os.environ.setdefault("BASELINE_VALUE_HEAD", "hybrid")

# Production default: trajectory chooser. v4 with wait_N>0 + wallclock
# budgeting hits 42/64 = 65.6pct Wlo=0.534 vs v15 (n=64), point-estimate
# +3pp over composite_a2's 40/64 = 62.5pct in the same A/B, with better
# max-turn-ms (1077 vs 1292). The trajectory path is deterministic on
# sun/oob/expired-comet failure modes (predict_fleet_fate filter) and
# was the architectural reframe completed in this session. Local A/B
# drivers can force the composite path by setting BASELINE_CHOOSER to
# any value other than "trajectory" (e.g. "composite").
os.environ.setdefault("BASELINE_CHOOSER", "trajectory")

# Direction B v3 (2026-05-18 PM): joint candidate enumeration enabled
# by default. 2P A/B: joint vs hybrid = 38/64 = 59.4pct, Wlo=0.471,
# Whi=0.705 (INCONCLUSIVE-but-positive). 2P-only gate in chooser
# (num_seats <= 2 check) preserves 4P behaviour (4P regressed without
# gate at 12.5pct first-place). Wallclock OK: bench max=891ms,
# p95=703ms, zero >1000ms. Set BASELINE_JOINT=0 to disable.
os.environ.setdefault("BASELINE_JOINT", "1")

# Kinematic precomputation table (Phase γ of
# /root/.claude/plans/do-it-thoroughly-consider-tingly-fox.md). Replaces
# the per-call position-rebuild inside predict_fleet_fate with a
# per-turn-cached lookup. Bit-parity verified by 564 brute-force
# (FleetFate-level) and 2 full-game byte-identical assertions
# (seeds 42, 7); wall-clock saves 47-114 ms/step in measured runs.
os.environ.setdefault("KINEMATIC_TABLE_ENABLED", "1")

# H1 — post-chooser idle drain (2026-05-18) — DISABLED BY DEFAULT.
# Audit `audit/replays/idle-trajectory-2026-05-17.md` measured 43.8pct
# isolated ship-turns in trajectory champion (mu=1271.8). H1 attempted
# to drain rear sources via post-chooser reinforce launches. A/B vs
# hybrid reference at n=32: **11/32 = 34.4pct, Wlo=0.204, max-ms=1528
# — FAIL**. The chooser's decision to leave rear planets idle is
# CORRECTLY calibrated reserve-holding; H1's forced emissions weaken
# defense without compensating capture-EV. Spatial-leaf head (commit
# b5f5296) failed for the same root cause. The 43.8 pct isolated is
# not a leak — it's correctly-held reserve. See audit/2026-05-18-
# spatial-leaf-negative-result.md and audit/2026-05-18-h1-idle-drain-
# negative-result.md. Default OFF; opt-in via BASELINE_IDLE_DRAIN=1.
IDLE_DRAIN_THRESHOLD = int(os.environ.get("BASELINE_IDLE_DRAIN_THRESHOLD", "30"))
IDLE_REAR_THRESHOLD = float(os.environ.get("BASELINE_IDLE_REAR_THRESHOLD", "35.0"))
IDLE_DRAIN_RESERVE = int(os.environ.get("BASELINE_IDLE_DRAIN_RESERVE", "5"))
IDLE_DRAIN_ENABLED = os.environ.get("BASELINE_IDLE_DRAIN", "0") == "1"

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from lib.fast_sim import from_obs as fs_from_obs
from lib.intent import World
from lib.world_model import WorldModel

# Import by explicit names so the bundler's per-line import-stripping regex
# can handle them. Single-line form is mandatory — the regex matches one
# line at a time, so multi-line parenthesised imports would leak their
# continuation lines as indented orphans. Friction tag
# `bundler-modular-agent-namespace-access-breaks-bundle` (2026-05-17).
from agents.baseline.chooser import build_idle_baseline, choose, WALLCLOCK_BUDGET_MS
from agents.baseline.proposer import propose, MAX_HORIZON, MIN_HORIZON


_PARITY_ENV_VAR = "ORBIT_WARS_PARITY_WALLCLOCK_MS"


def _as_dict(obs) -> dict:
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


def _num_seats(planets, fleets) -> int:
    max_owner = -1
    for p in planets:
        if int(p.owner) > max_owner:
            max_owner = int(p.owner)
    for f in fleets:
        if int(f.owner) > max_owner:
            max_owner = int(f.owner)
    return 4 if max_owner >= 2 else 2


def _wallclock_ms() -> float:
    override = os.environ.get(_PARITY_ENV_VAR)
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    try:
        return float(os.environ.get("BASELINE_WALLCLOCK_MS", WALLCLOCK_BUDGET_MS))
    except ValueError:
        return WALLCLOCK_BUDGET_MS


def _gamma() -> float:
    try:
        return float(os.environ.get("BASELINE_GAMMA", 0.99))
    except ValueError:
        return 0.99


def drain_idle_rear(moves, planets, my_id: int, world, model) -> list:
    """H1: append reinforce launches for rear sources the chooser didn't use.

    Idempotent post-chooser pass. Fires only when ALL of:
      - source is one of MY planets AND not in `moves`
      - source.ships > IDLE_DRAIN_THRESHOLD
      - source's min-distance to any non-our planet > IDLE_REAR_THRESHOLD
      - source has no enemy threat (model.time_to_enemy_threat is None)
      - there is an own planet strictly closer to the action than source
    Emits one launch toward that closer own planet, ships = source.ships
    minus IDLE_DRAIN_RESERVE. Each `move` is `[src_id, angle, ships]`.
    """
    if not IDLE_DRAIN_ENABLED:
        return moves
    used_srcs = set()
    for m in moves:
        try:
            used_srcs.add(int(m[0]))
        except (TypeError, IndexError):
            pass
    non_our_xy = [(float(p.x), float(p.y)) for p in planets
                  if int(p.owner) != my_id]
    if not non_our_xy:
        return moves
    my_planets = [p for p in planets if int(p.owner) == my_id]
    if len(my_planets) < 2:
        return moves  # no closer own target available

    def d_action(p):
        return min(math.hypot(float(p.x) - tx, float(p.y) - ty)
                   for tx, ty in non_our_xy)

    extras = []
    for src in my_planets:
        if int(src.id) in used_srcs:
            continue
        if int(src.ships) <= IDLE_DRAIN_THRESHOLD:
            continue
        src_d = d_action(src)
        if src_d <= IDLE_REAR_THRESHOLD:
            continue
        if model.time_to_enemy_threat(int(src.id), my_id, world) is not None:
            continue
        best_target = None
        best_d = src_d  # strict-less-than → require improvement
        for q in my_planets:
            if int(q.id) == int(src.id):
                continue
            qd = d_action(q)
            if qd >= best_d:
                continue
            best_d = qd
            best_target = q
        if best_target is None:
            continue
        ships = int(src.ships) - IDLE_DRAIN_RESERVE
        if ships < 1:
            continue
        angle = math.atan2(float(best_target.y) - float(src.y),
                           float(best_target.x) - float(src.x))
        extras.append([int(src.id), float(angle), int(ships)])
    return list(moves) + extras


def agent(obs, configuration=None):
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
    # Phase β opt-in: prime the kinematic precomputation singleton when
    # KINEMATIC_TABLE_ENABLED is set. No caller consumes it yet (Phase γ
    # wires predict_fleet_fate); this just confirms the per-turn build
    # runs without behaviour change. Plan reference:
    # /root/.claude/plans/do-it-thoroughly-consider-tingly-fox.md
    if os.environ.get("KINEMATIC_TABLE_ENABLED", "").strip().lower() in (
        "1", "true", "on", "yes",
    ):
        from lib.kinematic_table import begin_turn as _kt_begin_turn
        _kt_begin_turn(world)
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0))
    num_seats = _num_seats(planets, fleets)
    gamma = _gamma()
    wallclock_ms = _wallclock_ms()

    threatened_mine = [
        p for p in my_planets
        if model.time_to_enemy_threat(int(p.id), me, world) is not None
    ]
    target_pool = other_planets + threatened_mine

    snap_base = fs_from_obs(obs, num_seats=num_seats)

    # Trajectory-first chooser opt-in (2026-05-17). Deterministic
    # admissibility + single-tick combat prediction; no K-step rollout,
    # no leaf-value approximation. See knowledge-base/concepts/
    # trajectory-first-architecture.md. Default chooser remains the
    # K-step rollout for backward compat with the v15-line A/B baseline.
    if os.environ.get("BASELINE_CHOOSER", "").strip().lower() == "trajectory":
        # Trajectory chooser doesn't need baseline_favors (no idle baseline);
        # propose still wants a baseline_len for shape but value doesn't
        # affect the trajectory chooser's scoring.
        prerank = propose(
            my_planets, target_pool, world, model, me, omega,
            baseline_len=MAX_HORIZON + 1,
        )
        from agents.baseline.chooser_trajectory import choose_trajectory
        moves = choose_trajectory(
            snap_base, prerank, None,
            me, num_seats, wallclock_ms,
            MIN_HORIZON, MAX_HORIZON, gamma,
            world, model,
        )
        return drain_idle_rear(moves, planets, me, world, model)

    # LP chooser opt-in (2026-05-20 slice 10). Joint bipartite-
    # assignment over the whole turn's move-set. The LP replaces the
    # per-candidate greedy emit with a Hungarian/scipy solver. See
    # /root/.claude/plans/take-the-lens-of-magical-shore.md §16.
    if os.environ.get("BASELINE_CHOOSER", "").strip().lower() == "lp":
        prerank = propose(
            my_planets, target_pool, world, model, me, omega,
            baseline_len=MAX_HORIZON + 1,
        )
        # Same migration injection as the differential branch — the
        # LP needs the same candidate space to pick from.
        if os.environ.get("BASELINE_MIGRATION", "1").strip() != "0":
            from agents.baseline.migration_solver import propose_migrations
            migrations = propose_migrations(world, model, me)
            prerank = list(prerank) + list(migrations)
        from agents.baseline.chooser_lp import choose_lp
        moves = choose_lp(
            snap_base, prerank, None,
            me, num_seats, wallclock_ms,
            MIN_HORIZON, MAX_HORIZON, gamma,
            world, model,
        )
        return drain_idle_rear(moves, planets, me, world, model)

    # Differential chooser opt-in (2026-05-19 slice 8). Closed-form
    # WorldModel-projection leaf eval; no fast_sim rollout. See
    # /root/.claude/plans/take-the-lens-of-magical-shore.md §13.
    if os.environ.get("BASELINE_CHOOSER", "").strip().lower() == "differential":
        prerank = propose(
            my_planets, target_pool, world, model, me, omega,
            baseline_len=MAX_HORIZON + 1,
        )
        # Slice 9: append closed-form ship-migration candidates to
        # the prerank. Fills the missing "reposition ships toward
        # action" candidate class that the proposer doesn't emit.
        # The differential chooser detects migration tuples (tgt.owner
        # == me, no inbound threat) and uses their cheap_delta as the
        # score directly (Δ-favor projection would be 0 for own→own).
        # Default-on under BASELINE_CHOOSER=differential; opt-out via
        # BASELINE_MIGRATION=0 for ablation.
        if os.environ.get("BASELINE_MIGRATION", "1").strip() != "0":
            from agents.baseline.migration_solver import propose_migrations
            migrations = propose_migrations(world, model, me)
            prerank = list(prerank) + list(migrations)
        from agents.baseline.chooser_differential import choose_differential
        moves = choose_differential(
            snap_base, prerank, None,
            me, num_seats, wallclock_ms,
            MIN_HORIZON, MAX_HORIZON, gamma,
            world, model,
        )
        return drain_idle_rear(moves, planets, me, world, model)

    # ROI chooser opt-in (2026-05-19). Closed-form ROI prior + N-way
    # coalition + opp-modifier posterior; no fast_sim rollout. See
    # agents/baseline/chooser_roi.py and the plan at
    # /root/.claude/plans/okay-we-can-do-elegant-lampson.md.
    if os.environ.get("BASELINE_CHOOSER", "").strip().lower() == "roi":
        prerank = propose(
            my_planets, target_pool, world, model, me, omega,
            baseline_len=MAX_HORIZON + 1,
        )
        from agents.baseline.chooser_roi import choose_roi
        step = int(obs_d.get("step", 0))
        moves = choose_roi(
            snap_base, prerank,
            me, num_seats, wallclock_ms,
            MIN_HORIZON, MAX_HORIZON, gamma,
            world, model, step,
        )
        return drain_idle_rear(moves, planets, me, world, model)

    # Layered chooser opt-in (2026-05-19 slice 2). Layer-0 closed-form
    # predicates (W1/W2 commit, L1/L2 discard) over a pluggable inner
    # chooser selected via BASELINE_INNER_CHOOSER (default "trajectory").
    # See /root/.claude/plans/take-the-lens-of-magical-shore.md §9.
    if os.environ.get("BASELINE_CHOOSER", "").strip().lower() == "layered":
        inner_name = os.environ.get(
            "BASELINE_INNER_CHOOSER", "trajectory",
        ).strip().lower()
        prerank = propose(
            my_planets, target_pool, world, model, me, omega,
            baseline_len=MAX_HORIZON + 1,
        )
        # Inner chooser's appetite for baseline_favors differs: ROI
        # ignores it; trajectory accepts None; composite needs it.
        if inner_name == "composite":
            baseline_favors = build_idle_baseline(
                snap_base, me, num_seats, MAX_HORIZON, gamma,
            )
        else:
            baseline_favors = None
        from agents.baseline.chooser_layered import choose_layered
        step = int(obs_d.get("step", 0))
        moves = choose_layered(
            snap_base, prerank, baseline_favors,
            me, num_seats, wallclock_ms,
            MIN_HORIZON, MAX_HORIZON, gamma,
            world, model, step,
            inner_chooser_name=inner_name,
        )
        return drain_idle_rear(moves, planets, me, world, model)

    baseline_favors = build_idle_baseline(
        snap_base, me, num_seats, MAX_HORIZON, gamma,
    )

    prerank = propose(
        my_planets, target_pool, world, model, me, omega,
        baseline_len=len(baseline_favors),
    )

    moves = choose(
        snap_base, prerank, baseline_favors,
        me, num_seats, wallclock_ms,
        MIN_HORIZON, MAX_HORIZON, gamma,
    )
    return drain_idle_rear(moves, planets, me, world, model)
