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

# H1 — post-chooser idle drain (2026-05-18).
# Audit `audit/replays/idle-trajectory-2026-05-17.md` measured 43.8pct of
# our ship-turns sit on planets > 50 units from any non-our planet ("isolated"
# in the audit terminology). Spatial leaf head (favor_hybrid_spatial) tried
# to fix this in the chooser's Δ scoring but failed A/B (40.6pct 2P,
# 9.4pct 4P first-place — see audit/2026-05-18-spatial-leaf-negative-
# result.md). H1 is a strictly POST-CHOOSER heuristic: for OUR planets the
# chooser chose not to use, with idle surplus, no incoming threat, and
# "rear" position, emit one extra reinforce launch toward our closest
# non-rear own planet. This does NOT perturb chooser Δ — only drains idle
# garrisons when the chooser would have done nothing for that source.
IDLE_DRAIN_THRESHOLD = int(os.environ.get("BASELINE_IDLE_DRAIN_THRESHOLD", "30"))
IDLE_REAR_THRESHOLD = float(os.environ.get("BASELINE_IDLE_REAR_THRESHOLD", "35.0"))
IDLE_DRAIN_RESERVE = int(os.environ.get("BASELINE_IDLE_DRAIN_RESERVE", "5"))
IDLE_DRAIN_ENABLED = os.environ.get("BASELINE_IDLE_DRAIN", "1") != "0"

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
