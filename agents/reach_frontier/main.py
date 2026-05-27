"""reach_frontier — closed-form chooser targeting Σ p̃·τ_p^me.

Framework REPLACEMENT for the v9 K=10 rollout (doctrine §6). Per turn:

  1. parse obs -> World
  2. build my reach table ρ_me(p) = min_{src,k} arrival(s,p,k) + k/p̃_s
  3. estimate opponent reach ρ_opp(p) via WorldModel.time_to_enemy_threat
  4. hold_time(p) = ρ_opp(p) - ρ_me(p)  (or - t_now for owned planets)
  5. per-candidate reward R = p̃·hold - λ_loss·losses - λ_risk·risk
  6. Hungarian assignment over (sources × targets + noop slot per source)
  7. physics-validate via predict_fleet_fate; drop anything not outcome=="target"
  8. emit env-shape [[src_id, angle_rad, ships], ...]

See `knowledge-base/concepts/reach-frontier-doctrine.md` for the math
and `knowledge-base/concepts/reach-frontier-chooser-design.md` for the
implementation rationale.

B4 milestone: multi-source via Hungarian assignment. Defends via lp.py's
diagonal noop column (cost 0); explicit reinforce-mine logic is a v2
axis. 4P branch (doctrine §8.3) is also deferred to v2 — v1 runs in 4P
without crashing but per-turn decisions are 2P-shaped.
"""

from __future__ import annotations

import os

# Per design §9 mitigation (b): the kinematic-table fast path is required
# for the closed-form ρ-table sweep + physics-validate-each-candidate flow
# to fit the 1 s/turn budget. Set BEFORE importing lib.trajectory so the
# env-var gate `_kinematic_table_enabled()` returns True. The bundler
# strips this os.environ.setdefault line as a no-op (it doesn't match the
# intra-package import regex). Setdefault means an external A/B harness
# (fast.py, A/B drivers, scripts that pre-set the var) wins.
os.environ.setdefault("KINEMATIC_TABLE_ENABLED", "1")

from lib.kinematic_table import begin_turn
from lib.intent import World
from lib.world_model import WorldModel

from agents.reach_frontier.assignment import pick_actions
from agents.reach_frontier.hold import compute_hold_times, LAMBDA_LOSS_DEFAULT, LAMBDA_RISK_DEFAULT
from agents.reach_frontier.opponent_reach import estimate_opp_reach
from agents.reach_frontier.reach import build_reach_table


# Game length used as the cap on hold_time. The env's hard cap is 500
# steps. doctrine §2: S_i(T) integrates p̃·τ from capture-time to T.
GAME_HORIZON: int = 500


def _as_dict(obs) -> dict:
    """Coerce obs (dict or namedtuple-like) to a plain dict."""
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


def agent(obs, configuration=None):
    obs_d = _as_dict(obs)
    world = World.from_obs(obs_d)
    me = int(world.my_id)
    step_now = int(world.step)

    planets = list(world.planets_by_id.values())
    my_sources = [p for p in planets
                  if int(p.owner) == me and int(p.ships) > 0]
    targets = [p for p in planets if int(p.owner) != me]
    if not my_sources or not targets:
        return []

    # Prime the kinematic-table singleton for this turn so every
    # predict_fleet_fate call inside build_reach_table + pick_actions
    # uses the cached planet positions instead of rebuilding inline.
    # Idempotent per-turn (fingerprint-keyed); first call rebuilds,
    # subsequent calls within the same obs are O(1).
    begin_turn(world)

    world_model = WorldModel.from_world(world)

    # validate_physics=True drops OOB / sun / non-target candidates from
    # the column set before the Hungarian sees them. Without this, the
    # closed-form aim solver produces self-consistent-but-OOB intercepts
    # (doctrine §8.4 failure mode) that pass the Hungarian and then get
    # filtered post-hoc — wasting the assignment slot. Cost is bounded
    # by k-frac dedup + the no-launch-from-empty-sources prefilter; in
    # practice ~30-80 predict_fleet_fate calls/turn = well inside budget.
    my_reach = build_reach_table(
        my_sources, targets, world, world_model, me_id=me,
        validate_physics=True,
    )
    if not my_reach:
        return []

    opp_reach = estimate_opp_reach(world, me, world_model)
    hold_times = compute_hold_times(
        world, me, my_reach, opp_reach, step_now,
        game_horizon=GAME_HORIZON,
    )

    return pick_actions(
        my_reach, hold_times, world,
        me=me,
        lambda_risk=LAMBDA_RISK_DEFAULT,
        lambda_loss=LAMBDA_LOSS_DEFAULT,
    )
