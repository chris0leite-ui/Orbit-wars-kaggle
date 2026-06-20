"""least_resistance — simulation-driven forward-expansion agent for Orbit Wars.

Plain-English strategy
----------------------
Be smart by *simulating*, not by hand-tuned weights. Every turn the agent:

  1. Lists the sensible coordinated moves it could make — capture a planet
     (from one source, or several ganging up when one can't afford it). The
     list is ordered by the "path of least resistance to production" (most
     production per turn of travel), tie-broken toward whatever shortens our
     distance to the nearest opponent. This ordering is the strategy's
     *flavour* — which moves we try first.

  2. Decides which moves to actually make by scoring each candidate plan with
     a forward-projecting evaluator and keeping a launch only if it improves
     the projected outcome. We greedily build a coordinated multi-fleet plan
     until nothing further helps.

The evaluator is the key to being smart. We use the PRODUCER's
(`orbit_lite`, our strongest agent) garrison-flow scorer
`score_candidates`: it projects every planet's garrison + production +
in-flight combat forward ~18 turns and returns each candidate's competitive
*net ship gain* (mine minus opponents'). This is production-aware and
policy-free — it doesn't depend on a weak rollout policy, so reserves,
gang-up-vs-solo, attack-vs-expand, and "don't bleed ships" all fall out of
the projected ship-delta with no strategy weights. A capture only commits if
its projected payoff over the horizon clears a small floor (the producer's
ROI threshold).

If `orbit_lite` / torch isn't importable (e.g. a stripped environment), the
agent falls back to a `lib/fast_sim` rollout under `lite_greedy_policy` with a
production-aware leaf — weaker, but keeps the agent running anywhere.

Physics + machinery reused
--------------------------
  - agents/producer/orbit_lite  the producer's garrison-flow scorer
                                (single_obs_to_tensor / PlanetMovement /
                                 score_candidates) — producer-strength leaf
  - lib.aim          orbit-aware lead intercept (aim_orbiting / aim_comet)
  - lib.fleet/orbit/geometry  speed, ETA, moving-planet prediction, plus a
                    cheap `path_clears_sun` candidate pre-filter
  - lib.world_model comet path / lifetime helpers
  - lib.fast_sim / lib.opp_model / lib.value_heads  fallback evaluator only

The only parameters are compute bounds (projection horizon, candidate cap,
per-turn budget) and the producer's ROI floor — not strategy tuning.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from lib.geometry import dist, path_clears_sun
from lib.fleet import speed as fleet_speed
from lib.orbit import is_orbiting
from lib.aim import aim_orbiting, aim_comet, estimate_eta
from lib.world_model import comet_remaining_lifetime, _comet_paths_by_id


# --------------------------------------------------------------------------
# Optional producer (orbit_lite) evaluator. Imported lazily / defensively so
# the agent still loads where torch isn't installed (falls back to fast_sim).
# --------------------------------------------------------------------------
_ORBIT_OK = False
try:
    # Resolve where the orbit_lite engine + the producer entry live. Works in
    # two layouts: (a) in-repo dev (agents/least_resistance/ with sibling
    # agents/producer/), and (b) a flat submission tar.gz (orbit_lite/ +
    # producer_main.py sit next to this file).
    try:
        _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:               # kaggle execs agents without __file__
        _THIS_DIR = (sys.path[-1] if sys.path
                     and os.path.isfile(os.path.join(sys.path[-1], "main.py"))
                     else os.getcwd())
    _dev = os.path.abspath(os.path.join(_THIS_DIR, "..", "producer"))
    if (os.path.isfile(os.path.join(_dev, "main.py"))
            and os.path.isdir(os.path.join(_dev, "orbit_lite"))):
        _PRODUCER_DIR = _dev                                  # dev layout
        _PRODUCER_MAIN = os.path.join(_dev, "main.py")
    else:
        _PRODUCER_DIR = _THIS_DIR                             # flat submission
        _PRODUCER_MAIN = os.path.join(_THIS_DIR, "producer_main.py")
    if _PRODUCER_DIR not in sys.path:
        sys.path.insert(0, _PRODUCER_DIR)
    import torch as _torch
    from orbit_lite.adapter import single_obs_to_tensor as _single_obs_to_tensor
    from orbit_lite.adapter import sparse_action_row_to_moves as _sparse_action_row_to_moves
    from orbit_lite.movement import MovementConfig as _MovementConfig
    from orbit_lite.movement_step import ensure_planet_movement as _ensure_planet_movement
    from orbit_lite.planner_core import (
        make_launch_set as _make_launch_set,
        score_candidates as _score_candidates,
        largest_initial_player_count as _largest_initial_player_count,
    )
    from orbit_lite.distance_cache import build_distance_cache as _build_distance_cache
    from orbit_lite.native_forward import (
        build_candidate_trajectories as _nf_build_traj,
        reachable_enemy_mass as _nf_reach_mass,
        hazard_ownership_value as _nf_hazard_value,
        _inflight_by_owner as _nf_inflight,
    )
    import importlib.util as _ilu
    _pm_spec = _ilu.spec_from_file_location("_lr_producer_main", _PRODUCER_MAIN)
    _producer_main = _ilu.module_from_spec(_pm_spec)
    sys.modules["_lr_producer_main"] = _producer_main
    _pm_spec.loader.exec_module(_producer_main)
    _ORBIT_OK = True
except Exception:
    _ORBIT_OK = False

# Fallback evaluator deps (pure Python, no torch).
from lib.fast_sim import from_obs, clone, step, ship_totals
from lib.opp_model import lite_greedy_policy
from lib.value_heads import inflight_value


# --------------------------------------------------------------------------
# Compute bounds (NOT strategy weights).
# --------------------------------------------------------------------------
def _i(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _leader_relative_4p():
    """Default-OFF gate. In 4-player games, score a position by the gap to the
    single STRONGEST opponent (win-equity / overtake-the-leader) instead of the
    gap to the SUM of all opponents (material / safe-2nd). 2-player is
    byte-identical (one opponent IS the leader). Read at call time."""
    return os.environ.get("LR_LEADER_RELATIVE_4P", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _native_leaf():
    """Default-OFF gate (PI 2026-06-20). When ON, the 2-ply search leaf becomes the
    dropout-NATIVE expected-SHIP-MARGIN value under a per-step flip-hazard forward
    model (orbit_lite.native_forward, value_mode='ships') instead of the ship-count
    snapshot. This prices captures that won't hold (the opponent retakes) and gives
    the native model the multi-ply it lacked as a standalone one-ply scorer (the
    2026-06-19 plateau). OFF = byte-identical ship-count path. Read at call time."""
    return os.environ.get("LR_NATIVE_LEAF", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _native_strict():
    """Default ON. A native-leaf error RAISES instead of silently falling back to
    the ship-count leaf -- the 2026-06-19 trap where a scorer that threw every turn
    made the whole evaluation secretly measure the fallback. Set LR_NATIVE_STRICT=0
    only to tolerate fallback deliberately."""
    return os.environ.get("LR_NATIVE_STRICT", "1").strip().lower() in (
        "1", "true", "on", "yes")


def _dropout_prune():
    """PI 2026-06-20: decline a candidate ATTACK whose flip-back ('dropout')
    probability is high -- the target sits where incoming enemy fleets / a nearby
    opponent will retake it (and far / high-EDA launches, which give the enemy more
    time to contest). A gate on OUR attacks only (avoid wasteful fleets), NOT a
    defensive buff on our standing planets (that lever regressed). Default tracks
    the native leaf; LR_DROPOUT_PRUNE overrides. Read at call time."""
    v = os.environ.get("LR_DROPOUT_PRUNE")
    if v is None:
        return _native_leaf()
    return v.strip().lower() in ("1", "true", "on", "yes")


def _wallclock_ms():
    """Per-turn budget, read at CALL time. The bundle parity gate sets
    ORBIT_WARS_PARITY_WALLCLOCK_MS huge so the greedy loop never bails
    mid-list, making the agent a pure function of obs."""
    override = os.environ.get("ORBIT_WARS_PARITY_WALLCLOCK_MS")
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    return _f("LR_WALLCLOCK_MS", 700.0)


PROJECT_HORIZON_2P = _i("LR_HORIZON_2P", 18)   # orbit_lite garrison-flow window (2P)
PROJECT_HORIZON_4P = _i("LR_HORIZON_4P", 13)   # 4P
ROI_FLOOR = _f("LR_ROI_FLOOR", 1.5)            # min projected net-ship gain to commit (producer's value)
MAX_CANDIDATES = _i("LR_MAX_CANDIDATES", 28)
FRONTIER_REF_SHIPS = _f("LR_FRONTIER_REF_SHIPS", 30.0)
RANK_HINT_SHIPS = 20
# Fallback (no-torch) evaluator knobs.
FALLBACK_HORIZON = _i("LR_FALLBACK_HORIZON", 10)
FORCE_EVAL = os.environ.get("LR_EVAL", "").strip().lower()  # "orbit" | "fallback" | ""
# 2-ply lookahead vs the producer (2P only): evaluate candidate full-plans by
# applying my move + the producer's predicted reply, then a turn of
# producer-vs-producer, and scoring the resulting position. Catches moves the
# producer punishes next turn (which the 1-ply scorer over-rates). The
# producer's own move is always a candidate, so we never do worse than it.
TWOPLY = os.environ.get("LR_TWOPLY", "1").strip().lower() in ("1", "true", "on", "yes")
TWOPLY_BUDGET_MS = _f("LR_TWOPLY_MS", 450.0)


def _anytime():
    """Lever 3 gate (default OFF): spend per-turn / overage-bank headroom by
    widening the 2-ply plan set and letting the 2-ply budget draw on the bank."""
    return os.environ.get("LR_ANYTIME", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _twoply_budget(obs_d):
    """2-ply time budget (ms). Default TWOPLY_BUDGET_MS. With LR_ANYTIME on,
    draw a slice of the episode overage bank (obs.remainingOverageTime, seconds)
    so pivotal turns think longer on otherwise-wasted headroom. Self-limiting:
    spends a fixed fraction of the *spendable* bank (keeping a reserve), so it
    tapers as the bank drains and never exceeds the cap."""
    base = TWOPLY_BUDGET_MS
    if not _anytime():
        return base
    try:
        bank_s = float(obs_d.get("remainingOverageTime"))
    except (TypeError, ValueError):
        return base
    spendable = max(0.0, bank_s - _f("LR_ANYTIME_BANK_FLOOR_S", 8.0))
    extra = min(spendable * 1000.0 * _f("LR_ANYTIME_BANK_FRAC", 0.03),
                _f("LR_ANYTIME_EXTRA_CAP_MS", 1200.0))
    return base + extra


def _value_commit():
    """Fundamental gate (default OFF, both modes): commit captures in order of
    their VALUE under the objective (highest win-equity first) rather than
    cheapness -- scoring each candidate once with spare compute."""
    return os.environ.get("LR_VALUE_COMMIT", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _value_budget(obs_d, base):
    """Budget (ms) for value-ordered commitment, which scores every candidate up
    front. Draw a self-limiting slice of the overage bank so the extra scoring
    does not starve the commit pass."""
    if not _value_commit():
        return base
    try:
        bank_s = float(obs_d.get("remainingOverageTime"))
    except (TypeError, ValueError):
        return base
    spendable = max(0.0, bank_s - _f("LR_ANYTIME_BANK_FLOOR_S", 8.0))
    extra = min(spendable * 1000.0 * _f("LR_VALUE_BANK_FRAC", 0.02),
                _f("LR_VALUE_EXTRA_CAP_MS", 700.0))
    return base + extra


def _rollout_depth():
    """Deep-search gate (default 0 = OFF -> use the 2-ply pick). A value >= 2
    turns on the K-turn producer-rollout search (see _deep_pick)."""
    return _i("LR_ROLLOUT_DEPTH", 0)


def _deep_opp():
    """Per-node opponent model for the search rollouts (_twoply_pick/_deep_pick).
    0 = producer mirror (accurate, ~10-50ms/call -- the per-node bottleneck);
    1 = lite_greedy (cheap, ~1-2ms, still models expansion) so deeper search
    fits the 1000ms wall. Default 0 keeps current behaviour byte-identical."""
    return _i("LR_DEEP_OPP", 0)


def _skip_comets():
    """Skip COMET targets in candidate generation (default 0 = target them, as
    today). ON because comet intercept (aim_comet) can mis-predict on a moving
    target and there is no oob/accuracy guard, so a missed comet shot sails
    off-board (wasted). Temporary: disable comet targeting until the comet aim
    is fixed."""
    return _i("LR_SKIP_COMETS", 0) >= 1


def _iterdeepen():
    """Anytime iterative deepening for the deep rollout (default 0 = OFF, fixed
    depth). When ON, _deep_pick deepens d=1..LR_ROLLOUT_DEPTH within the timebox,
    extending each candidate's rollout one ply per level (no re-roll), adopting
    only the deepest COMPLETED level -- so a high LR_ROLLOUT_DEPTH cap is bounded
    by time per turn and never breaches the wall."""
    return _i("LR_ITERDEEPEN", 0) >= 1


def _opponent_move_fn(tier=None):
    """Return a callable (obs, seat) -> [[src,angle,ships],...] for the per-node
    opponent move, matching _producer_move_obs' signature. lite_greedy reads the
    seat from obs.player, so the seat arg is ignored for it."""
    if tier is None:
        tier = _deep_opp()
    if int(tier) == 1:
        return lambda obs_any, seat: lite_greedy_policy(obs_any)
    return _producer_move_obs


def _deep_budget(obs_d):
    """Per-turn budget (ms) for deep rollout search. Draws a self-limiting slice
    of the episode overage bank (obs.remainingOverageTime) so pivotal turns can
    search deeper -- this is what finally spends the headroom. Tapers as the
    bank drains; never exceeds base + cap."""
    base = _wallclock_ms()
    try:
        bank_s = float(obs_d.get("remainingOverageTime"))
    except (TypeError, ValueError):
        return base
    spendable = max(0.0, bank_s - _f("LR_DEEP_BANK_FLOOR_S", 6.0))
    extra = min(spendable * 1000.0 * _f("LR_DEEP_BANK_FRAC", 0.04),
                _f("LR_DEEP_EXTRA_CAP_MS", 2500.0))
    return base + extra


# --------------------------------------------------------------------------
# Obs parsing.
# --------------------------------------------------------------------------
def _as_dict(obs):
    if isinstance(obs, dict):
        return obs
    return {
        "player": getattr(obs, "player", 0),
        "step": getattr(obs, "step", 0),
        "planets": list(getattr(obs, "planets", []) or []),
        "fleets": list(getattr(obs, "fleets", []) or []),
        "comets": list(getattr(obs, "comets", []) or []),
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", []) or []),
        "angular_velocity": float(getattr(obs, "angular_velocity", 0.0) or 0.0),
    }


def _num_seats(planets, fleets):
    max_owner = -1
    for p in planets:
        max_owner = max(max_owner, int(p.owner))
    for f in fleets:
        max_owner = max(max_owner, int(f.owner))
    return 4 if max_owner >= 2 else 2


# --------------------------------------------------------------------------
# Aim with the correct accurate physics per body type.
# --------------------------------------------------------------------------
def _plan_shot(src, tgt, world_comet_ids, comet_paths, omega, ships):
    """Return (aim_angle, eta_turns, arrival_xy) or None if no intercept."""
    s_xy = (float(src.x), float(src.y))
    t_tuple = [int(tgt.id), int(tgt.owner), float(tgt.x), float(tgt.y),
               float(tgt.radius), float(tgt.ships), float(tgt.production)]
    ships = max(1, int(ships))
    if int(tgt.id) in world_comet_ids:
        entry = comet_paths.get(int(tgt.id))
        if entry is None:
            return None
        path, path_index = entry
        res = aim_comet(s_xy, float(src.radius), t_tuple, float(tgt.radius),
                        ships, path, path_index)
    elif omega != 0.0 and is_orbiting(t_tuple):
        res = aim_orbiting(s_xy, float(src.radius), t_tuple, float(tgt.radius),
                           ships, omega)
    else:
        eta_f = estimate_eta(s_xy, float(src.radius), (float(tgt.x), float(tgt.y)),
                             float(tgt.radius), ships)
        if eta_f is None:
            return None
        angle = math.atan2(float(tgt.y) - float(src.y),
                           float(tgt.x) - float(src.x))
        res = (angle, (float(tgt.x), float(tgt.y)), eta_f)
    if res is None:
        return None
    angle, arrival_xy, eta_f = res
    return angle, max(1, int(math.ceil(eta_f))), arrival_xy


def _sun_clear(src, arrival_xy):
    """Cheap geometry pre-filter; full path safety is left to the evaluator."""
    return path_clears_sun((float(src.x), float(src.y)), arrival_xy)


# --------------------------------------------------------------------------
# Producer (orbit_lite) leaf scorer — built once per turn.
# --------------------------------------------------------------------------
def _strongest_opp_weights(obs_tensors, me, pc):
    """One-hot ``[pc]`` weight on the strongest opponent by current ship total
    (0 at ``me``, sums to 1 over opponents) — the scorer's ``opp_weights``
    contract. Turns the competitive score from "me - sum(opponents)" into
    "me - strongest_opponent"."""
    planets = obs_tensors["planets"]                 # [P, 7]; owner col 1, ships col 5
    owner = planets[:, 1].long()
    ships = planets[:, 5].to(_torch.float32)
    best, best_v = None, -1.0
    for pl in range(int(pc)):
        if pl == int(me):
            continue
        tot = float((ships * (owner == pl).to(_torch.float32)).sum())
        if tot > best_v:
            best_v, best = tot, pl
    w = _torch.zeros(int(pc), dtype=_torch.float32)
    if best is not None:
        w[best] = 1.0
    return w


def _build_orbit_scorer(obs, me):
    """Return (score_units_fn, id2slot) or None on any failure.

    score_units_fn(units) -> float, where units is a list of
    (src_slot, tgt_slot, ships, eta) describing a coordinated launch plan;
    the score is the producer's competitive net-ship-delta over the horizon.
    """
    obs_tensors = _single_obs_to_tensor(obs, player_id=int(me))
    pc = _largest_initial_player_count(obs_tensors)
    H = PROJECT_HORIZON_4P if int(pc) >= 4 else PROJECT_HORIZON_2P
    cfg = _MovementConfig(
        movement_horizon=int(H), drift_epsilon=1e-3, track_fleets=True,
        player_count=int(pc), max_tracked_fleets=128,
    )
    movement = _ensure_planet_movement(
        obs_tensors=obs_tensors, expected_cfg=cfg, cached_movement=None,
    )
    status = movement.garrison_status(max_horizon=int(H))
    prod = movement.planet_prod
    alive_by_step = movement.alive_by_step[: int(H) + 1]
    ids = obs_tensors["planets"][:, 0].long().tolist()
    id2slot = {int(v): i for i, v in enumerate(ids)}

    # Default-OFF: leader-relative opponent weighting in 4P (gap-to-strongest).
    opp_w = None
    if _leader_relative_4p() and int(pc) >= 4:
        opp_w = _strongest_opp_weights(obs_tensors, me, int(pc))

    def score_units(units):
        if not units:
            return 0.0
        a = _torch.tensor([[u[0] for u in units]], dtype=_torch.long)
        b = _torch.tensor([[u[1] for u in units]], dtype=_torch.long)
        sh = _torch.tensor([[float(u[2]) for u in units]])
        et = _torch.tensor([[float(max(1, u[3])) for u in units]])
        va = _torch.ones((1, len(units)), dtype=_torch.bool)
        ls = _make_launch_set(source_slots=a, target_slots=b, ships=sh,
                              eta=et, valid=va, player_id=int(me))
        with _torch.no_grad():
            sc = _score_candidates(
                status, prod=prod, alive_by_step=alive_by_step,
                player_count=int(pc), launches=ls, player_id=int(me),
                opp_weights=opp_w,
            )
        return float(sc.reshape(-1)[0])

    return score_units, id2slot


# --------------------------------------------------------------------------
# 2-ply lookahead vs the producer.
# --------------------------------------------------------------------------
def _producer_move_obs(obs_any, seat):
    """The producer's launches for `seat` given any obs (dict or Struct),
    using a fresh memory (single-turn prediction, no shared state)."""
    try:
        ot = _single_obs_to_tensor(obs_any, player_id=int(seat))
        runtime = _producer_main.ProducerLiteRuntime()
        with _torch.no_grad():
            row = runtime.tensor_action(ot)
        return _sparse_action_row_to_moves(row, obs_any, player_id=int(seat))
    except Exception:
        return []


def _project_value(obs_any, me):
    """Position value: project the board `H` turns forward (all in-flight
    fleets + production + combat, no new launches) and return our garrison
    advantage (our ships - opponents') at the horizon. The producer's own
    garrison-flow projector used as a state evaluator."""
    if _native_leaf():
        # PI 2026-06-20: dropout-native expected SHIP-MARGIN under a flip-hazard
        # forward model -> prices captures the opponent would retake.
        return _native_value(obs_any, me)
    ot = _single_obs_to_tensor(obs_any, player_id=int(me))
    pc = _largest_initial_player_count(ot)
    H = PROJECT_HORIZON_4P if int(pc) >= 4 else PROJECT_HORIZON_2P
    cfg = _MovementConfig(
        movement_horizon=int(H), drift_epsilon=1e-3, track_fleets=True,
        player_count=int(pc), max_tracked_fleets=128,
    )
    mv = _ensure_planet_movement(obs_tensors=ot, expected_cfg=cfg, cached_movement=None)
    status = mv.garrison_status(max_horizon=int(H))
    owner = status.owner[:, int(H)]
    ships = status.ships[:, int(H)].to(_torch.float32)
    mine = float((ships * (owner == int(me)).to(_torch.float32)).sum())
    if _leader_relative_4p() and int(pc) >= 4:
        # Win-equity: gap to the single strongest opponent, not the whole field.
        theirs = 0.0
        for pl in range(int(pc)):
            if pl == int(me):
                continue
            tot = float((ships * (owner == pl).to(_torch.float32)).sum())
            if tot > theirs:
                theirs = tot
    else:
        theirs = float((ships * ((owner != int(me)) & (owner >= 0)).to(_torch.float32)).sum())
    return mine - theirs


def _project_outcome(obs_any, me, horizon=None):
    """One garrison projection -> (win, margin). Project the board H turns forward
    (the producer's strong static evaluator that makes the baseline value
    expansion correctly), then read PER-PLAYER projected ship totals: `win` =
    1.0 if I'm projected rank #1, 0.5 if tied for top, else 0.0; `margin` = my
    projected ships minus the strongest rival's. Used as the leaf of the robust
    ensemble so the agnostic opponent + rank objective sit on a strong evaluator
    rather than a crude greedy rollout. `horizon` overrides the default
    phase-horizon (longer sight values held production + in-flight retakes)."""
    ot = _single_obs_to_tensor(obs_any, player_id=int(me))
    pc = _largest_initial_player_count(ot)
    if horizon is not None:
        H = int(horizon)
    else:
        H = PROJECT_HORIZON_4P if int(pc) >= 4 else PROJECT_HORIZON_2P
    cfg = _MovementConfig(
        movement_horizon=int(H), drift_epsilon=1e-3, track_fleets=True,
        player_count=int(pc), max_tracked_fleets=128,
    )
    mv = _ensure_planet_movement(obs_tensors=ot, expected_cfg=cfg, cached_movement=None)
    status = mv.garrison_status(max_horizon=int(H))
    owner = status.owner[:, int(H)]
    ships = status.ships[:, int(H)].to(_torch.float32)
    # Terminal production credit: a planet I still own at the horizon keeps
    # producing for the rest of the game, so its value is ships PLUS its future
    # output (LR_ROBUST_PROD_CREDIT turns' worth). Without this the short
    # projection only credits ~H turns of production, so holding a planet always
    # looks worse than grabbing one -> the agent never defends. With it, holding
    # a high-production planet can out-value a low-value new grab.
    pc_i = int(pc)
    owner_l = owner.to(_torch.long)
    prod = mv.planet_prod.to(_torch.float32).reshape(-1)
    pcred = _f("LR_ROBUST_PROD_CREDIT", 12.0)
    # CAPABILITY value per player, not raw ship count:
    #   base = ships + production-credit (a held planet keeps producing), then
    #   + DEFENSIBILITY (negative when my planets are out-massed by enemy force
    #     that can reach them -> finally makes holding/defending score), and
    #   + REACH (capturable production within strike range -> values board
    #     position / expansion potential, not just current stuff).
    base_val = ships + pcred * prod                       # [P]
    cap = _torch.zeros(pc_i, dtype=_torch.float32)
    for pl in range(pc_i):
        cap[pl] = (base_val * (owner_l == pl).to(_torch.float32)).sum()

    if os.environ.get("LR_ROBUST_CAP", "1").strip().lower() in ("1", "true", "on", "yes"):
        try:
            xy = ot["planets"][:, 2:4].to(_torch.float32)        # [P,2] positions
            d = _torch.cdist(xy, xy)                              # [P,P]
            near_def = (d <= _f("LR_ROBUST_DEF_RANGE", 35.0)).to(_torch.float32)
            near_reach = (d <= _f("LR_ROBUST_REACH_RANGE", 45.0)).to(_torch.float32)
            w_def = _f("LR_ROBUST_W_DEF", 1.0)
            w_reach = _f("LR_ROBUST_W_REACH", 0.5)
            for pl in range(pc_i):
                mine_mask = (owner_l == pl).to(_torch.float32)   # [P]
                enemy_mask = ((owner_l != pl) & (owner_l >= 0)).to(_torch.float32)
                reach_enemy = near_def @ (ships * enemy_mask)     # enemy mass that can reach each planet
                vuln = _torch.clamp(reach_enemy - ships, min=0.0)  # how out-massed each of my planets is
                defens = -(vuln * mine_mask).sum()               # less vulnerable -> higher (closer to 0)
                reach_prod = near_reach @ (prod * (owner_l != pl).to(_torch.float32))
                reachv = (reach_prod * mine_mask).sum()          # capturable production in range
                cap[pl] = cap[pl] + w_def * defens + w_reach * reachv
        except Exception:
            pass

    mine = float(cap[int(me)])
    rivals = [float(cap[pl]) for pl in range(pc_i) if pl != int(me)]
    if not rivals:
        return 1.0, mine
    # Placement = fraction of rivals I out-CAPABILITY (tie = half). 2P = win/loss;
    # 4P rewards out-positioning each rival -> rank-aligned but defense-sensitive.
    beaten = sum(1.0 if mine > r else (0.5 if mine == r else 0.0) for r in rivals)
    return beaten / len(rivals), (mine - max(rivals))


_NATIVE_LEAF_CALLS = 0    # proof the native leaf actually executed (env swallows stderr)


def _native_value(obs_any, me):
    """Dropout-NATIVE position value: expected SHIP-MARGIN under a per-step
    flip-hazard forward model (PI 2026-06-20; reuses agents/producer/orbit_lite/
    native_forward.py, the 2026-06-19 ship-margin reformulation).

    Evaluate the DO-NOTHING trajectory from this position (no new launches) and let
    the flip hazard leak ownership of planets the opponent can reach -> a position
    where our gains are exposed scores lower; a holdable position scores its full
    forward ship stream. This is the absolute board value (used as the 2-ply leaf,
    so the search itself becomes hazard-evaluated). Returns a scalar (higher
    better). Raises on failure unless LR_NATIVE_STRICT=0 (never silently fall back
    to the ship-count leaf -- the 2026-06-19 trap)."""
    global _NATIVE_LEAF_CALLS
    ot = _single_obs_to_tensor(obs_any, player_id=int(me))
    pc = int(_largest_initial_player_count(ot))
    H = _i("LR_NATIVE_HORIZON", PROJECT_HORIZON_4P if pc >= 4 else PROJECT_HORIZON_2P)
    cfg = _MovementConfig(
        movement_horizon=int(H), drift_epsilon=1e-3, track_fleets=True,
        player_count=pc, max_tracked_fleets=128,
    )
    mv = _ensure_planet_movement(obs_tensors=ot, expected_cfg=cfg, cached_movement=None)
    status = mv.garrison_status(max_horizon=int(H))
    prod = mv.planet_prod.to(_torch.float32).reshape(-1)
    alive_by_step = mv.alive_by_step[: int(H) + 1]
    cache = _build_distance_cache(mv, max_k=int(H))
    owner0 = status.owner[:, 0].to(_torch.long)
    ships0 = status.ships[:, 0].to(_torch.float32)
    is_enemy = (owner0 != int(me)) & (owner0 >= 0)
    # Do-nothing trajectory: one empty candidate (src/tgt = -1, no launches).
    empty_l = _torch.full((1, 1), -1, dtype=_torch.long)
    owner_traj, ships_traj, arr_c = _nf_build_traj(
        init_owner=owner0, init_ships=ships0, prod=prod,
        alive_by_step=alive_by_step,
        background_arrivals=status.arrivals_by_owner[..., 1:, :],
        src=empty_l, tgt=empty_l, ships=_torch.zeros(1, 1),
        eta=_torch.ones(1, 1), owner=_torch.zeros(1, 1, dtype=_torch.long),
        valid=_torch.zeros(1, 1, dtype=_torch.bool),
    )
    atk_reach = _nf_reach_mass(
        cross_dist=cache.cross_dist, ships=ships0, is_enemy=is_enemy, H=int(H),
        prod=prod, growth_alpha=_f("LR_NATIVE_THREAT_GROWTH", 0.0),
    )
    val = _nf_hazard_value(
        owner=owner_traj, ships=ships_traj, prod=prod, atk_reach=atk_reach,
        me=int(me), steepness=_f("LR_NATIVE_STEEPNESS", 5.0),
        discount=_f("LR_NATIVE_DISCOUNT", 1.0), value_mode="ships",
        inflight=_nf_inflight(arr_c), terminal=_f("LR_NATIVE_TERMINAL", 12.0),
    )
    _NATIVE_LEAF_CALLS += 1
    return float(val.reshape(-1)[0])


def _twoply_pick(obs, configuration, me, num_seats, candidate_plans, budget_ms=None,
                 opp_move_fn=None):
    """Pick the candidate full-plan with the best 2-ply value: apply [my plan,
    opponent's predicted reply] this turn, then a turn of opponent pressure,
    then score the resulting position. `candidate_plans` always includes the
    producer's own move (the >=-producer floor). Returns the chosen plan. The
    per-node opponent is `opp_move_fn` (default: the LR_DEEP_OPP selection)."""
    if budget_ms is None:
        budget_ms = TWOPLY_BUDGET_MS
    if opp_move_fn is None:
        opp_move_fn = _opponent_move_fn()
    snap = from_obs(obs, configuration, num_seats=num_seats)
    opps = [i for i in range(num_seats) if i != int(me)]
    # Every opponent's predicted move this turn; shared across candidates since
    # moves are simultaneous (they don't see my plan). 2P (one opp) and 4P (three).
    opp_now = {i: opp_move_fn(snap.state[i].observation, i) for i in opps}

    def value(plan):
        s = clone(snap)
        acts = [[] for _ in range(num_seats)]
        acts[int(me)] = list(plan)
        for i in opps:
            acts[i] = list(opp_now[i])
        step(s, acts, in_place=True)
        if not s.fake_env.done:
            # One more turn of the opponents' pressure (each replies; we stay
            # idle -- conservative, and it surfaces the next-turn punishment the
            # 1-ply scorer misses).
            nxt = [[] for _ in range(num_seats)]
            for i in opps:
                nxt[i] = opp_move_fn(s.state[i].observation, i)
            step(s, nxt, in_place=True)
        if _native_leaf() and _native_strict():
            # Strict: a native-leaf error must surface, never silently degrade the
            # whole pick to the producer floor (the 2026-06-19 trap).
            return _project_value(s.state[int(me)].observation, me)
        try:
            return _project_value(s.state[int(me)].observation, me)
        except Exception:
            return None

    best_plan, best_v = candidate_plans[0], None
    t0 = time.perf_counter()
    for plan in candidate_plans:
        if best_v is not None and (time.perf_counter() - t0) * 1000.0 > budget_ms:
            break
        v = value(plan)
        if v is None:
            continue
        if best_v is None or v > best_v:
            best_v, best_plan = v, plan
    return best_plan


def _deep_pick(obs, configuration, me, num_seats, candidate_plans, depth, budget_ms=None,
               opp_move_fn=None):
    """Deeper search vs a fixed-policy opponent. For each candidate first-move,
    roll the game out `depth` turns -- turn 1 = my move + each opponent's reply;
    turns 2..depth = EVERY seat (incl. me) plays the opponent policy -- then score
    with the analytic leaf. Plans are tried in the given (value-ranked) order;
    best-so-far kept; time-guarded (anytime-safe). The per-node opponent is
    `opp_move_fn` (default: the LR_DEEP_OPP selection -- producer mirror, or the
    cheap lite_greedy so deeper search fits the wall). The producer's own move is
    always among `candidate_plans` (>=-producer floor)."""
    if budget_ms is None:
        budget_ms = TWOPLY_BUDGET_MS
    if opp_move_fn is None:
        opp_move_fn = _opponent_move_fn()
    snap = from_obs(obs, configuration, num_seats=num_seats)
    opps = [i for i in range(num_seats) if i != int(me)]
    opp_now = {i: opp_move_fn(snap.state[i].observation, i) for i in opps}

    def rollout_value(plan):
        s = clone(snap)
        acts = [[] for _ in range(num_seats)]
        acts[int(me)] = list(plan)
        for i in opps:
            acts[i] = list(opp_now[i])
        step(s, acts, in_place=True)
        for _ in range(max(0, int(depth) - 1)):
            if s.fake_env.done:
                break
            nxt = [opp_move_fn(s.state[i].observation, i)
                   for i in range(num_seats)]
            step(s, nxt, in_place=True)
        try:
            return _project_value(s.state[int(me)].observation, me)
        except Exception:
            return None

    if _iterdeepen():
        # Anytime iterative deepening with incremental rollout extension: keep
        # each candidate's Snapshot after d plies and extend ONE ply to reach
        # d+1 (no re-roll); deepen d=1..depth within the timebox; adopt only the
        # deepest COMPLETED level's best; always hold a legal move.
        t0 = time.perf_counter()
        cand = []                                   # [plan, snapshot-after-d-plies]
        for plan in candidate_plans:
            s = clone(snap)
            acts = [[] for _ in range(num_seats)]
            acts[int(me)] = list(plan)
            for i in opps:
                acts[i] = list(opp_now[i])
            step(s, acts, in_place=True)            # ply 1 = my move + opp_now
            cand.append([plan, s])
        best_plan = candidate_plans[0]
        for d in range(1, int(depth) + 1):
            level_best_plan, level_best_v = None, None
            completed = True
            for entry in cand:
                if (time.perf_counter() - t0) * 1000.0 > budget_ms:
                    completed = False
                    break
                plan, s = entry
                if d > 1 and not s.fake_env.done:
                    nxt = [opp_move_fn(s.state[i].observation, i)
                           for i in range(num_seats)]
                    step(s, nxt, in_place=True)      # extend one ply
                try:
                    v = _project_value(s.state[int(me)].observation, me)
                except Exception:
                    v = None
                if v is not None and (level_best_v is None or v > level_best_v):
                    level_best_v, level_best_plan = v, plan
            if completed and level_best_plan is not None:
                best_plan = level_best_plan
            if not completed:
                break
        return best_plan

    best_plan, best_v = candidate_plans[0], None
    t0 = time.perf_counter()
    for plan in candidate_plans:
        if best_v is not None and (time.perf_counter() - t0) * 1000.0 > budget_ms:
            break
        v = rollout_value(plan)
        if v is None:
            continue
        if best_v is None or v > best_v:
            best_v, best_plan = v, plan
    return best_plan


# --------------------------------------------------------------------------
# Comet-path helper that works off a plain obs dict (no lib.intent World).
# --------------------------------------------------------------------------
class _ObsRawShim:
    """Minimal shim exposing `.obs_raw` for lib.world_model comet helpers."""
    __slots__ = ("obs_raw",)

    def __init__(self, obs_d):
        self.obs_raw = obs_d


def _comet_paths_by_id_safe(obs_d):
    try:
        return _comet_paths_by_id(_ObsRawShim(obs_d))
    except Exception:
        return {}


# --------------------------------------------------------------------------
# Robust-ensemble pick (opponent-agnostic; default OFF behind LR_ROBUST).
#
# Instead of assuming the opponent is the producer, we judge a candidate plan by
# how it fares across a DISTRIBUTION of plausible futures. Each future samples
# every rival's move from a stochastic "least-resistance" picker (they expand
# onto neutrals AND attack -- a two-sided, opponent-agnostic, self-play-style
# model), then plays the game forward a few turns with all seats (including us)
# continuing on a cheap greedy policy, and scores the result by the REAL win
# quantity: total ships (planets + fleets), measured against the SINGLE
# STRONGEST rival in 4P (beat-the-leader = rank #1, not safe-2nd).
#
# All candidates are scored against the SAME sampled futures (paired -> the
# argmax is low-noise), and we keep the plan whose WORST-fraction outcome is
# best (a robust lower-quantile / CVaR statistic): "an action that scores well
# against ~most of the simulated futures". This is the piece the dropout line
# recommended but never built; it is two-sided (so it does not go passive) and
# candidate-dependent (so the distribution actually reorders the choice).
# --------------------------------------------------------------------------
def _robust():
    return os.environ.get("LR_ROBUST", "0").strip().lower() in (
        "1", "true", "on", "yes")


# Compute bounds + the one risk dial (NOT strategy weights).
ROBUST_SAMPLES = _i("LR_ROBUST_SAMPLES", 16)   # number of sampled futures
ROBUST_K = _i("LR_ROBUST_K", 14)               # plies rolled forward per future (long enough for expansion to convert to ships)
ROBUST_QUANTILE = _f("LR_ROBUST_QUANTILE", 0.25)  # average the worst this-fraction (CVaR); ->0 = worst-case, 1 = mean
ROBUST_TEMP = _f("LR_ROBUST_TEMP", 0.7)        # rival target-sampling temperature (spread)
ROBUST_MS = _f("LR_ROBUST_MS", 650.0)          # per-turn time budget for the pick
ROBUST_HORIZON = _i("LR_ROBUST_HORIZON", 0)    # leaf projection horizon override; 0 = phase default (18/13)


def _robust_budget(obs_d):
    """Per-turn budget (ms), drawing a self-limiting slice of the episode
    overage bank so pivotal turns can simulate more futures on otherwise-wasted
    headroom. Tapers as the bank drains; never exceeds base + cap."""
    base = ROBUST_MS
    try:
        bank_s = float(obs_d.get("remainingOverageTime"))
    except (TypeError, ValueError):
        return base
    spendable = max(0.0, bank_s - _f("LR_ROBUST_BANK_FLOOR_S", 8.0))
    extra = min(spendable * 1000.0 * _f("LR_ROBUST_BANK_FRAC", 0.03),
                _f("LR_ROBUST_EXTRA_CAP_MS", 1500.0))
    return base + extra


def _board_seed(obs_d):
    """Deterministic per-board RNG seed (owners + rounded ships + step), so the
    sampled futures are reproducible -- a pure function of the observation."""
    parts = [int(obs_d.get("step", 0))]
    for p in (obs_d.get("planets", []) or []):
        parts.append(int(p[1]))            # owner
        parts.append(int(round(float(p[5]))))   # ships
    return hash(tuple(parts)) & 0x7FFFFFFF


def _stochastic_greedy(obs, rng, temp):
    """A rival's sampled move: least-resistance capture picker (production over
    distance), but each owned planet picks its target by softmax SAMPLING over
    attractiveness (temperature `temp`) instead of the hard argmax -- so across
    futures the rivals diverge (the spread that makes robustness meaningful).
    Sizing/affordability mirror lite_greedy_policy (expansion-aware: they grab
    neutrals AND attack). Returns env-format launches."""
    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    planets = obs.get("planets") if isinstance(obs, dict) else getattr(obs, "planets", None)
    if not planets:
        return []
    targets = [p for p in planets if int(p[1]) != int(player)]
    if not targets:
        return []
    moves = []
    for src in planets:
        if int(src[1]) != int(player) or float(src[5]) < 10:
            continue
        sx, sy = float(src[2]), float(src[3])
        scored = []
        for t in targets:
            if int(t[0]) == int(src[0]):
                continue
            dx, dy = float(t[2]) - sx, float(t[3]) - sy
            d = math.sqrt(dx * dx + dy * dy)
            if d < 1e-6:
                continue
            scored.append((float(t[6]) / (d + 1.0), t, d))
        if not scored:
            continue
        # Softmax sample a target (temperature `temp`); temp<=0 -> hard argmax.
        if temp <= 1e-6:
            best = max(scored, key=lambda e: e[0])
        else:
            mx = max(s for s, _, _ in scored)
            weights = [math.exp((s - mx) / temp) for s, _, _ in scored]
            tot = sum(weights)
            r = rng.random() * tot
            acc = 0.0
            best = scored[-1]
            for w, e in zip(weights, scored):
                acc += w
                if acc >= r:
                    best = e
                    break
        _, tgt, d = best
        budget = int(src[5])
        agg = max(5, int(budget * 0.7))
        if agg > budget:
            agg = budget
        spd = fleet_speed(agg)
        if spd <= 0:
            continue
        flight = max(0.0, d - float(src[4]) - float(tgt[4]) - 0.1)
        eta = max(1, int(math.ceil(flight / spd)))
        if int(tgt[1]) == -1:
            defenders = float(tgt[5])
        else:
            defenders = float(tgt[5]) + float(tgt[6]) * eta
        needed = int(math.ceil(defenders)) + 1
        if needed > budget:
            continue
        ships = min(budget, max(agg, needed))
        if ships < 5:
            continue
        angle = math.atan2(float(tgt[3]) - sy, float(tgt[2]) - sx)
        moves.append([int(src[0]), float(angle), int(ships)])
    return moves


def _leader_margin(snap, me, num_seats, leader_relative):
    """Our ships minus the strongest rival's (leader-relative, 4P win-equity) or
    minus the sum of all rivals (material). Uses the engine's real score head."""
    tot = ship_totals(snap)
    ours = tot.get(int(me), 0.0)
    rivals = [v for k, v in tot.items() if int(k) != int(me)]
    if not rivals:
        return ours
    return ours - (max(rivals) if leader_relative else sum(rivals))


def _winprob_at(snap, me):
    """Placement of one future: the fraction of rivals I outscore (tie = half).
    In 2P this is win/loss; in 4P it rewards staying ahead of each rival, so it
    is sensitive to falling behind (not blind to defense like strict rank-#1)."""
    tot = ship_totals(snap)
    mine = tot.get(int(me), 0.0)
    rivals = [v for k, v in tot.items() if int(k) != int(me)]
    if not rivals:
        return 1.0
    return sum(1.0 if mine > r else (0.5 if mine == r else 0.0) for r in rivals) / len(rivals)


def _make_robust_value(snap, me, num_seats, opps, futures, leader_relative):
    """Build the `value(plan) -> (win_fraction, mean_margin)` scorer over a fixed
    set of sampled futures. For each future: apply [my plan, each rival's sampled
    move] this turn, then roll the tail with RIVALS expanding (cheap greedy) and
    ME IDLE -- so a capture I make NOW is judged against rivals who keep taking
    the map, and "I'd grab it later anyway" can't make a real capture look
    worthless (the passivity trap). `win_fraction` = how often I end up rank #1
    (the objective: win in the most futures); `mean_margin` (ships vs the leader)
    is the continuous tie-break only."""
    greedy = lite_greedy_policy
    use_orbit_leaf = _ORBIT_OK and os.environ.get("LR_ROBUST_LEAF", "orbit").strip().lower() != "rollout"

    def value(plan):
        wins = 0.0
        margin = 0.0
        n = 0
        for fut in futures:
            s = clone(snap)
            acts = [[] for _ in range(num_seats)]
            acts[int(me)] = list(plan)
            for i in opps:
                acts[i] = list(fut[i])
            step(s, acts, in_place=True)
            if use_orbit_leaf and not s.fake_env.done:
                # Strong leaf: the producer's garrison projection from the
                # post-move position (the evaluator that makes expansion pay).
                try:
                    w, m = _project_outcome(s.state[int(me)].observation, me,
                                            horizon=(ROBUST_HORIZON or None))
                    wins += w
                    margin += m
                    n += 1
                    continue
                except Exception:
                    pass
            # Fallback leaf: roll the tail with rivals expanding (greedy), me idle.
            for _ in range(max(0, ROBUST_K - 1)):
                if s.fake_env.done:
                    break
                a = [[] for _ in range(num_seats)]
                for j in opps:
                    a[j] = greedy(s.state[j].observation)
                step(s, a, in_place=True)
            wins += _winprob_at(s, me)
            margin += _leader_margin(s, me, num_seats, leader_relative)
            n += 1
        if n == 0:
            return (0.0, 0.0)
        return (wins / n, margin / n)

    return value


def _pack_caps(caps, avail0):
    """Concatenate captures' launches under a single source-availability budget,
    skipping any capture whose sources are already spent. Keeps a combined plan
    (e.g. defense + attacks) legal -- never double-spends a planet's garrison.
    Returns (emit, remaining_avail)."""
    avail = dict(avail0)
    emit = []
    for c in caps:
        srcs = c.get("srcs", {}) or {}
        if all(avail.get(s, 0) >= sz for s, sz in srcs.items()):
            emit.extend(c.get("emit") or [])
            for s, sz in srcs.items():
                avail[s] = avail.get(s, 0) - sz
    return emit, avail


def _robust_debug():
    return os.environ.get("LR_ROBUST_DEBUG", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _sample_futures(snap, me, opps, rng):
    """Sample `LR_ROBUST_SAMPLES` futures once (shared across candidates so the
    comparison is paired). A future = each rival's move THIS turn; rivals move
    simultaneously, blind to my plan, so the sample is shared across my plans."""
    n_samples = max(1, ROBUST_SAMPLES)
    return [{i: _stochastic_greedy(snap.state[i].observation, rng, ROBUST_TEMP)
             for i in opps} for _ in range(n_samples)]


def _robust_pick(obs, configuration, me, num_seats, candidate_plans,
                 budget_ms=None, labels=None):
    """Pick the whole candidate plan with the best robust value across the
    ensemble. Time-guarded; falls back to candidate_plans[0]. `labels` (optional,
    parallel to candidate_plans) annotate the LR_ROBUST_DEBUG trace."""
    if budget_ms is None:
        budget_ms = ROBUST_MS
    obs_d = _as_dict(obs)
    leader_relative = num_seats >= 4
    rng = random.Random(_board_seed(obs_d))
    snap = from_obs(obs, configuration, num_seats=num_seats)
    opps = [i for i in range(num_seats) if i != int(me)]
    futures = _sample_futures(snap, me, opps, rng)
    value = _make_robust_value(snap, me, num_seats, opps, futures, leader_relative)

    best_plan, best_v, best_label = candidate_plans[0], None, None
    dbg = []
    t0 = time.perf_counter()
    for idx, plan in enumerate(candidate_plans):
        if best_v is not None and (time.perf_counter() - t0) * 1000.0 > budget_ms:
            break
        try:
            v = value(plan)                  # (win_fraction, mean_margin); tuple compare
        except Exception:
            continue
        lbl = labels[idx] if labels and idx < len(labels) else str(idx)
        if dbg is not None:
            dbg.append((lbl, v))
        if best_v is None or v > best_v:
            best_v, best_plan, best_label = v, plan, lbl
    if _robust_debug():
        step_no = int(_as_dict(obs).get("step", 0))
        parts = " | ".join("%s win=%.2f m=%+.0f" % (l, v[0], v[1]) for l, v in dbg)
        sys.stderr.write("[robust step=%d seats=%d] chose=%s :: %s\n" % (
            step_no, num_seats, best_label, parts))
    return best_plan


def _robust_impact_pick(obs, configuration, me, num_seats, committed_caps,
                        budget_ms=None):
    """Impact-filtered robust pick. Build the emitted plan ONE CAPTURE AT A TIME,
    accepting a capture only if it improves the robust (CVaR) ship-margin across
    the sampled futures by at least the meaningful-impact floor (LR_ROBUST_IMPACT,
    in ships). A capture whose fleet is too small, too far (arrives after the
    simulation window), or redundant does not move the simulated outcome, so it
    is dropped; idle is the default when nothing clears the floor. This is the
    modelling-correct way to "avoid fleet sending without meaningful impact" --
    impact is measured by simulation, not by a hard size/distance cap. Captures
    are kept whole (gang-up launches stay together), never split into ineffective
    fragments. Returns env-format launches; time-guarded."""
    if budget_ms is None:
        budget_ms = ROBUST_MS
    obs_d = _as_dict(obs)
    leader_relative = num_seats >= 4
    rng = random.Random(_board_seed(obs_d))
    snap = from_obs(obs, configuration, num_seats=num_seats)
    opps = [i for i in range(num_seats) if i != int(me)]
    futures = _sample_futures(snap, me, opps, rng)
    value = _make_robust_value(snap, me, num_seats, opps, futures, leader_relative)

    # A capture is kept only if it MEANINGFULLY improves the outcome: it wins at
    # least `win_floor` more of the futures, OR (when the win-fraction is tied)
    # it improves the ship-margin tie-break by at least `margin_floor`. Tiny /
    # far / redundant captures move neither, so they're dropped; idle is the
    # default. This is "win in more futures" turned into an accept test.
    win_floor = _f("LR_ROBUST_WIN_FLOOR", 0.03)      # fraction of futures (~half a future at N=16)
    margin_floor = _f("LR_ROBUST_MARGIN_FLOOR", 8.0)  # ships, tie-break only

    def better(new, cur):
        wn, mn = new
        wc, mc = cur
        if wn - wc > win_floor:
            return True
        if wc - wn > win_floor:
            return False
        return mn > mc + margin_floor                 # win-fraction tied -> need real ship gain

    plan = []
    try:
        cur_v = value(plan)                          # outcome of doing nothing
    except Exception:
        return []
    t0 = time.perf_counter()
    for cap in committed_caps:
        if (time.perf_counter() - t0) * 1000.0 > budget_ms:
            break
        emit = cap.get("emit") or []
        if not emit:
            continue
        try:
            v = value(plan + emit)
        except Exception:
            continue
        if better(v, cur_v):                          # capture must win more futures (or clearly gain ships)
            plan = plan + emit
            cur_v = v
    return plan


# --------------------------------------------------------------------------
# The agent.
#
# IMPORTANT: kaggle_environments loads an agent file by picking the LAST
# top-level callable in the module (see kaggle_environments/agent.py:
# `[v for v in env.values() if callable(v)][-1]`). `agent` MUST therefore
# remain the final def in this file — do NOT add any module-level function or
# class below it, or that helper becomes the entry point and the agent idles
# every turn (it returns a non-move value, which the env silently drops).
# --------------------------------------------------------------------------
def agent(obs, configuration=None):
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    raw_planets = obs_d.get("planets", []) or []
    if not raw_planets:
        return []
    raw_fleets = obs_d.get("fleets", []) or []
    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if int(p.owner) == me]
    targets = [p for p in planets if int(p.owner) != me]
    if not my_planets or not targets:
        return []

    omega = float(obs_d.get("angular_velocity", 0.0) or 0.0)
    num_seats = _num_seats(planets, fleets)
    comet_ids = frozenset(int(c) for c in (obs_d.get("comet_planet_ids", []) or []))
    comet_paths = _comet_paths_by_id_safe(obs_d) if comet_ids else {}

    # ---- choose the evaluator (producer-strength orbit_lite, else fast_sim) ----
    use_orbit = _ORBIT_OK and FORCE_EVAL != "fallback"
    orbit = None
    if use_orbit:
        try:
            orbit = _build_orbit_scorer(obs, me)
        except Exception:
            orbit = None
    if orbit is None and FORCE_EVAL == "orbit":
        return []                       # explicit orbit-only request but unavailable
    score_units, id2slot = (orbit if orbit is not None else (None, None))

    # Fallback fast_sim scorer (used only when orbit_lite is unavailable).
    fb_snap = None
    if orbit is None:
        fb_snap = from_obs(obs, configuration, num_seats=num_seats)

    def value_fallback(emit_launches):
        s = clone(fb_snap)
        first = [list(emit_launches) if i == me else lite_greedy_policy(s.state[i].observation)
                 for i in range(num_seats)]
        step(s, first, in_place=True)
        for _ in range(FALLBACK_HORIZON - 1):
            if s.fake_env.done:
                break
            acts = [lite_greedy_policy(s.state[i].observation) for i in range(num_seats)]
            step(s, acts, in_place=True)
        return inflight_value(s.state[me].observation, me)

    # ---- candidate moves (least-resistance physics + ordering) ----
    opp_xy = [(float(p.x), float(p.y)) for p in planets
              if int(p.owner) != me and int(p.owner) != -1]
    ref_speed = max(1e-6, fleet_speed(FRONTIER_REF_SHIPS))

    def frontier_eta(xy):
        if not opp_xy:
            return 0.0
        return min(dist(xy, o) for o in opp_xy) / ref_speed

    available = {int(p.id): int(p.ships) for p in my_planets}
    by_id = {int(p.id): p for p in planets}
    # each candidate: emit=[[src_id,angle,ships],...], units=[(src_slot,tgt_slot,ships,eta),...],
    #                 srcs={src_id:ships}, rank, front
    candidates = []
    # Lever 2 (default 1.0 = off; 4P-only -- 2P is already strong and the scorer
    # focuses the one opponent correctly there): boost enemy-owned targets so
    # denial captures (taking from opponents) outrank equal-production neutrals.
    enemy_boost = _f("LR_ENEMY_BOOST", 1.0) if num_seats >= 4 else 1.0
    # Hold-sizing (default 0.5; confirmed vs Producer V2): size enemy captures to
    # take AND HOLD -- add surplus garrison to survive the opponent's retake.
    # Larger sizes force source-combining, so fewer / bigger fleets (concentration)
    # emerge naturally. Set LR_HOLD_MARGIN=0 to disable.
    hold_margin = _f("LR_HOLD_MARGIN", 0.5)

    def units_for(launch_triples):
        # launch_triples: list of (src_id, tgt_id, ships, eta)
        if id2slot is None:
            return None
        out = []
        for (sid, tid, sh, eta) in launch_triples:
            if sid not in id2slot or tid not in id2slot:
                return None
            out.append((id2slot[sid], id2slot[tid], int(sh), int(eta)))
        return out

    # Dropout-prune (PI 2026-06-20): strongest single enemy mass that can REACH a
    # target by about our arrival + a retake window -> probability the capture is
    # flipped back. Decline high-dropout attacks (incoming fleets / nearby opponent;
    # far/high-EDA launches give more time to contest) so we don't waste fleets.
    _dp_on = _dropout_prune()
    _enemy_pl = [((float(p.x), float(p.y)), float(p.ships)) for p in planets
                 if int(p.owner) != me and int(p.owner) != -1] if _dp_on else []
    _enemy_fl = [((float(f.x), float(f.y)), float(f.ships)) for f in fleets
                 if int(f.owner) != me and int(f.owner) != -1] if _dp_on else []
    _dp_buf = _f("LR_RETAKE_BUFFER", 4.0)
    _dp_steep = _f("LR_DROPOUT_STEEPNESS", 5.0)
    _dp_max = _f("LR_DROPOUT_MAX", 0.8)   # only decline clearly-doomed attacks

    def _attack_dropout(txy, eta, our_hold):
        horizon = float(eta) + _dp_buf
        reach = 0.0
        for (qxy, qsh) in _enemy_pl:
            if qsh > reach and dist(txy, qxy) <= horizon * fleet_speed(qsh):
                reach = qsh
        for (fxy, fsh) in _enemy_fl:
            if fsh > reach and dist(txy, fxy) <= horizon * fleet_speed(fsh):
                reach = fsh
        if reach <= 0.0:
            return 0.0
        bal = (reach - our_hold) / (reach + our_hold + 1.0)
        return 1.0 / (1.0 + math.exp(-_dp_steep * bal))

    for tgt in targets:
        tid = int(tgt.id)
        is_enemy = int(tgt.owner) != -1
        is_comet = tid in comet_ids
        if is_comet and _skip_comets():
            continue                       # comet aim can miss -> oob waste; disabled for now
        prod = float(tgt.production)

        shots = []   # (eta, size, sid, src, angle)
        for src in my_planets:
            shot = _plan_shot(src, tgt, comet_ids, comet_paths, omega, RANK_HINT_SHIPS)
            if shot is None:
                continue
            angle, eta, arr = shot
            if not _sun_clear(src, arr):
                continue
            if is_comet:
                life = comet_remaining_lifetime(tid, _ObsRawShim(obs_d))
                if life is not None and life <= eta:
                    continue
            defenders = prod * eta + tgt.ships if is_enemy else tgt.ships
            size = int(math.ceil(defenders)) + 1
            if is_enemy and hold_margin > 0.0:
                size += int(math.ceil(hold_margin * defenders))   # surplus to hold
            shots.append((eta, size, int(src.id), src, angle))
        if not shots:
            continue
        shots.sort(key=lambda x: x[0])
        if _dp_on:
            eta0, size0 = shots[0][0], shots[0][1]
            defenders0 = (prod * eta0 + tgt.ships) if is_enemy else tgt.ships
            our_hold = max(0.0, size0 - defenders0) + prod * _dp_buf
            if _attack_dropout((float(tgt.x), float(tgt.y)), eta0, our_hold) > _dp_max:
                continue                       # likely retaken -> don't waste the fleet
        rank = prod / max(1.0, shots[0][0])
        if is_enemy and enemy_boost != 1.0:
            rank *= enemy_boost
        front = frontier_eta((float(tgt.x), float(tgt.y)))

        # Solo capture from the cheapest affordable source — re-aim at the
        # actual size so the emit angle and the scorer eta are accurate.
        solo = None
        for (eta, size, sid, src, angle) in shots:
            if available[sid] >= size:
                shot = _plan_shot(src, tgt, comet_ids, comet_paths, omega, size)
                if shot is None:
                    continue
                a2, eta2, _ = shot
                triples = [(sid, tid, size, eta2)]
                units = units_for(triples)
                if units is None and id2slot is not None:
                    continue
                solo = {"emit": [[sid, float(a2), size]],
                        "units": units, "srcs": {sid: size},
                        "rank": rank, "front": front}
                break
        if solo is not None:
            candidates.append(solo)
            continue

        # Gang-up the nearest sources when none can solo (neutral attrition or
        # near-simultaneous enemy wave; the evaluator validates either way).
        need = shots[0][1]
        emit, triples, srcs, acc = [], [], {}, 0
        for (eta, size, sid, src, angle) in shots:
            if sid in srcs:
                continue
            take = min(available[sid], need - acc)
            if take <= 0:
                continue
            shot = _plan_shot(src, tgt, comet_ids, comet_paths, omega, take)
            if shot is None:
                continue
            a2, eta2, _ = shot
            emit.append([sid, float(a2), take])
            triples.append((sid, tid, take, eta2))
            srcs[sid] = take
            acc += take
            if acc >= need:
                break
        if acc >= need and emit:
            units = units_for(triples)
            if not (units is None and id2slot is not None):
                candidates.append({"emit": emit, "units": units, "srcs": srcs,
                                   "rank": rank, "front": front})

    # Regroup / defense (default ON; confirmed vs Producer V2): reinforce our own
    # planets that an enemy fleet is bearing down on with enough force to flip --
    # keep HELD production instead of only ever grabbing new planets (the move we
    # were structurally blind to). Set LR_DEFEND=0 to disable.
    if (os.environ.get("LR_DEFEND", "1").strip().lower() in ("1", "true", "on", "yes")
            and my_planets):
        defend_range = _f("LR_DEFEND_RANGE", 35.0)
        enemy_fleets = [f for f in fleets
                        if int(f.owner) != me and int(f.owner) != -1]
        for mine in my_planets:
            mxy = (float(mine.x), float(mine.y))
            threat = sum(float(f.ships) for f in enemy_fleets
                         if dist(mxy, (float(f.x), float(f.y))) <= defend_range)
            if threat <= float(mine.ships):
                continue                                  # not under real threat
            deficit = int(math.ceil(threat - float(mine.ships))) + 1
            donors = sorted((p for p in my_planets if int(p.id) != int(mine.id)),
                            key=lambda p: dist(mxy, (float(p.x), float(p.y))))
            d_emit, d_triples, d_srcs, acc = [], [], {}, 0
            for d in donors:
                take = min(available.get(int(d.id), 0), deficit - acc)
                if take <= 0:
                    continue
                shot = _plan_shot(d, mine, comet_ids, comet_paths, omega, take)
                if shot is None:
                    continue
                a2, eta2, arr = shot
                if not _sun_clear(d, arr):
                    continue
                d_emit.append([int(d.id), float(a2), take])
                d_triples.append((int(d.id), int(mine.id), take, eta2))
                d_srcs[int(d.id)] = take
                acc += take
                if acc >= deficit:
                    break
            if d_emit:
                units = units_for(d_triples)
                if not (units is None and id2slot is not None):
                    candidates.append({"emit": d_emit, "units": units,
                                       "srcs": d_srcs,
                                       "rank": float(mine.production) * 2.0,
                                       "front": 0.0, "kind": "defend"})

    if not candidates:
        return []

    candidates.sort(key=lambda c: (-c["rank"], c["front"]))
    candidates = candidates[:MAX_CANDIDATES]

    # ---- greedy plan construction by projected value ----
    committed_emit = []
    committed_units = []
    committed_caps = []                  # accepted captures (kept whole) for the robust impact filter
    avail = dict(available)
    if orbit is not None:
        current = 0.0                       # score of the empty plan
        floor = ROI_FLOOR
    else:
        current = value_fallback([])
        floor = 0.5
    budget_ms = _value_budget(obs_d, _wallclock_ms())
    t0 = time.perf_counter()

    # Fundamental (default OFF, both modes): order captures by their VALUE under
    # the objective (highest win-equity first) instead of cheapness, spending
    # spare compute to score each once -- funds the captures that actually win
    # before scattered cheap neutrals (principled replacement for enemy-boost).
    if _value_commit() and orbit is not None and len(candidates) > 1:
        scored = []
        for c in candidates:
            if c["units"] is None or (time.perf_counter() - t0) * 1000.0 > budget_ms:
                scored.append((float("-inf"), c))      # unscored -> keep after scored
            else:
                scored.append((score_units(c["units"]), c))
        scored.sort(key=lambda e: -e[0])               # highest marginal value first
        candidates = [c for _, c in scored]

    for c in candidates:
        if (time.perf_counter() - t0) * 1000.0 > budget_ms:
            break
        if any(avail.get(s, 0) < sz for s, sz in c["srcs"].items()):
            continue
        if orbit is not None:
            v = score_units(committed_units + c["units"])
        else:
            v = value_fallback(committed_emit + c["emit"])
        if v > current + floor:
            committed_emit = committed_emit + c["emit"]
            committed_units = committed_units + (c["units"] or [])
            committed_caps.append(c)
            current = v
            for s, sz in c["srcs"].items():
                avail[s] = avail.get(s, 0) - sz

    # ---- 2-ply lookahead pick (2P only): choose among a few full-plans by
    #      their value AFTER the producer's reply + a producer-vs-producer turn,
    #      so moves the producer punishes next turn are correctly down-rated.
    if orbit is not None and TWOPLY and num_seats >= 2:
        try:
            producer_me = _producer_move_obs(obs, me)
        except Exception:
            producer_me = []
        # Levers 2/3 are 4P-only: 2P is our strength and these regress it.
        anytime_on = _anytime() and num_seats >= 4
        plans = [producer_me, committed_emit, []]   # producer floor first
        if _native_leaf() and len(committed_emit) > 1:
            # Offer every CONCENTRATION level (1..n-1 launches) so the native
            # ship-margin leaf can keep only the captures that hold and drop the
            # scattered tail the flip hazard says the opponent retakes.
            plans.extend(committed_emit[:k] for k in range(1, len(committed_emit)))
        elif anytime_on:
            # Lever 3: spend headroom -- offer every aggression level of the
            # committed plan, so extra compute becomes more plans evaluated.
            plans.extend(committed_emit[:k] for k in range(1, len(committed_emit)))
        elif len(committed_emit) > 2:
            # One milder aggression level of my plan for the lookahead to weigh.
            plans.append(committed_emit[:len(committed_emit) // 2])
        # De-dup (by repr) preserving order.
        seen, uniq = set(), []
        for p in plans:
            key = repr(p)
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        try:
            if _robust() and num_seats >= 4 and not _native_leaf():
                # Opponent-agnostic robust-ensemble pick (default OFF), 4-PLAYER
                # ONLY. Capability leaf (production + defensibility + reach), rank
                # placement objective. In 2P we fall through to the proven 2-ply
                # path (the capability/defensibility leaf is too cautious for the
                # 2P knife-fight, where aggressive expansion wins).
                # LR_ROBUST_FILTER=1 instead builds the plan capture-by-capture.
                if _i("LR_ROBUST_FILTER", 0) >= 1:
                    return _robust_impact_pick(obs, configuration, me, num_seats,
                                               committed_caps,
                                               budget_ms=_robust_budget(obs_d))
                # Build the robust candidate set INCLUDING standalone defense, so
                # the win-prob leaf can choose to HOLD a planet about to flip
                # rather than only ever counterattacking (PI observation). Defense
                # captures are source-packed first so defend+attack stays legal.
                defend_caps = [c for c in candidates if c.get("kind") == "defend"]
                defend_emit, av_after_def = _pack_caps(defend_caps, available)
                atk_on_def, _ = _pack_caps(committed_caps, av_after_def)
                rplans = [committed_emit, producer_me]
                rlabels = ["attack", "producer"]
                if defend_emit:
                    rplans.append(defend_emit); rlabels.append("defend")
                    rplans.append(defend_emit + atk_on_def); rlabels.append("defend+attack")
                rplans.append([]); rlabels.append("idle")
                seen2, runiq, rlab = set(), [], []
                for p, l in zip(rplans, rlabels):
                    k = repr(p)
                    if k not in seen2:
                        seen2.add(k); runiq.append(p); rlab.append(l)
                return _robust_pick(obs, configuration, me, num_seats, runiq,
                                    budget_ms=_robust_budget(obs_d), labels=rlab)
            depth = _rollout_depth()
            if depth >= 2:
                return _deep_pick(obs, configuration, me, num_seats, uniq,
                                  depth, budget_ms=_deep_budget(obs_d))
            return _twoply_pick(obs, configuration, me, num_seats, uniq,
                                budget_ms=_twoply_budget(obs_d) if anytime_on else None)
        except Exception:
            return committed_emit

    return committed_emit
