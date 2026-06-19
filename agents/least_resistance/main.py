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
    import importlib.util as _ilu
    _pm_spec = _ilu.spec_from_file_location("_lr_producer_main", _PRODUCER_MAIN)
    _producer_main = _ilu.module_from_spec(_pm_spec)
    sys.modules["_lr_producer_main"] = _producer_main
    _pm_spec.loader.exec_module(_producer_main)
    _ORBIT_OK = True
except Exception:
    _ORBIT_OK = False

# Fallback evaluator deps (pure Python, no torch).
from lib.fast_sim import from_obs, clone, step
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
    """Deep-search opponent model (Phase 1). Default 0 = the producer mirror
    (`_producer_move_obs`, byte-identical to the shipped agent). 1 = the cheap
    ROI-greedy expansion policy (`lib.opp_model.lite_greedy_policy`, ~1-2 ms vs
    the mirror's ~10-50 ms per node), so the K-turn rollout can afford more
    search depth under the 1000 ms wall. Read at call time."""
    return _i("LR_DEEP_OPP", 0)


def _contagion_reach_ticks():
    """Per-step reach window for the contagion opponent (LR_DEEP_OPP=2): a rival
    source can overrun a target within `fleet_speed(ships) * reach` of it this
    step. Read at call time."""
    return _f("LR_CONTAGION_REACH_TICKS", 3.0)


def _contagion_thin():
    """Garrison at/below which one of MY planets counts as THINLY HELD -- inherently
    exposed, so the contagion can overrun it from extended range (and unbounded). This
    PUNISHES over-extension (grab-all fragmentation) in the rollout, but at too high a
    threshold it also sweeps legitimately-small early holdings and over-prunes good
    plans -- so it is DEFAULT OFF (0 = inert, original contagion); enable+tune via the
    A/B. Read at call time."""
    return _f("LR_CONTAGION_THIN", 0.0)


def _contagion_thin_reach():
    """Extended reach window (multiplier on fleet_speed) for overrunning MY thinly
    held planets -- a thin, scattered capture is vulnerable to rivals farther away
    than a well-garrisoned one. Read at call time."""
    return _f("LR_CONTAGION_THIN_REACH", 8.0)


def _wide_candidates():
    """Wide candidate generation (default OFF). When on, the deep search chooses
    among a DIVERSE pool of full-turn plans (different aggression / order / theme)
    instead of ~5 near-identical variants of the single greedy plan. Read at call
    time."""
    return os.environ.get("LR_WIDE_CANDIDATES", "0").strip().lower() in (
        "1", "true", "on", "yes")


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


def _deep_opp_move(obs_any, seat, mode):
    """One node's opponent (and rollout-continuation) move for the deep search,
    dispatched by LR_DEEP_OPP (`mode`, read once per turn). Mode 0 = the producer
    mirror (default, byte-identical). Mode 1 = the cheap lite_greedy expansion
    policy -- it reads the acting seat from `obs_any.player`, which fast_sim sets
    to `seat` on every per-seat observation. Falls back to [] on any error so a
    bad model can never crash the rollout (the search keeps the producer-floor
    candidate)."""
    if mode == 1:
        try:
            return lite_greedy_policy(obs_any)
        except Exception:
            return []
    return _producer_move_obs(obs_any, seat)


def _apply_contagion(snap, me):
    """Mode-2 opponent (LR_DEEP_OPP=2): a deterministic, model-free CONTAGION flip
    applied once per rollout step, REPLACING explicit opponent launches. Rivals
    expand onto NEUTRALS and overrun MY under-defended planets; each target flips
    toward the single STRONGEST reachable rival (max-aggregate threat, per the
    dropout-plan-review learnings -- NOT the summed enemy mass). Each rival source
    flips at most one target per step (bounded rate), so the front snowballs across
    steps rather than swallowing the board in one tick; newly-flipped planets become
    sources next step (compounding). No RNG -> rollout/CPU-GPU deterministic.

    Operates in-place on the shared mutable planet rows
    `[id, owner, x, y, radius, ships, production]` (the interpreter respects
    in-place owner/ships edits). My reinforced/held planets out-mass the local rival
    and survive, so the ranking signal is 'which planets does my move let me hold'."""
    me = int(me)
    planets = snap.state[0].observation.planets
    reach = _contagion_reach_ticks()
    thin = _contagion_thin()
    thin_reach = _contagion_thin_reach()
    rivals = [p for p in planets if int(p[1]) >= 0 and int(p[1]) != me]
    if not rivals:
        return
    # Strongest rivals first so the first reachable match IS the max-threat source.
    rivals.sort(key=lambda p: float(p[5]), reverse=True)
    used = set()                       # rival source ids that already flipped this step
    for tgt in planets:
        owner = int(tgt[1])
        if owner != -1 and owner != me:
            continue                   # only neutrals and my planets can be overrun
        # step() already accrued this turn's production, so current ships = defense.
        defense = float(tgt[5])
        # A thinly-held planet of MINE is exposed: the contagion reaches it from
        # FARTHER (extended reach) AND a rival can mop up MANY thin planets in one
        # step (it bypasses the one-flip-per-source bound -- thin captures are nearly
        # free). This is what PUNISHES grab-all fragmentation in the rollout, so the
        # deep search stops picking scattered tiny-fleet plans. Well-garrisoned
        # planets keep the tighter base reach, the per-source bound, and the out-mass
        # protection below.
        is_thin = (owner == me and defense <= thin)
        eff_reach = thin_reach if is_thin else reach
        tpt = (float(tgt[2]), float(tgt[3]))
        trad = float(tgt[4])
        for q in rivals:
            qid = int(q[0])
            if qid == int(tgt[0]) or (qid in used and not is_thin):
                continue
            qships = float(q[5])
            if qships <= defense:
                continue               # this single rival can't out-mass the target
            d = dist((float(q[2]), float(q[3])), tpt) - float(q[4]) - trad
            spd = fleet_speed(qships)
            if spd <= 0.0 or d > spd * eff_reach:
                continue               # not reachable this step
            tgt[1] = int(q[1])         # flip ownership to the strongest reachable rival
            tgt[5] = max(1.0, qships - defense)   # garrison with the landing surplus
            if not is_thin:
                used.add(qid)          # normal captures stay bounded; thin mop-up is free
            break


def _reachable_rival_mass(target_row, planets, me, reach=None):
    """Strongest single rival (max-aggregate threat, NOT the sum) that can reach
    `target_row` within `fleet_speed(ships) * reach` this step -> that rival's ship
    count (0.0 if none). The shared 'what can overrun this planet' measure: used to
    ORDER captures by holdability (take what rivals can't retake) and conceptually
    mirrors the contagion opponent's flip test. Indexes rows as
    [id, owner, x, y, radius, ships, production] -- works on Planet namedtuples and
    raw obs rows alike."""
    me = int(me)
    if reach is None:
        reach = _contagion_reach_ticks()
    tid = int(target_row[0])
    tx, ty, trad = float(target_row[2]), float(target_row[3]), float(target_row[4])
    best = 0.0
    for q in planets:
        if int(q[1]) < 0 or int(q[1]) == me or int(q[0]) == tid:
            continue
        qs = float(q[5])
        if qs <= best:
            continue
        d = dist((float(q[2]), float(q[3])), (tx, ty)) - float(q[4]) - trad
        spd = fleet_speed(qs)
        if spd > 0.0 and d <= spd * reach:
            best = qs
    return best


def _greedy_commit_cheap(candidates, order_key, available, max_captures=None):
    """Score-free plan builder: commit affordable captures in `order_key` order until
    sources run out (or `max_captures` reached). Diversity for the deep search comes
    from the ORDER / SUBSET / cardinality chosen here, NOT from re-scoring -- the
    contagion rollout in _deep_pick does the real (expensive) evaluation, so plan
    generation stays cheap. Returns the env-format emit `[[src_id, angle, ships], ...]`."""
    avail = dict(available)
    emit, n = [], 0
    for c in sorted(candidates, key=order_key):
        if max_captures is not None and n >= max_captures:
            break
        srcs = c["srcs"]
        if any(avail.get(s, 0) < sz for s, sz in srcs.items()):
            continue
        emit = emit + c["emit"]
        for s, sz in srcs.items():
            avail[s] = avail.get(s, 0) - sz
        n += 1
    return emit


def _wide_candidate_plans(candidates, producer_me, committed_emit, available,
                          planets, by_id, me, max_plans=None):
    """A diverse, de-duped, capped pool of full-turn plans for _deep_pick to score.
    Spans aggression (cardinality), commit ORDER (rank / holdability / nearest), and
    THEME (enemy-denial / neutral-expansion), plus the safe anchors (producer floor,
    the shipped greedy plan, hold-everything). Ordered safe/strong first so the
    time-guarded deep search keeps the best of what it can afford (anytime-safe)."""
    if max_plans is None:
        max_plans = _i("LR_WIDE_MAX", 12)
    rank_order = lambda c: (-float(c["rank"]), float(c["front"]))
    near_order = lambda c: float(c["front"])
    def hold_order(c):
        tid = c.get("tid")
        return _reachable_rival_mass(by_id[tid], planets, me) if tid in by_id else 1e18
    pool = [producer_me, committed_emit, []]            # safe/strong anchors first
    pool.append(_greedy_commit_cheap(candidates, rank_order, available))
    pool.append(_greedy_commit_cheap(candidates, hold_order, available))   # take-and-hold
    pool.append(_greedy_commit_cheap(candidates, rank_order, available, max_captures=1))
    pool.append(_greedy_commit_cheap(candidates, hold_order, available, max_captures=3))
    pool.append(_greedy_commit_cheap(candidates, near_order, available))
    for kind in ("enemy", "neutral"):                  # denial vs expansion themes
        sub = [c for c in candidates if c.get("kind") == kind]
        if sub:
            pool.append(_greedy_commit_cheap(sub, rank_order, available))
    seen, uniq = set(), []
    for p in pool:
        key = repr(sorted(p, key=lambda e: (int(e[0]), round(float(e[1]), 3), int(e[2]))))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
        if len(uniq) >= max_plans:
            break
    return uniq


def _project_value(obs_any, me):
    """Position value: project the board `H` turns forward (all in-flight
    fleets + production + combat, no new launches) and return our garrison
    advantage (our ships - opponents') at the horizon. The producer's own
    garrison-flow projector used as a state evaluator."""
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


def _twoply_pick(obs, configuration, me, num_seats, candidate_plans, budget_ms=None):
    """Pick the candidate full-plan with the best 2-ply value: apply [my plan,
    producer's predicted reply] this turn, then a turn of producer-vs-producer,
    then score the resulting position. `candidate_plans` always includes the
    producer's own move (the >=-producer floor). Returns the chosen plan."""
    if budget_ms is None:
        budget_ms = TWOPLY_BUDGET_MS
    snap = from_obs(obs, configuration, num_seats=num_seats)
    opps = [i for i in range(num_seats) if i != int(me)]
    # Every opponent's predicted move this turn (each modelled as the producer);
    # shared across candidates since moves are simultaneous (they don't see my
    # plan). Works for 2P (one opponent) and 4P (three opponents).
    opp_now = {i: _producer_move_obs(snap.state[i].observation, i) for i in opps}

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
                nxt[i] = _producer_move_obs(s.state[i].observation, i)
            step(s, nxt, in_place=True)
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


def _deep_pick(obs, configuration, me, num_seats, candidate_plans, depth, budget_ms=None):
    """Deeper search vs a fixed-policy opponent (the producer). For each candidate
    first-move, roll the game out `depth` turns -- turn 1 = my move + each
    opponent's producer reply; turns 2..depth = EVERY seat (incl. me) plays the
    producer -- then score with the analytic leaf. Plans are tried in the given
    (value-ranked) order; best-so-far kept; time-guarded (anytime-safe). Modelling
    opponents AS the producer is exact for the 'beat the producer' goal, and the
    producer's own move is always among `candidate_plans` (>=-producer floor)."""
    if budget_ms is None:
        budget_ms = TWOPLY_BUDGET_MS
    snap = from_obs(obs, configuration, num_seats=num_seats)
    opps = [i for i in range(num_seats) if i != int(me)]
    opp_mode = _deep_opp()  # read the opponent-model knob once per turn
    # Mode 2 (contagion) REPLACES opponent launches with a per-step ownership flip,
    # so it never calls the (expensive) opponent move models -- skip the opp cache.
    contagion = (opp_mode == 2)
    opp_now = ({} if contagion else
               {i: _deep_opp_move(snap.state[i].observation, i, opp_mode)
                for i in opps})

    def rollout_value(plan):
        s = clone(snap)
        acts = [[] for _ in range(num_seats)]
        acts[int(me)] = list(plan)
        if not contagion:
            for i in opps:
                acts[i] = list(opp_now[i])
        step(s, acts, in_place=True)
        if contagion:
            _apply_contagion(s, me)
        for _ in range(max(0, int(depth) - 1)):
            if s.fake_env.done:
                break
            if contagion:
                # I take-and-hold (no further launches); rivals snowball via the flip.
                step(s, [[] for _ in range(num_seats)], in_place=True)
                _apply_contagion(s, me)
            else:
                nxt = [_deep_opp_move(s.state[i].observation, i, opp_mode)
                       for i in range(num_seats)]
                step(s, nxt, in_place=True)
        try:
            return _project_value(s.state[int(me)].observation, me)
        except Exception:
            return None

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

    for tgt in targets:
        tid = int(tgt.id)
        is_enemy = int(tgt.owner) != -1
        is_comet = tid in comet_ids
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
                a2, eta2, arr2 = shot
                if not _sun_clear(src, arr2):
                    continue            # re-aim at the real size can clip the sun
                triples = [(sid, tid, size, eta2)]
                units = units_for(triples)
                if units is None and id2slot is not None:
                    continue
                solo = {"emit": [[sid, float(a2), size]],
                        "units": units, "srcs": {sid: size},
                        "rank": rank, "front": front,
                        "kind": "enemy" if is_enemy else "neutral", "tid": tid}
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
            a2, eta2, arr2 = shot
            if not _sun_clear(src, arr2):
                continue                # re-aim at the real size can clip the sun
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
                                   "rank": rank, "front": front,
                                   "kind": "enemy" if is_enemy else "neutral",
                                   "tid": tid})

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
                                       "front": 0.0,
                                       "kind": "defend", "tid": int(mine.id)})

    if not candidates:
        return []

    candidates.sort(key=lambda c: (-c["rank"], c["front"]))
    candidates = candidates[:MAX_CANDIDATES]

    # ---- greedy plan construction by projected value ----
    committed_emit = []
    committed_units = []
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
        if _wide_candidates():
            # Wide generation: hand the deep search a DIVERSE pool of full-plans to
            # score, not ~5 variants of one greedy plan (the candidate-breadth lever).
            uniq = _wide_candidate_plans(candidates, producer_me, committed_emit,
                                         available, planets, by_id, me)
        else:
            plans = [producer_me, committed_emit, []]   # producer floor first
            if anytime_on:
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
            depth = _rollout_depth()
            if depth >= 2:
                return _deep_pick(obs, configuration, me, num_seats, uniq,
                                  depth, budget_ms=_deep_budget(obs_d))
            return _twoply_pick(obs, configuration, me, num_seats, uniq,
                                budget_ms=_twoply_budget(obs_d) if anytime_on else None)
        except Exception:
            return committed_emit

    return committed_emit
