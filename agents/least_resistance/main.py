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
# Threat-aware capture sizing (LR_THREAT_SIZE): radius within which an enemy
# fleet/garrison counts as able to recapture a planet, and how much of a parked
# garrison to count (opponents won't commit all of it -- doing so empties their
# own planet). 40 / 0.5 chosen from the recapture analysis (100% of ladder
# recaptures had threat within 40; in-flight is the dominant term).
THREAT_RADIUS = _f("LR_THREAT_RADIUS", 40.0)
STATIONED_DISCOUNT = _f("LR_STATIONED_DISCOUNT", 0.5)
RECAP_RANGE = _f("LR_RECAP_RANGE", 45.0)   # opponent-model recapture reach (lookahead only)
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


def _rollout_eval():
    """FUNDAMENTAL foresight gate (default OFF). Score candidate plans by a real
    fast_sim rollout under the producer policy -- where opponents RESPOND, incl.
    launching at our thinly-held captures -- instead of the orbit_lite garrison-
    FLOW projection, which is blind to discrete recaptures. A real-state probe
    over 40 ladder recaptures showed the flow leaf foresees ~0% while a producer-
    opponent rollout reproduces ~68% of them. This is where the compute headroom
    finally buys actual lookahead. Set LR_ROLLOUT_EVAL=1 to enable."""
    return os.environ.get("LR_ROLLOUT_EVAL", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _threat_size():
    """FUNDAMENTAL fix for the 4P recapture churn (default OFF). Size every
    capture to beat the enemy force ALREADY able to reach that planet (in-flight
    fleets + discounted parked garrison) instead of minimum force, so a planet we
    take is one we can HOLD. Unaffordable (truly contested) captures are dropped
    by the existing affordability check -> we skip the doomed grab and expand
    elsewhere, rather than churning ships into planets that flip straight back.
    The threat is 100% foreseeable from the observation (measured), so this is
    analytic -- no rollout needed. Set LR_THREAT_SIZE=1 to enable."""
    return os.environ.get("LR_THREAT_SIZE", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _arrival_cap():
    """Default OFF. Skip generating ENEMY captures whose ETA exceeds the
    evaluation horizon -- the garrison-flow leaf scores those with the fleet still
    in transit (capture unresolved), so far attacks look 'free' and we over-extend
    (measured: 77% of enemy attacks fly to ETA ~20-25 vs a 13-turn 4P horizon). The
    champion agent had this launch-arrival ceiling; least_resistance dropped it.
    Set LR_ARRIVAL_CAP=1 to enable."""
    return os.environ.get("LR_ARRIVAL_CAP", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _hold_neutral():
    """Default OFF. Apply the hold-margin to NEUTRAL captures too, not just enemy
    ones. Our captures are neutral-dominated and min-force (median fleet ~20) while
    V2 consolidates (median ~36) and holds them; the enemy-only margin never moved
    our sizes. Only bites when hold_margin>0 (i.e. 2P). Set LR_HOLD_NEUTRAL=1."""
    return os.environ.get("LR_HOLD_NEUTRAL", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _holdability():
    """Default OFF. The recapture PENALTY (distinct from sizing): in the greedy, a
    capture whose post-landing garrison can't survive the visible reachable enemy
    force is penalized by its foregone production, so the greedy SKIPS doomed
    captures and spends our (scarce) ships on holdable ones instead of churning.
    Selection, not sizing -- fits a ship-constrained agent. Set LR_HOLDABILITY=1."""
    return os.environ.get("LR_HOLDABILITY", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _recapture_opp():
    """FUNDAMENTAL fix for the recapture churn (default OFF). The lookahead's
    opponent model (the producer policy) recaptures our thinly-held captures only
    ~13% of the time, but the real ladder recaptures ~43% -- so the simulation
    thinks thin captures are safe and the agent makes them. With this on, the
    opponent's modeled REPLY also launches to retake the weak planets we just took
    (a competent reactive recapture), so the production-aware flow-leaf terminal
    correctly penalizes captures that will not hold, and the agent prefers the
    holdable plan. Selection, not sizing (no over-commit); production-aware (no
    defeatism). Set LR_RECAP_OPP=1 to enable."""
    return os.environ.get("LR_RECAP_OPP", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _confident_only():
    """PIVOT (default OFF). Commit only HIGH-CONFIDENCE actions. Of the launches
    we are about to play (usually the producer's move), keep a capture only if we
    can HOLD the planet we take -- the garrison we land with (ships sent minus the
    target's defenders) is at least the enemy force that can still reach it
    (in-flight fleets + nearby parked garrison, discounted). Safe grabs (nothing
    can reach them) and reinforcements of our own planets are always kept; thin
    contested grabs that would flip straight back are dropped, and those ships stay
    home. No count cap, no concentration -- just confidence. Set LR_CONFIDENT=1."""
    return os.environ.get("LR_CONFIDENT", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _recapture_moves(obs_dict, seat, exclude_srcs=()):
    """A competent opponent's reactive recapture, used ONLY inside the lookahead's
    opponent replies: `seat` launches from its nearest unused source to retake the
    cheapest weakly-held rival planets within reach (one launch per source). This
    is what real ladder opponents do and the producer model omits."""
    raw = obs_dict.get("planets", []) or []
    if not raw:
        return []
    planets = [Planet(*p) for p in raw]
    mine = [p for p in planets
            if int(p.owner) == seat and int(p.id) not in exclude_srcs]
    rivals = [p for p in planets if int(p.owner) != seat and int(p.owner) != -1]
    if not mine or not rivals:
        return []
    omega = float(obs_dict.get("angular_velocity", 0.0) or 0.0)
    # Thread real comet data so a rival-held comet is aimed along its path, not at
    # its stale current position (it moves each turn).
    comet_ids = frozenset(int(c) for c in (obs_dict.get("comet_planet_ids", []) or []))
    comet_paths = _comet_paths_by_id_safe(obs_dict) if comet_ids else {}
    avail = {int(p.id): float(p.ships) for p in mine}
    out, used = [], set()
    for tgt in sorted(rivals, key=lambda p: float(p.ships)):   # cheapest retakes first
        need = int(tgt.ships) + 1
        txy = (float(tgt.x), float(tgt.y))
        for src in sorted(mine, key=lambda s: dist((float(s.x), float(s.y)), txy)):
            sid = int(src.id)
            if sid in used or avail.get(sid, 0.0) < need:
                continue
            if dist((float(src.x), float(src.y)), txy) > RECAP_RANGE:
                break                                          # nearest is already too far
            shot = _plan_shot(src, tgt, comet_ids, comet_paths, omega, need)
            if shot is None:
                continue
            ang, _eta, _arr = shot
            out.append([sid, float(ang), need])
            used.add(sid)
            avail[sid] -= need
            break
    return out


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
    """Seats to model = (highest live seat index + 1), clamped to >= 2. A seat is
    live if it owns a planet OR has an in-flight fleet, so a fully-eliminated
    high-index seat is dropped (its slot would only idle) while every seat with
    any presence is covered. Correct for both the lookahead seat count and the
    2P-vs-multi-front strategy gate. (Was `4 if max_owner>=2 else 2`, which
    over-counted a 3-player board as 4 and conflated owner-index with count.)"""
    max_owner = -1
    for p in planets:
        max_owner = max(max_owner, int(p.owner))
    for f in fleets:
        max_owner = max(max_owner, int(f.owner))
    return max(2, max_owner + 1)


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
            # 1-ply scorer misses). With LR_RECAP_OPP, each opponent ALSO launches
            # to retake the weak planets we just took (the reactive recapture the
            # producer model omits) -- so thin captures get retaken in-sim and the
            # flow-leaf terminal penalizes them.
            recap_on = _recapture_opp()
            nxt = [[] for _ in range(num_seats)]
            for i in opps:
                pm = _producer_move_obs(s.state[i].observation, i)
                if recap_on:
                    used = {int(l[0]) for l in pm if isinstance(l, (list, tuple)) and l}
                    pm = pm + _recapture_moves(s.state[i].observation, i, exclude_srcs=used)
                nxt[i] = pm
            step(s, nxt, in_place=True)
        try:
            return _project_value(s.state[int(me)].observation, me)
        except Exception:
            return None

    best_plan, best_v = candidate_plans[0], None
    t0 = time.perf_counter()
    for plan in candidate_plans:
        if (time.perf_counter() - t0) * 1000.0 > budget_ms:
            break               # time check is unconditional: a run of None-valued
            #                     plans must not let the loop ignore the budget
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
    opp_now = {i: _producer_move_obs(snap.state[i].observation, i) for i in opps}

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
            nxt = [_producer_move_obs(s.state[i].observation, i)
                   for i in range(num_seats)]
            step(s, nxt, in_place=True)
        try:
            return _project_value(s.state[int(me)].observation, me)
        except Exception:
            return None

    best_plan, best_v = candidate_plans[0], None
    t0 = time.perf_counter()
    for plan in candidate_plans:
        if (time.perf_counter() - t0) * 1000.0 > budget_ms:
            break               # time check is unconditional: a run of None-valued
            #                     plans must not let the loop ignore the budget
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


def _reachable_enemy_force(tx, ty, tid, planets, fleets, me):
    """Enemy force that can still reach the planet at (tx, ty) after we take it:
    in-flight enemy fleets within THREAT_RADIUS (full weight) + nearby parked
    enemy garrison (STATIONED_DISCOUNT), EXCLUDING the target itself (tid -- its
    garrison is what we just captured)."""
    t = 0.0
    for fl in fleets:
        o = int(fl.owner)
        if o == me or o == -1:
            continue
        if dist((tx, ty), (float(fl.x), float(fl.y))) <= THREAT_RADIUS:
            t += float(fl.ships)
    for p in planets:
        o = int(p.owner)
        if o == me or o == -1 or int(p.id) == tid:
            continue
        if dist((tx, ty), (float(p.x), float(p.y))) <= THREAT_RADIUS:
            t += STATIONED_DISCOUNT * float(p.ships)
    return t


def _keep_confident_launches(move, planets, fleets, by_id, me,
                             comet_ids, comet_paths, omega):
    """Keep only HIGH-CONFIDENCE actions. Launches are grouped by their target
    (matched via the agent's own intercept aim, so a coordinated gang-up is judged
    as one capture). A capture of an enemy/neutral planet is kept only if the
    garrison we land with -- (total ships sent) minus the target's current
    defenders -- is at least the enemy force that can still reach it (so we can
    HOLD it). Reinforcements of our own planets are always kept; a launch we cannot
    confidently classify is kept (never dropped on a guess). Dropped launches'
    ships stay home. No count cap, no concentration."""
    if not move:
        return move
    groups = {}                 # tid -> list of launch indices
    unclassified = []           # launches we keep regardless
    for launch in move:
        sid = int(launch[0])
        ang = float(launch[1])
        sh = launch[2]
        src = by_id.get(sid)
        if src is None:
            unclassified.append(launch)
            continue
        best, bd = None, 1e9
        for p in planets:
            if int(p.id) == sid:
                continue
            shot = _plan_shot(src, p, comet_ids, comet_paths, omega, max(1, int(sh)))
            if shot is None:
                continue
            d = abs(shot[0] - ang)
            d = min(d, 2.0 * math.pi - d)
            if d < bd:
                bd, best = d, p
        if best is None or bd > 0.10:
            unclassified.append(launch)      # can't classify -> keep
        else:
            groups.setdefault(int(best.id), []).append(launch)
    kept = list(unclassified)
    for tid, launches in groups.items():
        tgt = by_id[tid]
        if int(tgt.owner) == me:
            kept.extend(launches)            # reinforcing our own planet -> keep
            continue
        total = sum(float(l[2]) for l in launches)
        surplus = total - float(tgt.ships)   # garrison left after taking it
        threat = _reachable_enemy_force(float(tgt.x), float(tgt.y), tid,
                                        planets, fleets, me)
        if surplus >= threat:                # we can hold it -> high confidence
            kept.extend(launches)
        # else: thin contested grab that flips back -> drop, ships stay home
    return kept


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
    _turn_t0 = time.perf_counter()      # single per-turn clock shared by all phases
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

    # Rollout scorer. Used as the FALLBACK when orbit_lite is unavailable, and --
    # when LR_ROLLOUT_EVAL is on -- as the PRIMARY evaluator over the flow leaf,
    # because a real-engine rollout foresees recaptures the flow projection misses.
    rollout_eval = _rollout_eval()
    fb_snap = None
    if orbit is None or rollout_eval:
        fb_snap = from_obs(obs, configuration, num_seats=num_seats)

    def value_fallback(emit_launches):
        s = clone(fb_snap)
        first = [list(emit_launches) if i == me else lite_greedy_policy(s.state[i].observation)
                 for i in range(num_seats)]
        step(s, first, in_place=True)
        for _ in range(FALLBACK_HORIZON - 1):
            if s.fake_env.done:
                break
            # `me` stays idle after turn 1 (like the 2-ply pick) so the candidate's
            # first-move signal isn't washed out by an identical greedy continuation
            # across every candidate; only the opponents apply pressure.
            acts = [lite_greedy_policy(s.state[i].observation) if i != me else []
                    for i in range(num_seats)]
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

    confident_only = _confident_only()

    def _confident(move):
        if not confident_only:
            return move
        return _keep_confident_launches(move, planets, fleets, by_id, me,
                                        comet_ids, comet_paths, omega)
    # each candidate: emit=[[src_id,angle,ships],...], units=[(src_slot,tgt_slot,ships,eta),...],
    #                 srcs={src_id:ships}, rank, front
    candidates = []
    # Lever 2 (default 1.0 = off; 4P-only -- 2P is already strong and the scorer
    # focuses the one opponent correctly there): boost enemy-owned targets so
    # denial captures (taking from opponents) outrank equal-production neutrals.
    enemy_boost = _f("LR_ENEMY_BOOST", 1.0) if num_seats >= 4 else 1.0
    # Hold-sizing: size enemy captures to take AND HOLD (surplus garrison to
    # survive the opponent's retake), which forces source-combining => fewer /
    # bigger concentrated fleets. This is a DUEL tactic: on the real ladder it is
    # a 2P WIN (70% vs the breadth-first agent's 61%) but a 4P DISASTER (10.5% vs
    # 31.2%) -- 4P's three fronts punish concentrate-and-hold (under-expand at
    # step 30, then collapse by step 90 because a margin sized for ONE opponent
    # cannot hold against THREE). So default ON in 2P, OFF in 4P (4P falls back to
    # breadth-first minimum force, which was above fair share). See
    # knowledge-base/thoughts/2026-06-17-take-and-hold-is-a-2P-win-and-a-4P-disaster.md.
    hold_margin = _f("LR_HOLD_MARGIN", 0.5 if num_seats <= 2 else 0.0)
    threat_size = _threat_size()
    # Arrival-horizon cap (default OFF): don't generate ENEMY captures whose ETA is
    # past the projection horizon -- the evaluator can't see them resolve, so they
    # look free and we over-extend. Per-mode horizon; neutral expansion untouched.
    arrival_cap = _arrival_cap()
    hold_neutral = _hold_neutral()   # extend hold-margin to neutral captures (2P consolidation)
    holdability = _holdability()     # recapture penalty: skip captures we can't hold
    recap_k = _f("LR_RECAP_K", 1.0)  # penalty weight (x foregone production)
    horizon_cap = PROJECT_HORIZON_4P if num_seats >= 4 else PROJECT_HORIZON_2P

    def reachable_threat(tx, ty):
        """Visible enemy force that can recapture a planet at (tx, ty): in-flight
        enemy fleets near it (full weight) + nearby parked enemy garrison
        (discounted). Drives threat-aware capture sizing so we take planets with
        enough to HOLD against what is already on the board."""
        t = 0.0
        for fl in fleets:
            o = int(fl.owner)
            if o == me or o == -1:
                continue
            if dist((tx, ty), (float(fl.x), float(fl.y))) <= THREAT_RADIUS:
                t += float(fl.ships)
        for p in planets:
            o = int(p.owner)
            if o == me or o == -1:
                continue
            if dist((tx, ty), (float(p.x), float(p.y))) <= THREAT_RADIUS:
                t += STATIONED_DISCOUNT * float(p.ships)
        return t

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

    _gen_cap = _wallclock_ms() * 0.6    # leave the rest of the per-turn budget for scoring
    for tgt in targets:
        if (time.perf_counter() - _turn_t0) * 1000.0 > _gen_cap:
            break                       # bound candidate generation on dense boards
        tid = int(tgt.id)
        is_enemy = int(tgt.owner) != -1
        is_comet = tid in comet_ids
        prod = float(tgt.production)
        # Threat-aware sizing: the enemy force already able to retake this planet.
        tgt_reach = (reachable_threat(float(tgt.x), float(tgt.y))
                     if (threat_size or holdability) else 0.0)
        tgt_threat = tgt_reach if threat_size else 0.0

        shots = []   # (eta, size, sid, src, angle)
        for src in my_planets:
            shot = _plan_shot(src, tgt, comet_ids, comet_paths, omega, RANK_HINT_SHIPS)
            if shot is None:
                continue
            angle, eta, arr = shot
            if not _sun_clear(src, arr):
                continue
            if arrival_cap and is_enemy and eta > horizon_cap:
                continue            # don't fling fleets at enemies past the eval horizon
            if is_comet:
                life = comet_remaining_lifetime(tid, _ObsRawShim(obs_d))
                if life is not None and life <= eta:
                    continue
            defenders = prod * eta + tgt.ships if is_enemy else tgt.ships
            if threat_size:
                # Size to take AND hold against the visible incoming + reachable
                # threat. Unaffordable (truly contested) captures get dropped by
                # the affordability check below -> skip the doomed grab.
                size = int(math.ceil(defenders + tgt_threat)) + 1
            else:
                size = int(math.ceil(defenders)) + 1
                if hold_margin > 0.0 and (is_enemy or hold_neutral):
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
                a2, eta2, _ = shot
                triples = [(sid, tid, size, eta2)]
                units = units_for(triples)
                if units is None and id2slot is not None:
                    continue
                _surplus = size - ((prod * eta + float(tgt.ships)) if is_enemy else float(tgt.ships))
                _pen = prod * horizon_cap * recap_k if (holdability and tgt_reach > _surplus) else 0.0
                solo = {"emit": [[sid, float(a2), size]],
                        "units": units, "srcs": {sid: size},
                        "rank": rank, "front": front, "recap_penalty": _pen}
                break
        if solo is not None:
            candidates.append(solo)
            continue

        # Gang-up the nearest sources when none can solo. Provision for the
        # defenders at the LATEST contributing fleet's ACTUAL arrival: an enemy
        # keeps producing, and the partial fleets fly slower than the 20-ship
        # ranking hint -- both make the old "fastest source's size" total too
        # small, so the wave lands under-strength and the capture flips straight
        # back. Size against the slowest fleet actually used.
        def _required(latest_eta):
            d = (prod * latest_eta + float(tgt.ships)) if is_enemy else float(tgt.ships)
            return int(math.ceil(d + tgt_threat)) + 1
        emit, triples, srcs, acc, max_eta = [], [], {}, 0, 0.0
        for (eta, size, sid, src, angle) in shots:
            if sid in srcs:
                continue
            need = _required(max(max_eta, float(eta)))
            if acc >= need:
                break
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
            max_eta = max(max_eta, float(eta2))
        if emit and acc >= _required(max_eta):
            units = units_for(triples)
            if not (units is None and id2slot is not None):
                _surplus = acc - ((prod * max_eta + float(tgt.ships)) if is_enemy else float(tgt.ships))
                _pen = prod * horizon_cap * recap_k if (holdability and tgt_reach > _surplus) else 0.0
                candidates.append({"emit": emit, "units": units, "srcs": srcs,
                                   "rank": rank, "front": front, "recap_penalty": _pen})

    # Regroup / defense: reinforce our own planets an enemy fleet is about to flip
    # -- keep HELD production instead of only grabbing new planets. Same duel
    # tactic as hold-sizing, gated the same way: default ON in 2P, OFF in 4P
    # (in 4P it turtles while three rivals out-expand -- part of the 4P collapse).
    _defend_on = os.environ.get("LR_DEFEND", "1" if num_seats <= 2 else "0")
    if (_defend_on.strip().lower() in ("1", "true", "on", "yes")
            and my_planets):
        defend_range = _f("LR_DEFEND_RANGE", 35.0)
        enemy_fleets = [f for f in fleets
                        if int(f.owner) != me and int(f.owner) != -1]
        for mine in my_planets:
            mxy = (float(mine.x), float(mine.y))
            threat = 0.0
            for f in enemy_fleets:
                fx, fy = float(f.x), float(f.y)
                if dist(mxy, (fx, fy)) > defend_range:
                    continue
                # Only count fleets CLOSING on us -- a fleet within range but
                # heading away (already past, or aimed elsewhere) is not a real
                # threat, and reinforcing against it wastes ships (and these
                # defense candidates outrank captures).
                if math.cos(float(f.angle) - math.atan2(mxy[1] - fy, mxy[0] - fx)) <= 0.0:
                    continue
                threat += float(f.ships)
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
                                       "front": 0.0})

    if not candidates:
        return []

    candidates.sort(key=lambda c: (-c["rank"], c["front"]))
    candidates = candidates[:MAX_CANDIDATES]

    # ---- greedy plan construction by projected value ----
    committed_emit = []
    committed_units = []
    avail = dict(available)
    use_rollout_score = (orbit is None) or rollout_eval
    if not use_rollout_score:
        current = 0.0                       # score of the empty plan
        floor = ROI_FLOOR
    else:
        current = value_fallback([])
        floor = 0.5
    budget_ms = _value_budget(obs_d, _wallclock_ms())
    # Share the turn clock so candidate-gen + greedy are bounded together by ONE
    # per-turn budget (not a fresh budget per phase, which summed past 1s).
    t0 = _turn_t0

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
        if use_rollout_score:
            v = value_fallback(committed_emit + c["emit"])
        else:
            v = score_units(committed_units + c["units"])
        v -= c.get("recap_penalty", 0.0)   # holdability: down-rate doomed captures
        if v > current + floor:
            committed_emit = committed_emit + c["emit"]
            committed_units = committed_units + (c["units"] or [])
            current = v
            for s, sz in c["srcs"].items():
                avail[s] = avail.get(s, 0) - sz

    # ---- rollout-based final pick (LR_ROLLOUT_EVAL): choose among a few full
    #      plans by their value AFTER opponents respond in the REAL engine, so a
    #      committed plan whose captures get recaptured is rejected for a milder
    #      plan or the producer move. Replaces the flow-based 2-ply (blind to it).
    if use_rollout_score:
        try:
            producer_me = _producer_move_obs(obs, me)
        except Exception:
            producer_me = []
        plans = [committed_emit, producer_me, []]
        if len(committed_emit) > 2:
            plans.append(committed_emit[:len(committed_emit) // 2])
        # Own time budget + timer: the greedy loop above already consumed
        # `budget_ms` against `t0`, so sharing it would skip this re-rank entirely.
        # Guard each rollout so one bad terminal state can't crash the turn.
        t_pick = time.perf_counter()
        best, best_v = committed_emit, None
        for p in plans:
            if best_v is not None and (time.perf_counter() - t_pick) * 1000.0 > TWOPLY_BUDGET_MS:
                break
            try:
                v = value_fallback(p)
            except Exception:
                continue
            if best_v is None or v > best_v:
                best, best_v = p, v
        return _confident(best)

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
        # Bound the WHOLE turn: the pick gets only the time left under the per-turn
        # budget, so candidate-gen + greedy + pick <= _wallclock_ms() instead of
        # three separate budgets that could sum past the 1s actTimeout.
        _rem = max(80.0, _wallclock_ms() - (time.perf_counter() - _turn_t0) * 1000.0)
        try:
            depth = _rollout_depth()
            if depth >= 2:
                return _confident(_deep_pick(obs, configuration, me, num_seats,
                                  uniq, depth, budget_ms=_deep_budget(obs_d)))
            return _confident(_twoply_pick(obs, configuration, me, num_seats, uniq,
                              budget_ms=_twoply_budget(obs_d) if anytime_on else _rem))
        except Exception:
            return _confident(committed_emit)

    return _confident(committed_emit)
