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
    # Dropout-perturbed score (default-OFF): primitives to bake the enemy's
    # physically-reachable mass onto our planets as a credit-only "drop" and
    # re-score, so holding a defensive reserve out-scores draining a source.
    from orbit_lite.garrison_launch import _run_exact_recurrence as _lr_recurrence
    from orbit_lite.movement import PlanetGarrisonStatus as _PGS
    from orbit_lite.distance_cache import build_distance_cache as _build_dist_cache
    from orbit_lite.native_forward import reachable_enemy_mass as _reach_enemy_mass
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


def _hold_source():
    """Defensive-reserve gate (default OFF, byte-identical when OFF). When ON, a
    planet may only contribute ships to an OFFENSIVE capture down to a floor =
    the enemy mass physically reachable to THAT source within
    LR_HOLD_SOURCE_REACH turns, scaled by LR_HOLD_SOURCE_MARGIN.

    Diagnosis it fixes (PI ladder replays, seeds 25260880/788834306/1576908455):
    the one-ply garrison-flow scorer models production + in-flight combat but NO
    new opponent launches, so draining a source to fund a capture looks free --
    the counterattack that flips the emptied source is invisible. The greedy
    commit loop then strips our planets bare and we lose the sources ("we expose
    our planets by attacking, not holding defense -- short-sighted"). This is the
    SOURCE-side mirror of the existing target-side LR_HOLD_MARGIN: we already
    size captures to HOLD THE TARGET; this makes us also HOLD THE SOURCE. Uses
    the same reachable-enemy-mass signal as smart dropout, but binds it as a hard
    sourcing constraint (so it changes the chosen action, unlike a soft score
    perturbation, which prior sessions found inert)."""
    return _i("LR_HOLD_SOURCE", 0) >= 1


def _dropout_score():
    """Dropout-perturbed-score gate (default OFF, byte-identical when OFF).

    The "smart dropout opponent replacement" applied at the SCORE level. The
    one-ply garrison-flow scorer models NO opponent launches, so reserve ships
    left at home contribute zero projected value -- there is no reward for
    defence, only for capturing, and the greedy loop over-commits (drains
    sources -> loses them; too many fleets). This perturbs the projection: the
    enemy's physically-reachable mass is baked onto each of our planets as a
    credit-only "drop", and every candidate is scored a SECOND time in that
    pessimist world. The two scores are blended (LR_DROPOUT_W, default 0.5). A
    candidate that drains a source below its reachable enemy mass sees that
    source FALL in the pessimist world -> lower score; reinforcing or simply
    HOLDING a reserve makes the planet survive -> higher score. So "hold
    defence" wins and the greedy loop stops sooner -- addressing the value
    function, not the sourcing (where the LR_HOLD_SOURCE cap was inert)."""
    return _i("LR_DROPOUT_SCORE", 0) >= 1


def _response_veto():
    """Response-veto gate (default OFF, byte-identical when OFF). During the
    greedy commit, a capture that DRAINS a source below the enemy mass reachable
    to it is committed only if it does not regress the REAL two-ply value -- my
    move + each opponent's producer-mirror reply, then one opponent reaction turn
    (so the now-undefended source is actually taken). This is the accurate
    counterattack check that the one-ply scorer (and the inert dropout proxy)
    cannot be: it sees the opponent punish the drain and skips the capture, so we
    stop stripping sources / over-committing. Cost is bounded -- only DRAINING
    candidates trigger it, the shared opponent reply is computed once, the
    reaction turn uses the cheap expansion policy, and it respects the per-turn
    budget."""
    return _i("LR_RESPONSE_VETO", 0) >= 1


def _lr_drop_status(garrison_status, *, drop_tgt, drop_tick, drop_ships,
                    drop_owner, prod, alive_by_step):
    """Baseline garrison trajectories with our exposed planets flipped to the
    rival (credit-only enemy arrivals, NO friendly source debit). Ported from
    producer_plus._dropout_adjusted_status; replays the exact production->combat
    recurrence so a flip propagates (lost production + garrison after the drop
    tick). ``alive_by_step`` is ``[H+1, P]``."""
    owner0 = garrison_status.owner[..., 0]                       # [P]
    ships0 = garrison_status.ships[..., 0]                       # [P]
    arr = garrison_status.arrivals_by_owner                      # [P, H+1, A]
    P, H1, A = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    H = H1 - 1
    fdtype = ships0.dtype if ships0.is_floating_point() else _torch.float32

    tgt = drop_tgt.clamp(0, max(P - 1, 0))
    own = drop_owner.clamp(0, max(A - 1, 0))
    ships = drop_ships.to(fdtype)
    tick = drop_tick.long().clamp(min=1)

    init_ships = ships0.to(fdtype).clone()                       # NO debit (credit-only)
    arr_delta = arr[:, 1:, :].to(fdtype).clone()                 # [P, H, A]
    in_h = tick <= H
    if bool(in_h.any()):
        arr_delta.index_put_(
            (tgt[in_h], tick[in_h] - 1, own[in_h]), ships[in_h], accumulate=True,
        )
    owner_t, ships_t, pre_o, pre_s = _lr_recurrence(
        init_owner=owner0.unsqueeze(0),
        init_ships=init_ships.unsqueeze(0),
        prod=prod.to(fdtype).unsqueeze(0),
        alive=alive_by_step.transpose(0, 1).unsqueeze(0),
        arrivals=arr_delta.unsqueeze(0),
    )
    return _PGS(
        owner=owner_t[0], ships=ships_t[0],
        pre_combat_owner=pre_o[0], pre_combat_ships=pre_s[0],
        arrivals_by_owner=_torch.cat([arr[:, :1, :].to(fdtype), arr_delta], dim=1),
    )


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

    # Dropout-perturbed score (default OFF): build ONE pessimist baseline where
    # the enemy's physically-reachable mass falls on each of our planets (a
    # credit-only "drop", strongest single hammer per planet). It is
    # candidate-independent, so build it once; each candidate's launches are
    # then scored against BOTH `status` and `pess_status` and blended below.
    pess_status = None
    drop_w = 0.0
    if _dropout_score():
        try:
            drop_w = _f("LR_DROPOUT_W", 0.5)
            planets_t = obs_tensors["planets"]
            owner_col = planets_t[:, 1].long()
            ships_col = planets_t[:, 5].to(_torch.float32)
            # single_obs_to_tensor pads to a fixed slot count; padding slots carry
            # owner 0, so mask to ALIVE real planets or the drop lands on phantoms.
            alive0 = alive_by_step[0].to(_torch.bool)
            is_enemy = (owner_col != int(me)) & (owner_col >= 0) & alive0
            is_mine = (owner_col == int(me)) & alive0
            rival, best_v = 0, -1.0
            for pl in range(int(pc)):
                if pl == int(me):
                    continue
                tot = float((ships_col * (owner_col == pl).to(_torch.float32)).sum())
                if tot > best_v:
                    best_v, rival = tot, pl
            cache = _build_dist_cache(movement, max_k=int(H))
            reach = _reach_enemy_mass(cross_dist=cache.cross_dist, ships=ships_col,
                                      is_enemy=is_enemy, H=int(H), aggregate="max")  # [P,H+1]
            pos = reach > 0.0                                       # [P, H+1]
            Hh = int(H)
            tick_idx = (_torch.arange(Hh + 1, dtype=_torch.float32)
                        .unsqueeze(0).expand_as(reach))
            masked = _torch.where(pos, tick_idx,
                                  _torch.full_like(reach, float(Hh + 1)))
            first_tick = masked.min(dim=1).values                  # [P] float
            drop_mask = is_mine & pos.any(dim=1)                   # [P]
            sel = drop_mask.nonzero(as_tuple=True)[0]
            if drop_w > 0.0 and int(sel.numel()) > 0:
                pess_status = _lr_drop_status(
                    status, drop_tgt=sel.long(),
                    drop_tick=first_tick[sel].long().clamp(min=1),
                    drop_ships=reach[sel, Hh],
                    drop_owner=_torch.full((int(sel.numel()),), int(rival),
                                           dtype=_torch.long),
                    prod=prod, alive_by_step=alive_by_step,
                )
        except Exception:
            pess_status = None

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
            val = float(sc.reshape(-1)[0])
            if pess_status is not None:
                sp = _score_candidates(
                    pess_status, prod=prod, alive_by_step=alive_by_step,
                    player_count=int(pc), launches=ls, player_id=int(me),
                    opp_weights=opp_w,
                )
                val = (1.0 - drop_w) * val + drop_w * float(sp.reshape(-1)[0])
        return val

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

    # Defensive-reserve floor (default OFF -> avail_off is `available`, so the
    # code path below is byte-identical). When LR_HOLD_SOURCE is on, an
    # OFFENSIVE capture may not draw a source below the enemy mass that can
    # physically reach THAT source within LR_HOLD_SOURCE_REACH turns (scaled by
    # LR_HOLD_SOURCE_MARGIN). Threat = the single strongest reachable enemy
    # hammer (planet garrison or in-flight fleet) -- the worst single wave a
    # source must survive -- summed with fleets already inbound to it. This
    # stops us stripping a planet bare to grab a target and then losing the
    # emptied source (the counterattack the 1-ply scorer can't see).
    avail_off = available
    if _hold_source():
        reach = _f("LR_HOLD_SOURCE_REACH", 8.0)
        src_margin = _f("LR_HOLD_SOURCE_MARGIN", 1.0)
        enemy_planets = [(float(p.x), float(p.y), float(p.ships)) for p in planets
                         if int(p.owner) != me and int(p.owner) != -1]
        enemy_flts = [(float(f.x), float(f.y), float(f.ships)) for f in fleets
                      if int(f.owner) != me and int(f.owner) != -1]
        avail_off = {}
        for p in my_planets:
            pxy = (float(p.x), float(p.y))
            # strongest single reachable enemy planet (the worst single wave)
            hammer = 0.0
            for (ex, ey, es) in enemy_planets:
                if dist((ex, ey), pxy) / ref_speed <= reach and es > hammer:
                    hammer = es
            # plus enemy fleets already bearing down within the reach window
            inbound = sum(es for (ex, ey, es) in enemy_flts
                          if dist((ex, ey), pxy) / ref_speed <= reach)
            reserve = int(math.ceil((hammer + inbound) * src_margin))
            avail_off[int(p.id)] = max(0, available[int(p.id)] - reserve)

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
        rank = prod / max(1.0, shots[0][0])
        if is_enemy and enemy_boost != 1.0:
            rank *= enemy_boost
        front = frontier_eta((float(tgt.x), float(tgt.y)))

        # Solo capture from the cheapest affordable source — re-aim at the
        # actual size so the emit angle and the scorer eta are accurate.
        solo = None
        for (eta, size, sid, src, angle) in shots:
            if avail_off[sid] >= size:
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
            take = min(avail_off[sid], need - acc)
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
                                       "front": 0.0})

    if not candidates:
        return []

    candidates.sort(key=lambda c: (-c["rank"], c["front"]))
    candidates = candidates[:MAX_CANDIDATES]

    # ---- greedy plan construction by projected value ----
    committed_emit = []
    committed_units = []
    avail = dict(avail_off)
    if orbit is not None:
        current = 0.0                       # score of the empty plan
        floor = ROI_FLOOR
    else:
        current = value_fallback([])
        floor = 0.5
    budget_ms = _value_budget(obs_d, _wallclock_ms())
    t0 = time.perf_counter()

    # Response veto (default OFF): set up the reachable-enemy-mass per source and
    # the shared opponent reply ONCE, so a draining capture can be re-checked
    # against the real two-ply value inside the loop.
    veto_on = _response_veto() and orbit is not None and num_seats >= 2
    veto_reach = {}
    v2snap = v2opps = v2opp_now = None
    if veto_on:
        try:
            vreach = _f("LR_VETO_REACH", 8.0)
            e_pl = [(float(p.x), float(p.y), float(p.ships)) for p in planets
                    if int(p.owner) != me and int(p.owner) != -1]
            e_fl = [(float(f.x), float(f.y), float(f.ships)) for f in fleets
                    if int(f.owner) != me and int(f.owner) != -1]
            for p in my_planets:
                pxy = (float(p.x), float(p.y))
                ham = 0.0
                for (ex, ey, es) in e_pl:
                    if dist((ex, ey), pxy) / ref_speed <= vreach and es > ham:
                        ham = es
                inb = sum(es for (ex, ey, es) in e_fl
                          if dist((ex, ey), pxy) / ref_speed <= vreach)
                veto_reach[int(p.id)] = ham + inb
            v2snap = from_obs(obs, configuration, num_seats=num_seats)
            v2opps = [i for i in range(num_seats) if i != me]
            _omf = _opponent_move_fn()
            v2opp_now = {i: _omf(v2snap.state[i].observation, i) for i in v2opps}
        except Exception:
            veto_on = False

    def _veto_2ply(emit_plan):
        """Value of emit_plan after my move + each opponent's (precomputed) mirror
        reply, then one cheap opponent reaction turn (so a drained source is
        actually taken). Higher = better for me; None on failure."""
        try:
            s = clone(v2snap)
            acts = [[] for _ in range(num_seats)]
            acts[int(me)] = list(emit_plan)
            for i in v2opps:
                acts[i] = list(v2opp_now[i])
            step(s, acts, in_place=True)
            if not s.fake_env.done:
                nxt = [[] for _ in range(num_seats)]
                for i in v2opps:
                    nxt[i] = lite_greedy_policy(s.state[i].observation)
                step(s, nxt, in_place=True)
            return _project_value(s.state[int(me)].observation, me)
        except Exception:
            return None

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
            # Response veto: a capture that drains a source below its reachable
            # enemy mass must survive the real opponent reply (two-ply). If the
            # opponent punishes the drain, skip this capture (keep the reserve).
            if (veto_on and (time.perf_counter() - t0) * 1000.0 <= budget_ms
                    and any((avail.get(s, 0) - sz) < veto_reach.get(s, 0.0)
                            for s, sz in c["srcs"].items())):
                base_2ply = _veto_2ply(committed_emit)
                cand_2ply = _veto_2ply(committed_emit + c["emit"])
                if (base_2ply is not None and cand_2ply is not None
                        and cand_2ply < base_2ply - _f("LR_VETO_MARGIN", 0.0)):
                    continue                         # opponent punishes the drain
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
