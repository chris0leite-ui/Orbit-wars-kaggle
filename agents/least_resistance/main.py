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

# Optional V2 opponent model for the search (selected by LR_DEEP_OPP=2). V2 is
# pure-Python (lib.* only), stateless, and its agent(obs) already returns the
# [[src, angle, ships], ...] move format step() consumes -- no conversion needed.
# Resolved in two layouts like the producer above: in-repo dev (sibling
# ../v2/main.py) and a flat submission tar (v2_main.py next to this file).
_V2_OK = False
try:
    try:
        _V2_THIS = os.path.dirname(os.path.abspath(__file__))
    except NameError:               # kaggle execs agents without __file__
        _V2_THIS = (sys.path[-1] if sys.path
                    and os.path.isfile(os.path.join(sys.path[-1], "main.py"))
                    else os.getcwd())
    _v2_dev = os.path.abspath(os.path.join(_V2_THIS, "..", "v2", "main.py"))
    _v2_flat = os.path.join(_V2_THIS, "v2_main.py")
    _v2_path = _v2_dev if os.path.isfile(_v2_dev) else _v2_flat
    import importlib.util as _ilu2
    _v2_spec = _ilu2.spec_from_file_location("_lr_v2_main", _v2_path)
    _v2_mod = _ilu2.module_from_spec(_v2_spec)
    sys.modules["_lr_v2_main"] = _v2_mod
    _v2_spec.loader.exec_module(_v2_mod)
    _v2_agent = _v2_mod.agent
    _V2_OK = True
except Exception:
    _V2_OK = False


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
    fits the 1000ms wall; 2 = V2, the actual benchmark opponent (pure-Python,
    stateless) -- models the policy we are scored against directly. Default 0
    keeps current behaviour byte-identical."""
    return _i("LR_DEEP_OPP", 0)


def _win_leaf():
    """Win-equity leaf (default 0 = OFF, byte-identical). When ON, the leaf
    evaluator returns our CONTROL SHARE at the horizon -- (ours - theirs) /
    (ours + theirs) in [-1, 1] -- instead of the raw ship margin (ours - theirs).
    Rationale: the ladder scores WINS, not ship surplus, so a +5000 blowout and a
    +1 squeaker count the same; the linear-margin leaf rates the blowout 5000x
    higher and so trades robustness for expected magnitude (the fat negative tail).
    The share is non-monotone in the margin (it depends on the contested total),
    so it genuinely re-ranks plans: among equal-margin plans it prefers the one
    that wins with a SMALLER contested pool (more decisive, less exposed control),
    and once dominant it stops gambling for extra surplus. Parameter-free."""
    return _i("LR_WIN_LEAF", 0) >= 1


def _robust_search():
    """Robustness-aware multi-reply search (default 0 = OFF, byte-identical).
    When ON, the 2-ply pick scores each candidate plan against a SET of plausible
    turn-1 opponent replies and keeps the WORST case (min leaf) rather than a
    single predicted reply. Rationale: our losses vs a strong peer come from
    opponent-response uncertainty -- a plan that looks best against the one
    predicted reply can lose badly against a slightly different one (the fat
    negative tail). Worst-casing over a small reply set prefers plans that hold up
    no matter which way the opponent jumps. The replies are precomputed ONCE and
    shared across all candidate plans -- valid because moves are simultaneous (the
    opponent cannot see our plan). Reply set: the producer mirror plus lite_greedy
    (two genuinely different policies; both already bundled). OFF -> the set is the
    single base reply, so min-of-one reproduces today's pick exactly."""
    return _i("LR_ROBUST_SEARCH", 0) >= 1


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


def _hold_search():
    """Hold-search (default OFF). When ON, add a few stronger-hold whole-turn
    candidate plans for CONTESTED held planets to the 2-ply/deep rollout menu, so
    that holding a contested planet decisively EMERGES from simulating the
    opponent's retake (a thin hold gets flipped in the rollout and scores worse)
    rather than from a reinforcement-sizing heuristic. The 1-ply launch scorer
    cannot value a *prevented loss* (reinforcing our own planet shows ~0 net-ship
    gain); the 2-ply/deep rollout against the opponent model can. OFF =
    byte-identical (no extra plans on the menu)."""
    return _i("LR_HOLD_SEARCH", 0) >= 1


def _hold_search_range():
    """Distance within which an in-flight enemy fleet counts toward the pressure
    on one of our planets, for selecting which held planets to offer stronger-hold
    plans for. Wider than the reactive-defense range so the rollout is offered the
    decisive option before the wave is already on top of us."""
    return _f("LR_HOLD_SEARCH_RANGE", 55.0)


def _hold_search_levels():
    """Hold-variant strengths (multiples of the detected incoming threat) offered
    to the rollout for a contested planet. Default (1.25, 1.75) -- gentler than the
    original (1.5, 2.5) so the menu cannot propose an over-commit that drains the
    rest of the position. Override via LR_HOLD_SEARCH_LEVELS (comma-separated)."""
    raw = os.environ.get("LR_HOLD_SEARCH_LEVELS", "")
    if raw.strip():
        try:
            vals = tuple(float(x) for x in raw.split(",") if x.strip())
            if vals:
                return vals
        except ValueError:
            pass
    return (1.25, 1.75)


def _hold_donor_keep():
    """Fraction of its own garrison each donor must RETAIN when feeding a
    hold-search reinforcement (default 0.5). Caps the drain so a stronger-hold
    variant can never strip a donor bare -- the over-hold catastrophic tail."""
    return _f("LR_HOLD_DONOR_KEEP", 0.5)


def _dropout():
    """Dropout-risk rollout penalty (default 0 = OFF, byte-identical). When ON,
    each candidate plan's leaf value is docked a MARGINAL penalty for the exposure
    *this turn's action* creates: planets it CAPTURES that stay reachable by the
    strongest rival ("attack far -> it falls back"), and sources it STRIPS thin
    ("attack -> garrison falls -> flip"). The standing frontier (planets we already
    held and did not weaken) is NOT penalised -- so over-extension/over-aggression
    self-penalise without making us passive. Deterministic, no RNG. Mirrors the
    producer_plus per-capture reflip rather than a blanket state penalty."""
    return _i("LR_DROPOUT", 0) >= 1


def _dropout_weight():
    """Scale on the marginal exposure penalty (default 0.5). leaf_value -=
    weight * (captured-exposure + stripped-source-exposure), in ship units."""
    return _f("LR_DROPOUT_WEIGHT", 0.5)


def _dropout_reach_pad():
    """Slack (turns) on the reach test: a rival garrison can contest our planet
    iff travel_turns <= H + pad. Default 0 (strict within-horizon)."""
    return _f("LR_DROPOUT_REACH_PAD", 0.0)


def _opponent_move_fn(tier=None):
    """Return a callable (obs, seat) -> [[src,angle,ships],...] for the per-node
    opponent move, matching _producer_move_obs' signature. lite_greedy reads the
    seat from obs.player, so the seat arg is ignored for it."""
    if tier is None:
        tier = _deep_opp()
    if int(tier) == 1:
        return lambda obs_any, seat: lite_greedy_policy(obs_any)
    if int(tier) == 2 and _V2_OK:
        return _v2_move_obs
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


def _v2_move_obs(obs_any, seat):
    """V2's launches for `seat` -- the actual benchmark opponent used as the
    search's per-node opponent model (LR_DEEP_OPP=2). V2 is stateless and reads
    its seat from obs.player (like lite_greedy), so `seat` is unused. agent(obs)
    already returns the [[src, angle, ships], ...] format step() consumes."""
    try:
        return _v2_agent(obs_any)
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
    if _win_leaf():
        # Control share in [-1, 1]: optimize probability of finishing ahead, not
        # expected ship surplus. Saturates, so the search secures leads instead of
        # gambling for magnitude. denom <= 0 (no live military) -> neutral 0.
        denom = mine + theirs
        if denom <= 0.0:
            return 0.0
        return (mine - theirs) / denom
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
    # Turn-1 opponent replies, precomputed ONCE and shared across all candidate
    # plans -- valid because moves are simultaneous (the opponent can't see our
    # plan). 2P (one opp) and 4P (three). Default: the single base reply (the
    # LR_DEEP_OPP selection). Robust mode: a set of genuinely different policies
    # (producer mirror + lite_greedy), scored worst-case in value() below.
    if _robust_search():
        reply_fns = [_producer_move_obs, lambda o, s: lite_greedy_policy(o)]
    else:
        reply_fns = [opp_move_fn]
    opp_replies = [{i: fn(snap.state[i].observation, i) for i in opps}
                   for fn in reply_fns]
    drop_on = _dropout()
    pre_owned, pre_ships = _dropout_prestate(obs, me) if drop_on else (None, None)

    def _leaf(leaf_obs):
        v = _project_value(leaf_obs, me)
        if drop_on:
            v -= _dropout_penalty(leaf_obs, me, pre_owned, pre_ships, num_seats)
        return v

    def value(plan):
        # Score the plan against every turn-1 reply; keep the WORST case (min).
        # OFF -> opp_replies is a single element, so this is min-of-one == today.
        worst = None
        for opp_now in opp_replies:
            s = clone(snap)
            acts = [[] for _ in range(num_seats)]
            acts[int(me)] = list(plan)
            for i in opps:
                acts[i] = list(opp_now[i])
            step(s, acts, in_place=True)
            if not s.fake_env.done:
                # One more turn of the opponents' pressure (base model; each
                # replies, we stay idle -- conservative, surfaces the next-turn
                # punishment the 1-ply scorer misses). Only the turn-1 reply
                # varies across the set, to keep the search cost bounded.
                nxt = [[] for _ in range(num_seats)]
                for i in opps:
                    nxt[i] = opp_move_fn(s.state[i].observation, i)
                step(s, nxt, in_place=True)
            try:
                v = _leaf(s.state[int(me)].observation)
            except Exception:
                v = None
            if v is None:
                continue
            if worst is None or v < worst:
                worst = v
        return worst

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
    drop_on = _dropout()
    pre_owned, pre_ships = _dropout_prestate(obs, me) if drop_on else (None, None)

    def _leaf(leaf_obs):
        v = _project_value(leaf_obs, me)
        if drop_on:
            v -= _dropout_penalty(leaf_obs, me, pre_owned, pre_ships, num_seats)
        return v

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
            return _leaf(s.state[int(me)].observation)
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
                    v = _leaf(s.state[int(me)].observation)
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


def _reinforce_emit(target, total, avail, my_planets, comet_ids, comet_paths, omega):
    """Build launches sending up to `total` ships to one of our own planets
    (`target`) from our other planets, nearest first, drawing only on ships not
    already committed this turn (`avail`). Does NOT mutate `avail` -- each
    hold-search variant is an ALTERNATIVE plan, so they each draw independently
    from the same pool; the rollout, not the bookkeeping, decides between them."""
    txy = (float(target.x), float(target.y))
    donors = sorted(
        (p for p in my_planets
         if int(p.id) != int(target.id) and avail.get(int(p.id), 0) > 0),
        key=lambda p: dist(txy, (float(p.x), float(p.y))))
    keep = _hold_donor_keep()
    emit, acc = [], 0
    for d in donors:
        # leave each donor at least `keep` of its own garrison -- a hold variant
        # must never strip a donor bare (the over-hold drain that lost the war).
        spare = int(avail.get(int(d.id), 0)) - int(math.ceil(keep * float(d.ships)))
        take = min(max(0, spare), int(total) - acc)
        if take <= 0:
            continue
        shot = _plan_shot(d, target, comet_ids, comet_paths, omega, take)
        if shot is None:
            continue
        angle, _eta, arr = shot
        if not _sun_clear(d, arr):
            continue
        emit.append([int(d.id), float(angle), int(take)])
        acc += take
        if acc >= int(total):
            break
    return emit


def _hold_search_plans(committed_emit, avail, my_planets, fleets, me,
                       comet_ids, comet_paths, omega,
                       max_targets=2, levels=None):
    """Stronger-hold whole-turn plan variants for the rollout menu. For the most
    valuable CONTESTED held planets (incoming enemy mass exceeds our garrison),
    propose plans that pour extra ships into them at a couple of strengths, ON TOP
    of the greedy plan. These are only *candidates*: the 2-ply/deep rollout scores
    each against the opponent model and keeps one only if it actually survives the
    retake -- so the hold LEVEL is chosen by simulation, and over-stripping a donor
    is self-punished (the donor falls in the rollout -> lower leaf value)."""
    if levels is None:
        levels = _hold_search_levels()
    enemy_fleets = [f for f in fleets
                    if int(f.owner) != int(me) and int(f.owner) != -1]
    if not enemy_fleets or not my_planets:
        return []
    rng = _hold_search_range()
    contested = []
    for mine in my_planets:
        mxy = (float(mine.x), float(mine.y))
        threat = sum(float(f.ships) for f in enemy_fleets
                     if dist(mxy, (float(f.x), float(f.y))) <= rng)
        if threat > float(mine.ships):
            contested.append((float(mine.production), threat, mine))
    if not contested:
        return []
    # Most valuable / most pressured first; a high-production planet under heavy
    # incoming mass is exactly the take-and-hold target we were losing.
    contested.sort(key=lambda c: (c[0], c[1]), reverse=True)
    out = []
    for _prod, threat, mine in contested[:max_targets]:
        for k in levels:
            need = int(math.ceil(k * threat)) - int(mine.ships)
            if need <= 0:
                continue
            reinforce = _reinforce_emit(mine, need, avail, my_planets,
                                        comet_ids, comet_paths, omega)
            if reinforce:
                out.append(list(committed_emit) + reinforce)
    return out


def _dropout_prestate(obs_any, me):
    """Snapshot which planets we own and their garrisons BEFORE our move, so the
    rollout penalty can charge only the exposure our action creates this turn."""
    d = _as_dict(obs_any)
    planets = d.get("planets", []) or []
    owned, ships = set(), {}
    for p in planets:
        if int(p[1]) == int(me):
            owned.add(int(p[0]))
            ships[int(p[0])] = float(p[5])
    return owned, ships


def _dropout_penalty(leaf_obs, me, pre_owned, pre_ships, num_seats):
    """Marginal exposure penalty (>=0) for a candidate plan, read off the rollout
    leaf. Charges, in ship units scaled by _dropout_weight(): (a) planets we now
    hold that we did NOT own pre-move (captures) and that the strongest rival can
    still REACH within the horizon -> far/over-extended captures; (b) planets we
    owned pre-move whose garrison FELL (sources we stripped) and are rival-
    reachable -> over-aggression. The standing, un-weakened frontier is untouched."""
    d = _as_dict(leaf_obs)
    planets = d.get("planets", []) or []
    rows = [(int(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[5]))
            for p in planets]
    tot = {}
    for _pid, o, _x, _y, sh in rows:
        if o != int(me) and o >= 0:
            tot[o] = tot.get(o, 0.0) + sh
    if not tot:
        return 0.0
    rival = max(tot.items(), key=lambda kv: (kv[1], -kv[0]))[0]  # deterministic
    rival_planets = [(x, y, sh) for _pid, o, x, y, sh in rows if o == rival]
    if not rival_planets:
        return 0.0
    H = PROJECT_HORIZON_4P if int(num_seats) >= 4 else PROJECT_HORIZON_2P
    ref = max(1e-6, float(fleet_speed(FRONTIER_REF_SHIPS)))
    budget = (float(int(H)) + _dropout_reach_pad()) * ref
    eps, pen = 1e-6, 0.0
    for pid, o, x, y, sh in rows:
        if o != int(me):
            continue
        reach = sum(q for (qx, qy, q) in rival_planets
                    if dist((x, y), (qx, qy)) <= budget)
        if reach <= 0.0:
            continue
        contest = reach / (reach + sh + eps)
        if pid not in pre_owned:
            pen += contest * sh                       # captured & still exposed
        else:
            drop = pre_ships.get(pid, sh) - sh
            if drop > 0.0:
                pen += contest * drop                 # source we stripped, exposed
    return _dropout_weight() * pen


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
                                       "front": 0.0})

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
        plans = [producer_me, committed_emit, []]   # producer floor first
        if anytime_on:
            # Lever 3: spend headroom -- offer every aggression level of the
            # committed plan, so extra compute becomes more plans evaluated.
            plans.extend(committed_emit[:k] for k in range(1, len(committed_emit)))
        elif len(committed_emit) > 2:
            # One milder aggression level of my plan for the lookahead to weigh.
            plans.append(committed_emit[:len(committed_emit) // 2])
        if _hold_search():
            # Decisive holding emerges from the rollout, not a sizing rule: offer
            # the search stronger-hold variants for contested held planets and let
            # it keep one only if the opponent model can't flip it.
            plans.extend(_hold_search_plans(
                committed_emit, avail, my_planets, fleets, me,
                comet_ids, comet_paths, omega))
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
