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

# ---- SUBMISSION BAKE (PI 2026-06-20) -------------------------------------
# Ship the native ship-margin leaf + reinforcement-race holdability as the
# default for this submission. Gates still read os.environ at call time, so an
# explicit env var overrides this (e.g. LR_NATIVE_LEAF=0 reverts to the
# ship-count leaf). Remove this block to restore the default-OFF live path.
os.environ.setdefault("LR_NATIVE_LEAF", "1")
os.environ.setdefault("LR_NATIVE_REINFORCE", "1")
os.environ.setdefault("LR_CONCENTRATE", "1")   # additive: decisive captures + value-ordered commit
os.environ.setdefault("LR_NATIVE_OFFENSE", "1")  # credit massing/holding; far dribble grabs score negative
# (LR_SKIP_COMETS already defaults to 1.)
# --------------------------------------------------------------------------

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


def _native_allocate():
    """Default-OFF gate (PI 2026-06-20). When ON (with the native leaf), the
    flip-hazard threat per planet is a CONSERVED allocation: pool all reachable
    enemy strength (planets + landed reinforcements) and distribute it across our
    planets concentrated on the close / big / weakly-held ones, with the shares
    summing to the enemy's total ('integrates to enemy strength'). Replaces the
    per-planet 'max' threat, which over-counts (one enemy looks like a full threat
    to every planet at once -> the board looks doomed and the agent freezes).
    Also makes the pot reinforcement-aware (the second-order retake). OFF =
    byte-identical 'max' path. Read at call time."""
    return os.environ.get("LR_NATIVE_ALLOCATE", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _native_offense():
    """Default-OFF gate (PI 2026-06-20). With the native leaf, ADD the symmetric
    mirror of the defensive flip-hazard: OUR reachable mass onto enemy/neutral
    planets, priced as expected capture-gain (offensive POTENTIAL). Computed on each
    plan's own trajectory, so massing ships near the frontier earns value (HOLDING /
    patience is rewarded), dribbling into far neutrals routes ships away and destroys
    it (small far grabs lose to holding), and a concentrated capture converts it into
    realized margin (decisive action wins). One coherent value term for the dribble /
    no-concentration / launch-every-round behaviour. OFF = byte-identical. Read at
    call time."""
    return os.environ.get("LR_NATIVE_OFFENSE", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _native_reinforce():
    """Default-OFF gate (PI 2026-06-20). With the native leaf, holdability becomes a
    REINFORCEMENT RACE: the flip-hazard denominator counts not just a planet's own
    garrison but how much our OTHER planets can route to support it (conserved, so
    finite reinforcement is shared across our holdings). A capture sticks only if our
    reinforcement reach beats the enemy's attack reach there -> far/unsupportable
    captures flip and score negative, close/supportable ones hold; overextension
    self-penalizes (support spread thin). Routes the enemy threat through allocate
    too (apples-to-apples conserved contest). OFF = byte-identical. Read at call
    time."""
    return os.environ.get("LR_NATIVE_REINFORCE", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _native_dynamic():
    """Default-OFF gate (PI 2026-06-20, 'see the opponent coming'). With the native
    leaf, make the enemy threat AND our reinforcement DYNAMIC over the whole
    look-ahead: at each step use the projected garrison at that step (which already
    lands in-flight fleets + accrues production) as the per-source mass, instead of a
    single frozen near-step snapshot. So an opponent building up / arriving later in
    the horizon becomes visible, and the reinforcement-race correctly flags captures
    we'll lose. OFF = the frozen-snapshot path. Read at call time."""
    return os.environ.get("LR_NATIVE_DYNAMIC", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _native_threat_max():
    """Default-OFF gate (PI 2026-06-21, 'see the attack coming'). With the native
    leaf, compute the ENEMY ATTACK threat per planet as the worst-case SINGLE enemy
    planet that can reach it (the `max` aggregation: the full ships+production that
    one opponent planet can land there), instead of the conserved `allocate` split.

    Observed failure (replay 2026-06-21): an enemy stronghold (~95 ships) within
    reach of one of our planets is diluted by `allocate` across ALL our reachable
    planets, so each sees only a fraction (~17 of 95). No planet ever looks
    endangered, so holding a source is never valued and the search happily drains it
    into a doomed attack -- then the opponent CONCENTRATES the full stronghold and
    takes the emptied planet. A real opponent concentrates; `allocate` assumes it
    disperses. `max` restores the worst-case single-source view so a planet the
    opponent can flip with one decisive fleet reads as genuinely threatened, which
    makes HOLDING that garrison the high-value move and punishes draining it.

    Scope: ONLY the enemy attack reach. Our reinforcement reach (def_reach) stays
    conserved `allocate` -- our finite support genuinely is split across the planets
    we must cover, so the defense reading stays conservative (enemy concentrates, we
    spread). OFF = the conserved-allocate threat (byte-identical). Read at call
    time."""
    return os.environ.get("LR_NATIVE_THREAT_MAX", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _garrison_floor():
    """Default-OFF gate (PI 2026-06-21, 'hold only under threat'). A planet that
    a single enemy can clearly hit reserves enough garrison to win that fight and
    may only spend its SURPLUS on attacks; an unthreatened planet drains freely
    (reserve 0, byte-identical). This is enforced in plan GENERATION -- it caps
    each source's spendable ships BEFORE candidates are built -- so the plan menu
    naturally contains 'attack elsewhere, hold the threatened planet', the option
    the 2-ply chooser is otherwise never offered. Read at call time.

    Observed failure (PI replay, 2026-06-21): we launch FIRST, draining a large
    planet at a small/far target, then lose it to the opponent's obvious attack.
    The drain is committed by the greedy plan-builder (the producer net-ship-delta
    scorer, which has no hold/threat concept), so the leaf-level threat fix cannot
    undo it. The floor stops the drain at the source.

    Threat = the worst-case SINGLE attacker (no group-ups, per the PI): the larger
    of (a) the strongest enemy PLANET that can reach it within the window, sized at
    its ships + production accrued by arrival, and (b) the strongest enemy FLEET
    already in flight and CLOSING on it (the in-flight attack the leaf's threat
    model is blind to)."""
    return os.environ.get("LR_GARRISON_FLOOR", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _native_builder():
    """Default-OFF gate (PI 2026-06-21). When ON, the greedy plan-builder scores
    candidate launches with the native flip-hazard leaf (== the 2-ply chooser leaf
    _native_value) instead of the producer net-ship-delta scorer, so far thin grabs
    and exposed-planet drains are never BUILT in the first place.

    Observed failure (PI replays): the bad target-selection (drain a large planet at
    a far/low-value target) is committed by the greedy builder, which scores with the
    producer net-ship-delta scorer -- no flip-hazard, no 'does this capture hold?',
    no 'is this planet threatened?'. The 2-ply chooser uses the good leaf but only
    trims whole-plan prefixes, so it cannot repair a bad plan. Scoring the BUILDER
    with the same flip-hazard value fixes target selection at the source.

    OFF = byte-identical producer-scorer path. Most effective with LR_VALUE_COMMIT /
    LR_CONCENTRATE on (so the ordering pass uses the native leaf too) and the full
    chooser config (LR_NATIVE_LEAF/REINFORCE/OFFENSE) so builder leaf == chooser
    leaf. Read at call time."""
    return os.environ.get("LR_NATIVE_BUILDER", "0").strip().lower() in (
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
    cheapness -- scoring each candidate once with spare compute.
    `LR_CONCENTRATE` turns this on as one of its additive levers."""
    return _concentrate() or os.environ.get("LR_VALUE_COMMIT", "0").strip().lower() in (
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
    """Skip COMET targets in candidate generation (default 1 = skip; PI 2026-06-20).
    Comet intercept (aim_comet) can mis-predict on a moving target and there is no
    oob/accuracy guard (`_sun_clear` only checks the sun, not the board edge), so a
    missed comet shot sails off-board -- a wasted fleet the PI spotted in a replay.
    Verified neutral on the live path (no outcome change on the tested seeds, fewer
    launches). Temporary: target comets again (LR_SKIP_COMETS=0) once the comet aim
    has an out-of-bounds guard."""
    return _i("LR_SKIP_COMETS", 1) >= 1


def _decisive_capture():
    """Default-OFF gate (PI 2026-06-20). When ON, take a target with ONE decisive
    fleet from our STRONGEST source, sized to beat not just the target's garrison but
    the strongest enemy force that can reach it by ~our arrival (the contest) -- a
    bigger fleet is also FASTER (fleet speed grows with size), so it wins the race and
    holds, like the opponent's single big strike. Falls back to the minimal solo/gang
    when no single source can field it (no forced passivity). Read at call time.
    `LR_CONCENTRATE` turns this on as one of its additive levers."""
    return _concentrate() or os.environ.get("LR_DECISIVE_CAPTURE", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _concentrate():
    """Default-OFF gate (PI 2026-06-20 redesign). When ON (with the native leaf), the
    2-ply chooses among a CONCENTRATED, value-ordered menu only -- idle + the top-1..m
    best objectives -- with the scattered producer-floor move DROPPED. So the emitted
    move is at most m best objectives (or idle/hold), structurally concentrated; the
    leaf can no longer escape to the producer's scattered move. OFF = original menu.
    Read at call time."""
    return os.environ.get("LR_CONCENTRATE", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _teamup():
    """Default-OFF gate (PI 2026-06-21, LR_TEAMUP). With the decisive capture, when
    NO single source can field the decisive (take + beat-the-retake + hold) size,
    COMBINE the nearest-ETA sources into one strike that reaches that size -- so a
    contested corner is taken with enough force to HOLD, instead of falling through
    to the thin gang-up (sized to defenders only, no contest/hold) that gets
    retaken. The native builder leaf validates the combined strike actually holds
    (a too-staggered team-up scores low and isn't committed). Standalone flag (NOT
    folded into LR_CONCENTRATE) so the shipped concentrate stack stays byte-
    identical. Read at call time."""
    return os.environ.get("LR_TEAMUP", "0").strip().lower() in (
        "1", "true", "on", "yes")


def _frontier():
    """Default-OFF gate (PI 2026-06-21, LR_FRONTIER). Skip ENEMY targets that sit
    deep in enemy territory -- where our reinforcement can't follow, so a capture is
    retaken. A target's DEPTH = dist(target, nearest OUR planet) / dist(target,
    nearest OTHER enemy planet); depth > LR_FRONTIER_DEPTH (default 1.5) means the
    target is much closer to the enemy than to us. We then attack only frontier
    enemy planets (the contested border, depth ~1) we can support, instead of
    hurling large fleets into the enemy's home cluster (the measured failure: mean
    attack depth 2.28, 100-210-ship strikes 3-4x deep). NEUTRAL expansion is
    untouched (filter is enemy-only). Read at call time."""
    return os.environ.get("LR_FRONTIER", "0").strip().lower() in (
        "1", "true", "on", "yes")


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
    if _native_allocate() or _native_reinforce():
        # Conserved per-source allocation (PI 2026-06-20). Pot is REINFORCEMENT-
        # aware: use the do-nothing projected garrison at a near step (the
        # second-order retake -- incoming enemy fleets land into the source
        # garrison and into our defense). In-flight fleets are NOT double-counted:
        # they land into ships_traj via the recurrence, not added separately here.
        # The reinforcement-race gate (_native_reinforce) also routes the enemy
        # threat through allocate so attack vs reinforcement is an apples-to-apples
        # conserved contest.
        k_ref = min(int(H), _i("LR_NATIVE_REINF_STEP", 3))
        ships_ref = ships_traj[0, :, k_ref].to(_torch.float32)
        # Dynamic threat ("see the opponent coming"): per-step projected garrison as
        # the source mass, so the threat/reinforcement rises over the horizon as the
        # opponent (and we) build up, instead of the frozen k_ref snapshot.
        dyn_mbs = ships_traj[0].to(_torch.float32) if _native_dynamic() else None
        if _native_threat_max():
            # Concentrated threat (PI 2026-06-21): each planet faces the worst-case
            # SINGLE enemy planet that can reach it (full ships+production), not the
            # conserved split -- so a stronghold within reach reads as a real threat
            # and holding the targeted garrison becomes the valued move. Dynamic mass
            # / growth still apply; the allocate-only knobs are inert under `max`.
            atk_reach = _nf_reach_mass(
                cross_dist=cache.cross_dist, ships=ships_ref, is_enemy=is_enemy,
                H=int(H), prod=prod, growth_alpha=_f("LR_NATIVE_THREAT_GROWTH", 0.0),
                aggregate="max", mass_by_step=dyn_mbs,
            )
        else:
            atk_reach = _nf_reach_mass(
                cross_dist=cache.cross_dist, ships=ships_ref, is_enemy=is_enemy,
                H=int(H), prod=prod, growth_alpha=_f("LR_NATIVE_THREAT_GROWTH", 0.0),
                aggregate="allocate", our_garrison=ships_ref,
                alloc_w_prox=_f("LR_ALLOC_W_PROX", 1.0),
                alloc_w_val=_f("LR_ALLOC_W_VAL", 1.0),
                alloc_w_def=_f("LR_ALLOC_W_DEF", 0.5),
                alloc_eps=_f("LR_ALLOC_EPS", 1.0),
                mass_by_step=dyn_mbs,
            )
    else:
        atk_reach = _nf_reach_mass(
            cross_dist=cache.cross_dist, ships=ships0, is_enemy=is_enemy, H=int(H),
            prod=prod, growth_alpha=_f("LR_NATIVE_THREAT_GROWTH", 0.0),
        )
    off_reach = None
    off_weight = 0.0
    off_steepness = 5.0
    if _native_offense():
        # Symmetric mirror: OUR routable mass onto enemy/neutral planets (our planets
        # are the "sources"). Conserved `allocate` so held mass isn't double-counted
        # across many enemy targets. Reference garrison at the same near step as the
        # defensive snapshot. This is the capture-POTENTIAL the leaf credits.
        k_off = min(int(H), _i("LR_NATIVE_OFFENSE_STEP", 3))
        ships_off = ships_traj[0, :, k_off].to(_torch.float32)
        is_mine_t = (owner0 == int(me))
        off_reach = _nf_reach_mass(
            cross_dist=cache.cross_dist, ships=ships_off, is_enemy=is_mine_t,
            H=int(H), prod=prod, aggregate="allocate", our_garrison=ships_off,
            alloc_w_prox=_f("LR_ALLOC_W_PROX", 1.0),
            alloc_w_val=_f("LR_ALLOC_W_VAL", 1.0),
            alloc_w_def=_f("LR_ALLOC_W_DEF", 0.5),
            alloc_eps=_f("LR_ALLOC_EPS", 1.0),
        )
        off_weight = _f("LR_NATIVE_OFFENSE_W", 0.5)
        off_steepness = _f("LR_NATIVE_OFFENSE_STEEPNESS", 5.0)
    def_reach = None
    def_weight = 0.0
    if _native_reinforce():
        # Reinforcement race (PI 2026-06-20): how much OUR OTHER planets can route to
        # support each of our planets -- conserved, so finite reinforcement is split
        # across our holdings (overextension spreads it thin -> captures flip; a few
        # close captures concentrate it -> they hold). Holdability becomes our
        # reinforcement reach vs the enemy's attack reach. ships_ref is the projected
        # garrison, so a plan that drained its planets on attacks shows weak support.
        is_mine_d = (owner0 == int(me))
        def_reach = _nf_reach_mass(
            cross_dist=cache.cross_dist, ships=ships_ref, is_enemy=is_mine_d,
            H=int(H), prod=prod, aggregate="allocate", our_garrison=ships_ref,
            alloc_w_prox=_f("LR_ALLOC_W_PROX", 1.0),
            alloc_w_val=_f("LR_ALLOC_W_VAL", 1.0),
            alloc_w_def=_f("LR_ALLOC_W_DEF", 0.5),
            alloc_eps=_f("LR_ALLOC_EPS", 1.0),
            target_mask=is_mine_d, exclude_self=True,
            mass_by_step=dyn_mbs,
        )
        def_weight = _f("LR_NATIVE_DEF_W", 1.0)
    val = _nf_hazard_value(
        owner=owner_traj, ships=ships_traj, prod=prod, atk_reach=atk_reach,
        me=int(me), steepness=_f("LR_NATIVE_STEEPNESS", 5.0),
        discount=_f("LR_NATIVE_DISCOUNT", 1.0), value_mode="ships",
        inflight=_nf_inflight(arr_c), terminal=_f("LR_NATIVE_TERMINAL", 12.0),
        off_reach=off_reach, off_weight=off_weight, off_steepness=off_steepness,
        def_reach=def_reach, def_weight=def_weight,
    )
    _NATIVE_LEAF_CALLS += 1
    return float(val.reshape(-1)[0])


def _build_native_scorer(obs, me):
    """Return (score_units_fn, id2slot) or raise -- the BUILDER counterpart of
    _build_orbit_scorer (main.py), but scoring launch-sets with the native
    flip-hazard leaf (== the 2-ply chooser leaf _native_value) instead of the
    producer net-ship-delta scorer (PI 2026-06-21, LR_NATIVE_BUILDER).

    The expensive movement/threat tensors (movement build, garrison status, distance
    cache, the do-nothing trajectory, and the atk/off/def reach tensors -- all of
    which depend on the current board's do-nothing projection, NOT on the candidate)
    are built ONCE here; the returned closure does only the per-candidate trajectory
    recurrence + hazard value. Score is MARGINAL over do-nothing (empty -> 0.0), so
    it slots into the greedy's `v > current + floor` accept test exactly like the
    producer scorer. Mirrors _native_value's blocks verbatim (duplicated rather than
    refactored, to keep the shipped chooser path byte-identical)."""
    ot = _single_obs_to_tensor(obs, player_id=int(me))
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
    background = status.arrivals_by_owner[..., 1:, :]
    ids = ot["planets"][:, 0].long().tolist()
    id2slot = {int(v): i for i, v in enumerate(ids)}

    # Do-nothing trajectory + reach tensors -- identical to _native_value, computed once.
    empty_l = _torch.full((1, 1), -1, dtype=_torch.long)
    base_owner, base_ships, base_arr = _nf_build_traj(
        init_owner=owner0, init_ships=ships0, prod=prod, alive_by_step=alive_by_step,
        background_arrivals=background, src=empty_l, tgt=empty_l,
        ships=_torch.zeros(1, 1), eta=_torch.ones(1, 1),
        owner=_torch.zeros(1, 1, dtype=_torch.long),
        valid=_torch.zeros(1, 1, dtype=_torch.bool),
    )
    if _native_allocate() or _native_reinforce():
        k_ref = min(int(H), _i("LR_NATIVE_REINF_STEP", 3))
        ships_ref = base_ships[0, :, k_ref].to(_torch.float32)
        dyn_mbs = base_ships[0].to(_torch.float32) if _native_dynamic() else None
        if _native_threat_max():
            atk_reach = _nf_reach_mass(
                cross_dist=cache.cross_dist, ships=ships_ref, is_enemy=is_enemy,
                H=int(H), prod=prod, growth_alpha=_f("LR_NATIVE_THREAT_GROWTH", 0.0),
                aggregate="max", mass_by_step=dyn_mbs,
            )
        else:
            atk_reach = _nf_reach_mass(
                cross_dist=cache.cross_dist, ships=ships_ref, is_enemy=is_enemy,
                H=int(H), prod=prod, growth_alpha=_f("LR_NATIVE_THREAT_GROWTH", 0.0),
                aggregate="allocate", our_garrison=ships_ref,
                alloc_w_prox=_f("LR_ALLOC_W_PROX", 1.0),
                alloc_w_val=_f("LR_ALLOC_W_VAL", 1.0),
                alloc_w_def=_f("LR_ALLOC_W_DEF", 0.5),
                alloc_eps=_f("LR_ALLOC_EPS", 1.0), mass_by_step=dyn_mbs,
            )
    else:
        ships_ref = ships0
        dyn_mbs = None
        atk_reach = _nf_reach_mass(
            cross_dist=cache.cross_dist, ships=ships0, is_enemy=is_enemy, H=int(H),
            prod=prod, growth_alpha=_f("LR_NATIVE_THREAT_GROWTH", 0.0),
        )
    off_reach = None
    off_weight = 0.0
    off_steepness = 5.0
    if _native_offense():
        k_off = min(int(H), _i("LR_NATIVE_OFFENSE_STEP", 3))
        ships_off = base_ships[0, :, k_off].to(_torch.float32)
        is_mine_t = (owner0 == int(me))
        off_reach = _nf_reach_mass(
            cross_dist=cache.cross_dist, ships=ships_off, is_enemy=is_mine_t,
            H=int(H), prod=prod, aggregate="allocate", our_garrison=ships_off,
            alloc_w_prox=_f("LR_ALLOC_W_PROX", 1.0),
            alloc_w_val=_f("LR_ALLOC_W_VAL", 1.0),
            alloc_w_def=_f("LR_ALLOC_W_DEF", 0.5), alloc_eps=_f("LR_ALLOC_EPS", 1.0),
        )
        off_weight = _f("LR_NATIVE_OFFENSE_W", 0.5)
        off_steepness = _f("LR_NATIVE_OFFENSE_STEEPNESS", 5.0)
    def_reach = None
    def_weight = 0.0
    if _native_reinforce():
        is_mine_d = (owner0 == int(me))
        def_reach = _nf_reach_mass(
            cross_dist=cache.cross_dist, ships=ships_ref, is_enemy=is_mine_d,
            H=int(H), prod=prod, aggregate="allocate", our_garrison=ships_ref,
            alloc_w_prox=_f("LR_ALLOC_W_PROX", 1.0),
            alloc_w_val=_f("LR_ALLOC_W_VAL", 1.0),
            alloc_w_def=_f("LR_ALLOC_W_DEF", 0.5), alloc_eps=_f("LR_ALLOC_EPS", 1.0),
            target_mask=is_mine_d, exclude_self=True, mass_by_step=dyn_mbs,
        )
        def_weight = _f("LR_NATIVE_DEF_W", 1.0)

    steep = _f("LR_NATIVE_STEEPNESS", 5.0)
    disc = _f("LR_NATIVE_DISCOUNT", 1.0)
    term = _f("LR_NATIVE_TERMINAL", 12.0)

    def _hazard(owner_traj, ships_traj, arr):
        return _nf_hazard_value(
            owner=owner_traj, ships=ships_traj, prod=prod, atk_reach=atk_reach,
            me=int(me), steepness=steep, discount=disc, value_mode="ships",
            inflight=_nf_inflight(arr), terminal=term,
            off_reach=off_reach, off_weight=off_weight, off_steepness=off_steepness,
            def_reach=def_reach, def_weight=def_weight,
        )

    base_val = float(_hazard(base_owner, base_ships, base_arr).reshape(-1)[0])

    def score_native_units(units):
        if not units:
            return 0.0
        L = len(units)
        src = _torch.tensor([[int(u[0]) for u in units]], dtype=_torch.long)
        tgt = _torch.tensor([[int(u[1]) for u in units]], dtype=_torch.long)
        sh = _torch.tensor([[float(u[2]) for u in units]])
        et = _torch.tensor([[float(max(1, int(u[3]))) for u in units]])
        ow = _torch.full((1, L), int(me), dtype=_torch.long)   # arriving fleets are ours
        va = _torch.ones((1, L), dtype=_torch.bool)
        owner_traj, ships_traj, arr = _nf_build_traj(
            init_owner=owner0, init_ships=ships0, prod=prod,
            alive_by_step=alive_by_step, background_arrivals=background,
            src=src, tgt=tgt, ships=sh, eta=et, owner=ow, valid=va,
        )
        v = float(_hazard(owner_traj, ships_traj, arr).reshape(-1)[0])
        return v - base_val            # MARGINAL over do-nothing (empty -> 0)

    return score_native_units, id2slot


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

    # Plan-builder scorer (default OFF): score candidate launches with the native
    # flip-hazard leaf (== the 2-ply chooser) instead of the producer net-ship-delta
    # scorer, so far thin grabs / exposed-planet drains are never built. Rebind
    # score_units only; id2slot is derived identically so units_for stays valid.
    if _native_builder() and orbit is not None:
        try:
            _nat = _build_native_scorer(obs, me)
        except Exception:
            _nat = None
        if _nat is not None:
            score_units = _nat[0]

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

    # Garrison floor (default OFF): a planet a single enemy can clearly hit keeps
    # enough garrison to win that fight and may spend only its SURPLUS; unthreatened
    # planets are untouched (drain freely). Caps `available` BEFORE candidates are
    # built, so the menu gains 'attack elsewhere, hold the threatened planet'.
    if _garrison_floor():
        window = _f("LR_FLOOR_WINDOW", 18.0)        # steps of look-ahead for "can reach"
        enemy_planets = [p for p in planets if int(p.owner) != me and int(p.owner) != -1]
        enemy_fleets = [f for f in fleets if int(f.owner) != me and int(f.owner) != -1]
        for mp in my_planets:
            mid = int(mp.id)
            mxy = (float(mp.x), float(mp.y))
            threat = 0.0
            eta_threat = window
            # (a) strongest enemy PLANET that can reach within the window:
            #     ships + production accrued by arrival (the "count ships+prod" view).
            for q in enemy_planets:
                eta = dist(mxy, (float(q.x), float(q.y))) / max(1e-6, fleet_speed(float(q.ships)))
                if eta <= window:
                    t = float(q.ships) + float(q.production) * eta
                    if t > threat:
                        threat, eta_threat = t, eta
            # (b) strongest enemy FLEET already in flight and CLOSING on this planet
            #     (the in-flight attack the leaf threat model misses).
            for f in enemy_fleets:
                fxy = (float(f.x), float(f.y))
                eta = dist(mxy, fxy) / max(1e-6, fleet_speed(float(f.ships)))
                if eta > window:
                    continue
                closing = (math.cos(float(f.angle)) * (mxy[0] - fxy[0])
                           + math.sin(float(f.angle)) * (mxy[1] - fxy[1])) > 0.0
                if closing and float(f.ships) > threat:
                    threat, eta_threat = float(f.ships), eta
            if threat <= 0.0:
                continue                              # not under threat -> no reserve
            # Net of REINFORCEMENT: our OTHER planets that can route ships here BEFORE
            # the threat lands share the defense, so an isolated planet reserves the
            # most and a well-supported one barely reserves (frees surplus to expand).
            # A single distant enemy is thus no longer a full threat to every planet.
            support = 0.0
            for o in my_planets:
                if int(o.id) == mid:
                    continue
                if dist(mxy, (float(o.x), float(o.y))) / max(1e-6, fleet_speed(float(o.ships))) <= eta_threat:
                    support += float(o.ships)
            need = threat - _f("LR_FLOOR_SUPPORT_W", 1.0) * support
            if need <= 0.0:
                continue                              # reinforcement covers it -> no reserve
            reserve = min(int(mp.ships), int(math.ceil(need)))
            available[mid] = max(0, int(mp.ships) - reserve)

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
    # Neutral mass margin (ported from r48nve d3049d1f; default 0.0 = OFF =>
    # byte-identical). Expansion (neutral) captures are otherwise sized to JUST take
    # the empty planet -- a small, by the size->speed law SLOW fleet that lands thin,
    # so we crawl into an undefendable spread. With this margin a neutral capture is
    # sized bigger: faster arrival AND surplus garrison to hold the new planet / stage
    # the next push. Scaled by the planet's value (production) and travel time so far /
    # valuable neutrals get the most mass. Enemy captures already get hold_margin.
    neutral_margin = _f("LR_NEUTRAL_MARGIN", 0.0)

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
        # Frontier filter (default OFF): don't attack an ENEMY planet that sits much
        # deeper in enemy territory than ours -- we can't reinforce it, so it's
        # retaken. depth = (dist to nearest OUR planet) / (dist to nearest OTHER
        # enemy planet); skip when depth > LR_FRONTIER_DEPTH. Neutrals unaffected.
        if is_enemy and _frontier():
            txy = (float(tgt.x), float(tgt.y))
            d_ours = min((dist(txy, (float(p.x), float(p.y))) for p in my_planets),
                         default=float("inf"))
            d_enemy = min((dist(txy, (float(p.x), float(p.y))) for p in planets
                           if int(p.owner) != me and int(p.owner) != -1
                           and int(p.id) != tid), default=float("inf"))
            if d_enemy < float("inf") and d_ours > _f("LR_FRONTIER_DEPTH", 1.5) * d_enemy:
                continue                   # too deep in enemy territory -> unholdable
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
            elif (not is_enemy) and neutral_margin > 0.0:
                size += int(math.ceil(neutral_margin * (tgt.ships + prod * eta)))  # mass expansion
            shots.append((eta, size, int(src.id), src, angle))
        if not shots:
            continue
        shots.sort(key=lambda x: x[0])
        rank = prod / max(1.0, shots[0][0])
        if is_enemy and enemy_boost != 1.0:
            rank *= enemy_boost
        front = frontier_eta((float(tgt.x), float(tgt.y)))

        # Decisive capture (PI 2026-06-20): ONE big fleet from our STRONGEST source,
        # sized to beat the target's defenders AND the strongest enemy force that can
        # reach it by ~our arrival (the contest) -- bigger => faster (size->speed) =>
        # wins the race and holds, like the opponent's single decisive strike. Falls
        # through to the minimal solo/gang below if no single source can field it.
        if _decisive_capture():
            eta0 = float(shots[0][0])
            defenders0 = (prod * eta0 + float(tgt.ships)) if is_enemy else float(tgt.ships)
            horizon = eta0 + _f("LR_DECISIVE_BUF", 3.0)
            txy = (float(tgt.x), float(tgt.y))
            contest = 0.0
            for q in planets:
                if int(q.owner) != me and int(q.owner) != -1:
                    if dist(txy, (float(q.x), float(q.y))) <= horizon * fleet_speed(float(q.ships)):
                        contest = max(contest, float(q.ships))
            for fl in fleets:
                if int(fl.owner) != me and int(fl.owner) != -1:
                    if dist(txy, (float(fl.x), float(fl.y))) <= horizon * fleet_speed(float(fl.ships)):
                        contest = max(contest, float(fl.ships))
            dec_size = int(math.ceil(defenders0 + contest)) + 1
            if is_enemy and hold_margin > 0.0:
                dec_size += int(math.ceil(hold_margin * defenders0))
            elif (not is_enemy) and neutral_margin > 0.0:
                dec_size += int(math.ceil(neutral_margin * (float(tgt.ships) + prod * eta0)))
            # Strongest affordable source (most ships) -> big, fast, decisive.
            dec = None
            for (eta, size, sid, src, angle) in sorted(
                    shots, key=lambda s: -available.get(s[2], 0)):
                if available[sid] >= dec_size:
                    shot = _plan_shot(src, tgt, comet_ids, comet_paths, omega, dec_size)
                    if shot is None:
                        continue
                    a2, eta2, _ = shot
                    units = units_for([(sid, tid, dec_size, eta2)])
                    if units is None and id2slot is not None:
                        continue
                    dec = {"emit": [[sid, float(a2), dec_size]],
                           "units": units, "srcs": {sid: dec_size},
                           "rank": rank, "front": front}
                    break
            if dec is None and _teamup():
                # TEAM UP (PI 2026-06-21): no single source can field the decisive
                # (take + contest + hold) size -> pool the nearest-ETA sources
                # (arrivals clustered) until the combined mass reaches dec_size, so
                # the corner is taken with enough force to HOLD instead of the thin
                # gang-up below. The native builder leaf validates holdability.
                t_emit, t_triples, t_srcs, acc = [], [], {}, 0
                for (eta, size, sid, src, angle) in sorted(shots, key=lambda s: s[0]):
                    if sid in t_srcs:
                        continue
                    take = min(available[sid], dec_size - acc)
                    if take <= 0:
                        continue
                    shot = _plan_shot(src, tgt, comet_ids, comet_paths, omega, take)
                    if shot is None:
                        continue
                    a2, eta2, _ = shot
                    t_emit.append([sid, float(a2), take])
                    t_triples.append((sid, tid, take, eta2))
                    t_srcs[sid] = take
                    acc += take
                    if acc >= dec_size:
                        break
                if acc >= dec_size and len(t_emit) >= 2:
                    units = units_for(t_triples)
                    if not (units is None and id2slot is not None):
                        dec = {"emit": t_emit, "units": units, "srcs": t_srcs,
                               "rank": rank, "front": front, "kind": "teamup"}
            if dec is not None:
                candidates.append(dec)
                continue

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
        # The native-builder marginal is a discounted-MEAN ship-margin, so a capture
        # that flips after a few steps still scores positive (it was owned a while).
        # A 0 floor therefore accepts doomed captures -> over-commit -> the frontier
        # collapses (captures not maintained). Require a REAL margin (~3 ships) so
        # only captures that actually stick are committed. Read at call time (env
        # override) -- unlike the producer ROI_FLOOR this is tunable per run.
        floor = _f("LR_NATIVE_BUILDER_FLOOR", 3.0) if _native_builder() else ROI_FLOOR
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
            # scattered tail the flip hazard says the opponent retakes. Under
            # LR_CONCENTRATE these prefixes are VALUE-ordered (best-first) and the
            # captures DECISIVE (big fast fleets), so the leaf has strong concentrated
            # options to prefer -- WITHOUT removing the producer floor (anti-passivity).
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
