"""copycat — broad candidate enumeration + K=10 fast-brain argmax.

The strategy PI described: don't IMPOSE structure (e.g., sigma-pair),
let the strongest move EMERGE from a decision process that compares
candidates by projected gain. We just give the search a rich,
diverse candidate set and pick the highest-scoring one.

Architecture:

  Each turn, build a candidate pool from multiple sources:
    1. geo-style strategic stances - incumbent + tilts (opening_boost,
       enemy_focus, front_reinforce) + archetypes (concentrated,
       saturation, gang_up) + drop-one variants. Each is a genuinely
       different strategic shape.
    2. v7_0_drop_one's chosen action - the K=10 argmax over v3.5.1's
       mission set + drop-one variants. A solid heuristic floor.

  Score every candidate via lib.v7_search.score_candidate at K=10
  (the scalar fast brain on lib/fast_sim's 183x snapshot engine).
  Take the argmax. No tau gate, no sigma-pair constraint.

History note (this branch, ceb0710..50a0a3e): an earlier design used
sigma-paired perturbations as the candidate set. Phase 2 showed those
perturbations contributed zero net wins - the tweaks (angle nudges,
+-15% ship counts, drop-pair) were too cosmetic to materially change
the strategic shape and beat the floor. Removed in this revision per
PI directive: "don't impose; let it emerge from gain."

Per-turn budget:
  - sense_state + WorldModel build:    ~10 ms
  - geo-style tilt application (6-8):  ~15 ms
  - v7_0_drop_one chooser (capped):    ~400 ms
  - score_candidate K=10 x 8-10:       ~240-300 ms
  - Total:                              ~700 ms (under 1000 ms cap)

Env vars:
  COPYCAT_K                int   (default 10) - lookahead depth.
  COPYCAT_WALLCLOCK_MS     float (default 750) - total turn budget.
  COPYCAT_V7_WALLCLOCK_MS  float (default 400) - cap on v7_0_drop_one's
                                                 internal chooser.
  COPYCAT_USE_V7           "1"/"0" (default "1") - include v7_0_drop_one.
  COPYCAT_USE_GEO          "1"/"0" (default "1") - include geo-style
                                                   enumeration.

2P uses K=10; 4P uses K=8 (matches geo).
"""

from __future__ import annotations

import math
import os
import time
from typing import Callable, Optional


def _env_float(name, default):
    v = os.environ.get(name)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


def _env_int(name, default):
    v = os.environ.get(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


def _env_bool(name, default):
    v = os.environ.get(name, default)
    return str(v).strip() not in ("0", "false", "False", "")


_K = _env_int("COPYCAT_K", 10)
_K_4P = _env_int("COPYCAT_K_4P", 8)
_WALLCLOCK_MS = _env_float("COPYCAT_WALLCLOCK_MS", 750.0)
_V7_WALLCLOCK_MS = _env_float("COPYCAT_V7_WALLCLOCK_MS", 400.0)
_USE_V7 = _env_bool("COPYCAT_USE_V7", "1")
_USE_GEO = _env_bool("COPYCAT_USE_GEO", "1")
_PER_SCORE_TIMEOUT_MS = _env_float("COPYCAT_PER_SCORE_TIMEOUT_MS", 700.0)


# ---------------------------------------------------------------------------
# Action helpers.
# ---------------------------------------------------------------------------


def _action_key(action):
    return tuple((int(r[0]), round(float(r[1]), 5), int(r[2])) for r in action)


# ---------------------------------------------------------------------------
# Candidate generators (lazy-imported so unit tests can stub).
# ---------------------------------------------------------------------------


def _geo_candidates(obs, configuration):
    """Build geo's full candidate set WITHOUT geo's K=10 lookahead.

    Returns list[(name, action)]. We reuse geo's proposers + tilts +
    archetypes + drop-one helpers directly; the scoring happens later
    in our pool.
    """
    from agents.geo.main import (
        _action_from_intents,
        _build_base_missions,
        _concentrated_archetype_tilt,
        _drop_one_capped,
        _enemy_focus_tilt,
        _front_reinforce_tilt,
        _gang_up_action,
        _opening_boost_tilt,
        _saturation_archetype_action,
        _settle_with_tilt,
        MAX_DROP_ONE_VARIANTS,
    )
    from lib.geo.sense import sense_state
    from lib.intent import World
    from lib.planner import settle_plan
    from lib.world_model import WorldModel

    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    sense = sense_state(world, model)

    base = _build_base_missions(world, model)
    incumbent_intents = settle_plan(base, world, model)
    incumbent_action = _action_from_intents(incumbent_intents, obs, model)

    seen: set[tuple] = {_action_key(incumbent_action)}
    out: list[tuple[str, list]] = [("geo_incumbent", incumbent_action)]

    def add(name: str, action):
        if not action:
            return
        k = _action_key(action)
        if k in seen:
            return
        seen.add(k)
        out.append((name, action))

    # Tilts (per-mission transforms).
    for name, tilt_fn in [
        ("opening_boost", _opening_boost_tilt(world)),
        ("enemy_focus", _enemy_focus_tilt(world)),
        ("concentrated", _concentrated_archetype_tilt(world)),
        ("front_reinforce", _front_reinforce_tilt(sense)),
    ]:
        if tilt_fn is None:
            continue
        try:
            add(name, _settle_with_tilt(base, world, model, tilt_fn))
        except Exception:
            continue

    # Archetypes (alternative settlements).
    try:
        gu = _gang_up_action(base, world, model)
        if gu is not None:
            add("gang_up", gu)
    except Exception:
        pass
    try:
        sat = _saturation_archetype_action(base, world, model, sense)
        add("saturation", sat)
    except Exception:
        pass

    # Drop-one variants of the incumbent (proven v7_0 floor).
    for variant in _drop_one_capped(incumbent_action, MAX_DROP_ONE_VARIANTS):
        add("drop_one", variant)

    return out


def _v7_drop_one_action(obs, configuration):
    """Run v7_0_drop_one and return its chosen action.

    Capped internal wallclock to fit our total turn budget.
    """
    from lib.v7_search import choose
    return choose(
        obs, configuration,
        enumerator_mode="drop_one",
        K=10,
        wallclock_ms=_V7_WALLCLOCK_MS,
    )


# ---------------------------------------------------------------------------
# Scoring with per-call SIGALRM safety (copy of geo's _score_with_timeout).
# ---------------------------------------------------------------------------


class _ScoreTimeout(Exception):
    pass


def _score_with_timeout(score_fn, timeout_ms, *args, **kwargs):
    import signal
    try:
        signal.signal
    except AttributeError:
        return score_fn(*args, **kwargs)
    if timeout_ms <= 0:
        return score_fn(*args, **kwargs)

    def _handler(signum, frame):
        raise _ScoreTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_ms / 1000.0)
    try:
        return score_fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# Fallback (when scoring fails).
# ---------------------------------------------------------------------------


def _fallback(obs):
    from lib.opp_model import top_tier_mirror_policy
    try:
        return top_tier_mirror_policy(obs)
    except Exception:
        return []


def _planets_from_obs(obs):
    if isinstance(obs, dict):
        return obs.get("planets", []) or []
    return list(getattr(obs, "planets", []) or [])


def _player_id(obs):
    if isinstance(obs, dict):
        return int(obs.get("player", 0))
    return int(getattr(obs, "player", 0))


def _num_seats(obs):
    from lib.mirror import detect_num_players
    n = detect_num_players(_planets_from_obs(obs))
    return n if n in (2, 4) else 2


# ---------------------------------------------------------------------------
# Agent entry.
# ---------------------------------------------------------------------------


def agent(obs, configuration=None):
    t0 = time.perf_counter()
    num_seats = _num_seats(obs)
    my_id = _player_id(obs)

    # 1. Build the candidate pool.
    candidates: list[tuple[str, list]] = []
    seen: set[tuple] = set()

    def add(name, action):
        if not action:
            # Empty is a valid candidate (stand pat); allow once.
            k = ()
        else:
            k = _action_key(action)
        if k in seen:
            return
        seen.add(k)
        candidates.append((name, action))

    if _USE_GEO:
        try:
            for name, act in _geo_candidates(obs, configuration):
                add(name, act)
        except Exception:
            pass

    if _USE_V7:
        try:
            v7_action = _v7_drop_one_action(obs, configuration)
            add("v7_0_drop_one", v7_action)
        except Exception:
            pass

    if not candidates:
        return _fallback(obs)

    # 2. Score the pool.
    try:
        from lib.fast_sim import from_obs as fs_from_obs
        snap = fs_from_obs(obs, configuration, num_seats=num_seats)
    except Exception:
        # Snapshot failed - play the first candidate (incumbent-like).
        return candidates[0][1]

    if num_seats == 2:
        from lib.v7_search import score_candidate as score_fn
        K = _K
        score_kwargs = {"my_id": my_id, "K": K, "opp_tier": 1}
    elif num_seats == 4:
        from lib.v7_search import score_candidate_4p as score_fn
        K = _K_4P
        score_kwargs = {"my_id": my_id, "K": K}
    else:
        return candidates[0][1]

    best_action = candidates[0][1]
    best_score = -math.inf
    scored_any = False
    for name, action in candidates:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > _WALLCLOCK_MS:
            break
        try:
            s = _score_with_timeout(
                score_fn, _PER_SCORE_TIMEOUT_MS,
                snap, action, **score_kwargs,
            )
        except _ScoreTimeout:
            continue
        except Exception:
            continue
        if not scored_any or s > best_score:
            scored_any = True
            best_score = float(s)
            best_action = action

    return best_action
