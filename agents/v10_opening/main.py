"""v10_opening — v7_0_drop_one architecture + opening-conditional scoring.

Synthesis of two threads:

1. Search base (from v7_0_drop_one, live μ≈1030.4):
   v3.5.1 incumbent (aggressive snipe + reinforce) → drop-one
   enumeration → fast_sim K=10 rollout against top-tier mirror
   opp → argmax with parity-floor fallback.

2. Opening fix (from v9_opening, audit/2026-05-12-opening-analysis.md):
   v3 is myopic about opp targets in steps 4-20; contested neutral
   captures produce -38 to +59 ship swings per seed vs precision.
   When step < OPENING_HORIZON and game is 2P, BEFORE settle_plan:
   - NEUTRAL_BONUS=1.5× on neutral mission scores (re-enables the
     globally-disabled boost in the opening window only)
   - Opp-aware demotion 0.1× on targets opp reaches first (predict
     opp via v3.5.1-from-opp-POV mission proposal, same pattern
     v7_minimax used for its opp model)

The opening adjustments modify the mission scores BEFORE
settle_plan picks the incumbent. The drop-one enumerator then
explores around that better incumbent, scored by fast_sim. Outside
the opening window the behavior is identical to v7_0_drop_one.

σ-equiv layer is NOT used (reverted per v7.6 bisect — regresses
drop-one architecture by ~54pp).
"""

from __future__ import annotations

import math
import time

from lib.fast_sim import clone as fs_clone
from lib.fast_sim import from_obs as fs_from_obs
from lib.fleet import speed as fleet_speed
from lib.intent import Intent, World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.mission import Mission
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.v7_search import enumerate_candidates, score_candidate, _infer_num_seats
from lib.world_model import WorldModel


OPENING_HORIZON = 15
NEUTRAL_BONUS_OPENING = 1.5
COMET_BONUS_OPENING = 1.3
OPP_CONTESTED_DEMOTION = 0.1

K_ROLLOUT = 10
WALLCLOCK_MS = 700.0


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _swap_obs_player(obs, opp_id: int):
    if isinstance(obs, dict):
        obs2 = dict(obs)
        obs2["player"] = opp_id
        return obs2
    keys = ("player", "planets", "fleets", "angular_velocity",
            "initial_planets", "comet_planet_ids", "comets",
            "step", "next_fleet_id", "remainingOverageTime")
    obs2 = {}
    for k in keys:
        v = getattr(obs, k, None)
        if v is not None:
            obs2[k] = v
    obs2["player"] = opp_id
    return obs2


def _predict_opp_first_targets(obs, opp_id: int) -> dict[int, int]:
    """Run v3.5.1 mission proposal from opp's POV; return
    {target_id: opp_eta} for the missions opp's settle_plan would pick.
    """
    swapped_obs = _swap_obs_player(obs, opp_id)
    opp_world = World.from_obs(swapped_obs)
    if not opp_world.planets_by_id:
        return {}
    opp_model = WorldModel.from_world(opp_world)
    opp_missions = (
        propose_snipe_missions(opp_world, opp_model, aggressive=True)
        + propose_reinforce_missions(opp_world, opp_model)
    )
    opp_intents = settle_plan(opp_missions, opp_world, opp_model)
    out: dict[int, int] = {}
    for it in opp_intents:
        src = opp_world.planets_by_id.get(it.src_id)
        tgt = opp_world.planets_by_id.get(it.target_id)
        if src is None or tgt is None:
            continue
        d = math.hypot(tgt.x - src.x, tgt.y - src.y)
        v = fleet_speed(int(it.ships))
        eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
        prev = out.get(it.target_id)
        if prev is None or eta < prev:
            out[it.target_id] = eta
    return out


def _apply_opening_adjustments(missions: list, world: World,
                                opp_targets: dict[int, int]) -> list:
    """Score-rewrite missions: NEUTRAL_BONUS / COMET_BONUS / opp demotion."""
    out = []
    for m in missions:
        t = world.planets_by_id.get(m.target_id)
        if t is None:
            out.append(m)
            continue
        score = m.score
        is_neutral = t.owner == -1
        is_comet = t.id in world.comet_ids
        if is_comet:
            score *= COMET_BONUS_OPENING
        elif is_neutral:
            score *= NEUTRAL_BONUS_OPENING
        if m.target_id in opp_targets:
            opp_eta = opp_targets[m.target_id]
            if opp_eta <= m.eta:
                score *= OPP_CONTESTED_DEMOTION
        out.append(Mission(
            mission_class=m.mission_class,
            src_id=m.src_id,
            target_id=m.target_id,
            ships=m.ships,
            score=score,
            eta=m.eta,
        ))
    return out


def _build_missions(world: World, model: WorldModel) -> list:
    return (
        propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )


def agent(obs, configuration=None):
    t_start = time.perf_counter()

    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    my_id = world.my_id
    model = WorldModel.from_world(world)
    step = int(world.step)

    missions = _build_missions(world, model)

    if step < OPENING_HORIZON:
        owners = {p.owner for p in world.planets_by_id.values() if p.owner != -1}
        if len(owners) == 2 and my_id in owners:
            opp_id = next(iter(owners - {my_id}))
            try:
                opp_targets = _predict_opp_first_targets(obs, opp_id)
            except Exception:
                opp_targets = {}
            missions = _apply_opening_adjustments(missions, world, opp_targets)
        else:
            missions = _apply_opening_adjustments(missions, world, {})

    incumbent_intents = settle_plan(missions, world, model)
    incumbent_action = realize(
        incumbent_intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model
    )

    if _infer_num_seats(world) != 2:
        return incumbent_action

    try:
        snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=2)
    except Exception:
        return incumbent_action

    candidates = enumerate_candidates(
        world, model,
        enumerator_mode="drop_one",
        incumbent_intents=incumbent_intents,
        incumbent_action=incumbent_action,
        obs=obs,
    )
    if len(candidates) <= 1:
        return incumbent_action

    best_action = incumbent_action
    best_score = float("-inf")
    incumbent_scored = False
    for cand in candidates:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if elapsed_ms > WALLCLOCK_MS:
            break
        try:
            score = score_candidate(snap, cand, my_id=my_id, K=K_ROLLOUT, opp_tier=1)
        except Exception:
            continue
        if not incumbent_scored:
            incumbent_scored = True
            best_score = score
            best_action = list(cand)
            continue
        if score > best_score:
            best_score = score
            best_action = list(cand)
    return best_action
