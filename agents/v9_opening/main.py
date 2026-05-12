"""v9_opening — v3 base + opening-conditional opp-aware adjustments.

Diagnostic (audit/2026-05-12-opening-analysis.md) showed v3 falls
behind in opening (steps 4-20) vs aggressive opps (roi, precision)
with HIGH per-seed variance — some games down 38 ships at step 30
despite +20 mean across seeds. Root cause: v3's snipe scoring is
myopic — picks best ROI target ignoring that opp also targets the
same neutral. Contested-capture fights resolve unpredictably.

v9's two opening-conditional changes (step < OPENING_HORIZON only):

A. NEUTRAL_BONUS reactivation in opening only. The global
   NEUTRAL_BONUS=1.5 regressed 28% (lib/missions/snipe.py:46-51)
   because contested enemy planets were the binding constraint
   late game. In opening, no enemy planets are reachable — so
   the global regression cause doesn't apply. Re-enable in opening
   only.

B. Opp-aware target deduplication. Predict opp's first launch
   (via v3 from opp POV — same pattern as v7_minimax) and DEMOTE
   targets opp would reach before us. We send fleet to opp's
   non-targets instead, capturing uncontested.

After step OPENING_HORIZON: identical to v3 (no mid/late-game
regression). σ-equivariance preserved (uses the same lib/planner
patches).
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.geometry import sym_hypot
from lib.intent import World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.mission import Mission
from lib.planner import settle_plan
from lib.world_model import WorldModel


# ───────────────────────────────────────────────────────────────────────────
# Opening-conditional parameters
# ───────────────────────────────────────────────────────────────────────────

OPENING_HORIZON = 15           # step < this triggers opening logic
NEUTRAL_BONUS_OPENING = 1.5    # multiplier for neutral targets in opening
COMET_BONUS_OPENING = 1.3      # multiplier for comet targets (moot in step<15)
OPP_CONTESTED_DEMOTION = 0.1   # score multiplier for targets opp reaches first


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _swap_obs_player(obs, opp_id: int):
    """Return shallow-copied obs with player swapped — for opp-from-opp-POV."""
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
    """Run v3 mission-proposal on opp's POV; return {target_id: opp_eta}
    for the missions opp's settle_plan would PICK.

    Delegates to the same lib.missions/planner with my_id swapped — we
    get exactly what opp would do if opp = us (v3-class).
    """
    swapped_obs = _swap_obs_player(obs, opp_id)
    opp_world = World.from_obs(swapped_obs)
    opp_model = WorldModel.from_world(opp_world)
    opp_missions = (
        propose_snipe_missions(opp_world, opp_model)
        + propose_reinforce_missions(opp_world, opp_model)
    )
    opp_intents = settle_plan(opp_missions, opp_world, opp_model)
    out: dict[int, int] = {}
    for it in opp_intents:
        src = opp_world.planets_by_id.get(it.src_id)
        tgt = opp_world.planets_by_id.get(it.target_id)
        if src is None or tgt is None:
            continue
        d = sym_hypot(tgt.x - src.x, tgt.y - src.y)
        v = fleet_speed(int(it.ships))
        eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
        # Keep the SHORTEST eta if multiple sources target same target.
        prev = out.get(it.target_id)
        if prev is None or eta < prev:
            out[it.target_id] = eta
    return out


def _apply_opening_adjustments(missions: list, world: World,
                                opp_targets: dict[int, int]) -> list:
    """Apply NEUTRAL_BONUS + OPP_CONTESTED_DEMOTION to mission scores.

    Returns NEW mission list (Mission is a frozen-ish dataclass we
    reconstruct).
    """
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
        # Opp-contested demotion: if opp targets same AND opp arrives <= us
        if m.target_id in opp_targets:
            opp_eta = opp_targets[m.target_id]
            if opp_eta <= m.eta:
                score *= OPP_CONTESTED_DEMOTION
        new_m = Mission(
            mission_class=m.mission_class,
            src_id=m.src_id,
            target_id=m.target_id,
            ships=m.ships,
            score=score,
            eta=m.eta,
        )
        out.append(new_m)
    return out


def agent(obs):
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    missions = (
        propose_snipe_missions(world, model)
        + propose_reinforce_missions(world, model)
    )

    # Opening-conditional adjustments.
    step = int(world.step)
    if step < OPENING_HORIZON:
        # Detect 2P; opp-prediction in 4P is more complex (multiple opps).
        owners = {p.owner for p in world.planets_by_id.values() if p.owner != -1}
        if len(owners) == 2 and world.my_id in owners:
            opp_id = next(iter(owners - {world.my_id}))
            try:
                opp_targets = _predict_opp_first_targets(obs, opp_id)
            except Exception:
                opp_targets = {}
            missions = _apply_opening_adjustments(missions, world, opp_targets)
        else:
            # 4P: still apply NEUTRAL_BONUS / COMET_BONUS; no opp prediction
            missions = _apply_opening_adjustments(missions, world, {})

    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
