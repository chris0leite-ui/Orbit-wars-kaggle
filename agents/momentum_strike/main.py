"""momentum_strike V4 — production-first expansion + defense + synchronized salvo.

Architecture (V4 adds the salvo + cross-turn ledger; V2/V3 unchanged):

  1. Tick the per-seat ledger first. Any wait-N commit whose
     `wait_remaining` decrements to 0 fires this turn (re-aimed against
     current geometry). Sources holding pending commits are reserved.
  2. Defense pass — reinforce planets predicted to flip.
  3. Phase trigger — STRIKE when our planet count ≥ STRIKE_PLANET_RATIO ×
     max-per-opponent planet count AND we have ≥ STRIKE_MIN_PLANETS.
  4. In STRIKE: plan a synchronized salvo at the production leader.
     Emit wait_N==0 intents now; register wait_N>0 intents in the
     ledger (the fast sources actually hold fire).
  5. Expansion pass for remaining sources — production-first +
     ENEMY_MULTIPLIER when behind on planet count.

All emissions route through `lib.intent.realize` with
`DEFAULT_MECHANISMS`, getting auto-aim + auto-sizing + sun avoidance +
path-clears-planets for free.

Cross-turn state: `_PENDING_LAUNCHES: dict[int, list[dict]]` keyed by
player id, cleared on `step == 0`. Pattern mirrors
`agents/baseline/main.py::_tick_ledger` (lines ~225-304).
"""

from __future__ import annotations

import math
import os
from collections import Counter

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import Intent, World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.world_model import WorldModel

from agents.momentum_strike.proposer import propose_defense, propose_expand, propose_salvo

DEBUG = os.environ.get("MOMENTUM_DEBUG", "0") == "1"

# Phase-trigger knobs.
# V4 attempted to wire the synchronized salvo via cross-turn ledger.
# 8-game A/B vs `agents/baseline`: 0/8 — unchanged from V3. Simple-panel
# early-elim regressed 27/32 → 26/32 (uncapped wait) or 20/32 (no cap).
# The salvo reserves source-ships for many turns, starving expansion
# velocity faster than the synchronized landing wins back. Conclusion:
# salvo is at best neutral here, marginally regressive on PI's primary
# metric (elimination ≤250 turns).
#
# Gated OFF by default — set default ratio to 999.0 so _detect_phase
# never returns STRIKE; behavior reverts to V3 (production-first
# expand + defense + enemy_multiplier). Code (proposer.propose_salvo,
# _tick_ledger, lib/salvo.SalvoPlan.wait_Ns) is retained for future
# re-evaluation via MOMENTUM_STRIKE_RATIO=1.20 + MOMENTUM_SALVO_MAX_WAIT=8.
STRIKE_PLANET_RATIO = float(os.environ.get("MOMENTUM_STRIKE_RATIO", "999.0"))
STRIKE_MIN_PLANETS = int(os.environ.get("MOMENTUM_STRIKE_MIN", "3"))

# Per-seat ledger of pending wait-N commits (mirrors baseline pattern).
# Each entry: {src_id, tgt_id, ships_planned, wait_remaining, t_commit}.
_PENDING_LAUNCHES: dict[int, list[dict]] = {}


def _as_dict(obs) -> dict:
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


def _detect_phase(my_planets, enemy_planets) -> str:
    """STRIKE iff we have ≥ STRIKE_MIN_PLANETS AND our count ≥
    STRIKE_PLANET_RATIO × max-per-opponent. Else EXPAND.
    """
    if len(my_planets) < STRIKE_MIN_PLANETS or not enemy_planets:
        return "EXPAND"
    per_opp = Counter(int(p.owner) for p in enemy_planets)
    max_opp = max(per_opp.values()) if per_opp else 0
    if len(my_planets) >= STRIKE_PLANET_RATIO * max(1, max_opp):
        return "STRIKE"
    return "EXPAND"


def _tick_ledger(me: int, planets_by_id: dict) -> tuple[list[Intent], list[dict], set[int]]:
    """Decrement pending wait-N commits; emit ones whose timer hit 0.

    Returns `(due_intents, surviving_pending, pending_srcs)`:
      - `due_intents` — re-aimed (`aim_angle=None` so realize/lead_aim
        computes the right angle for current geometry).
      - `surviving_pending` — entries with `wait_remaining > 0` after
        the decrement.
      - `pending_srcs` — set of src_ids in surviving_pending PLUS those
        firing this turn (so defense/expand don't double-emit).

    Drop semantics: same as `agents/baseline/main.py::_tick_ledger`:
      - src no longer ours, tgt now ours, src empty, ships ≤ 0 → drop.
      - Otherwise emit Intent with new aim computed via realize pipeline.
    """
    pending = _PENDING_LAUNCHES.get(int(me), [])
    if not pending:
        return [], [], set()
    due: list[Intent] = []
    survivors: list[dict] = []
    pending_srcs: set[int] = set()
    for entry in pending:
        entry["wait_remaining"] = int(entry["wait_remaining"]) - 1
        if entry["wait_remaining"] > 0:
            survivors.append(entry)
            pending_srcs.add(int(entry["src_id"]))
            continue
        # Fire-now validation.
        sid = int(entry["src_id"])
        tid = int(entry["tgt_id"])
        src = planets_by_id.get(sid)
        tgt = planets_by_id.get(tid)
        if src is None or tgt is None:
            if DEBUG: print(f"[momentum] ledger drop: planet_missing src={sid} tgt={tid}", flush=True)
            continue
        if int(src.owner) != int(me):
            if DEBUG: print(f"[momentum] ledger drop: src_lost src={sid}", flush=True)
            continue
        if int(tgt.owner) == int(me):
            if DEBUG: print(f"[momentum] ledger drop: tgt_now_ours src={sid} tgt={tid}", flush=True)
            continue
        available = int(src.ships)
        if available <= 0:
            if DEBUG: print(f"[momentum] ledger drop: src_empty src={sid}", flush=True)
            continue
        ships = min(int(entry["ships_planned"]), available)
        if ships <= 0:
            continue
        # Emit as an Intent; realize+DEFAULT_MECHANISMS computes the aim.
        due.append(Intent(
            src_id=sid, target_id=tid, ships=int(ships),
            note=f"salvo_fire:committed_at={entry.get('t_commit', '?')}",
        ))
        pending_srcs.add(sid)
        if DEBUG:
            print(f"[momentum] ledger fire: src={sid} tgt={tid} ships={ships}", flush=True)
    return due, survivors, pending_srcs


def agent(obs, configuration=None):
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    step = int(obs_d.get("step", 0))

    raw_planets = obs_d.get("planets", []) or []
    if not raw_planets:
        return []

    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if int(p.owner) == me]
    if not my_planets:
        return []
    enemy_planets = [p for p in planets if int(p.owner) != me and int(p.owner) >= 0]
    neutrals = [p for p in planets if int(p.owner) == -1]
    planets_by_id = {int(p.id): p for p in planets}

    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0))

    # 1. Clear ledger on new game.
    if step == 0:
        _PENDING_LAUNCHES.pop(me, None)

    # 2. Tick ledger first. Sources with pending/firing commits are
    #    reserved against defense and expand emissions.
    due_intents, surviving_pending, pending_srcs = _tick_ledger(me, planets_by_id)
    _PENDING_LAUNCHES[me] = surviving_pending

    intents: list[Intent] = list(due_intents)
    used: set[int] = set(pending_srcs)

    # 3. Defense — reinforce planets predicted to flip.
    intents.extend(propose_defense(my_planets, world, model, me, used))

    # 4. Phase trigger — STRIKE iff we're sufficiently ahead.
    phase = _detect_phase(my_planets, enemy_planets)

    # 5. Salvo (STRIKE phase only). Emit wait==0 now; register wait>0
    #    in the ledger for cross-turn firing.
    if phase == "STRIKE" and enemy_planets:
        salvo_intents, wait_Ns = propose_salvo(
            my_planets, enemy_planets, world, model, me, omega,
            used, pending_srcs,
        )
        for it, wait_N in zip(salvo_intents, wait_Ns):
            if wait_N <= 0:
                intents.append(it)
                used.add(int(it.src_id))
                if DEBUG:
                    print(f"[momentum] salvo fire-now: src={it.src_id} tgt={it.target_id} ships={it.ships}", flush=True)
            else:
                _PENDING_LAUNCHES[me].append({
                    "src_id": int(it.src_id),
                    "tgt_id": int(it.target_id),
                    "ships_planned": int(it.ships),
                    "wait_remaining": int(wait_N),
                    "t_commit": step,
                })
                used.add(int(it.src_id))
                if DEBUG:
                    print(f"[momentum] salvo register: src={it.src_id} tgt={it.target_id} ships={it.ships} wait={wait_N}", flush=True)

    # 6. Expand the rest — production-first + ENEMY_MULTIPLIER if behind.
    intents.extend(propose_expand(
        my_planets, neutrals, enemy_planets, world, model, me, used,
    ))

    if DEBUG and (step % 20 == 0 or len(intents) > 0):
        n_def = sum(1 for i in intents if "defense" in i.note)
        n_exp = sum(1 for i in intents if i.note == "expand")
        n_fire = sum(1 for i in intents if i.note.startswith("salvo_fire"))
        n_salvo_now = sum(1 for i in intents if i.note.startswith("salvo:"))
        n_pending = len(_PENDING_LAUNCHES.get(me, []))
        print(f"[momentum] step={step} phase={phase} my_p={len(my_planets)} "
              f"enemies={len(enemy_planets)} intents={len(intents)} "
              f"(def={n_def} exp={n_exp} salvo_now={n_salvo_now} ledger_fire={n_fire} pending={n_pending})",
              flush=True)

    return realize(intents, obs_d, mechanisms=DEFAULT_MECHANISMS, model=model)
