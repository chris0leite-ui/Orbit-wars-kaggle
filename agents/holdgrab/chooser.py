"""Greedy per-turn chooser — selects capture intents.

Per turn:
  A. Reserve a defensive floor on every one of my planets (committed in-flight
     threats) -> spendable budget per source.
  B. Enumerate (source -> nearest targets) capture candidates; for each, size
     the fleet (Lanchester capture-and-hold against the enemy's bounded full
     reachable counter) and GATE by affordability; score by the
     production-time-integral value.
  C. Greedily commit highest-value candidates under per-source budget +
     one-capture-per-target dedup.

The chooser emits abstract ``Intent`` objects and lets the shared mechanism
layer (``lib.mechanism.DEFAULT_MECHANISMS`` via ``realize`` in main) turn them
into physically-valid actions — lead-aim, production-aware sizing, sun-avoid,
path-detour, OOB-guard. That is the same "obvious-rule wins" stack the simple
baselines use; hand-rolling aim + raw collision-rejection instead (an earlier
holdgrab draft) silently dropped ~90% of shots on a dense board.

The pass is deterministic and always returns a legal (possibly empty) intent
list. Recapture, reinforcement-by-not-draining, and adaptation all emerge from
re-running this every turn.
"""

from __future__ import annotations

import math
from collections import defaultdict

from lib.fleet import eta_turns
from lib.intent import Intent

from agents.holdgrab.sizing import ships_to_capture, ships_to_capture_and_hold
from agents.holdgrab.threat import contest_force, defense_follow_on, opp_reach_tick
from agents.holdgrab.value import planet_value


def _nearest_targets(src, targets, n):
    if len(targets) <= n:
        return list(targets)
    sx, sy = float(src.x), float(src.y)
    return sorted(
        targets,
        key=lambda t: (float(t.x) - sx) ** 2 + (float(t.y) - sy) ** 2,
    )[:n]


def _seed_eta(src, tgt) -> int:
    """Cheap straight-line ETA seed (the mechanism layer does the real
    lead-aim). Sized for a minimum-cover fleet; good enough to read
    garrison-at-arrival and value off the timeline."""
    return max(1, eta_turns(
        (float(src.x), float(src.y)), (float(tgt.x), float(tgt.y)),
        int(tgt.ships) + 1,
    ))


def _budget(view, cfg) -> dict:
    """Spendable ships per source after reserving each planet's defensive
    floor (committed in-flight enemy fleets)."""
    out = {}
    for s in view.my_sources:
        floor = defense_follow_on(view, s, cfg.defense_floor_horizon)
        out[int(s.id)] = max(0, int(s.ships) - int(math.ceil(floor)))
    return out


def select(view, cfg) -> list:
    """Return the committed list of ``Intent`` for this turn."""
    me = view.me
    model = view.model
    budget = _budget(view, cfg)

    intents: list = []
    candidates = []  # (value, eta, src_id, tgt_id, ships)
    for src in view.my_sources:
        b = budget[int(src.id)]
        if b <= 0:
            continue
        for tgt in _nearest_targets(src, view.targets, cfg.max_targets_per_source):
            if int(src.id) == int(tgt.id):
                continue
            eta = _seed_eta(src, tgt)
            if eta > cfg.max_arrival_lead:
                continue

            owner_arr = model.owner_at(int(tgt.id), eta)
            owner_arr = int(tgt.owner) if owner_arr is None else int(owner_arr)
            if owner_arr == me:
                continue  # already ours by arrival; recapture re-emerges next turn
            garrison = model.ships_at(int(tgt.id), eta)
            garrison = float(tgt.ships) if garrison is None else float(garrison)

            remaining = max(0, cfg.game_horizon - view.step - eta)
            if remaining <= 0:
                continue

            # --- SIZING + TWO-TIER VALUE ---
            # Tier 1 (HOLD): bring enough to beat the enemy's full reachable
            # counter -> we hold it for the rest of the game (full integral).
            # Tier 2 (PRESSURE): can't afford that, but can afford the capture
            # (G+1) -> spend surplus on denial + a short hold instead of
            # hoarding idle ships. Prefer Tier 1; fall back to Tier 2.
            f_contest = contest_force(view, tgt, eta, cfg.contest_window)
            need_hold = ships_to_capture_and_hold(
                garrison, f_contest, float(tgt.production), cfg.contest_window,
            )
            need_cap = ships_to_capture(garrison)
            if b >= need_hold:
                ships = need_hold
                hold_ticks = remaining
            elif b >= need_cap:
                ships = need_cap
                opp = opp_reach_tick(view, int(tgt.id), eta)
                if opp is None:
                    hold_ticks = remaining
                else:
                    hold_ticks = max(1, min(remaining, int(opp) - eta))
            else:
                continue  # can't even capture

            v = planet_value(view, tgt, hold_ticks, cfg)
            if v <= 0:
                continue
            candidates.append((float(v), int(eta), int(src.id), int(tgt.id), int(ships)))

    # Deterministic order: value desc, then earliest arrival, then ids.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))

    spent: dict[int, int] = defaultdict(int)
    taken: set[int] = set()
    for (_v, _eta, src_id, tgt_id, ships) in candidates:
        if tgt_id in taken:
            continue
        if spent[src_id] + ships > budget[src_id]:
            continue
        intents.append(Intent(src_id=src_id, target_id=tgt_id, ships=ships))
        spent[src_id] += ships
        taken.add(tgt_id)

    return intents
