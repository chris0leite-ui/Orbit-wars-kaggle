"""Unified per-turn chooser — one action space, one value, one greedy.

Every move is the same primitive (launch K ships A->B) scored by the same
objective (production banked over the game). There is no offense/defense split:

  - CAPTURE (target not mine)  -> value = production *gained* over the hold.
  - REINFORCE (my planet the timeline says will FALL) -> value = production
    *preserved* from the fall to game-end.

Both go into one candidate list, ordered by value-per-ship, and committed by
one greedy. Defense is not a subsystem — it is just the candidates whose value
is production preserved, plus one piece of correct cost-accounting: a planet
holding its ships in place is the implicit baseline a launch must out-value, so
a source keeps enough to survive its own committed in-flight threat before
spending the rest (``_spendable``).

The chooser emits abstract ``Intent`` objects; the shared mechanism layer
(``DEFAULT_MECHANISMS`` via ``realize`` in main) turns them into physically-
valid actions (lead-aim, sun-avoid, path-detour, OOB-guard). Reinforce intents
(target.owner == me) pass through ``arrival_size`` unchanged.

Deterministic; always returns a legal (possibly empty) list. Recapture and
adaptation emerge from re-running every turn.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from lib.fleet import eta_turns
from lib.intent import Intent

from agents.holdgrab.sizing import ships_to_capture, ships_to_capture_and_hold
from agents.holdgrab.threat import contest_force, opp_reach_tick, peak_inflight_threat
from agents.holdgrab.value import planet_value, preserve_value


@dataclass
class Cand:
    value: float
    ships: int
    eta: int
    src_id: int
    tgt_id: int
    kind: str   # "capture" | "reinforce"


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
    lead-aim); enough to read garrison-at-arrival and value off the timeline."""
    return max(1, eta_turns(
        (float(src.x), float(src.y)), (float(tgt.x), float(tgt.y)),
        int(tgt.ships) + 1,
    ))


def _spendable(view, cfg) -> dict:
    """Ships each source can spend after its own self-defense first-claim: a
    planet keeps enough to survive its strongest committed in-flight threat
    (holding-in-place is the baseline a launch must beat)."""
    out = {}
    for s in view.my_sources:
        force, _eta = peak_inflight_threat(view, s, cfg.defense_floor_horizon)
        out[int(s.id)] = max(0, int(s.ships) - int(math.ceil(force)))
    return out


def _capture_candidates(view, cfg, spendable) -> list:
    me = view.me
    model = view.model
    out = []
    for src in view.my_sources:
        b = spendable[int(src.id)]
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
                continue
            garrison = model.ships_at(int(tgt.id), eta)
            garrison = float(tgt.ships) if garrison is None else float(garrison)

            remaining = max(0, cfg.game_horizon - view.step - eta)
            if remaining <= 0:
                continue

            # Tier 1 (HOLD) preferred; Tier 2 (PRESSURE) spends surplus instead
            # of hoarding when full hold is unaffordable.
            f_contest = contest_force(view, tgt, eta, cfg.contest_window)
            need_hold = ships_to_capture_and_hold(
                garrison, f_contest, float(tgt.production), cfg.contest_window,
            )
            need_cap = ships_to_capture(garrison)
            opp = opp_reach_tick(view, int(tgt.id), eta)
            if b >= need_hold:
                ships = need_hold
                hold_ticks = remaining
            elif b >= need_cap:
                ships = need_cap
                hold_ticks = remaining if opp is None else max(1, min(remaining, int(opp) - eta))
            else:
                continue

            v = planet_value(view, tgt, hold_ticks, eta, opp, cfg)
            if v <= 0:
                continue
            out.append(Cand(float(v), int(ships), int(eta), int(src.id), int(tgt.id), "capture"))
    return out


def _reinforce_candidates(view, cfg, spendable):
    """For each of my planets the timeline says will FALL, a reinforcement
    from every donor that can arrive before the fall. Returns (candidates,
    need) where need[B] is the shortfall to cap reinforcement at."""
    out = []
    need: dict[int, int] = {}
    for B in view.my_planets:
        force, fall_eta = peak_inflight_threat(view, B, cfg.defense_floor_horizon)
        shortfall = int(math.ceil(force)) - int(B.ships)
        if shortfall < 1 or fall_eta <= 0:
            continue
        val = preserve_value(view, B, fall_eta, cfg)
        if val <= 0:
            continue
        need[int(B.id)] = shortfall
        for A in view.my_sources:
            if int(A.id) == int(B.id) or spendable[int(A.id)] <= 0:
                continue
            eta_ab = eta_turns((float(A.x), float(A.y)), (float(B.x), float(B.y)), shortfall)
            if eta_ab >= fall_eta:
                continue  # too slow to save it
            ships = min(shortfall, spendable[int(A.id)])
            if ships < 1:
                continue
            out.append(Cand(float(val), int(ships), int(eta_ab), int(A.id), int(B.id), "reinforce"))
    return out, need


def select(view, cfg) -> list:
    """Return the committed list of ``Intent`` for this turn."""
    spendable = _spendable(view, cfg)

    candidates = _capture_candidates(view, cfg, spendable)
    reinforce, need = _reinforce_candidates(view, cfg, spendable)
    candidates.extend(reinforce)

    # One ordering for both kinds. value-per-ship (ROI) is knapsack-correct
    # under the per-source budget; deterministic tiebreak.
    if cfg.order_by_roi:
        candidates.sort(key=lambda c: (-(c.value / max(1, c.ships)), c.eta, c.src_id, c.tgt_id))
    else:
        candidates.sort(key=lambda c: (-c.value, c.eta, c.src_id, c.tgt_id))

    spent: dict[int, int] = defaultdict(int)
    taken: set[int] = set()                 # captured-this-turn targets (dedup)
    reinforced: dict[int, int] = defaultdict(int)
    intents: list = []
    for c in candidates:
        room = spendable[c.src_id] - spent[c.src_id]
        if room <= 0:
            continue
        if c.kind == "capture":
            if c.tgt_id in taken or c.ships > room:
                continue
            intents.append(Intent(src_id=c.src_id, target_id=c.tgt_id, ships=c.ships))
            spent[c.src_id] += c.ships
            taken.add(c.tgt_id)
        else:  # reinforce: cap total help to a planet at its shortfall
            still = need[c.tgt_id] - reinforced[c.tgt_id]
            give = min(c.ships, still, room)
            if give < 1:
                continue
            intents.append(Intent(src_id=c.src_id, target_id=c.tgt_id, ships=int(give)))
            spent[c.src_id] += int(give)
            reinforced[c.tgt_id] += int(give)

    return intents
