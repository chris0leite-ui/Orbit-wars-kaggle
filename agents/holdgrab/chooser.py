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


@dataclass
class Coalition:
    """A mass-to-HOLD candidate: two+ sources pool budget to capture AND HOLD a
    target neither can hold alone. ``legs`` is ``[(src_id, ships), ...]`` sized
    to land jointly at ``common_eta`` and clear the hold threshold."""
    tgt_id: int
    common_eta: int
    legs: list          # [(src_id, ships), ...]
    value: float


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


# ---------------------------------------------------------------------------
# Mass-to-HOLD consolidation (Plan v5) — shared enumerator used by both the
# STEP-1 census probe and (when GO) the STEP-3 proposer. Reuses the EXACT
# sizing primitives _capture_candidates uses, so a coalition's hold threshold
# matches the per-source proposer's gate.
# ---------------------------------------------------------------------------


def _hold_sizing_at(view, cfg, tgt, eta):
    """Mirror _capture_candidates' per-(src,tgt) sizing, parameterised on an
    explicit arrival ``eta`` (the same garrison/contest/opp_reach reads). Returns
    a dict, or ``None`` if the launch is invalid at that eta (lands on us, too
    far, or game over)."""
    me = view.me
    model = view.model
    if eta > cfg.max_arrival_lead:
        return None
    owner_arr = model.owner_at(int(tgt.id), eta)
    owner_arr = int(tgt.owner) if owner_arr is None else int(owner_arr)
    if owner_arr == me:
        return None
    garrison = model.ships_at(int(tgt.id), eta)
    garrison = float(tgt.ships) if garrison is None else float(garrison)
    remaining = max(0, cfg.game_horizon - view.step - eta)
    if remaining <= 0:
        return None
    f_contest = contest_force(view, tgt, eta, cfg.contest_window)
    need_hold = ships_to_capture_and_hold(
        garrison, f_contest, float(tgt.production), cfg.contest_window,
    )
    need_cap = ships_to_capture(garrison)
    opp = opp_reach_tick(view, int(tgt.id), eta)
    return {
        "eta": int(eta), "garrison": float(garrison), "need_hold": int(need_hold),
        "need_cap": int(need_cap), "opp": opp, "remaining": int(remaining),
    }


def _size_legs(legs_src, need_hold_total, spendable):
    """Split ``need_hold_total`` ships across the chosen sources, proportional to
    each source's spendable budget, each leg capped by its spendable. Deterministic
    (largest-remainder by src_id). Returns ``[(src_id, ships), ...]`` or ``None``
    if the legs can't jointly cover the threshold."""
    caps = [(sid, int(spendable[sid])) for sid, _hc in legs_src]
    total_cap = sum(c for _s, c in caps)
    if total_cap < need_hold_total:
        return None
    legs = []
    assigned = 0
    for sid, cap in caps:
        share = int(need_hold_total * cap / total_cap) if total_cap > 0 else 0
        share = min(share, cap)
        legs.append([sid, share])
        assigned += share
    # distribute the rounding shortfall onto sources with remaining headroom
    i = 0
    order = sorted(range(len(legs)), key=lambda j: legs[j][0])
    while assigned < need_hold_total and i < len(order) * 4:
        j = order[i % len(order)]
        sid, cap = caps[j]
        if legs[j][1] < cap:
            legs[j][1] += 1
            assigned += 1
        i += 1
    legs = [(sid, ships) for sid, ships in legs if ships > 0]
    if len(legs) < 2 or assigned < need_hold_total:
        return None
    return legs


def consolidation_opportunities(view, cfg, spendable) -> list:
    """Enumerate mass-to-HOLD coalitions: high-value ENEMY targets where NO single
    source can HOLD (some can only PRESSURE), but the nearest 2..max_legs sources'
    pooled budget clears the hold threshold at a synchronised arrival. Returns the
    top-``consolidate_max_targets`` by ``planet_value``. Deterministic."""
    me = view.me
    targets = []
    for tgt in view.targets:
        owner = int(tgt.owner)
        if owner != me and owner != -1:
            targets.append(tgt)               # enemy: double value
        elif cfg.consolidate_neutral and owner == -1:
            targets.append(tgt)               # contested neutral (opt-in)

    scored = []
    for tgt in targets:
        # per-source single-source sizing (at each source's own straight-line eta)
        infos = {}
        for s in view.my_sources:
            sid = int(s.id)
            if spendable[sid] <= 0 or sid == int(tgt.id):
                continue
            hc = _hold_sizing_at(view, cfg, tgt, _seed_eta(s, tgt))
            if hc is not None:
                infos[sid] = hc
        if len(infos) < 2:
            continue
        # (2) no single source can solo-HOLD (else the per-source proposer has it)
        if any(spendable[sid] >= hc["need_hold"] for sid, hc in infos.items()):
            continue
        # (3) at least one source can solo-CAPTURE today (drops to Tier-2 PRESSURE)
        if not any(spendable[sid] >= hc["need_cap"] for sid, hc in infos.items()):
            continue
        # choose the minimal coalition (nearest legs) that clears HOLD in sync
        ranked = sorted(infos.items(), key=lambda kv: (kv[1]["eta"], kv[0]))
        coalition = None
        max_k = min(cfg.consolidate_max_legs, len(ranked))
        for k in range(2, max_k + 1):
            legs_src = ranked[:k]
            etas = [hc["eta"] for _sid, hc in legs_src]
            common_eta = max(etas)
            if common_eta - min(etas) > cfg.consolidate_max_eta_gap:
                continue
            nh = _hold_sizing_at(view, cfg, tgt, common_eta)
            if nh is None:
                continue
            if sum(spendable[sid] for sid, _hc in legs_src) >= nh["need_hold"]:
                coalition = (common_eta, legs_src, nh)
                break
        if coalition is None:
            continue
        common_eta, legs_src, nh = coalition
        value = planet_value(view, tgt, nh["remaining"], common_eta, nh["opp"], cfg)
        if value <= 0:
            continue
        scored.append((float(value), int(tgt.id), common_eta, legs_src, nh))

    scored.sort(key=lambda x: (-x[0], x[1]))
    out = []
    for value, tgt_id, common_eta, legs_src, nh in scored[: cfg.consolidate_max_targets]:
        legs = _size_legs(legs_src, nh["need_hold"], spendable)
        if legs is None:
            continue
        out.append(Coalition(tgt_id=tgt_id, common_eta=int(common_eta),
                             legs=legs, value=float(value)))
    return out
