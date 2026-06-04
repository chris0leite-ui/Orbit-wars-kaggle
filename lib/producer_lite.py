"""Producer-lite — a faithful pure-python port of the public "Producer"
torch agent's *attack* policy, for use as a rollout opponent model.

Why this exists
---------------
Our chooser rolls every opponent seat forward with a weak opponent model
(`lib.opp_model.lite_greedy_policy`). The vendored public **Producer** torch
agent beats our line ~60%, at the same rate against both the champion and the
refine chooser — a chooser-independent blind spot: the search never anticipates
a Producer-class aggressive attacker, so it under-defends and mis-calibrates
capture-safety. This module ports Producer's attack phase into a cheap
pure-python `Policy` (`Callable[[obs], list[[src_id, angle, ships]]]`) so the
chooser can roll out against a Producer-like aggressor (gated `BASELINE_OPP_TIER=2`,
default OFF → champion bundle byte-identical).

Fidelity model
--------------
Producer's torch pipeline (agents/producer/) is, per turn:
  1. build a do-nothing garrison projection over H turns, resolving in-flight
     fleets (`movement.garrison_status`);
  2. `safe_drain(s)` — max ships a source can shed while staying held over H
     (`planner_core.py:587`);
  3. shortlist sources (top by ships) + offensive targets (top by proximity) +
     defensive targets (my planets that flip within H, by urgency);
  4. one candidate per (source,target), size = safe_drain(s);
  5. `capture_floor` = ceil(projected_defenders_at_arrival + 1.0) (`:186`);
  6. `competitive_score = Δnet_me − Σ Δnet_opp` via an exact sparse flow delta
     (`:75`);
  7. greedy top-6 waves, one per target, role-mutex, fire iff score > 1.5
     (`_greedy_select`, `:363`).

This port keeps 1–4, 7 faithful and **compacts** the exact flow delta (6) into a
production-integral proxy:
    score(s→t) = prod_t · (H − eta) · flip_mult − defenders_cleared
This is THE main fidelity risk (it drops cascade / combat-timing); the
winrate-transfer acceptance gate measures it. Defensive (reinforcement) targets
are scored by flip-urgency instead, mirroring `friendly_flip_targets` (`:244`).
Producer's regroup / pressure-gradient *defense* phase is intentionally skipped
in v1 (the opponent model predicts attacks; add later only if defense-prediction
proves to matter).

Attribution
-----------
Re-implemented (not copied) from the public Producer Kaggle notebook; see
`agents/producer/PROVENANCE.md`. Licensing: code-reuse-with-attribution cleared
(2026-06-04, competition rule §2.6 external public tools). The torch source under
`agents/producer/` is the read-only oracle this port mirrors.
"""

from __future__ import annotations

import math
from typing import Any

from lib.aim import aim_orbiting, estimate_eta
from lib.fleet import speed as fleet_speed
from lib.orbit import is_orbiting

# --- Producer config (main.py:53-76 for 2P; :305-312 overrides for 4P) -------
# 2-player defaults.
_H_2P = 18
_MAX_SOURCES_2P = 12
_MAX_DEF_TARGETS_2P = 4
# 4-player (FFA) overrides.
_H_4P = 13
_MAX_SOURCES_4P = 6
_MAX_DEF_TARGETS_4P = 2
# Shared across player counts.
_MAX_OFFENSIVE_TARGETS = 12
_MAX_WAVES_PER_TURN = 6
_ROI_THRESHOLD = 1.5
_MIN_SHIPS_TO_LAUNCH = 4.0
_CAPTURE_OVERHEAD = 1.0


def _get(obs: Any, key: str, default: Any = None) -> Any:
    """Dual dict-or-attr obs read (mirrors lib.opp_model / fast_sim)."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _player_count(obs: Any, planets: list) -> int:
    """2 or 4 — distinct player owners in initial_planets, else current board."""
    initial = _get(obs, "initial_planets", None)
    owners: set[int] = set()
    src = initial if initial else planets
    for p in src:
        o = int(p[1])
        if o >= 0:
            owners.add(o)
    return 4 if len(owners) >= 4 else 2


def _build_projection(planets, fleets_raw, omega, H, me):
    """Do-nothing forward projection of every planet over turns 0..H,
    resolving in-flight fleet arrivals — the pure-python analog of
    Producer's `movement.garrison_status` (the substrate `safe_drain` and
    `capture_floor` read from).

    Returns `(owner, ships, flip_turn)`:
      - `owner[i][t]`, `ships[i][t]` for planet index i at turn t in [0, H].
      - `flip_turn[i]` = earliest turn in [1, H] where a *currently-mine*
        planet stops being mine (else None).

    Production accrues for player-owned planets only (neutrals are constant —
    env rule, orbit_wars.py:511-514). Combat at each arrival turn is resolved
    sequentially (friendly arrivals reinforce; the strongest hostile arrival
    captures if it exceeds the garrison) — a faithful-enough compaction of the
    env's exact multi-player combat for an opponent model.
    """
    P = len(planets)
    owner = [[0] * (H + 1) for _ in range(P)]
    ships = [[0.0] * (H + 1) for _ in range(P)]
    prod = [float(p[6]) for p in planets]
    is_mine = [int(p[1]) == me for p in planets]
    # Cache planet centres/radii once (tuple indexing in tight loops is slow).
    px = [float(p[2]) for p in planets]
    py = [float(p[3]) for p in planets]
    pr = [float(p[4]) for p in planets]
    for i, p in enumerate(planets):
        owner[i][0] = int(p[1])
        ships[i][0] = float(p[5])

    # Attribute each in-flight fleet to (target planet, arrival turn) by a cheap
    # closed-form straight-line ray-cast — the plan's sanctioned fallback.
    # world_model.fleet_target_planet does a per-tick orbital scan to
    # DEFAULT_HORIZON=250 (~2M predict_relative calls/board on dense boards,
    # 17ms/call); orbital drift over the few turns to arrival is negligible for
    # an opponent threat estimate, and exact aim is re-applied on fired waves.
    # arrivals[t] -> {planet_idx: [(fleet_owner, fleet_ships), ...]}.
    arrivals: list[dict[int, list[tuple[int, float]]]] = [dict() for _ in range(H + 1)]
    for f in fleets_raw:
        f_owner = int(f[1])
        fx = float(f[2]); fy = float(f[3])
        f_ships = float(f[6])
        spd = fleet_speed(f_ships)
        if spd <= 0:
            continue
        dir_x = math.cos(float(f[4]))
        dir_y = math.sin(float(f[4]))
        best_turns = None
        best_i = -1
        for i in range(P):
            dx = px[i] - fx
            dy = py[i] - fy
            proj = dx * dir_x + dy * dir_y
            if proj < 0.0:
                continue  # planet is behind the fleet
            r = pr[i]
            perp_sq = dx * dx + dy * dy - proj * proj
            if perp_sq >= r * r:
                continue  # ray misses the planet disc
            hit_d = proj - math.sqrt(r * r - perp_sq)
            if hit_d < 0.0:
                hit_d = 0.0
            turns = hit_d / spd
            if turns <= H and (best_turns is None or turns < best_turns):
                best_turns = turns
                best_i = i
        if best_i < 0:
            continue
        k = max(1, min(H, int(math.ceil(best_turns))))
        arrivals[k].setdefault(best_i, []).append((f_owner, f_ships))

    flip_turn: list[int | None] = [None] * P
    for t in range(1, H + 1):
        arr_t = arrivals[t]
        for i in range(P):
            o = owner[i][t - 1]
            s = ships[i][t - 1]
            if o >= 0:  # player-owned planets produce; neutrals (-1) do not
                s += prod[i]
            inc = arr_t.get(i)
            if inc:
                for (fo, fs) in inc:
                    if fo == o:
                        s += fs  # reinforce
                    elif fs > s:
                        o = fo
                        s = fs - s  # captured
                    else:
                        s -= fs  # repelled
            owner[i][t] = o
            ships[i][t] = s
            if flip_turn[i] is None and is_mine[i] and o != me:
                flip_turn[i] = t
    return owner, ships, flip_turn


def _safe_drain(idx, owner, ships, H, me, source_ships):
    """Max ships source `idx` can shed while staying held over [1, H]
    (planner_core.py:587). = min over held turns of projected ships, capped
    by current ships, floored at 0. A doomed source (never held) → send all.
    """
    min_slack = math.inf
    for t in range(1, H + 1):
        if owner[idx][t] == me and ships[idx][t] > 0.0:
            if ships[idx][t] < min_slack:
                min_slack = ships[idx][t]
    drain = min(min_slack, source_ships)
    return max(0.0, drain)


def producer_lite_policy(obs: Any) -> list:
    """Producer-lite attack policy. Returns `[[src_id, angle, ships], ...]`."""
    me = int(_get(obs, "player", 0) or 0)
    planets = _get(obs, "planets", None)
    if not planets:
        return []
    omega = float(_get(obs, "angular_velocity", 0.0) or 0.0)
    fleets_raw = _get(obs, "fleets", []) or []
    comet_ids = set(int(c) for c in (_get(obs, "comet_planet_ids", []) or []) if int(c) >= 0)

    player_count = _player_count(obs, planets)
    H = _H_4P if player_count >= 4 else _H_2P
    max_sources = _MAX_SOURCES_4P if player_count >= 4 else _MAX_SOURCES_2P
    max_def = _MAX_DEF_TARGETS_4P if player_count >= 4 else _MAX_DEF_TARGETS_2P

    P = len(planets)
    owner, ships_proj, flip_turn = _build_projection(planets, fleets_raw, omega, H, me)

    # --- shortlist sources: owned, ships >= min, top-N by current ships ------
    src_cands = [
        i for i in range(P)
        if int(planets[i][1]) == me and float(planets[i][5]) >= _MIN_SHIPS_TO_LAUNCH
    ]
    if not src_cands:
        return []
    src_cands.sort(key=lambda i: -float(planets[i][5]))
    src_cands = src_cands[:max_sources]

    # safe_drain per source (the single fleet size Producer uses).
    drain = {i: math.floor(_safe_drain(i, owner, ships_proj, H, me, float(planets[i][5])))
             for i in src_cands}

    # --- shortlist offensive targets: enemy|neutral, non-comet, top by -------
    # min straight-line distance to any source ------------------------------
    off_cands = []
    for j in range(P):
        oj = int(planets[j][1])
        if oj == me:
            continue
        if int(planets[j][0]) in comet_ids:
            continue
        # nearest source distance (centre-to-centre is fine for ranking)
        best_d = math.inf
        jx, jy = float(planets[j][2]), float(planets[j][3])
        for i in src_cands:
            dx = jx - float(planets[i][2])
            dy = jy - float(planets[i][3])
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d
        off_cands.append((best_d, j))
    off_cands.sort(key=lambda kv: kv[0])
    offensive = [j for _d, j in off_cands[:_MAX_OFFENSIVE_TARGETS]]

    # --- shortlist defensive targets: my planets flipping within H, by ------
    # urgency = prod*(H - flip_turn) + ships_now -----------------------------
    def_cands = []
    for i in range(P):
        if int(planets[i][1]) != me or flip_turn[i] is None:
            continue
        urgency = float(planets[i][6]) * (H - flip_turn[i]) + float(planets[i][5])
        def_cands.append((urgency, i))
    def_cands.sort(key=lambda kv: -kv[0])
    defensive = [i for _u, i in def_cands[:max_def]]

    targets = offensive + defensive
    if not targets:
        return []

    # --- one candidate per (source, target); score; greedy select -----------
    # candidate tuple: (score, src_idx, tgt_idx, size, eta, is_reinforce)
    cands: list[tuple] = []
    for i in src_cands:
        size = drain[i]
        if size < 1.0:
            continue
        sx, sy, sr = float(planets[i][2]), float(planets[i][3]), float(planets[i][4])
        for j in targets:
            if i == j:
                continue
            tx, ty, tr = float(planets[j][2]), float(planets[j][3]), float(planets[j][4])
            eta = estimate_eta((sx, sy), sr, (tx, ty), tr, size)
            if eta is None or eta > H:
                continue
            k = max(1, min(H, int(math.ceil(eta))))
            owner_at_arr = owner[j][k]
            defenders_at_arr = max(0.0, ships_proj[j][k])
            cur_owner = int(planets[j][1])
            is_reinforce = (owner_at_arr == me)
            # capture floor (planner_core.py:186): owned-at-arrival → 1.
            if is_reinforce:
                floor = 1.0
            else:
                floor = math.ceil(defenders_at_arr + _CAPTURE_OVERHEAD)
            if size < floor:
                continue
            remaining = max(0.0, H - eta)
            if cur_owner == me:
                # defensive reinforcement — score by flip-urgency (the swing
                # prevented), mirroring friendly_flip_targets.
                ft = flip_turn[j] if flip_turn[j] is not None else H
                score = float(planets[j][6]) * (H - ft) + float(planets[j][5])
            else:
                # offensive — production-integral proxy for competitive_score.
                flip_mult = 2.0 if cur_owner >= 0 else 1.0  # enemy swing vs neutral
                defenders_cleared = 0.0 if is_reinforce else defenders_at_arr
                score = float(planets[j][6]) * remaining * flip_mult - defenders_cleared
            cands.append((score, i, j, size, eta, is_reinforce))

    if not cands:
        return []
    cands.sort(key=lambda c: -c[0])

    # --- greedy top-6, one wave per target, role-mutex, fire iff score>1.5 ---
    budget = {i: float(planets[i][5]) for i in src_cands}
    target_taken: set[int] = set()
    used_src: set[int] = set()
    defended: set[int] = set()
    fired: list[tuple] = []  # (src_idx, tgt_idx, size)
    for (score, i, j, size, eta, is_reinforce) in cands:
        if len(fired) >= _MAX_WAVES_PER_TURN:
            break
        if score <= _ROI_THRESHOLD:
            break  # candidates are sorted desc → nothing below fires
        if j in target_taken:
            continue
        if budget.get(i, 0.0) < size:
            continue
        # role mutex: a reinforced planet can't be a source; a used source
        # can't be a target.
        if i in defended:
            continue
        if j in used_src:
            continue
        budget[i] -= size
        target_taken.add(j)
        used_src.add(i)
        if is_reinforce or int(planets[j][1]) == me:
            defended.add(j)
        fired.append((i, j, size))

    if not fired:
        return []

    # --- exact aim on the <=6 fired waves only ------------------------------
    moves: list = []
    for (i, j, size) in fired:
        sx, sy, sr = float(planets[i][2]), float(planets[i][3]), float(planets[i][4])
        tx, ty, tr = float(planets[j][2]), float(planets[j][3]), float(planets[j][4])
        ships_int = int(size)
        if ships_int < 1:
            continue
        # Orbital lead ONLY for actually-orbiting targets. Static planets
        # (orbital_radius + radius >= ROTATION_RADIUS_LIMIT) do not rotate;
        # leading them via aim_orbiting/predict_relative aims at a bogus
        # future position and routes the fleet into a wall planet (this was
        # the opening-capture failure: a static target got over-led 0.5 rad).
        # Mirrors lib.opp_model.me_defensive_action.
        target_tuple = (
            int(planets[j][0]), int(planets[j][1]), tx, ty, tr,
            int(planets[j][5]), int(planets[j][6]),
        )
        angle = None
        if omega != 0.0 and is_orbiting(target_tuple):
            aim = aim_orbiting((sx, sy), sr, target_tuple, tr, ships_int, omega)
            if aim is not None:
                angle = float(aim[0])
        if angle is None:
            angle = math.atan2(ty - sy, tx - sx)
        moves.append([int(planets[i][0]), float(angle), ships_int])
    return moves
