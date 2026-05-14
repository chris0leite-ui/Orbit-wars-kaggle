"""geo_all — geo v3.1 + drift discount + recapture wired + garrison.

Combined variant: all three single-axis fixes turned on. Run alongside
the single-axis variants to detect additive vs interaction effects.
Defaults match the single-axis variants exactly.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.mission import Mission
from lib.missions.opening import propose_opening_missions
from lib.missions.recapture import propose_recapture_missions
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.orbit import predict_relative, is_orbiting
from lib.world_model import WorldModel

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "geo_base_for_all", _REPO / "agents" / "geo" / "main.py",
)
_geo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geo)

# ---- shared tunables (match single-axis variants) ----
HOLD_HORIZON = 25
SAMPLE_STEPS = 5
MIN_DISCOUNT = 0.2
GARRISON_TARGET = 12
GARRISON_WINDOW = 8
MIN_SOURCE_GARRISON = 6
EPISODE_STEPS = 500


# =========================================================================
# Drift-discount hold-prob map
# =========================================================================


def _hold_prob_map(world: World) -> dict[int, float]:
    my_id = world.my_id
    omega = world.omega
    planets = list(world.planets_by_id.values())
    my_planets = [p for p in planets if p.owner == my_id]
    enemy_planets = [
        p for p in planets
        if p.owner != my_id and p.owner != -1
        and p.id not in world.comet_ids
    ]
    if not my_planets:
        return {p.id: 1.0 for p in planets}
    sample_ts = [
        HOLD_HORIZON * (i / max(1, SAMPLE_STEPS - 1))
        for i in range(SAMPLE_STEPS)
    ]

    def positions_at(group, t):
        out = []
        for p in group:
            tup = (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            if is_orbiting(tup):
                x, y = predict_relative(tup, omega, t)
            else:
                x, y = p.x, p.y
            out.append((x, y))
        return out

    my_pos_t = [positions_at(my_planets, t) for t in sample_ts]
    en_pos_t = (
        [positions_at(enemy_planets, t) for t in sample_ts]
        if enemy_planets else None
    )
    out: dict[int, float] = {}
    for p in planets:
        tup = (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
        in_cell = 0
        for i, t in enumerate(sample_ts):
            if is_orbiting(tup):
                tx, ty = predict_relative(tup, omega, t)
            else:
                tx, ty = p.x, p.y
            d_us = min(
                math.hypot(tx - fx, ty - fy) for fx, fy in my_pos_t[i]
            )
            if en_pos_t and en_pos_t[i]:
                d_them = min(
                    math.hypot(tx - ex, ty - ey)
                    for ex, ey in en_pos_t[i]
                )
            else:
                d_them = float("inf")
            if d_us <= d_them:
                in_cell += 1
        hp = in_cell / max(1, len(sample_ts))
        out[p.id] = max(MIN_DISCOUNT, hp)
    return out


# =========================================================================
# Garrison-on-capture state + proposer
# =========================================================================


class _GarrisonState:
    def __init__(self):
        self.last_step = -1
        self.last_owners: dict[int, int] = {}
        self.captured_at: dict[int, int] = {}

    def reset(self):
        self.last_step = -1
        self.last_owners = {}
        self.captured_at = {}


_STATE = _GarrisonState()


def _update_garrison_state(world: World) -> None:
    step = int(world.step)
    if step == 0 or step < _STATE.last_step:
        _STATE.reset()
    cur = {p.id: p.owner for p in world.planets_by_id.values()}
    my_id = world.my_id
    for pid, prev in _STATE.last_owners.items():
        cur_o = cur.get(pid)
        if cur_o is None:
            continue
        if cur_o == my_id and prev != my_id:
            _STATE.captured_at[pid] = step
        if cur_o != my_id and prev == my_id and pid in _STATE.captured_at:
            del _STATE.captured_at[pid]
    cutoff = step - GARRISON_WINDOW
    stale = [pid for pid, s in _STATE.captured_at.items() if s < cutoff]
    for pid in stale:
        del _STATE.captured_at[pid]
    _STATE.last_step = step
    _STATE.last_owners = cur


def _propose_garrison(world: World) -> list[Mission]:
    _update_garrison_state(world)
    if not _STATE.captured_at:
        return []
    my_id = world.my_id
    pbi = world.planets_by_id
    step_now = int(world.step)
    out: list[Mission] = []
    for pid, captured_step in list(_STATE.captured_at.items()):
        target = pbi.get(pid)
        if target is None or target.owner != my_id:
            continue
        if target.ships >= GARRISON_TARGET:
            del _STATE.captured_at[pid]
            continue
        deficit = GARRISON_TARGET - int(target.ships)
        age = step_now - captured_step
        if age > GARRISON_WINDOW:
            continue
        sources = [
            p for p in pbi.values()
            if p.owner == my_id and p.id != pid and p.ships > MIN_SOURCE_GARRISON
        ]
        if not sources:
            continue
        sources.sort(
            key=lambda s: math.hypot(s.x - target.x, s.y - target.y)
        )
        for src in sources[:2]:
            d = math.hypot(src.x - target.x, src.y - target.y)
            ships = max(deficit, 3)
            if src.ships - ships < MIN_SOURCE_GARRISON:
                continue
            v = fleet_speed(ships)
            eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
            remaining = max(1, EPISODE_STEPS - step_now - eta)
            value = float(target.production) * remaining
            score = (value / (ships + d + 1.0)) * 1.3
            out.append(Mission(
                mission_class="reinforce",
                src_id=src.id,
                target_id=target.id,
                ships=ships,
                score=score,
                eta=eta,
            ))
            break
    return out


# =========================================================================
# Combined _build_base_missions (recap + garrison) + drift discount
# =========================================================================


_geo_orig_build = _geo._build_base_missions


def agent(obs, configuration=None):
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    hp_map = _hold_prob_map(world)

    def patched_build(world_, model_):
        missions = (
            propose_opening_missions(world_, model_)
            + propose_snipe_missions(world_, model_, aggressive=True)
            + propose_reinforce_missions(world_, model_)
            + propose_recapture_missions(world_, model_)
            + _propose_garrison(world_)
        )
        missions = _geo._drop_comet_missions(missions, world_)
        rescaled = []
        for m in missions:
            hp = hp_map.get(m.target_id, 1.0)
            if hp >= 0.999 or m.target_id in world_.comet_ids:
                rescaled.append(m)
            else:
                rescaled.append(Mission(
                    mission_class=m.mission_class,
                    src_id=m.src_id, target_id=m.target_id,
                    ships=m.ships, score=m.score * hp,
                    eta=m.eta, note=m.note,
                ))
        return rescaled

    _geo._build_base_missions = patched_build
    try:
        return _geo.agent(obs, configuration)
    finally:
        _geo._build_base_missions = _geo_orig_build
