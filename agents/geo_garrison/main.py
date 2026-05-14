"""geo_garrison — geo v3.1 + post-capture garrison mission class.

Hypothesis: a planet captured at step t typically sits with 1-2 surplus
ships. Reinforce only fires when WorldModel PREDICTS a flip; by then
the recapture window is closing. A pre-emptive garrison fleet from the
NEXT-nearest friendly source pushes the planet over the GARRISON_TARGET
threshold before the opponent's counter-launch arrives.

Module-level state pattern (mirrors lib/missions/recapture.py): we
track planet ownership across turns and emit Reinforce-class missions
on the turn following a flip-to-us. Reset on step==0.

Single-axis variant: NO other change vs agents/geo/main.py.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.mission import Mission
from lib.world_model import WorldModel

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "geo_base_for_garrison", _REPO / "agents" / "geo" / "main.py",
)
_geo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geo)

GARRISON_TARGET = 12        # top-10 mean garrison-at-launch is ~11
GARRISON_WINDOW = 8         # propose only within N turns of capture
MIN_SOURCE_GARRISON = 6     # don't strand the source below this
EPISODE_STEPS = 500


class _GarrisonState:
    def __init__(self):
        self.last_step = -1
        self.last_owners: dict[int, int] = {}
        # pid -> step we (re)captured it
        self.captured_at: dict[int, int] = {}

    def reset(self):
        self.last_step = -1
        self.last_owners = {}
        self.captured_at = {}


_STATE = _GarrisonState()


def _update_state(world: World) -> None:
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
        # If we lose it, drop the bookmark (recapture handles re-take).
        if cur_o != my_id and prev == my_id and pid in _STATE.captured_at:
            del _STATE.captured_at[pid]
    # Evict bookmarks past the garrison window.
    cutoff = step - GARRISON_WINDOW
    stale = [pid for pid, s in _STATE.captured_at.items() if s < cutoff]
    for pid in stale:
        del _STATE.captured_at[pid]
    _STATE.last_step = step
    _STATE.last_owners = cur


def propose_post_capture_garrison(
    world: World, model: WorldModel,
) -> list[Mission]:
    _update_state(world)
    if not _STATE.captured_at:
        return []

    my_id = world.my_id
    pbi = world.planets_by_id
    step_now = int(world.step)
    out: list[Mission] = []

    for pid, captured_step in list(_STATE.captured_at.items()):
        target = pbi.get(pid)
        if target is None or target.owner != my_id:
            # already lost — recapture's problem now
            continue
        if target.ships >= GARRISON_TARGET:
            # already secured; drop bookmark
            del _STATE.captured_at[pid]
            continue
        deficit = GARRISON_TARGET - int(target.ships)
        # Score targets that are STILL within the window favourably.
        age = step_now - captured_step
        if age > GARRISON_WINDOW:
            continue
        # Pick second-nearest friendly source (not the planet itself)
        sources = [
            p for p in pbi.values()
            if p.owner == my_id and p.id != pid and p.ships > MIN_SOURCE_GARRISON
        ]
        if not sources:
            continue
        sources.sort(
            key=lambda s: math.hypot(s.x - target.x, s.y - target.y)
        )
        for src in sources[:2]:  # try two nearest
            d = math.hypot(src.x - target.x, src.y - target.y)
            ships = max(deficit, 3)
            if src.ships - ships < MIN_SOURCE_GARRISON:
                continue
            v = fleet_speed(ships)
            eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
            remaining = max(1, EPISODE_STEPS - step_now - eta)
            # Score on par with reinforce: production * remaining / (ships + d + 1).
            value = float(target.production) * remaining
            score = value / (ships + d + 1.0)
            # Bias scores up so they compete with snipe — but bounded.
            score *= 1.3
            out.append(Mission(
                mission_class="reinforce",  # reuse settle_plan's reinforce path
                src_id=src.id,
                target_id=target.id,
                ships=ships,
                score=score,
                eta=eta,
            ))
            break  # one garrison fleet per recently-captured target
    return out


_geo_orig_build = _geo._build_base_missions


def _build_with_garrison(world: World, model: WorldModel):
    missions = _geo_orig_build(world, model)
    extra = propose_post_capture_garrison(world, model)
    if extra:
        # Filter comets just in case (defensive — proposer skips them already).
        extra = [m for m in extra if m.target_id not in world.comet_ids]
        missions = missions + extra
    return missions


_geo._build_base_missions = _build_with_garrison


def agent(obs, configuration=None):
    return _geo.agent(obs, configuration)
