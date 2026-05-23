"""STRIKE phase — atomic-drop emission of a coordinated wave.

`step(world, plan)` re-validates every `Shot` in `plan.shots` against the
current world via `lib.trajectory.predict_fleet_fate`. ALL shots in a
wave fire on the same turn (the predicate searches for a single arrival
step T and produces shots with varying `eta` so they converge there).
There is no multi-turn persistence — the strike turn is one decisive
emission, and control returns to CONSOLIDATION next turn.

Two failure modes both trigger an atomic drop (return `[]`, log warning):

1. **Budget overflow** — Step-2 `evaluate_inflection` permits the same
   source planet to be counted toward multiple targets (the optimistic
   upper bound, see `predicates.py` module docstring). At emit time we
   enforce the actual constraint: per-source cumulative `ship_count`
   must not exceed the source's current garrison.

2. **Physics fail** — `predict_fleet_fate` returns any `outcome` other
   than `"target"`. Mirrors the joint-candidate atomic-drop in
   `agents/baseline/chooser_trajectory.py:612-628`.

The drop cost is one wasted predicate elect, not a wasted turn — the
dispatcher transitions back to CONSOLIDATION and emits normal moves
next turn.

Step 3b diagnostic: `BUILDUP_PLANNER_STRIKE_LOG=<path>` opt-in emits one
JSONL entry per `step` call so we can correlate strike outcomes with
game outcomes (decides between the three Step-3b modeling hypotheses).
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from lib.trajectory import predict_fleet_fate

from agents.buildup_planner.predicates import StrikePlan


_LOG = logging.getLogger("buildup_planner.strike")


def _strike_log_path() -> str | None:
    """Step 3b diagnostic env hook. Empty / unset = no log."""
    p = os.environ.get("BUILDUP_PLANNER_STRIKE_LOG", "")
    return p if p else None


def _log_strike(entry: dict) -> None:
    """Append one JSONL line to the strike-log; silent on any I/O error.

    Mirrors `main.py:_log_elect` — diagnostic must NEVER break the agent.
    """
    path = _strike_log_path()
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception:
        pass


def step(world, plan: StrikePlan, *,
         game_id: str = "unknown", step_now: int = -1) -> list[list]:
    """Emit the wave's moves, or atomic-drop on any failure.

    `game_id` and `step_now` are passed through to the strike-log entry
    (no behavioural use). Defaults make the new args optional so existing
    tests that call `strike.step(world, plan)` keep working.
    """
    if plan is None or not plan.shots:
        _log_strike({
            "game_id": game_id, "step": step_now,
            "outcome": "empty", "reason": "no_plan_or_no_shots",
            "num_emitted": 0,
        })
        return []

    target_ids_sorted = sorted(int(t) for t in plan.target_ids)
    arrival = int(plan.arrival_step)
    num_shots = len(plan.shots)

    # Pass 1: per-source ship-budget check. Step-2 predicate may have
    # double-counted a source across multi-target plans (|S|=2 case).
    usage: dict[int, int] = defaultdict(int)
    for shot in plan.shots:
        usage[int(shot.src_id)] += int(shot.ship_count)
    for src_id, used in usage.items():
        src = world.planets_by_id.get(int(src_id))
        if src is None:
            _LOG.warning(
                "atomic-drop: src %d not in world (target_ids=%s arrival=%d)",
                src_id, target_ids_sorted, arrival,
            )
            _log_strike({
                "game_id": game_id, "step": step_now,
                "outcome": "budget_overflow",
                "reason": f"src_missing:{src_id}",
                "num_emitted": 0, "num_shots": num_shots,
                "target_ids": target_ids_sorted, "arrival_step": arrival,
            })
            return []
        if used > int(src.ships):
            _LOG.warning(
                "atomic-drop: budget_overflow src=%d used=%d garrison=%d "
                "(target_ids=%s arrival=%d)",
                src_id, used, int(src.ships),
                target_ids_sorted, arrival,
            )
            _log_strike({
                "game_id": game_id, "step": step_now,
                "outcome": "budget_overflow",
                "reason": f"src={src_id} used={used} garrison={int(src.ships)}",
                "num_emitted": 0, "num_shots": num_shots,
                "target_ids": target_ids_sorted, "arrival_step": arrival,
            })
            return []

    # Pass 2: per-shot physics re-validation via the engine-mirroring
    # ray-cast. ANY non-"target" outcome drops the whole wave.
    for shot in plan.shots:
        src = world.planets_by_id.get(int(shot.src_id))
        tgt = world.planets_by_id.get(int(shot.tgt_id))
        if src is None or tgt is None:
            _LOG.warning(
                "atomic-drop: missing src=%s or tgt=%s",
                shot.src_id, shot.tgt_id,
            )
            _log_strike({
                "game_id": game_id, "step": step_now,
                "outcome": "physics_fail",
                "reason": f"missing_planet src={shot.src_id} tgt={shot.tgt_id}",
                "num_emitted": 0, "num_shots": num_shots,
                "target_ids": target_ids_sorted, "arrival_step": arrival,
            })
            return []
        fate = predict_fleet_fate(
            src, tgt, float(shot.angle), int(shot.ship_count), world,
        )
        if fate.outcome != "target":
            _LOG.warning(
                "atomic-drop: physics_fail src=%d tgt=%d outcome=%s hit=%s "
                "step=%d (target_ids=%s arrival=%d)",
                shot.src_id, shot.tgt_id, fate.outcome, fate.hit_planet_id,
                fate.step, target_ids_sorted, arrival,
            )
            _log_strike({
                "game_id": game_id, "step": step_now,
                "outcome": "physics_fail",
                "reason": f"src={shot.src_id} tgt={shot.tgt_id} fate={fate.outcome}",
                "num_emitted": 0, "num_shots": num_shots,
                "target_ids": target_ids_sorted, "arrival_step": arrival,
            })
            return []

    # All-pass: emit the wave.
    _log_strike({
        "game_id": game_id, "step": step_now,
        "outcome": "emit", "reason": "",
        "num_emitted": num_shots, "num_shots": num_shots,
        "target_ids": target_ids_sorted, "arrival_step": arrival,
    })
    return [[int(shot.src_id), float(shot.angle), int(shot.ship_count)]
            for shot in plan.shots]
