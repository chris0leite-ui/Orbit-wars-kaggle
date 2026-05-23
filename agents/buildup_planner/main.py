"""buildup_planner.main — phase-dispatch state machine.

Per-seat phase state lives in module-level `_PHASE_STATE`. State resets
on `obs.step == 0` (new game). The state machine is:

    Initial            : phase = "BUILDUP"
    BUILDUP            : try buildup.step(...); if it returns moves,
                         emit and stay in BUILDUP for next turn.
                         If None, transition to CONSOLIDATION SAME TURN
                         and emit the CONSOLIDATION moves.
    CONSOLIDATION      : evaluate_inflection search; log elect result to
                         audit/…strike_elect.jsonl; if a plan is found
                         AND BUILDUP_PLANNER_STRIKE_ENABLED=1, transition
                         to STRIKE same-turn and emit the wave. Otherwise
                         delegate to consolidation.step(...).
    STRIKE             : single-turn — atomic-drop validation via
                         strike.step(world, plan); transition back to
                         CONSOLIDATION at end of turn. Strike-only
                         emission (no consolidation moves that turn).

NO modifications to obs / configuration before CONSOLIDATION delegates
to the baseline pipeline — that pipeline handles its own parse, ledger,
and reset.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from lib.intent import World
from lib.world_model import WorldModel

# Re-use baseline's well-tested obs-dict + num_seats helpers rather than
# duplicating them. They are private-by-naming but stable: any change
# would also break agents/baseline/main.py, which is parity-tested.
from agents.baseline.main import _as_dict, _num_seats

from agents.buildup_planner import buildup, consolidation, predicates


# Phase tags (strings keep tracebacks + logs readable).
PHASE_BUILDUP = "BUILDUP"
PHASE_CONSOLIDATION = "CONSOLIDATION"
PHASE_STRIKE = "STRIKE"  # reserved for Step 3


# --- Observation-only logging (Step 2) ----------------------------------
# Path is relative to repo root by default; an env override lets tests
# point at /tmp or disable entirely (empty string = no logging).
# Default ON: the whole point of Step 2 is to measure elect-rate.
_ELECT_LOG_PATH_DEFAULT = "audit/buildup_planner_strike_elect.jsonl"


def _elect_log_path() -> str | None:
    """Resolve the audit log path; None = logging disabled."""
    p = os.environ.get("BUILDUP_PLANNER_ELECT_LOG", _ELECT_LOG_PATH_DEFAULT)
    return p if p else None


def _log_elect(entry: dict) -> None:
    """Append one JSON line to the elect-log; silent on any I/O error."""
    path = _elect_log_path()
    if not path:
        return
    try:
        # Make parent dir best-effort; missing parent should not crash.
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception:
        # Observation log MUST NOT break the agent. Drop silently.
        pass


def _strike_enabled() -> bool:
    """Step 3 wires STRIKE; Step 2 keeps it OFF by default."""
    return os.environ.get("BUILDUP_PLANNER_STRIKE_ENABLED", "0") == "1"

# Per-seat machine state. Keyed by `obs.player` (int).
_PHASE_STATE: dict[int, dict] = {}

# Monotonic per-process counter — included in elect-log entries so that
# `--workers >1` runs (where N games interleave in one log file) can be
# aggregated per-game by `(pid, game_idx)`. PID alone is insufficient
# because one worker process plays several games sequentially.
_GAME_COUNTER = 0


def _next_game_id() -> str:
    global _GAME_COUNTER
    _GAME_COUNTER += 1
    return f"{os.getpid()}_{_GAME_COUNTER}"


def _initial_state() -> dict:
    return {"phase": PHASE_BUILDUP, "strike_plan": None,
            "game_id": _next_game_id()}


def _reset_if_new_game(me: int, step: int) -> dict:
    """Ensure `_PHASE_STATE[me]` exists; reset on step==0."""
    if step == 0 or me not in _PHASE_STATE:
        _PHASE_STATE[me] = _initial_state()
    return _PHASE_STATE[me]


def agent(obs, configuration=None) -> list[list]:
    """Kaggle entrypoint — dispatch to the active phase."""
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    step = int(obs_d.get("step", 0))

    raw_planets = obs_d.get("planets", []) or []
    raw_fleets = obs_d.get("fleets", []) or []
    if not raw_planets:
        return []

    state = _reset_if_new_game(me, step)

    # --- BUILDUP branch -------------------------------------------------
    if state["phase"] == PHASE_BUILDUP:
        planets = [Planet(*p) for p in raw_planets]
        fleets = [Fleet(*f) for f in raw_fleets]
        my_planets = [p for p in planets if int(p.owner) == me]
        other_planets = [p for p in planets if int(p.owner) != me]
        if not my_planets or not other_planets:
            # Degenerate state — let CONSOLIDATION handle it; it will
            # also return [] but logs/tests treat it consistently.
            state["phase"] = PHASE_CONSOLIDATION
            return consolidation.step(obs, configuration)

        world = World.from_obs(obs_d)
        model = WorldModel.from_world(world)
        num_seats = _num_seats(planets, fleets)

        moves = buildup.step(world, model, me, num_seats, step)
        if moves is not None:
            # Stay in BUILDUP — the schedule will time out on its own
            # via the OPENING_HORIZON guard inside buildup.step.
            return moves

        # buildup returned None: opening exhausted or no fire-now entries.
        # Transition to CONSOLIDATION this same turn.
        state["phase"] = PHASE_CONSOLIDATION
        # Fall through.

    # --- CONSOLIDATION branch -------------------------------------------
    if state["phase"] == PHASE_CONSOLIDATION:
        # Step 2: run the real inflection predicate, log the result.
        # Strike transition is GATED by BUILDUP_PLANNER_STRIKE_ENABLED
        # (default OFF) — observation-only until Step 3 flips it ON.
        world_pred = World.from_obs(obs_d)
        opp_id = predicates.opp_id_2p(world_pred, me)
        # `.get` keeps existing dispatch tests (which build state manually
        # without game_id) green; live game state always has it set.
        gid = state.get("game_id", "unknown")
        if opp_id < 0:
            _log_elect({"game_id": gid, "step": step, "me": me,
                        "opp_id": -1, "skipped": "4p"})
        else:
            t0 = time.perf_counter()
            plan = None
            try:
                model = WorldModel.from_world(world_pred)
                plan = predicates.evaluate_inflection(
                    world_pred, model, me, opp_id
                )
            except Exception as exc:
                # Predicate errors must NOT break consolidation. Log and stay.
                _log_elect({
                    "game_id": gid, "step": step, "me": me, "opp_id": opp_id,
                    "error": repr(exc),
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                })
                plan = None
            else:
                _log_elect({
                    "game_id": gid, "step": step, "me": me, "opp_id": opp_id,
                    "plan_found": plan is not None,
                    "arrival_step": (int(plan.arrival_step) if plan else None),
                    "target_ids": (sorted(int(t) for t in plan.target_ids)
                                   if plan else None),
                    "num_shots": (len(plan.shots) if plan else 0),
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                })
            if plan is not None and _strike_enabled():
                state["phase"] = PHASE_STRIKE
                state["strike_plan"] = plan
                # Fall through to STRIKE this turn.

        if state["phase"] == PHASE_CONSOLIDATION:
            return consolidation.step(obs, configuration)

    # --- STRIKE branch ---------------------------------------------------
    # One-turn phase: all shots in `strike_plan.shots` fire same-turn (the
    # predicate sized each shot's eta so the wave converges at one arrival
    # step). Strike-only emission — defensive consolidation is skipped for
    # this single turn; the closed-form `is_winning_state_if_owned` gate
    # already certifies we win allowing for one turn of opp recovery.
    if state["phase"] == PHASE_STRIKE:
        from agents.buildup_planner import strike
        plan = state["strike_plan"]
        # Always transition back to CONSOLIDATION next turn, whether the
        # strike emits or atomic-drops. Clear the plan before emit so a
        # downstream exception can't leave stale state.
        state["phase"] = PHASE_CONSOLIDATION
        state["strike_plan"] = None
        # Build a fresh World here — don't rely on cross-block scope from
        # the CONSOLIDATION branch (only safe today because fall-through
        # is the only path that sets PHASE_STRIKE, but brittle).
        return strike.step(
            World.from_obs(obs_d), plan,
            game_id=state.get("game_id", "unknown"), step_now=step,
        )

    # Unknown phase — defensive fallback.
    state["phase"] = PHASE_CONSOLIDATION
    return consolidation.step(obs, configuration)
