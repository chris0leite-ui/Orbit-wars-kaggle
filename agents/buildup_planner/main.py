"""buildup_planner.main — phase-dispatch state machine.

Step 1 wiring: BUILDUP → CONSOLIDATION (no STRIKE).

Per-seat phase state lives in module-level `_PHASE_STATE`. State resets
on `obs.step == 0` (new game). The state machine is:

    Initial            : phase = "BUILDUP"
    BUILDUP            : try buildup.step(...); if it returns moves,
                         emit and stay in BUILDUP for next turn.
                         If None, transition to CONSOLIDATION SAME TURN
                         and emit the CONSOLIDATION moves.
    CONSOLIDATION      : evaluate_inflection (Step 1 stub → None);
                         delegate to consolidation.step(obs, configuration).
    STRIKE             : (Step 3) — never reached in Step 1.

NO modifications to obs / configuration before CONSOLIDATION delegates
to the baseline pipeline — that pipeline handles its own parse, ledger,
and reset.
"""
from __future__ import annotations

import os

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

# Per-seat machine state. Keyed by `obs.player` (int).
_PHASE_STATE: dict[int, dict] = {}


def _initial_state() -> dict:
    return {"phase": PHASE_BUILDUP, "strike_plan": None}


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
        # Inflection predicate (Step 1 stub: always None → stay).
        # World/model rebuilt here only when we need the predicate; once
        # the predicate is wired we'll always evaluate it. Step 1 skips
        # the rebuild as a small optimisation since the stub is no-op.
        if predicates.evaluate_inflection is not None:
            # Guard the stub call cheaply — short-circuit on `is None`.
            try:
                # Rebuild world only when the predicate would actually fire.
                # Step 1 stub returns None unconditionally, so we elide.
                plan = None  # placeholder for Step 2: evaluate_inflection(...)
            except Exception:
                plan = None
            if plan is not None:
                state["phase"] = PHASE_STRIKE
                state["strike_plan"] = plan
                # Fall through to STRIKE this turn.

        if state["phase"] == PHASE_CONSOLIDATION:
            return consolidation.step(obs, configuration)

    # --- STRIKE branch (Step 3 — not wired in Step 1) -------------------
    if state["phase"] == PHASE_STRIKE:
        # Unreachable in Step 1; defensive fallback to CONSOLIDATION.
        state["phase"] = PHASE_CONSOLIDATION
        state["strike_plan"] = None
        return consolidation.step(obs, configuration)

    # Unknown phase — defensive fallback.
    state["phase"] = PHASE_CONSOLIDATION
    return consolidation.step(obs, configuration)
