"""buildup_planner.main — phase-dispatch state machine.

Per-seat phase state lives in module-level `_PHASE_STATE`. State resets
on `obs.step == 0` (new game). The state machine is:

    Initial            : phase = "BUILDUP"
    BUILDUP            : if step >= OPENING_HORIZON, transition straight
                         to CONSOLIDATION (no World/Model build paid).
                         Else try buildup.step(...); if it returns moves,
                         emit and stay in BUILDUP for next turn. If None,
                         transition to CONSOLIDATION SAME TURN and emit
                         the CONSOLIDATION moves.
    CONSOLIDATION      : FIRST run the FINISHER pre-check
                         (`endgame.quick_trigger`); when opp owns
                         <=K_FINISH planets, search for a single-arrival
                         wave that captures EVERY opp planet and emit it
                         this turn (drives strict elimination, not just
                         score-leading). Otherwise, when STRIKE is enabled,
                         run `evaluate_inflection` and route to STRIKE on a
                         plan. Default fast path (STRIKE disabled, no
                         FINISHER trigger) delegates straight to
                         consolidation.step — the Step-3b observation-only
                         predicate path was retired since it paid a
                         ~600ms-per-turn wallclock tax vs strong opponents
                         without behavioral lift.
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

# Default BASELINE_ORBITAL_SAFETY on for this agent. Activates the
# rotation-aware opp-ETA path in lib/scoring.py, agents/baseline/proposer.py,
# lib/joint_solver/opening_planner.py and lib/joint_solver/value.py — see
# 2026-05-24 plan "rotation-aware opponent ETA in early-game planning".
# Explicit `BASELINE_ORBITAL_SAFETY=0` still wins (setdefault, not set).
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")

# Default KINEMATIC_TABLE_ENABLED on for this agent (2026-05-25). Wires
# the per-turn position cache (lib/kinematic_table.py) into
# predict_fleet_fate's inner loop via lib/trajectory._table_window_or_none.
# Bit-parity-gated (21/21 parity tests GREEN; see commit c48e143's
# 564 FleetFate assertions + 2 full-game parity result). Documented
# wallclock saving: 47-114 ms/step. Targets the consolidation review
# finding K1 (audit/2026-05-25-consolidation-review.md): predict_relative
# consumed 84s / 219 turns of the focal CPU; the table removes its
# per-(planet,step) rebuild. Explicit `KINEMATIC_TABLE_ENABLED=0` still
# wins (setdefault).
os.environ.setdefault("KINEMATIC_TABLE_ENABLED", "1")

# Default Layer Z v2 effective-landing prune on (2026-05-25). v2 fixes
# the v1 formula by subtracting `pred_ships` so the gate measures real
# headroom-over-defender, not total ship count. Applied in the proposer
# only — the opening_planner site was redundant with `gar_at_arr` and
# was dropped. See lib/joint_solver/opening_planner.py and
# agents/baseline/proposer.py:516. Explicit `BASELINE_EFFECTIVE_LANDING_PRUNE=0`
# still wins (setdefault, not set).
os.environ.setdefault("BASELINE_EFFECTIVE_LANDING_PRUNE", "1")

# Phi-1 leaf swap (2026-05-25): favor_phi adds the 2P elimination bonus
# (missing in `favor`) and uses 250-tick pv_horizon to match PI's
# fast-elim metric. HARD SET (not setdefault) because the bundler
# inlines `agents/baseline/main.py` BEFORE `agents/buildup_planner/main.py`
# (dependency order), and baseline's own setdefault to "hybrid" wins
# any setdefault race in the bundle. Hard-setting at agent-load time
# means BASELINE_VALUE_HEAD=composite passed in by an A/B harness will
# still be overridden — pass it AFTER the agent loads if you need
# a manual override. Local n=8 vs sub 52968889 lineage: 4/8 parity,
# Wilson [0.22, 0.79]. Submitted under explicit PI override of Rule 45.
os.environ["BASELINE_VALUE_HEAD"] = "phi"

# Sibling ESwSv per-ship-sort flag (cherry-picked from commit 0a8308f).
# Sub 53024913 settled at μ=1136.4 with this flag ON. The read is
# call-time inside chooser_trajectory.choose_trajectory (the call-time
# refactor closes the bundle-order timing gap that the module-level
# cache had). setdefault is fine here — no competing setdefault exists
# in the bundle for this key.
os.environ.setdefault("BASELINE_SORT_BY_EV_PER_SHIP", "1")

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from lib.intent import World
from lib.joint_solver.opening_planner import OPENING_HORIZON
from lib.world_model import WorldModel

# Re-use baseline's well-tested obs-dict + num_seats helpers rather than
# duplicating them. They are private-by-naming but stable: any change
# would also break agents/baseline/main.py, which is parity-tested.
from agents.baseline.main import _as_dict, _num_seats

from agents.buildup_planner import buildup, consolidation, dogpile, endgame, predicates
from agents.buildup_planner import strike as _strike_mod


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


def _buildup_enabled() -> bool:
    """Default ON. Env hook for ablation: when OFF, skip the BUILDUP MILP
    entirely and drop into CONSOLIDATION from turn 0 (so the agent is
    behaviorally identical to baseline + FINISHER). Used to test whether
    the open-loop MILP opening helps or hurts vs reactive opponents."""
    return os.environ.get("BUILDUP_PLANNER_OPENING_ENABLED", "1") == "1"

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
        # Past the opening horizon OR opening disabled by env: transition
        # straight to CONSOLIDATION without paying the World+Model build.
        # buildup.step short-circuits to None for step>=OPENING_HORIZON;
        # the ~25ms/turn build was wasted for the bulk of a 500-turn game.
        if step >= OPENING_HORIZON or not _buildup_enabled():
            state["phase"] = PHASE_CONSOLIDATION
        else:
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

            # Phase γ kinematic_table priming for BUILDUP turns. Mirrors
            # the same block in agents/baseline/main.py — without this,
            # the opening's `opening_plan() -> _build_candidates ->
            # predict_fleet_fate` path runs on an unprimed cache and
            # falls through to the inline build (the slow path the
            # K1 wiring exists to avoid). Per the K1 cProfile re-run,
            # opening turns (12-28) stayed at ~2s because BUILDUP never
            # primed; CONSOLIDATION turns improved by ~200ms p50.
            if os.environ.get("KINEMATIC_TABLE_ENABLED", "").strip().lower() in (
                "1", "true", "on", "yes",
            ):
                try:
                    from lib import kinematic_table as _kt
                    _kt.begin_turn(world)
                except Exception:
                    pass

            model = WorldModel.from_world(world)
            num_seats = _num_seats(planets, fleets)

            moves = buildup.step(world, model, me, num_seats, step)
            if moves is not None:
                # Stay in BUILDUP — the schedule will time out on its own
                # via the OPENING_HORIZON guard inside buildup.step.
                return moves

            # buildup returned None: either opening exhausted
            # (step >= OPENING_HORIZON) OR this turn's schedule has no
            # fire_step == step_now entries (the MILP wants to wait_N>0).
            # The 2026-05-25 fix: while we're still inside the opening
            # horizon, STAY in BUILDUP and emit no moves this turn so we
            # re-derive next turn. Pre-fix bug: the agent permanently
            # transitioned to CONSOLIDATION on the first turn the MILP
            # scheduled fire_step > step_now (i.e. *any* wait-N opening),
            # which made V3 and V1 unobservable in real games — the
            # planned step-3 launch was abandoned before step-3 arrived.
            if step < OPENING_HORIZON:
                return []

            # Opening genuinely exhausted: transition to CONSOLIDATION
            # this same turn.
            state["phase"] = PHASE_CONSOLIDATION
            # Fall through.

    # --- CONSOLIDATION branch -------------------------------------------
    if state["phase"] == PHASE_CONSOLIDATION:
        # FINISHER pre-check (cheap O(|planets|)). Drives elimination once
        # opp is reduced to <= K_FINISH planets — the closed-form gate
        # alone leaves opp with 1-2 planets at turn cap (score-win, not
        # elimination), which terminates as a "win" in reward space but
        # leaves opp alive. FINISHER closes that gap by planning a wave
        # that captures EVERY opp planet at one arrival step, dropping
        # opp to zero owned planets (combined with 0 in-flight ships
        # this triggers the env's elimination termination).
        eg_trigger = endgame.quick_trigger(raw_planets, me)
        if eg_trigger is not None:
            eg_opp_id, _ = eg_trigger
            eg_world = World.from_obs(obs_d)
            eg_model = WorldModel.from_world(eg_world)
            fin_plan = endgame.evaluate(
                eg_world, eg_model, me, eg_opp_id,
            )
            if fin_plan is not None:
                # Re-use strike.step's atomic-drop emission machinery —
                # both plan shapes share the same shot contract.
                try:
                    return _strike_mod.step(
                        eg_world, fin_plan,
                        game_id=state.get("game_id", "unknown"),
                        step_now=step,
                    )
                except Exception as exc:
                    import logging
                    logging.getLogger("buildup_planner.endgame").warning(
                        "atomic-drop: endgame strike.step raised %s: %s",
                        type(exc).__name__, exc,
                    )
                    return []

        # DOGPILE pre-check (cheap O(|planets|)). Fires mid-game on the
        # top-K opp planets by production when |opp planets| > K_FINISH
        # (NOT FINISHER's territory) and the post-capture production
        # advantage clears the relaxed gate. The atomic-drop emission
        # machinery is shared with FINISHER + STRIKE.
        dp_trigger = dogpile.quick_trigger(raw_planets, me)
        if dp_trigger is not None:
            dp_opp_id, _ = dp_trigger
            dp_world = World.from_obs(obs_d)
            dp_model = WorldModel.from_world(dp_world)
            dp_plan = dogpile.evaluate(
                dp_world, dp_model, me, dp_opp_id,
            )
            if dp_plan is not None:
                try:
                    return _strike_mod.step(
                        dp_world, dp_plan,
                        game_id=state.get("game_id", "unknown"),
                        step_now=step,
                    )
                except Exception as exc:
                    import logging
                    logging.getLogger("buildup_planner.dogpile").warning(
                        "atomic-drop: dogpile strike.step raised %s: %s",
                        type(exc).__name__, exc,
                    )
                    # Fall through to baseline consolidation rather than
                    # emit []; if dogpile's emission fails, we still want
                    # to play the turn normally.

        # Default fast path: STRIKE disabled → skip the predicate entirely
        # and delegate straight to consolidation. The Step-3b diagnostic
        # confirmed the observation-only predicate pays ~600ms/turn vs
        # strong opponents with no behavioral benefit when STRIKE is off.
        if not _strike_enabled():
            return consolidation.step(obs, configuration)

        # STRIKE-enabled path: run evaluate_inflection, log result, route.
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
            if plan is not None:
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
        # Wrap in try/except mirroring the CONSOLIDATION-branch guard on
        # evaluate_inflection (lines 170-176): predict_fleet_fate on a
        # degenerate Shot (ship_count=0 div-by-zero, stale src_id KeyError,
        # NaN angle) would otherwise propagate up and Kaggle would mark
        # the agent as ERROR for the game. Strike errors MUST NOT break
        # the agent — return [] on any exception. Logged for diagnosis.
        try:
            return strike.step(
                World.from_obs(obs_d), plan,
                game_id=state.get("game_id", "unknown"), step_now=step,
            )
        except Exception as exc:
            import logging
            logging.getLogger("buildup_planner.strike").warning(
                "atomic-drop: strike.step raised %s: %s",
                type(exc).__name__, exc,
            )
            return []

    # Unknown phase — defensive fallback.
    state["phase"] = PHASE_CONSOLIDATION
    return consolidation.step(obs, configuration)
