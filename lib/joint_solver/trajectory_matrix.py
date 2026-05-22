"""Trajectory matrix — game-start one-shot precompute of every viable
(src, tgt, launch_tick) → (angle, eta, capture-feasibility) tuple in the
opening phase.

Phase η of /root/.claude/plans/do-it-thoroughly-consider-tingly-fox.md.

The deterministic substrate (planet orbits + initial garrisons) lets us
enumerate all opening trajectories ONCE at game start, then look them
up in O(1) at every per-turn solve. This lifts the per-turn
`aim_and_eta + predict_fleet_fate` cost (≈ 0.5-2 ms per tuple) from
inside the solver loop to a one-shot precompute (~18 s on Kaggle's
allowed first-turn overage).

Design mirrors `lib/kinematic_table.py`'s singleton + fingerprint
pattern. One module-level instance shared across in-process agents;
tests instantiate `TrajectoryMatrix()` directly.

Layer responsibilities:
- `kinematic_table` (Phase γ): per-tick planet positions, O(1) lookup.
- `trajectory_matrix` (this file): per (src, tgt, T) viable trajectory
  (angle, eta, captures, ships_needed), O(1) lookup.
- `opening_search` (next file): candidate generation + MILP scheduling
  over the matrix.

Per-trajectory entry stores closed-form `ships_needed` (the minimum
capture size given the predicted defender garrison at arrival). The
opening_search layer enumerates ship-count variants on top
(`ships_needed`, `2 × ships_needed`, source-budget cap).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from agents.baseline.proposer import aim_and_eta
from lib.trajectory import predict_fleet_fate
from lib.world_model import predict_garrison_at


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum launch tick (relative to game start) we precompute. Matches
# OPENING_HORIZON in opening_planner.py — past this the LP takes over
# and the matrix is no longer consulted. Plus a small buffer to allow
# the opening_search to consider fires whose arrival is just past the
# horizon (production still accrues post-arrival).
DEFAULT_MAX_LAUNCH_TICK = 30
DEFAULT_ARRIVAL_BUFFER = 10        # ticks past OPENING_HORIZON we allow arrivals

# Defender guard — leave at least this many ships on each source (matches
# `OPENING_DEFENDER_GUARD` in opening_planner.py). Used here for the
# affordability check that drops obviously-infeasible (src, tgt, T) tuples.
DEFAULT_DEFENDER_GUARD = 2


# ---------------------------------------------------------------------------
# Entry dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryEntry:
    """One precomputed trajectory.

    `viable=True` iff: aim_and_eta returned a finite eta, predict_fleet_fate
    landed at tgt (not sun/oob/timeout), ships_needed ≤ source's ship
    budget at launch_tick, and arrival is within the matrix window.

    Non-viable entries are NOT stored in the matrix — `get()` returns
    None for them. The matrix holds only the viable subset.
    """
    src_id: int
    tgt_id: int
    launch_tick: int            # absolute env step at which the fire would launch
    ships_needed: int           # minimum capture size at arrival
    angle: float                # firing angle (radians)
    eta_flight: int             # ticks of flight from launch to arrival
    arrival_tick: int           # absolute env step the fleet lands
    arrival_owner: int          # owner at arrival (closed-form, before our fire)
    arrival_garrison: float     # garrison at arrival (before our fire)
    src_budget_at_launch: int   # source's available ships at launch_tick (pre-fire)


# ---------------------------------------------------------------------------
# Singleton class
# ---------------------------------------------------------------------------


class TrajectoryMatrix:
    """Game-start precomputed trajectory cache.

    Usage:

        matrix.begin_game(world, model, omega, my_id)
        entry = matrix.get(src_id, tgt_id, launch_tick)
        for entry in matrix.iter_viable(src_id=0):
            ...

    `begin_game` is idempotent within a game: if the fingerprint
    (n_planets, initial planet positions + ownerships, omega) matches
    the last build, no rebuild fires. On game boundary (new fingerprint)
    we wipe and re-precompute — the ~18 s cost is paid once per game.
    """

    def __init__(self) -> None:
        # Keyed by (src_id, tgt_id, launch_tick) → TrajectoryEntry. Only
        # viable entries stored; non-viable lookups return None.
        self._entries: dict[tuple[int, int, int], TrajectoryEntry] = {}
        self._fingerprint: Any = None
        self._max_launch_tick: int = DEFAULT_MAX_LAUNCH_TICK
        self._arrival_buffer: int = DEFAULT_ARRIVAL_BUFFER
        self._stats: dict[str, int] = {}

    # ---- lifecycle ----

    def reset(self) -> None:
        """Drop all state. Tests use this; production callers shouldn't."""
        self._entries = {}
        self._fingerprint = None
        self._stats = {}

    def begin_game(self, world, model, omega: float, my_id: int,
                   *, max_launch_tick: Optional[int] = None,
                   arrival_buffer: Optional[int] = None) -> bool:
        """Rebuild the matrix if the game fingerprint changed.

        Returns True iff a rebuild fired. The first call in a fresh
        game ALWAYS rebuilds; subsequent calls within the same game
        are no-ops (returns False).

        Per-game cost: one-shot, ~5-20 s depending on planet count and
        OPENING_HORIZON. Per-turn cost after build: 0 (only lookup).
        """
        if max_launch_tick is not None and int(max_launch_tick) != self._max_launch_tick:
            self._max_launch_tick = int(max_launch_tick)
            self._fingerprint = None
        if arrival_buffer is not None and int(arrival_buffer) != self._arrival_buffer:
            self._arrival_buffer = int(arrival_buffer)
            self._fingerprint = None

        new_fp = self._build_fingerprint(world, omega)
        if self._fingerprint == new_fp:
            return False
        self._rebuild(world, model, omega, int(my_id))
        self._fingerprint = new_fp
        return True

    @staticmethod
    def _build_fingerprint(world, omega: float) -> tuple:
        """Fingerprint anchors on initial planet positions + ownerships +
        omega. Stable across turns of a game; differs between games.
        """
        planets = world.planets_by_id
        # Coarse-rounded positions so floating-point drift in obs doesn't
        # spuriously change the fingerprint mid-game.
        positions = tuple(sorted(
            (int(pid), round(float(p.x), 3), round(float(p.y), 3),
             int(p.owner), int(p.production))
            for pid, p in planets.items()
        ))
        return ("traj_matrix", positions, round(float(omega), 6))

    # ---- query ----

    def get(self, src_id: int, tgt_id: int, launch_tick: int
            ) -> Optional[TrajectoryEntry]:
        """O(1) lookup. Returns None if the trajectory is not viable
        (bounce, OOB, sun-hit, unaffordable, or arrival past window)."""
        return self._entries.get((int(src_id), int(tgt_id), int(launch_tick)))

    def iter_viable(self, *, src_id: Optional[int] = None,
                    tgt_id: Optional[int] = None,
                    launch_tick: Optional[int] = None
                    ) -> Iterator[TrajectoryEntry]:
        """Iterate over viable entries optionally filtered by any subset
        of (src_id, tgt_id, launch_tick)."""
        for (s, t, T), entry in self._entries.items():
            if src_id is not None and s != int(src_id):
                continue
            if tgt_id is not None and t != int(tgt_id):
                continue
            if launch_tick is not None and T != int(launch_tick):
                continue
            yield entry

    def stats(self) -> dict:
        """Build-pass diagnostics (raw count, viable count, drop reasons)."""
        return dict(self._stats)

    def __len__(self) -> int:
        return len(self._entries)

    # ---- build pass ----

    def _rebuild(self, world, model, omega: float, my_id: int) -> None:
        """Enumerate every (src ∈ all planets, tgt ∈ non-mine non-comet,
        launch_tick ∈ [0, MAX_LAUNCH_TICK)). For each, compute the
        trajectory and store iff viable.
        """
        self._entries = {}
        stats = {
            "raw": 0, "viable": 0,
            "dropped_aim_fail": 0, "dropped_eta_oor": 0,
            "dropped_owner_at_arrival_mine": 0, "dropped_trajectory_bounce": 0,
            "dropped_unaffordable": 0, "dropped_arrival_oor": 0,
        }

        step_now = int(getattr(world, "step", 0) or 0)
        comet_ids = set(world.comet_ids) if world.comet_ids else set()

        # Sources: every planet — we precompute even for opp/neutral planets
        # because capture chains may make us the owner of any planet by
        # mid-opening. opening_search.py decides which sources to ACTUALLY
        # use per-turn based on current ownership + chain reachability.
        sources = list(world.planets_by_id.values())
        # Targets: everything that's not us and not a comet at game start.
        # (Comets are non-attackable: their `path_index` semantics make
        # capture-by-arrival unstable.)
        targets = [
            p for p in world.planets_by_id.values()
            if int(p.id) not in comet_ids
        ]

        max_T = self._max_launch_tick
        arrival_window = max_T + self._arrival_buffer

        for src in sources:
            src_initial_ships = int(src.ships)
            src_prod = int(src.production)
            for tgt in targets:
                if int(tgt.id) == int(src.id):
                    continue
                for launch_tick in range(0, max_T):
                    stats["raw"] += 1
                    entry = self._compute_one(
                        src, tgt, launch_tick, omega, world, model, my_id,
                        src_initial_ships, src_prod,
                        step_now=step_now,
                        arrival_window=arrival_window,
                        stats=stats,
                    )
                    if entry is not None:
                        self._entries[
                            (int(src.id), int(tgt.id), int(launch_tick))
                        ] = entry
                        stats["viable"] += 1

        self._stats = stats

    def _compute_one(self, src, tgt, launch_tick: int, omega: float,
                     world, model, my_id: int,
                     src_initial_ships: int, src_prod: int,
                     *, step_now: int, arrival_window: int,
                     stats: dict) -> Optional[TrajectoryEntry]:
        """Compute one (src, tgt, launch_tick) entry; return None on any
        drop reason and increment the matching stat.

        Strategy: iterate ships_est ↔ eta_flight to a fixed point. The
        estimated ship count affects fleet_speed → eta → arrival →
        predicted garrison → needed ship count. Loop until ships_est
        stops changing (capped at 5 iterations; 2-3 typically suffices,
        but cycles can occur at the floor/ceil boundary of ceil(gar)+1
        which 2-pass misses). After convergence, the STORED ships_needed
        is the converged ships_est and eta_flight is aim_and_eta's
        result for that ships count — so a direct call with the stored
        ships_needed reproduces the stored eta_flight byte-exact.
        """
        # Initial ship estimate. Use target's CURRENT ships + 1 as the
        # starting guess.
        ships_est = max(DEFAULT_DEFENDER_GUARD, int(tgt.ships) + 1)
        eta_flight: Optional[int] = None
        angle: Optional[float] = None
        gar_at_arr: float = 0.0
        owner_at_arr: int = -1

        MAX_REFINE_PASSES = 5
        for _refine_pass in range(MAX_REFINE_PASSES):
            try:
                res = aim_and_eta(src, tgt, ships_est, omega,
                                  wait_N=int(launch_tick))
            except Exception:
                res = None
            if res is None:
                stats["dropped_aim_fail"] += 1
                return None
            angle, eta_flight = res
            if eta_flight is None or eta_flight <= 0 or eta_flight > arrival_window:
                stats["dropped_eta_oor"] += 1
                return None

            arrival_total = int(launch_tick) + int(eta_flight)
            base_arrivals = list(model.ledger.get(int(tgt.id), []))
            try:
                owner_at_arr, gar_at_arr = predict_garrison_at(
                    tgt, arrival_total, base_arrivals,
                )
            except Exception:
                stats["dropped_aim_fail"] += 1
                return None

            if int(owner_at_arr) == int(my_id):
                # Already ours at arrival — not a capture; skip.
                stats["dropped_owner_at_arrival_mine"] += 1
                return None

            needed = max(1, int(math.ceil(float(gar_at_arr))) + 1)
            if needed == ships_est:
                # Converged: stored (eta_flight, ships_needed) is a fixed
                # point of aim_and_eta + predict_garrison_at.
                break
            ships_est = needed
        else:
            # Did not converge within MAX_REFINE_PASSES. May indicate a
            # 2-cycle at the ceil(gar)+1 boundary. Drop the entry —
            # better than storing a non-fixed-point that fails parity.
            stats.setdefault("dropped_no_converge", 0)
            stats["dropped_no_converge"] += 1
            return None

        if eta_flight is None or angle is None:
            stats["dropped_aim_fail"] += 1
            return None

        # ships_needed == ships_est (loop broke via `needed == ships_est`).
        ships_needed = int(ships_est)

        # Affordability: source must have ships_needed + DEFENDER_GUARD
        # available at launch_tick. budget = initial + prod × wait. Note:
        # this is an UPPER-BOUND check — actual feasibility depends on
        # prior fires from the same source eating budget. opening_search
        # enforces the per-tick budget constraint at MILP time. Here we
        # just drop trajectories that are NEVER affordable from a fresh
        # source budget.
        src_budget = src_initial_ships + src_prod * max(0, int(launch_tick))
        if ships_needed + DEFAULT_DEFENDER_GUARD > src_budget:
            stats["dropped_unaffordable"] += 1
            return None

        # Trajectory feasibility — collision check against orbital
        # geometry at fire-time. predict_fleet_fate advances positions by
        # wait_N=launch_tick orbital ticks so we check actual fire-time
        # geometry.
        try:
            fate = predict_fleet_fate(
                src, tgt, float(angle), int(ships_needed), world,
                wait_N=int(launch_tick),
            )
        except Exception:
            stats["dropped_trajectory_bounce"] += 1
            return None
        if fate is None or getattr(fate, "outcome", "") != "target":
            stats["dropped_trajectory_bounce"] += 1
            return None
        if int(getattr(fate, "hit_planet_id", -1)) != int(tgt.id):
            stats["dropped_trajectory_bounce"] += 1
            return None

        arrival_abs = step_now + arrival_total
        if arrival_abs - step_now > arrival_window:
            stats["dropped_arrival_oor"] += 1
            return None

        return TrajectoryEntry(
            src_id=int(src.id),
            tgt_id=int(tgt.id),
            launch_tick=int(launch_tick),
            ships_needed=int(ships_needed),
            angle=float(angle),
            eta_flight=int(eta_flight),
            arrival_tick=int(arrival_abs),
            arrival_owner=int(owner_at_arr),
            arrival_garrison=float(gar_at_arr),
            src_budget_at_launch=int(src_budget),
        )


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors kinematic_table's pattern)
# ---------------------------------------------------------------------------


_DEFAULT = TrajectoryMatrix()


def clear() -> None:
    """Reset the module-level singleton (tests + legacy entry point)."""
    _DEFAULT.reset()


def begin_game(world, model, omega: float, my_id: int,
               *, max_launch_tick: Optional[int] = None,
               arrival_buffer: Optional[int] = None) -> bool:
    """Game-start precompute — call once per game, idempotent within
    a game (fingerprint detects new games)."""
    return _DEFAULT.begin_game(
        world, model, omega, my_id,
        max_launch_tick=max_launch_tick, arrival_buffer=arrival_buffer,
    )


def get(src_id: int, tgt_id: int, launch_tick: int
        ) -> Optional[TrajectoryEntry]:
    """O(1) lookup on the module-level singleton."""
    return _DEFAULT.get(src_id, tgt_id, launch_tick)


def iter_viable(*, src_id: Optional[int] = None,
                tgt_id: Optional[int] = None,
                launch_tick: Optional[int] = None
                ) -> Iterator[TrajectoryEntry]:
    """Iterate over viable entries optionally filtered."""
    return _DEFAULT.iter_viable(
        src_id=src_id, tgt_id=tgt_id, launch_tick=launch_tick,
    )


def get_default() -> TrajectoryMatrix:
    """Accessor for the module-level singleton — for callers that want
    to drive `begin_game(...)` from a stage."""
    return _DEFAULT
