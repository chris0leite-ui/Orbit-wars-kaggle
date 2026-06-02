"""Per-turn kinematic precomputation table for Orbit Wars.

Lifts the inner position-computation loop of `lib.trajectory.predict_fleet_fate`
(and every other call site of `lib.orbit.predict_relative`) from "called
once per (fleet, step, planet)" to "computed once per turn, looked up
afterwards". 100% bit-parity with `predict_relative` by construction:
each entry is materialised by calling `predict_relative(p_tuple, omega, lead)`
with the SAME planet tuple the inline call site would have used.

Design (Layer 1 only — Phase α of the plan
/root/.claude/plans/do-it-thoroughly-consider-tingly-fox.md):

- Per-turn rebuild from `world.planets_by_id` at the start of each turn.
  This is the bit-parity-safe choice: callers that today call
  `predict_relative(world.planets_by_id[pid], world.omega, lead)` get the
  same float-for-float output via `table.lookup_relative(pid, lead)`.
  A game-anchor design (build once at turn 0, advance) would compose
  `predict_relative` across turns and could drift by ULPs through
  `atan2` round-trips. We pay ~2-3 ms per turn to rebuild and dodge
  the entire FP-composition risk surface.

- Static planets stored as a single constant + flag; orbital planets get
  a pre-allocated tuple-list across the configured `max_lead` window;
  comets are sourced from `obs["comets"]` path arrays with the existing
  `(-1e6, -1e6)` off-board sentinel reused from `lib/trajectory.py` so
  the `1daec97` expiry guard fires unchanged.

- Module-level singleton + thin function wrappers, modelled exactly on
  `lib/pipeline/pending_schedule.py`. Fingerprint-driven reset detects
  new turns (and new games, where `step` drops back to 0).

This module is NOT yet wired into any caller (per the phased plan;
Phase β / γ swap call sites). It can be exercised today via the unit
pin tests in `tests/test_kinematic_table_parity.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import math

from lib.geometry import ROTATION_RADIUS_LIMIT
from lib.orbit import is_orbiting, predict_relative


# Sentinel reused from `lib/trajectory.py:127`; we DO NOT import to avoid
# a circular dependency with the call site we'll later modify.
OFF_BOARD: tuple[float, float] = (-1e6, -1e6)

# Default lookup window. predict_fleet_fate uses `max_steps=200` as the
# ray-cast horizon; callers may also pass wait_N (fire-offset) up to ~50.
# 500 gives generous headroom (covers wait_N up to ~300 + max_steps=200)
# at ~250 KB total memory cost — negligible. Phase γ relies on this
# default being large enough that the table covers any predict_fleet_fate
# call without falling through to the slow path.
DEFAULT_MAX_LEAD: int = 500


# ---------------------------------------------------------------------------
# Class — per-instance container; tests instantiate directly.
# ---------------------------------------------------------------------------


@dataclass
class _PlanetEntry:
    """Per-planet position cache for one turn."""

    pid: int
    kind: str  # "static" | "orbital" | "comet"
    # For static: positions == None, static_pos holds the constant.
    static_pos: Optional[tuple[float, float]] = None
    # For orbital + comet: positions[t] is (x, y) at `lead = t` from
    # current obs; len(positions) == max_lead + 1.
    positions: Optional[list[tuple[float, float]]] = None
    # For comets only: the raw path + path_index from obs["comets"],
    # surfaced via `comet_paths_view` for callers replacing
    # `lib.world_model._comet_paths_by_id`.
    comet_path: Optional[list] = None
    comet_path_index: Optional[int] = None


class KinematicTable:
    """Per-instance kinematic position cache.

    One instance is held as a module-level singleton; tests can create
    isolated instances for parity assertions. Lifecycle:

        table.begin_turn(world)           # rebuild from current obs
        table.lookup_relative(pid, lead)  # (x, y) at `lead` ticks ahead
        table.window(pids, off, n)        # dict of position lists

    `begin_turn` is idempotent within a turn: if the (step, omega,
    planets) fingerprint matches the last build, no rebuild happens.
    """

    def __init__(self, max_lead: int = DEFAULT_MAX_LEAD) -> None:
        self._entries: dict[int, _PlanetEntry] = {}
        self._fingerprint: Any = None
        self._omega: float = 0.0
        self._step: int = -1
        self._max_lead: int = int(max_lead)

    # ---- lifecycle ----

    def reset(self) -> None:
        """Drop all state. Tests use this; production callers shouldn't."""
        self._entries = {}
        self._fingerprint = None
        self._omega = 0.0
        self._step = -1

    def begin_turn(self, world, *, max_lead: Optional[int] = None) -> bool:
        """Rebuild the table from `world` if the turn fingerprint changed.

        Returns True iff a rebuild fired (caller can log this for
        observability). Fingerprint:

            (step, omega, n_planets, sorted-tuple of (pid, id(planet_obj)))

        The `id(planet_obj)` term cheaply detects the per-turn obs
        rebuild — `World.from_obs` constructs fresh `Planet` instances
        every turn, so identities never repeat. On game boundary
        (`step` drops to 0 with different planet ids), fingerprint
        differs and we wipe.
        """
        if max_lead is not None and int(max_lead) != self._max_lead:
            # max_lead change forces rebuild even if obs is unchanged.
            self._max_lead = int(max_lead)
            self._fingerprint = None

        new_fp = self._build_fingerprint(world)
        if self._fingerprint == new_fp:
            return False
        self._rebuild(world)
        self._fingerprint = new_fp
        return True

    @staticmethod
    def _build_fingerprint(world) -> tuple:
        planets = world.planets_by_id
        pid_ids = tuple(sorted((int(pid), id(p)) for pid, p in planets.items()))
        return (int(world.step), float(world.omega), len(planets), pid_ids)

    def _rebuild(self, world) -> None:
        """Materialise per-planet position lists from `world`.

        For orbital planets, calls `predict_relative` per lead-tick — the
        SAME function `predict_fleet_fate`'s inner loop calls, with the
        SAME planet tuple shape, so bit-parity is guaranteed by
        construction. Static planets store a single constant; comets
        consult the obs path array with the off-board sentinel.
        """
        self._entries = {}
        self._omega = float(world.omega)
        self._step = int(world.step)
        comet_paths = _extract_comet_paths(world)
        max_lead = self._max_lead

        for pid, p in world.planets_by_id.items():
            pid_i = int(pid)
            if pid_i in comet_paths:
                path, path_index = comet_paths[pid_i]
                positions: list[tuple[float, float]] = []
                for t in range(max_lead + 1):
                    path_t = int(path_index) + t
                    if 0 <= path_t < len(path):
                        pt = path[path_t]
                        positions.append((float(pt[0]), float(pt[1])))
                    else:
                        positions.append(OFF_BOARD)
                self._entries[pid_i] = _PlanetEntry(
                    pid=pid_i, kind="comet",
                    positions=positions,
                    comet_path=path,
                    comet_path_index=int(path_index),
                )
                continue

            p_tuple = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
            if is_orbiting(p_tuple) and self._omega != 0.0:
                # Orbital. Per-lead call to predict_relative — identical
                # arithmetic path to the inline call site (scalar
                # math.cos/sin under the hood). Bit-parity by
                # construction.
                positions = [
                    predict_relative(p_tuple, self._omega, t)
                    for t in range(max_lead + 1)
                ]
                self._entries[pid_i] = _PlanetEntry(
                    pid=pid_i, kind="orbital",
                    positions=positions,
                )
            else:
                # Static (outer planet OR omega == 0). Single constant.
                self._entries[pid_i] = _PlanetEntry(
                    pid=pid_i, kind="static",
                    static_pos=(float(p.x), float(p.y)),
                )

    # ---- queries ----

    def has(self, pid: int) -> bool:
        return int(pid) in self._entries

    @property
    def max_lead(self) -> int:
        """Maximum `lead` value the table can answer without falling
        through. Use this to gate calls that need a large window."""
        return self._max_lead

    @property
    def step(self) -> int:
        """The absolute env step the table was last built for."""
        return self._step

    @property
    def n_planets(self) -> int:
        return len(self._entries)

    def covers(self, pids, max_needed_lead: int) -> bool:
        """Return True iff every pid is in the table AND the table's
        max_lead is >= max_needed_lead. Cheap pre-flight for the Phase γ
        predict_fleet_fate swap — on False, caller falls through to the
        slow inline build."""
        if max_needed_lead > self._max_lead:
            return False
        entries = self._entries
        for pid in pids:
            if int(pid) not in entries:
                return False
        return True

    def kind(self, pid: int) -> Optional[str]:
        entry = self._entries.get(int(pid))
        return entry.kind if entry is not None else None

    def lookup_relative(self, pid: int, lead: int) -> tuple[float, float]:
        """Return (x, y) at `lead` ticks after current obs.

        Bit-identical to
        `predict_relative(world.planets_by_id[pid], world.omega, lead)`
        for orbital planets, and to `(p.x, p.y)` for static planets.
        For comets, returns `path[path_index + lead]` if in range, else
        `OFF_BOARD`. Raises KeyError if `pid` is not in the table.
        """
        entry = self._entries.get(int(pid))
        if entry is None:
            raise KeyError(f"kinematic_table: pid={pid} not in current obs")
        if entry.kind == "static":
            return entry.static_pos  # type: ignore[return-value]
        positions = entry.positions
        if positions is None:
            raise RuntimeError(f"kinematic_table: pid={pid} has no positions cache")
        n = len(positions)
        i = int(lead)
        if 0 <= i < n:
            return positions[i]
        # Beyond the precomputed window: for orbital, this is a usage
        # bug (caller asked past max_lead). For comet, this is a real
        # case — the path may extend beyond max_lead. We fall through
        # to a slow-path computation that matches the inline behaviour.
        if entry.kind == "comet":
            path = entry.comet_path
            path_index = entry.comet_path_index
            assert path is not None and path_index is not None
            path_t = int(path_index) + i
            if 0 <= path_t < len(path):
                pt = path[path_t]
                return (float(pt[0]), float(pt[1]))
            return OFF_BOARD
        # Orbital out-of-range: compute on demand (bit-parity preserved
        # because we use the same predict_relative call).
        # Caller is asking past max_lead — re-derive from omega + the
        # stored first-position. We don't store the source `p_tuple`,
        # so we reconstruct from positions[0] which is the obs-step
        # position. NOTE: positions[0] == predict_relative(p_tuple, omega, 0)
        # which for static omega=0 case returns (p.x, p.y) exactly, and
        # for orbital case may differ from the raw obs (p.x, p.y) by ULPs
        # because of the atan2(cos(.), sin(.)) round-trip. To preserve
        # bit-parity we instead raise — out-of-range orbital lookups are
        # a contract violation and we want them surfaced, not silently
        # answered with possibly-drifted floats.
        raise IndexError(
            f"kinematic_table: lead={i} past max_lead={n - 1} for orbital "
            f"pid={pid}; increase max_lead at begin_turn"
        )

    def window(
        self,
        pids: Iterable[int],
        start_offset: int,
        length: int,
    ) -> dict[int, list[tuple[float, float]]]:
        """Return {pid: [position at lead=start_offset+t for t in range(length)]}.

        Mirrors the `planet_positions` dict built inline at
        `lib/trajectory.py:137-159`. Use the SAME `length = max_steps + 1`
        the inline code uses; `start_offset = wait_N` for the predict-
        fleet-fate use case.
        """
        out: dict[int, list[tuple[float, float]]] = {}
        for pid in pids:
            pid_i = int(pid)
            entry = self._entries.get(pid_i)
            if entry is None:
                # Match the inline behaviour: missing planet → skip
                # (callers iterate over world.planets_by_id, so this
                # shouldn't fire in practice).
                continue
            if entry.kind == "static":
                pos = entry.static_pos  # type: ignore[assignment]
                out[pid_i] = [pos] * int(length)
                continue
            assert entry.positions is not None
            positions = entry.positions
            n = len(positions)
            row: list[tuple[float, float]] = []
            for t in range(int(length)):
                k = int(start_offset) + t
                if 0 <= k < n:
                    row.append(positions[k])
                elif entry.kind == "comet":
                    # Slow-path lookup past max_lead.
                    path = entry.comet_path
                    path_index = entry.comet_path_index
                    assert path is not None and path_index is not None
                    path_t = int(path_index) + k
                    if 0 <= path_t < len(path):
                        pt = path[path_t]
                        row.append((float(pt[0]), float(pt[1])))
                    else:
                        row.append(OFF_BOARD)
                else:
                    # Orbital past max_lead — see note in lookup_relative.
                    raise IndexError(
                        f"kinematic_table: start_offset+t={k} past "
                        f"max_lead={n - 1} for orbital pid={pid_i}"
                    )
            out[pid_i] = row
        return out

    def comet_paths_view(self) -> dict[int, tuple[list, int]]:
        """{pid: (path, path_index)} for every comet in the current obs.

        Schema-identical to `lib/world_model._comet_paths_by_id(world)`;
        the integration in Phase γ swaps that function's body to read
        from here when the table is populated.
        """
        out: dict[int, tuple[list, int]] = {}
        for pid, entry in self._entries.items():
            if entry.kind == "comet":
                assert entry.comet_path is not None and entry.comet_path_index is not None
                out[pid] = (entry.comet_path, entry.comet_path_index)
        return out

    # ---- diagnostics ----

    def stats(self) -> dict:
        kinds = {"static": 0, "orbital": 0, "comet": 0}
        for e in self._entries.values():
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        return {
            "n_planets": len(self._entries),
            "kinds": kinds,
            "step": self._step,
            "omega": self._omega,
            "max_lead": self._max_lead,
        }


def _extract_comet_paths(world) -> dict[int, tuple[list, int]]:
    """Inline copy of `lib.world_model._comet_paths_by_id`'s body.

    Duplicated here to avoid a circular import at module-load time;
    Phase γ reverses this by having `_comet_paths_by_id` consult the
    table when populated.
    """
    raw = getattr(world, "obs_raw", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        comets = raw.get("comets", [])
    else:
        comets = getattr(raw, "comets", [])
    out: dict[int, tuple[list, int]] = {}
    for group in comets or []:
        if hasattr(group, "keys"):
            planet_ids = list(group["planet_ids"])
            paths = list(group["paths"])
            path_index = int(group["path_index"])
        else:
            planet_ids = list(group.planet_ids)
            paths = list(group.paths)
            path_index = int(group.path_index)
        for idx, pid in enumerate(planet_ids):
            out[int(pid)] = (paths[idx], path_index)
    return out


# ---------------------------------------------------------------------------
# Per-World attachment — the safe path. Each `World` owns its own
# `KinematicTable` via a lazy attribute slot. Eliminates the singleton
# contamination that broke in-process A/B harnesses
# (audit/2026-05-29-postmortem-three-abs-headroom-empty.md).
# ---------------------------------------------------------------------------

_WORLD_ATTR = "_kinematic_table"


def attach(world, *, max_lead: Optional[int] = None) -> KinematicTable:
    """Get-or-create the `KinematicTable` owned by `world`.

    Stable across repeated calls — returns the SAME instance for the
    same `world`. Per-World ownership is what makes this safe in
    in-process A/B harnesses (`fast.py eval`, `quick_ab.py`, pytest):
    seat A's world and seat B's world are distinct Python objects, so
    they get distinct tables — no cross-seat contamination.

    Live Kaggle ladder is unchanged: one process per seat, one world
    per turn, one attached table.
    """
    table = getattr(world, _WORLD_ATTR, None)
    if table is None:
        table = KinematicTable(max_lead=max_lead or DEFAULT_MAX_LEAD)
        setattr(world, _WORLD_ATTR, table)
    return table


def for_world(world) -> Optional[KinematicTable]:
    """Return the World's attached `KinematicTable`, or `None` if the
    world hasn't been primed yet.

    Callers that get `None` MUST take the inline fallback path. Never
    fall back to the module-global singleton — that's the contamination
    source this refactor exists to eliminate.
    """
    return getattr(world, _WORLD_ATTR, None)


# ---------------------------------------------------------------------------
# Module-level singleton + thin function wrappers.
#
# LEGACY: the singleton path is incorrect in multi-seat in-process A/Bs
# (two seats share one `_DEFAULT`). Kept transitionally so the existing
# wiring test (`tests/test_kinematic_table_baseline_wiring.py`) and the
# `test_module_singleton_wraps_class` parity test stay green. New code
# should use `attach(world)` / `for_world(world)` instead.
# ---------------------------------------------------------------------------

_DEFAULT = KinematicTable()


def clear() -> None:
    """Reset the module-level singleton (tests + the legacy entry point)."""
    _DEFAULT.reset()


def begin_turn(world, *, max_lead: Optional[int] = None) -> bool:
    """Prime the per-World table; ALSO prime the legacy singleton
    transitionally so existing callers that still read `get_default()`
    continue to work.

    The per-World priming is the load-bearing call — it's what makes
    `for_world(world)` return a fresh, isolated table for this seat.
    The `_DEFAULT` prime is a back-compat shim and should go away once
    `tests/test_kinematic_table_baseline_wiring.py` migrates to read
    via `for_world(world)`.
    """
    table = attach(world, max_lead=max_lead)
    rebuilt = table.begin_turn(world, max_lead=max_lead)
    # TODO(rule-50): remove once test_kinematic_table_baseline_wiring
    # migrates off `get_default()`. The singleton path is incorrect in
    # multi-seat in-process A/Bs.
    _DEFAULT.begin_turn(world, max_lead=max_lead)
    return rebuilt


def lookup_relative(pid: int, lead: int) -> tuple[float, float]:
    return _DEFAULT.lookup_relative(pid, lead)


def window(
    pids: Iterable[int],
    start_offset: int,
    length: int,
) -> dict[int, list[tuple[float, float]]]:
    return _DEFAULT.window(pids, start_offset, length)


def comet_paths_view() -> dict[int, tuple[list, int]]:
    return _DEFAULT.comet_paths_view()


def get_default() -> KinematicTable:
    """Accessor for the module-level singleton.

    LEGACY: prefer `for_world(world)` in new code. The singleton is
    shared process-wide and contaminates in-process A/Bs.
    """
    return _DEFAULT
