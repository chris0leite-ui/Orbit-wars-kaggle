"""Cross-turn memory implementations.

Kaggle invokes `agent(obs, configuration)` once per turn — stateless
on the surface. Module-level singletons survive within one game
subprocess but reset between games. These `Memory` impls give
strategies a typed way to carry state across the 500 turns of a game
without poking module globals directly.

Step 7 lands the three highest-value impls; `WarmStartMemory` and
`OppModelMemory` (search subtree reuse and opp archetype posterior)
are reserved for a follow-up step.

- `JitCacheMemory` — caches `jax.jit`-compiled closures so the cold
  compile on turn 1 (~600 ms) amortises over turns 2-500.
- `MissionMemory` — persists strategy-level intent (e.g., 3-wave
  snipe plans) across turns; prunes stale missions whose target was
  captured, source was lost, or all waves have fired.
- `CompositeMemory` — combines the above; what the live agent uses.
- `_MEMORY_KIND` enum is just for `__repr__` brevity.

The `EmptyMemory` no-op baseline lives in `lib/foundation/memory.py`
alongside the `Memory` protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Optional

import numpy as np

from lib.foundation.actions import ActionSpec
from lib.foundation.memory import Memory
from lib.game.jax.jax_types import MAX_PLANETS


# ---------------------------------------------------------------------------
# JitCacheMemory — JAX JIT closure cache
# ---------------------------------------------------------------------------


class JitCacheMemory:
    """Across-turn cache for `jax.jit`-compiled closures.

    Usage:
        memory = JitCacheMemory()
        my_jit_fn = memory.get_or_compile(
            "candidate_scorer",
            builder=lambda: jax.jit(some_function, static_argnames=("K",)),
        )
        # First call: builds and caches.
        # Subsequent: cache hit; no recompile.

    The cache is a plain dict mapping `str` keys to JIT'd functions
    (or any computed value, really). `update` is a no-op because the
    JIT cache doesn't depend on game state — the same function works
    for any state of the same Pytree structure.
    """

    __slots__ = ("_cache",)

    def __init__(self, cache: Optional[dict[str, Any]] = None) -> None:
        self._cache: dict[str, Any] = dict(cache) if cache else {}

    def get_or_compile(self, key: str, builder: Callable[[], Any]) -> Any:
        """Return the cached value under `key`, computing it via
        `builder()` on first access. The builder is called at most
        once per `(memory_instance, key)`."""
        if key not in self._cache:
            self._cache[key] = builder()
        return self._cache[key]

    def has(self, key: str) -> bool:
        return key in self._cache

    def update(self, state: Any, action_taken: Any) -> "JitCacheMemory":
        """JIT cache is state-independent — return self (identity)."""
        return self

    def reset(self) -> "JitCacheMemory":
        """Clear the cache. Called between games."""
        return JitCacheMemory()

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return f"JitCacheMemory(entries={len(self._cache)})"


# ---------------------------------------------------------------------------
# MissionMemory — strategy-level intent persistence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommittedMission:
    """A multi-wave mission committed on a past turn.

    A mission is a SEQUENCE of `ActionSpec` launches with increasing
    `launch_turn`. `waves_fired` tracks how many waves have already
    been launched; the next-to-fire is `waves[waves_fired]`.

    Fields:
        mission_id  — unique within the game (caller-assigned).
        seat        — which agent committed this (0..3).
        turn_committed — game step at which the mission was committed.
        target_planet_id — planet the mission is trying to capture /
                          influence.
        source_planet_id — primary source planet (for stale-source
                          pruning); waves themselves can have
                          per-wave source via `wave.from_planet_id`.
        waves       — tuple of `ActionSpec`, ordered by `launch_turn`.
                      `wave.launch_turn` is RELATIVE to
                      `turn_committed`.
        waves_fired — number of waves already launched.
    """

    mission_id: str
    seat: int
    turn_committed: int
    target_planet_id: int
    source_planet_id: int
    waves: tuple[ActionSpec, ...]
    waves_fired: int = 0

    @property
    def waves_remaining(self) -> int:
        return max(0, len(self.waves) - self.waves_fired)

    @property
    def is_complete(self) -> bool:
        return self.waves_fired >= len(self.waves)

    def next_wave(self) -> Optional[ActionSpec]:
        """Next wave to fire, or None if mission complete."""
        if self.is_complete:
            return None
        return self.waves[self.waves_fired]

    def absolute_launch_turn(self, wave_idx: int) -> int:
        """Game step at which wave `wave_idx` should fire."""
        return self.turn_committed + self.waves[wave_idx].launch_turn

    def with_fired(self, n: int = 1) -> "CommittedMission":
        return replace(self, waves_fired=self.waves_fired + n)


class MissionMemory:
    """Persists committed multi-turn missions across turns.

    Opt-in per strategy. Greedy strategies pass `EmptyMemory()` and
    re-propose every turn; planning strategies (`heuristic_v351`,
    `search_brute` in Step 8) carry plans here.

    Lifecycle:
        memory.commit(mission)
            → returns new memory with the mission added.
        memory.mark_wave_fired(mission_id)
            → returns new memory with the mission's `waves_fired`
              incremented (caller calls this AFTER launching).
        memory.update(state, action_taken)
            → returns new memory with missions PRUNED:
                * target planet no longer alive (game-ended for it)
                * target captured by `seat` (mission accomplished)
                * source planet no longer owned by `seat` (no fuel)
                * all waves fired AND grace period elapsed
              `action_taken` is unused here — let `mark_wave_fired`
              handle launch accounting explicitly.

    `update` does NOT auto-increment turn — that's the caller's
    responsibility via `advance_turn()`. (Decouples turn-counting
    from state-derived pruning; tests can set the turn directly.)
    """

    __slots__ = ("_missions", "_current_turn")

    def __init__(
        self,
        missions: Iterable[CommittedMission] = (),
        current_turn: int = 0,
    ) -> None:
        self._missions: tuple[CommittedMission, ...] = tuple(missions)
        self._current_turn: int = int(current_turn)

    # -- Read accessors ---------------------------------------------------

    @property
    def missions(self) -> tuple[CommittedMission, ...]:
        return self._missions

    @property
    def current_turn(self) -> int:
        return self._current_turn

    def get(self, mission_id: str) -> Optional[CommittedMission]:
        for m in self._missions:
            if m.mission_id == mission_id:
                return m
        return None

    def waves_to_fire(self, seat: int, *, at_turn: Optional[int] = None) -> list[ActionSpec]:
        """List of `ActionSpec` whose absolute launch turn equals
        `at_turn` (default: `current_turn`) and belong to `seat`.

        Strategies call this at the top of `emit()` to read the
        pre-committed waves for the current turn before proposing
        new ones."""
        if at_turn is None:
            at_turn = self._current_turn
        out: list[ActionSpec] = []
        for m in self._missions:
            if m.seat != seat:
                continue
            if m.is_complete:
                continue
            nw = m.next_wave()
            if nw is None:
                continue
            if m.absolute_launch_turn(m.waves_fired) == at_turn:
                out.append(nw)
        return out

    # -- Write transitions (return new MissionMemory) ---------------------

    def commit(self, mission: CommittedMission) -> "MissionMemory":
        """Add a new mission. Returns a new MissionMemory."""
        if any(m.mission_id == mission.mission_id for m in self._missions):
            raise ValueError(
                f"MissionMemory.commit: duplicate mission_id "
                f"{mission.mission_id!r}"
            )
        return MissionMemory(
            missions=self._missions + (mission,),
            current_turn=self._current_turn,
        )

    def mark_wave_fired(self, mission_id: str, n: int = 1) -> "MissionMemory":
        """Increment `waves_fired` for the named mission. Returns a
        new MissionMemory."""
        new_missions = tuple(
            m.with_fired(n) if m.mission_id == mission_id else m
            for m in self._missions
        )
        return MissionMemory(
            missions=new_missions,
            current_turn=self._current_turn,
        )

    def advance_turn(self, new_turn: Optional[int] = None) -> "MissionMemory":
        """Bump the internal turn counter. If `new_turn` is None,
        increments by 1; otherwise sets to `new_turn`. Returns a new
        MissionMemory."""
        if new_turn is None:
            new_turn = self._current_turn + 1
        return MissionMemory(
            missions=self._missions, current_turn=int(new_turn),
        )

    # -- State-driven pruning ---------------------------------------------

    def update(self, state: Any, action_taken: Any) -> "MissionMemory":
        """Prune missions whose target was captured by `seat`, target
        is gone (off-board / dead), source planet is no longer owned
        by `seat`, or all waves have fired (mission complete).

        `state` is a JAX `GameState`. `action_taken` is unused.
        """
        new_missions: list[CommittedMission] = []
        for m in self._missions:
            if _should_prune_mission(m, state):
                continue
            new_missions.append(m)
        return MissionMemory(
            missions=tuple(new_missions),
            current_turn=self._current_turn,
        )

    def reset(self) -> "MissionMemory":
        """Clear all missions and reset turn counter. Called between
        games."""
        return MissionMemory(missions=(), current_turn=0)

    def __len__(self) -> int:
        return len(self._missions)

    def __repr__(self) -> str:
        return (
            f"MissionMemory(missions={len(self._missions)}, "
            f"turn={self._current_turn})"
        )


def _should_prune_mission(m: CommittedMission, state: Any) -> bool:
    """Return True if mission `m` should be dropped this turn."""
    if m.is_complete:
        return True

    # Target planet status.
    target_owner = _planet_owner(state, m.target_planet_id)
    if target_owner is None:
        # Target planet no longer alive (off-board comet, etc.)
        return True
    if target_owner == m.seat:
        # We took it — mission accomplished.
        return True

    # Source planet status.
    source_owner = _planet_owner(state, m.source_planet_id)
    if source_owner is None:
        return True  # source destroyed
    if source_owner != m.seat:
        return True  # we lost the source — no fuel left to fire waves

    return False


def _planet_owner(state: Any, planet_id: int) -> Optional[int]:
    """Return owner of `planet_id` in JAX state, or None if not alive.
    Returns -1 for neutral planets."""
    try:
        alive = np.asarray(state.planets_alive)
        ids = np.asarray(state.planets_id)
        owner = np.asarray(state.planets_owner)
    except AttributeError:
        return None
    for i in range(MAX_PLANETS):
        if alive[i] and int(ids[i]) == planet_id:
            return int(owner[i])
    return None


# ---------------------------------------------------------------------------
# CompositeMemory — combines all sub-memories
# ---------------------------------------------------------------------------


@dataclass
class CompositeMemory:
    """Aggregates the per-aspect Memory impls into one object the
    live agent threads through `Strategy.emit`.

    Fields:
        jit_cache  — `JitCacheMemory` for `jax.jit` closures.
        missions   — `MissionMemory` for strategy-level intent.
        scratch    — free-form dict for ad-hoc per-game scratch
                     (turn-rate stats, observed-action history, etc.);
                     persists across turns but is opaque to the
                     framework.

    `update`/`reset` thread to each sub-memory.
    """

    jit_cache: JitCacheMemory = field(default_factory=JitCacheMemory)
    missions: MissionMemory = field(default_factory=MissionMemory)
    scratch: dict = field(default_factory=dict)

    def update(self, state: Any, action_taken: Any) -> "CompositeMemory":
        return CompositeMemory(
            jit_cache=self.jit_cache.update(state, action_taken),
            missions=self.missions.update(state, action_taken),
            scratch=self.scratch,
        )

    def reset(self) -> "CompositeMemory":
        return CompositeMemory(
            jit_cache=self.jit_cache.reset(),
            missions=self.missions.reset(),
            scratch={},
        )

    def with_missions(self, missions: MissionMemory) -> "CompositeMemory":
        """Replace the missions sub-memory (e.g., after commit /
        mark_wave_fired). Returns a new CompositeMemory."""
        return CompositeMemory(
            jit_cache=self.jit_cache,
            missions=missions,
            scratch=self.scratch,
        )

    def __repr__(self) -> str:
        return (
            f"CompositeMemory(jit_entries={len(self.jit_cache)}, "
            f"missions={len(self.missions)}, "
            f"scratch_keys={list(self.scratch.keys())})"
        )
