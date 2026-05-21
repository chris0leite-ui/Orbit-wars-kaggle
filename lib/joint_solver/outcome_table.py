"""Per-planet subset-enumeration outcome table.

Linearizes the non-convex combat resolution rule (largest-vs-second-largest)
for the joint LP. For each contested planet, enumerates all 2^k subsets
of candidate arrivals (k ≤ MAX_ENUMERATION_BITS), walks the closed-form
timeline via `lib.combat.resolve_arrivals`, and records the resulting
`(owner_T, ships_T, production-stream-by-owner)` at the planning horizon.

The LP then encodes "z_p = 1 iff me at horizon" via constraints over the
enumeration: for each subset S that yields owner_T == my_id,

    z_p ≥ Σ_{i ∈ S} x_i − |S| + 1

(standard OR-over-enumerated-subsets trick). Fixed arrivals (already
in-flight fleets, which the LP cannot un-fire) are always included.

Timeline semantics match `lib.world_model.simulate_planet_timeline`:
at each tick t in [1, horizon]
  1. If owner != neutral, garrison += production.
  2. Resolve same-tick arrivals via resolve_arrivals.
This matches the env interpreter's per-tick order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

from lib.combat import resolve_arrivals


# 2^6 = 64 subsets per planet — fits in a per-turn budget comfortably.
# >6 contesters at a single planet is rare (contested capitals only).
# Caller MUST route those planets to per-planet sub-MILP fallback.
MAX_ENUMERATION_BITS = 6


@dataclass(frozen=True)
class Arrival:
    """A single arrival event at a planet.

    `column_id` identifies the LP decision variable that gates this
    arrival's inclusion:
      - None  ⇒ fixed (in-flight fleet that already exists; always in the set).
      - int   ⇒ candidate (LP decides via x_{column_id} ∈ {0,1}).
    """
    eta: int
    owner: int
    ships: int
    column_id: Optional[int] = None


@dataclass(frozen=True)
class OutcomeRow:
    """Outcome of simulating one specific arrival set to the horizon."""
    subset: tuple[int, ...]
    owner_T: int
    ships_T: float
    prod_stream: dict[int, int] = field(hash=False)


def _simulate_one(
    initial_owner: int,
    initial_ships: float,
    production: int,
    horizon: int,
    arrivals: list[Arrival],
    subset_key: tuple[int, ...],
) -> OutcomeRow:
    """Walk the timeline for ONE specific arrival set. O(horizon)."""
    by_turn: dict[int, list[tuple[int, int]]] = {}
    for a in arrivals:
        if a.ships <= 0:
            continue
        bucket = max(1, int(a.eta))
        if bucket > horizon:
            continue
        by_turn.setdefault(bucket, []).append((int(a.owner), int(a.ships)))

    owner = int(initial_owner)
    garrison = float(initial_ships)
    prod_stream: dict[int, int] = {}

    for t in range(1, horizon + 1):
        if owner != -1:
            garrison += production
            prod_stream[owner] = prod_stream.get(owner, 0) + int(production)
        group = by_turn.get(t)
        if group:
            owner, garrison = resolve_arrivals(owner, garrison, group)

    return OutcomeRow(
        subset=subset_key,
        owner_T=owner,
        ships_T=max(0.0, garrison),
        prod_stream=prod_stream,
    )


def enumerate_outcomes(
    *,
    initial_owner: int,
    initial_ships: float,
    production: int,
    horizon: int,
    fixed_arrivals: list[Arrival],
    candidate_arrivals: list[Arrival],
) -> dict[tuple[int, ...], OutcomeRow]:
    """For each subset S ⊆ candidate_arrivals, simulate (fixed ∪ S) and
    return OutcomeRow keyed by sorted-tuple of S's column_ids.

    The empty subset key () corresponds to "no candidates fire".
    The full subset corresponds to "every candidate fires".

    Validates:
      - fixed arrivals have column_id is None
      - candidate arrivals have unique, non-None column_ids
      - len(candidate_arrivals) <= MAX_ENUMERATION_BITS
    """
    if len(candidate_arrivals) > MAX_ENUMERATION_BITS:
        raise ValueError(
            f"candidate_arrivals exceeds enumeration budget "
            f"({len(candidate_arrivals)} > {MAX_ENUMERATION_BITS}); "
            f"route to per-planet MILP fallback."
        )
    for f in fixed_arrivals:
        if f.column_id is not None:
            raise ValueError(
                f"fixed_arrival has column_id={f.column_id}; expected None."
            )
    cand_ids = [c.column_id for c in candidate_arrivals]
    if any(cid is None for cid in cand_ids):
        raise ValueError("candidate_arrivals must all have column_id != None.")
    if len(set(cand_ids)) != len(cand_ids):
        raise ValueError("candidate_arrivals have duplicate column_ids.")

    rows: dict[tuple[int, ...], OutcomeRow] = {}
    n = len(candidate_arrivals)
    indexed = list(enumerate(candidate_arrivals))
    for r in range(n + 1):
        for combo in combinations(indexed, r):
            chosen = [a for _, a in combo]
            key = tuple(sorted(int(a.column_id) for a in chosen))
            rows[key] = _simulate_one(
                initial_owner, initial_ships, production, horizon,
                fixed_arrivals + chosen, key,
            )
    return rows


def winning_subsets(table: dict[tuple[int, ...], OutcomeRow],
                    my_id: int) -> list[tuple[int, ...]]:
    """Subsets where I own the planet at the horizon."""
    return [s for s, row in table.items() if row.owner_T == my_id]


def empty_subset_outcome(table: dict[tuple[int, ...], OutcomeRow]) -> OutcomeRow:
    """The "no candidates fire" baseline outcome."""
    return table[()]
