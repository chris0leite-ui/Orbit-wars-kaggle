"""Joint allocator — replaces lib/planner.py:settle_plan's per-source greedy.

Two methods:

1. `allocate_greedy_multi(missions, world, sense, posture, model)`
   Global score-ordered greedy with multi-launch-per-source AND the
   same-turn arrival ledger from settle_plan. Captures the top-10
   "multi-launch turn" signal (48% of top-10 turns vs 38% midpack).

2. `allocate_lp(missions, world, sense, posture, model)`
   Continuous LP relaxation of (0, m.ships)-bounded variables with
   per-source capacity constraints. Solved via scipy.optimize.linprog
   (HiGHS). Rounds down + drops sub-capture missions, then applies the
   ledger check to drop redundant arrivals. Falls back to greedy on
   any failure (infeasible, timeout, no scipy).

The two methods produce similar outputs on most turns — LP is at most
~5pp better in expectation. We default to LP per the plan; greedy is
the documented fallback and ablation target.

Posture-aware source budgets
----------------------------
- OPENING: reserve 0 (empty home planets — top-10 garrison ≈ 10.6 vs midpack 22.0)
- DEFEND:  reserve = threat_budget(pid) for threatened planets
- BREAK:   reserve 0 (commit force)
- EXPAND:  reserve = MIN_GARRISON_EXPAND (small floor; never empty everything mid-game)
"""

from __future__ import annotations

import time
from collections import defaultdict

from lib.intent import Intent, World
from lib.mission import Mission
from lib.world_model import WorldModel

from lib.geo.posture import Posture
from lib.geo.sense import SenseState


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

MIN_GARRISON_EXPAND = 0    # empty everything — top-10 signal (garrison 10.6 vs midpack 22)
LP_WALLCLOCK_CAP_MS = 200  # hard cap on LP solve time; fall back to greedy beyond


# ---------------------------------------------------------------------------
# Source budget — posture-dependent
# ---------------------------------------------------------------------------


def _source_budget(world: World, sense: SenseState, posture: Posture) -> dict[int, int]:
    """Available ships per OUR source planet, after posture reserve."""
    budget: dict[int, int] = {}
    for p in world.planets_by_id.values():
        if p.owner != world.my_id:
            continue
        garrison = int(p.ships)
        if posture is Posture.OPENING or posture is Posture.BREAK:
            reserve = 0
        elif posture is Posture.DEFEND:
            reserve = int(sense.threat_budget.get(p.id, 0))
        else:  # EXPAND
            reserve = MIN_GARRISON_EXPAND
        budget[p.id] = max(0, garrison - reserve)
    return budget


# ---------------------------------------------------------------------------
# Ledger-aware commit check (mirrors settle_plan's logic)
# ---------------------------------------------------------------------------


def _ledger_admits(
    m: Mission,
    pending: dict[int, list[tuple[int, int]]],
    model: WorldModel,
) -> bool:
    """True if mission m can be committed without redundancy.

    Matches lib/planner.py:settle_plan lines 95-110.
    """
    already = sum(s for (e, s) in pending[m.target_id] if e <= m.eta)
    pred_enemy = model.ships_at(m.target_id, m.eta)
    if pred_enemy is None:
        pred_enemy = 0.0
    return already < pred_enemy + 1


def _commit(m: Mission, pending: dict[int, list[tuple[int, int]]]) -> None:
    pending[m.target_id].append((m.eta, m.ships))


# ---------------------------------------------------------------------------
# Greedy multi-launch allocator (always available, used as LP fallback)
# ---------------------------------------------------------------------------


def allocate_greedy_multi(
    missions: list[Mission],
    world: World,
    sense: SenseState,
    posture: Posture,
    model: WorldModel,
) -> list[Intent]:
    """Global score-ordered greedy. Multi-launch-per-source allowed."""
    if not missions:
        return []
    budget = _source_budget(world, sense, posture)
    sorted_missions = sorted(missions, key=lambda m: -m.score)
    pending: dict[int, list[tuple[int, int]]] = defaultdict(list)
    chosen: list[Mission] = []
    for m in sorted_missions:
        if budget.get(m.src_id, 0) < m.ships:
            continue
        if not _ledger_admits(m, pending, model):
            continue
        chosen.append(m)
        _commit(m, pending)
        budget[m.src_id] -= m.ships
    return [m.to_intent() for m in chosen]


# ---------------------------------------------------------------------------
# LP allocator — scipy.optimize.linprog (HiGHS)
# ---------------------------------------------------------------------------


def allocate_lp(
    missions: list[Mission],
    world: World,
    sense: SenseState,
    posture: Posture,
    model: WorldModel,
) -> list[Intent]:
    """Solve the LP relaxation; fall back to greedy on any failure.

    Variables: x_m ∈ [0, m.ships] (continuous) for each mission.
    Source caps: Σ_{m: src=s} x_m ≤ budget(s).
    Objective: maximize Σ (score_m / m.ships) × x_m.

    The relaxation of a 0-1 knapsack has at most one fractional variable
    per binding capacity constraint. After solving, we round x_m down
    to int and drop missions where the rounded ships fall below the
    mission's required floor (m.ships). The remaining fully-funded
    missions are run through the ledger check to drop redundant
    arrivals.
    """
    if not missions:
        return []
    try:
        import numpy as np
        from scipy.optimize import linprog
    except ImportError:
        return allocate_greedy_multi(missions, world, sense, posture, model)

    budget = _source_budget(world, sense, posture)
    # Only keep missions whose source has any budget AND m.ships >= 1.
    feasible = [m for m in missions if budget.get(m.src_id, 0) >= 1 and m.ships >= 1]
    if not feasible:
        return []

    n = len(feasible)
    # Objective: minimise -score_per_ship · x.
    c = np.array([-(m.score / max(1.0, float(m.ships))) for m in feasible], dtype=float)

    # Variable bounds: (0, m.ships).
    bounds = [(0.0, float(m.ships)) for m in feasible]

    # Source capacity constraints: one row per distinct source.
    src_ids = sorted({m.src_id for m in feasible})
    src_row: dict[int, int] = {s: i for i, s in enumerate(src_ids)}
    A_ub = np.zeros((len(src_ids), n), dtype=float)
    b_ub = np.zeros(len(src_ids), dtype=float)
    for i, m in enumerate(feasible):
        A_ub[src_row[m.src_id], i] = 1.0
    for s, i in src_row.items():
        b_ub[i] = float(budget.get(s, 0))

    t0 = time.perf_counter()
    try:
        res = linprog(
            c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
            method="highs",
            options={"time_limit": LP_WALLCLOCK_CAP_MS / 1000.0},
        )
    except Exception:
        return allocate_greedy_multi(missions, world, sense, posture, model)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if elapsed_ms > LP_WALLCLOCK_CAP_MS or not res.success or res.x is None:
        return allocate_greedy_multi(missions, world, sense, posture, model)

    # Round x_m down and drop missions that fall below their floor.
    pending: dict[int, list[tuple[int, int]]] = defaultdict(list)
    src_used: dict[int, int] = defaultdict(int)
    # Iterate by LP value descending so higher-scoring missions get the
    # source budget first when rounding creates contention.
    ordered = sorted(
        ((float(res.x[i]), feasible[i]) for i in range(n)),
        key=lambda t: -t[0],
    )
    chosen: list[Mission] = []
    for val, m in ordered:
        send = int(val + 1e-6)
        if send < m.ships:
            continue
        send = m.ships  # commit to the mission's required floor exactly
        if src_used[m.src_id] + send > budget.get(m.src_id, 0):
            continue
        if not _ledger_admits(m, pending, model):
            continue
        chosen.append(m)
        _commit(m, pending)
        src_used[m.src_id] += send
    return [m.to_intent() for m in chosen]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def allocate(
    missions: list[Mission],
    world: World,
    sense: SenseState,
    posture: Posture,
    model: WorldModel,
    method: str = "lp",
) -> list[Intent]:
    """Dispatch on `method` ∈ {'lp', 'greedy'}; falls back to greedy on LP fail."""
    if method == "greedy":
        return allocate_greedy_multi(missions, world, sense, posture, model)
    return allocate_lp(missions, world, sense, posture, model)
