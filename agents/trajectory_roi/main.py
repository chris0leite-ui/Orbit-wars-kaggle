"""trajectory_roi v2 — holistic analytical solver.

Five steps per turn, all closed-form, no rollouts:

  1. Project opp's launches via the SAME solver run from their seat
     (single-level, no nesting). Their projected launches become
     phantom fleets in our defender / threat ledger.
  2. Enumerate candidates over (target, launch_turn, allocation):
     - CAPTURE candidates for non-our planets — single-source with
       a wait-grid {0,1,2,5,10} plus a multi-source bundle at
       launch_turn=0.
     - DEFENSE candidates for our planets under threat — same shape
       but the "defenders to beat" includes opp's incoming attack.
  3. Score each candidate analytically: value = production held
     over (horizon - arrival).
  4. Joint-solve via 2-opt local search on a greedy seed.
  5. Emit launch_turn=0 actions only; re-plan next turn.

Compute envelope: ~50 ms / turn (vs 1000 ms env cap).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal

from lib.aim import aim_orbiting, flight_distance
from lib.fleet import speed as fleet_speed
from lib.trajectory_layer import World


# ---- tuneables ------------------------------------------------------------

HORIZON = 30
WAIT_GRID = (0, 1, 2, 5, 10)
TOP_CANDIDATES_PER_TARGET = 3
MIN_LAUNCH_SHIPS = 5
SAFETY_MARGIN = 1
MAX_ETA = 60
TWO_OPT_PASSES = 4  # bound the 2-opt loop

# Per-source minimum garrison reserved (anti-drain). Tuned empirically.
SOURCE_RESERVE = 0


# ---- core types -----------------------------------------------------------


@dataclass(frozen=True)
class Allocation:
    src_id: int
    ships: int


@dataclass(frozen=True)
class Candidate:
    flavor: Literal["capture", "defense"]
    target_id: int
    launch_turn: int
    arrival_turn: int
    allocations: tuple[Allocation, ...]
    aim_angles: tuple[float, ...]   # one per allocation
    value: float
    total_ships: int

    @property
    def roi(self) -> float:
        return self.value / (self.total_ships + 1.0)


@dataclass(frozen=True)
class PhantomFleet:
    """Projected opp launch from the mirror-opp pass."""
    owner: int        # opp_id
    src_id: int
    target_id: int
    arrival_turn: int  # turns from now
    ships: int


# ---- geometry / ETA primitives -------------------------------------------


def _aim_and_eta(src, target, ships: int, omega: float):
    """Compute (aim_angle, integer_eta) for `ships` from `src` to
    `target`. Returns None for invalid intercepts. Uses current
    positions (launch_turn=0 implicit); for launch_turn>0 we accept
    the approximation that geometry hasn't drifted meaningfully.
    """
    if ships < MIN_LAUNCH_SHIPS:
        return None
    v = fleet_speed(ships)
    if v <= 0:
        return None
    src_xy = (src.current_x, src.current_y)
    tgt_xy = (target.current_x, target.current_y)
    flight = flight_distance(src_xy, src.radius, tgt_xy, target.radius)
    if flight <= 0:
        return None
    if not getattr(target, "is_rotating", False) or omega == 0.0:
        eta = int(math.ceil(flight / v))
        if eta <= 0 or eta > MAX_ETA:
            return None
        ang = math.atan2(target.current_y - src.current_y,
                         target.current_x - src.current_x)
        return (ang, eta)
    # Orbital — fixed-point.
    target_tuple = (
        target.id, target.owner,
        target.current_x, target.current_y,
        target.radius, target.ships, target.production,
    )
    res = aim_orbiting(src_xy, src.radius, target_tuple, target.radius,
                       ships, omega)
    if res is None:
        return None
    angle, _arrival_xy, eta_f = res
    eta = int(math.ceil(eta_f))
    if eta <= 0 or eta > MAX_ETA:
        return None
    return (angle, eta)


def _fleet_eta(fleet, target_planet) -> int | None:
    """Static raycast eta for an in-flight fleet hitting a target."""
    fx, fy = fleet.current_x, fleet.current_y
    spd = fleet.speed
    if spd <= 0:
        return None
    tx, ty, tr = target_planet.current_x, target_planet.current_y, target_planet.radius
    dx, dy = tx - fx, ty - fy
    dir_x, dir_y = math.cos(fleet.angle), math.sin(fleet.angle)
    proj = dx * dir_x + dy * dir_y
    if proj < 0:
        return None
    perp_sq = dx * dx + dy * dy - proj * proj
    r_sq = tr * tr
    if perp_sq >= r_sq:
        return None
    hit_d = max(0.0, proj - math.sqrt(max(0.0, r_sq - perp_sq)))
    return int(math.ceil(hit_d / spd))


# ---- defender / arrival ledger --------------------------------------------


def _base_defenders(target, at_turn: int) -> float:
    """Closed-form: target's defenders at `at_turn` under grow-only rule.
    Neutrals don't accrete (env rule); owned planets grow by production.
    """
    if target.owner == -1:
        return float(target.ships)
    return float(target.ships) + float(target.production) * float(at_turn)


def _arrivals_at(world: World, target, by_turn: int, my_id: int,
                 phantoms: Iterable[PhantomFleet]):
    """Sum ships arriving at `target` by `by_turn`. Returns
    (our_ships, opp_ships) — real + phantom combined.
    """
    ours = 0.0
    theirs = 0.0
    for fleet in world.fleets:
        eta = _fleet_eta(fleet, target)
        if eta is None or eta > by_turn:
            continue
        if fleet.owner == my_id:
            ours += float(fleet.ships)
        else:
            theirs += float(fleet.ships)
    for p in phantoms:
        if p.target_id != target.id:
            continue
        if p.arrival_turn > by_turn:
            continue
        if p.owner == my_id:
            ours += float(p.ships)
        else:
            theirs += float(p.ships)
    return ours, theirs


def _net_defenders(world, target, arrival_turn: int, my_id: int,
                   phantoms: Iterable[PhantomFleet], target_is_ours: bool):
    """Net defenders that an attacker (us, by default) needs to beat
    to FLIP the target at `arrival_turn`. For our own target (defense
    flavour), `defenders` = opp incoming - our base.
    """
    base = _base_defenders(target, arrival_turn)
    ours, theirs = _arrivals_at(world, target, arrival_turn, my_id, phantoms)
    if target_is_ours:
        # opp's attack force must exceed our base + reinforcements
        # at the target. We want enough reinforcement so:
        # (our_base + our_reinforcement) > opp_attack
        return float(theirs) - (base + float(ours))
    # capture: attacker (us) must exceed defenders + opp reinforcement,
    # minus our prior in-flight credits.
    return base + float(theirs) - float(ours)


# ---- single-source capture / defense solver ------------------------------


def _solve_single_source(src, target, launch_turn: int, world: World,
                         my_id: int, phantoms: list[PhantomFleet],
                         target_is_ours: bool) -> Candidate | None:
    """Find the minimum-ship launch from src that wins the encounter
    at target, launched at launch_turn. Returns a Candidate or None.
    """
    omega = world.omega
    # Ship budget at launch_turn = current + production accrual.
    if src.owner != my_id:
        return None
    base_budget = int(src.ships) + int(src.production) * launch_turn
    base_budget = max(0, base_budget - SOURCE_RESERVE)
    if base_budget < MIN_LAUNCH_SHIPS:
        return None

    # Initial K guess by base defender count.
    K = max(MIN_LAUNCH_SHIPS, int(target.ships) + SAFETY_MARGIN)
    if target_is_ours:
        # Defense: estimate K as opp_incoming.
        ours, theirs = _arrivals_at(world, target, MAX_ETA, my_id, phantoms)
        K = max(MIN_LAUNCH_SHIPS, int(math.ceil(theirs)) + SAFETY_MARGIN)

    for _ in range(4):
        ae = _aim_and_eta(src, target, K, omega)
        if ae is None:
            return None
        angle, eta = ae
        arrival_turn = launch_turn + eta
        if arrival_turn > HORIZON:
            return None
        net_def = _net_defenders(
            world, target, arrival_turn, my_id, phantoms, target_is_ours,
        )
        if net_def < 0 and not target_is_ours:
            # Already over-captured by prior in-flight credits — no
            # new launch needed.
            return None
        if target_is_ours and net_def <= 0:
            # Defense not actually needed under current projection.
            return None
        K_needed = max(MIN_LAUNCH_SHIPS,
                       int(math.ceil(net_def)) + SAFETY_MARGIN)
        if K_needed <= K:
            K = K_needed
            break
        K = K_needed

    if K > base_budget:
        return None

    ae = _aim_and_eta(src, target, K, omega)
    if ae is None:
        return None
    angle, eta = ae
    arrival_turn = launch_turn + eta
    if arrival_turn > HORIZON:
        return None

    net_def_final = _net_defenders(
        world, target, arrival_turn, my_id, phantoms, target_is_ours,
    )
    if K < max(MIN_LAUNCH_SHIPS, net_def_final + SAFETY_MARGIN):
        return None

    # Value: production held over remaining horizon.
    held = max(0, HORIZON - arrival_turn)
    if target_is_ours:
        # Defense: we keep our planet AND avoid the ship loss.
        value = float(target.production) * float(held) + float(target.ships)
    else:
        value = float(target.production) * float(held)
    if value <= 0:
        return None

    flavor = "defense" if target_is_ours else "capture"
    return Candidate(
        flavor=flavor,
        target_id=target.id,
        launch_turn=launch_turn,
        arrival_turn=arrival_turn,
        allocations=(Allocation(src.id, K),),
        aim_angles=(angle,),
        value=value,
        total_ships=K,
    )


# ---- multi-source bundle (launch_turn=0 only) ----------------------------


def _solve_multi_source(target, world: World, my_id: int,
                        phantoms: list[PhantomFleet],
                        target_is_ours: bool) -> Candidate | None:
    """Build a multi-source bundle that captures (or defends) target
    at launch_turn=0. Pick sources greedily by ETA (closest first);
    they're fungible at arrival. Returns None if even the joint
    bundle can't fund the encounter.
    """
    my_planets = [p for p in world.planets if p.owner == my_id]
    if not my_planets:
        return None

    omega = world.omega

    # Per-source ETA at a midweight ship count (~50). We refine
    # per-source ship count after the bundle decision.
    feasible: list[tuple[int, int, float, int]] = []  # (eta, budget, angle, src_id)
    for src in my_planets:
        if src.id == target.id:
            continue
        if src.ships < MIN_LAUNCH_SHIPS:
            continue
        ae = _aim_and_eta(src, target, max(MIN_LAUNCH_SHIPS, int(src.ships)), omega)
        if ae is None:
            continue
        angle, eta = ae
        if eta > HORIZON:
            continue
        feasible.append((eta, int(src.ships - SOURCE_RESERVE), angle, src.id))

    if len(feasible) < 2:
        return None  # need at least 2 to be a "bundle"
    feasible.sort()  # by eta ascending

    # Sync all sources to the LATEST eta in the bundle so they all
    # arrive together (conservative — bigger ships fly faster, so the
    # constraint is the slowest source).
    best: Candidate | None = None
    for k_sources in range(2, min(len(feasible), 4) + 1):
        chosen = feasible[:k_sources]
        arrival_turn = chosen[-1][0]
        net_def = _net_defenders(
            world, target, arrival_turn, my_id, phantoms, target_is_ours,
        )
        need = max(MIN_LAUNCH_SHIPS, int(math.ceil(net_def)) + SAFETY_MARGIN)
        if target_is_ours and net_def <= 0:
            return None
        if net_def < 0 and not target_is_ours:
            return None

        # Allocate ships across sources closest-first up to budget.
        remaining = need
        allocations: list[Allocation] = []
        angles: list[float] = []
        for (eta, budget, ang, src_id) in chosen:
            if remaining <= 0:
                break
            take = min(remaining, budget)
            if take < MIN_LAUNCH_SHIPS:
                continue
            allocations.append(Allocation(src_id, take))
            angles.append(ang)
            remaining -= take
        if remaining > 0:
            continue  # bundle insufficient
        if len(allocations) < 2:
            continue  # collapsed to single-source

        total = sum(a.ships for a in allocations)
        held = max(0, HORIZON - arrival_turn)
        if target_is_ours:
            value = float(target.production) * float(held) + float(target.ships)
        else:
            value = float(target.production) * float(held)
        if value <= 0:
            continue
        c = Candidate(
            flavor="defense" if target_is_ours else "capture",
            target_id=target.id,
            launch_turn=0,
            arrival_turn=arrival_turn,
            allocations=tuple(allocations),
            aim_angles=tuple(angles),
            value=value,
            total_ships=total,
        )
        if best is None or c.roi > best.roi:
            best = c
    return best


# ---- candidate enumeration ------------------------------------------------


def enumerate_candidates(world: World, my_id: int,
                         phantoms: list[PhantomFleet]) -> list[Candidate]:
    """Step 2: produce all candidates (capture + defense, single +
    multi-source, over the wait-grid). Prune top-K per target by ROI."""
    candidates: dict[int, list[Candidate]] = {}  # target_id → top-K

    my_planets = [p for p in world.planets if p.owner == my_id]
    non_my = [p for p in world.planets if p.owner != my_id]
    threatened = _threatened_planets(world, my_id, phantoms, my_planets)

    targets_capture = non_my
    targets_defense = threatened

    def _add(c: Candidate):
        bucket = candidates.setdefault(c.target_id, [])
        bucket.append(c)
        bucket.sort(key=lambda x: -x.roi)
        del bucket[TOP_CANDIDATES_PER_TARGET:]

    # Capture candidates: single-source over wait-grid.
    for tgt in targets_capture:
        for src in my_planets:
            if src.id == tgt.id:
                continue
            for lt in WAIT_GRID:
                c = _solve_single_source(src, tgt, lt, world, my_id,
                                         phantoms, target_is_ours=False)
                if c is not None:
                    _add(c)
        # Multi-source bundle at launch_turn=0.
        mc = _solve_multi_source(tgt, world, my_id, phantoms,
                                 target_is_ours=False)
        if mc is not None:
            _add(mc)

    # Defense candidates: single-source for each threatened planet,
    # launch_turn=0 only.
    for tgt in targets_defense:
        for src in my_planets:
            if src.id == tgt.id:
                continue
            c = _solve_single_source(src, tgt, 0, world, my_id,
                                     phantoms, target_is_ours=True)
            if c is not None:
                _add(c)
        mc = _solve_multi_source(tgt, world, my_id, phantoms,
                                 target_is_ours=True)
        if mc is not None:
            _add(mc)

    flat: list[Candidate] = []
    for bucket in candidates.values():
        flat.extend(bucket)
    return flat


def _threatened_planets(world: World, my_id: int,
                        phantoms: list[PhantomFleet],
                        my_planets: list) -> list:
    """Our planets where projected opp arrivals exceed defenders."""
    out = []
    for p in my_planets:
        ours, theirs = _arrivals_at(world, p, MAX_ETA, my_id, phantoms)
        # If opp has more ships in flight to p than p will have at
        # the latest arrival turn, p is at risk.
        # Conservative: just check if ANY opp arrival exists with
        # ships > current garrison.
        if theirs <= 0:
            continue
        if theirs >= float(p.ships) - float(ours):
            out.append(p)
    return out


# ---- Step 1: mirror-opp ---------------------------------------------------


def project_opp_launches(world: World, opp_id: int) -> list[PhantomFleet]:
    """Run a trimmed solver from opp's seat to project their launches.
    No nested mirror (treat their opponent = grow-only-us). Returns
    phantom fleets representing opp's planned launches.

    Only single-source capture candidates with launch_turn ∈ {0, 1, 2}
    are enumerated for the opp — enough to see their immediate
    threats without doubling our compute.
    """
    opp_planets = [p for p in world.planets if p.owner == opp_id]
    if not opp_planets:
        return []

    targets = [p for p in world.planets if p.owner != opp_id]
    # No phantoms when projecting opp (avoids infinite regress).
    no_phantoms: list[PhantomFleet] = []

    raw: list[Candidate] = []
    for src in opp_planets:
        if src.ships < MIN_LAUNCH_SHIPS:
            continue
        for tgt in targets:
            for lt in (0, 1, 2):
                c = _solve_single_source(
                    src, tgt, lt, world, opp_id, no_phantoms,
                    target_is_ours=False,
                )
                if c is not None:
                    raw.append(c)

    # Greedy joint-pack from opp's POV.
    raw.sort(key=lambda c: -c.roi)
    budgets = {p.id: int(p.ships) for p in opp_planets}
    picked: list[Candidate] = []
    used_targets: set[int] = set()
    for c in raw:
        if c.target_id in used_targets:
            continue
        alloc = c.allocations[0]
        if budgets.get(alloc.src_id, 0) < alloc.ships:
            continue
        picked.append(c)
        budgets[alloc.src_id] -= alloc.ships
        used_targets.add(c.target_id)

    phantoms: list[PhantomFleet] = []
    for c in picked:
        for a in c.allocations:
            phantoms.append(PhantomFleet(
                owner=opp_id,
                src_id=a.src_id,
                target_id=c.target_id,
                arrival_turn=c.arrival_turn,
                ships=a.ships,
            ))
    return phantoms


# ---- Step 4: joint solve (greedy seed + 2-opt) ---------------------------


def _ships_drawn_from(c: Candidate, src_id: int) -> int:
    return sum(a.ships for a in c.allocations if a.src_id == src_id)


def _candidate_fits(c: Candidate, remaining: dict[int, int]) -> bool:
    for a in c.allocations:
        if remaining.get(a.src_id, 0) < a.ships:
            return False
    return True


def _apply(c: Candidate, remaining: dict[int, int], sign: int):
    """sign=+1 returns ships; sign=-1 spends them."""
    for a in c.allocations:
        remaining[a.src_id] = remaining.get(a.src_id, 0) + sign * a.ships


def joint_solve_2opt(candidates: list[Candidate],
                     my_planets: list) -> list[Candidate]:
    """Greedy seed + 2-opt local improvement under per-source budgets
    and one-launch-per-target dedup."""
    budgets = {p.id: int(p.ships - SOURCE_RESERVE) for p in my_planets}
    cands = sorted(candidates, key=lambda c: -c.roi)

    chosen: list[Candidate] = []
    used_target: set[int] = set()
    remaining = dict(budgets)
    for c in cands:
        if c.target_id in used_target:
            continue
        if not _candidate_fits(c, remaining):
            continue
        chosen.append(c)
        used_target.add(c.target_id)
        _apply(c, remaining, -1)

    # 2-opt: for each not-picked candidate, try swapping in by
    # removing 0, 1, or 2 lower-value chosen candidates.
    for _ in range(TWO_OPT_PASSES):
        improved = False
        for c in cands:
            if c in chosen:
                continue
            # Try drop 0: directly add.
            if c.target_id not in used_target and _candidate_fits(c, remaining):
                chosen.append(c)
                used_target.add(c.target_id)
                _apply(c, remaining, -1)
                improved = True
                continue
            # Try drop 1 + add c.
            best_swap = None
            for victim in chosen:
                if victim.target_id == c.target_id:
                    # Same target swap is allowed if c has higher value.
                    if c.value <= victim.value:
                        continue
                else:
                    if victim.value >= c.value:
                        continue
                temp = dict(remaining)
                _apply(victim, temp, +1)
                if _candidate_fits(c, temp):
                    delta = c.value - victim.value
                    if best_swap is None or delta > best_swap[0]:
                        best_swap = (delta, victim)
            if best_swap is not None:
                _, victim = best_swap
                chosen.remove(victim)
                used_target.discard(victim.target_id)
                _apply(victim, remaining, +1)
                if c.target_id not in used_target and _candidate_fits(c, remaining):
                    chosen.append(c)
                    used_target.add(c.target_id)
                    _apply(c, remaining, -1)
                    improved = True
        if not improved:
            break
    return chosen


# ---- Step 5: emit turn-0 actions -----------------------------------------


def emit_turn_0(plan: list[Candidate]) -> list[list]:
    """Emit env actions for plan items scheduled to launch this turn."""
    actions: list[list] = []
    for c in plan:
        if c.launch_turn != 0:
            continue
        for a, ang in zip(c.allocations, c.aim_angles):
            actions.append([a.src_id, ang, a.ships])
    return actions


# ---- main entry -----------------------------------------------------------


def agent(obs, configuration=None):
    world = World.from_obs(obs, configuration)
    my_id = world.my_id
    # 2P; for 4P this picks one opp deterministically. v3 generalises.
    other_owners = [p.owner for p in world.planets if p.owner not in (-1, my_id)]
    if not other_owners:
        return []
    opp_id = max(set(other_owners), key=other_owners.count)

    my_planets = [p for p in world.planets if p.owner == my_id]
    if not my_planets:
        return []

    phantoms = project_opp_launches(world, opp_id)
    candidates = enumerate_candidates(world, my_id, phantoms)
    if not candidates:
        return []
    plan = joint_solve_2opt(candidates, my_planets)
    return emit_turn_0(plan)
