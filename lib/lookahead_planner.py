"""Lookahead helpers for v4_planner — value function, adaptive K, comet truncation.

Four pure helpers consumed by `agents/v4_planner` (and v4_pef variant).

- `evaluate_value(observation, my_id, *, denial_weight, ships_weight,
  survivor_bonus, cluster_weight, frontier_weight, reach_weight) -> float`
  The leaf-state scoring head. Production-share is the main signal,
  denial captures forward-planning value (planets not in opp's hands),
  ships_share is a tie-break for short horizons, survivor bonus is the
  absorbing-state reward when we've eliminated all opponents.
  The PEF extensions (cluster / frontier / reach) add positional /
  defensibility primitives. All three default to weight 0 → bit-exact
  backward-compat with the original v4_planner value head.

- `_cluster_cohesion(planets, my_id)` — per-my-planet mean local
  production-share around me. Captures defensibility: my planets in
  dense friendly clusters score higher.
- `_frontier_exposure(planets, my_id)` — per-my-planet mean enemy-
  firepower fraction in the local neighborhood. Captures vulnerability:
  my planets surrounded by enemies are exposed.
- `_reach(planets, my_id)` — fraction of non-mine production within
  striking distance of any of my planets. Offensive option value.

- `adaptive_K(world) -> int`
  Entropy-conditioned rollout depth. The user's "N as a function of the
  ships flying around" — quiet boards get short rollouts (cheap, low
  variance); busy boards get longer rollouts so interactions have time
  to resolve. Bounded [8, 30] to stay under the 1 s/turn budget at 5
  candidates × ~5.6 ms/step × 2 player policies.

- `truncate_K_to_comet_boundary(K, step) -> int`
  Robustness. `env_from_obs` is bit-exact except across comet spawn
  boundaries (steps 50, 150, 250, 350, 450) where fresh env RNG diverges
  from the real game's RNG state. We shorten K so the rollout never
  crosses a boundary; floor 1 so we still get to apply our chosen action.
"""

from __future__ import annotations

import math


COMET_SPAWN_STEPS: tuple[int, ...] = (50, 150, 250, 350, 450)

# PEF positional-term defaults — radii in board units (board is 100x100),
# tau is the exp-decay length-scale.
_CLUSTER_RADIUS = 25.0
_CLUSTER_TAU = 12.0
_FRONTIER_RADIUS = 35.0
_FRONTIER_TAU = 12.0
_REACH_RADIUS = 40.0

# K bounds calibrated to the local box's measured per-step cost (~10 ms
# per forward step including 2x v3.5.1 policy calls). With env_from_obs
# ~105 ms one-time + ~24 ms clone() per candidate, the per-candidate
# wallclock is roughly (24 + 10*K) ms. At K_MAX=12 the budget supports
# ~5 candidates under 1 s; the watchdog in v4_planner cuts later
# candidates if needed. Phase 2 audit measurements (~5.6 ms/step) were
# on a faster 4-core box; these bounds adapt without changing the
# algorithm.
K_MIN = 6
K_MAX = 10


def evaluate_value(
    observation,
    my_id: int,
    *,
    denial_weight: float = 0.4,
    ships_weight: float = 0.05,
    survivor_bonus: float = 5.0,
    cluster_weight: float = 0.0,
    frontier_weight: float = 0.0,
    reach_weight: float = 0.0,
) -> float:
    """V(s) = prod_share + denial * prod_denied + ships * ships_share + survivor
              + cluster * cluster_cohesion - frontier * frontier_exposure + reach * reach.

    `observation` is the kaggle env's per-seat observation dict (the same
    shape `agent(obs)` receives). Fields used: `planets` (list of
    [id, owner, x, y, radius, ships, production]) and `fleets`
    (list of [id, owner, x, y, angle, from_planet_id, ships]).

    Empty world (no planets) → 0.0. Total production zero (all planets
    have prod=0, which doesn't happen in practice but bounds the math)
    → 0.0 share components, only survivor bonus can fire.

    PEF positional terms (cluster, frontier, reach) default to weight 0 →
    bit-exact backward-compat with the original v4_planner value head.
    Each new term is normalised to [0, 1] so weights are interpretable as
    relative importance against the base prod_share term.
    """
    planets = observation.get("planets", []) if isinstance(observation, dict) else getattr(observation, "planets", [])
    if not planets:
        return 0.0

    total_prod = 0.0
    my_prod = 0.0
    opp_prod = 0.0
    owners_with_planets: set[int] = set()
    garrison: dict[int, float] = {}
    for p in planets:
        owner = int(p[1])
        ships = float(p[5])
        prod = float(p[6])
        total_prod += prod
        if owner >= 0:
            owners_with_planets.add(owner)
            garrison[owner] = garrison.get(owner, 0.0) + ships
            if owner == my_id:
                my_prod += prod
            else:
                opp_prod += prod

    fleets = observation.get("fleets", []) if isinstance(observation, dict) else getattr(observation, "fleets", [])
    fleet_totals: dict[int, float] = {}
    for f in fleets:
        owner = int(f[1])
        ships = float(f[6])
        if owner >= 0:
            fleet_totals[owner] = fleet_totals.get(owner, 0.0) + ships

    totals: dict[int, float] = dict(garrison)
    for owner, ships in fleet_totals.items():
        totals[owner] = totals.get(owner, 0.0) + ships
    total_ships = sum(totals.values())
    my_ships = totals.get(my_id, 0.0)

    prod_share = (my_prod / total_prod) if total_prod > 0 else 0.0
    prod_denied = ((total_prod - opp_prod) / total_prod) if total_prod > 0 else 0.0
    ships_share = (my_ships / total_ships) if total_ships > 0 else 0.0
    lone = 1.0 if owners_with_planets == {my_id} else 0.0

    value = (
        prod_share
        + denial_weight * prod_denied
        + ships_weight * ships_share
        + survivor_bonus * lone
    )

    if cluster_weight or frontier_weight or reach_weight:
        if cluster_weight:
            value += cluster_weight * _cluster_cohesion(planets, my_id)
        if frontier_weight:
            value -= frontier_weight * _frontier_exposure(planets, my_id)
        if reach_weight:
            value += reach_weight * _reach(planets, my_id)
    return value


def _cluster_cohesion(
    planets,
    my_id: int,
    *,
    radius: float = _CLUSTER_RADIUS,
    tau: float = _CLUSTER_TAU,
) -> float:
    """Per-my-planet mean local production-share in [0, 1].

    For each planet I own, find all planets (mine + enemy + neutral)
    within `radius` board units, weight each by `prod * exp(-d/tau)`,
    and compute the fraction of that weighted production that belongs
    to me. Average over my planets. A planet sitting in a dense friendly
    cluster scores near 1.0; a lone planet in enemy/neutral space scores
    near 0.0. Self-pair excluded.
    """
    my_planets = [p for p in planets if int(p[1]) == my_id]
    if not my_planets:
        return 0.0
    radius_sq = radius * radius
    total = 0.0
    n_my = 0
    for q in my_planets:
        qx, qy = q[2], q[3]
        my_density = 0.0
        all_density = 0.0
        for r in planets:
            if r is q:
                continue
            dx = r[2] - qx
            dy = r[3] - qy
            d_sq = dx * dx + dy * dy
            if d_sq > radius_sq:
                continue
            d = math.sqrt(d_sq)
            w = math.exp(-d / tau) * float(r[6])
            all_density += w
            if int(r[1]) == my_id:
                my_density += w
        if all_density > 0:
            total += my_density / all_density
            n_my += 1
        # If no neighbours found, this planet contributes 0 (isolated).
        # We still count it in the denominator to penalise isolation.
        else:
            n_my += 1
    return total / n_my if n_my > 0 else 0.0


def _frontier_exposure(
    planets,
    my_id: int,
    *,
    radius: float = _FRONTIER_RADIUS,
    tau: float = _FRONTIER_TAU,
) -> float:
    """Per-my-planet mean enemy-firepower fraction in [0, 1].

    For each planet I own, sum nearby enemy garrison ships weighted by
    `exp(-d/tau)`, and divide by (enemy_weighted + my_defense + 1) so
    the ratio lives in [0, 1). High values mean enemies outnumber me
    locally — my planet is exposed. Neutrals contribute nothing (they
    don't threaten yet). +1 in the denominator avoids div-zero when a
    planet has zero garrison.
    """
    my_planets = [p for p in planets if int(p[1]) == my_id]
    if not my_planets:
        return 0.0
    radius_sq = radius * radius
    total = 0.0
    for q in my_planets:
        qx, qy = q[2], q[3]
        q_defense = float(q[5])
        enemy_threat = 0.0
        for e in planets:
            owner = int(e[1])
            if owner == my_id or owner < 0:
                continue
            dx = e[2] - qx
            dy = e[3] - qy
            d_sq = dx * dx + dy * dy
            if d_sq > radius_sq:
                continue
            d = math.sqrt(d_sq)
            enemy_threat += math.exp(-d / tau) * float(e[5])
        denom = enemy_threat + q_defense + 1.0
        total += enemy_threat / denom
    return total / len(my_planets)


def _reach(
    planets,
    my_id: int,
    *,
    radius: float = _REACH_RADIUS,
) -> float:
    """Fraction of non-mine production within striking distance, in [0, 1].

    For each non-mine planet (enemy or neutral), find the minimum
    Euclidean distance to any of my planets. If that minimum is ≤
    `radius`, the planet counts as "reachable". Result = sum of
    production of reachable non-mine planets / total non-mine production.
    Empty-target edge case → 0.0 (no offensive options).
    """
    my_planets = [p for p in planets if int(p[1]) == my_id]
    if not my_planets:
        return 0.0
    targets = [p for p in planets if int(p[1]) != my_id]
    if not targets:
        return 0.0
    total_prod = sum(float(p[6]) for p in targets)
    if total_prod <= 0:
        return 0.0
    radius_sq = radius * radius
    reachable = 0.0
    for t in targets:
        tx, ty = t[2], t[3]
        min_d_sq = float("inf")
        for q in my_planets:
            dx = q[2] - tx
            dy = q[3] - ty
            d_sq = dx * dx + dy * dy
            if d_sq < min_d_sq:
                min_d_sq = d_sq
                if min_d_sq <= radius_sq:
                    break
        if min_d_sq <= radius_sq:
            reachable += float(t[6])
    return reachable / total_prod


def adaptive_K(world: "World") -> int:
    """Entropy-adaptive rollout depth.

    entropy = fleets_in_flight + 0.5 * contested_planets
    K = clamp(round(8 + 1.5 * entropy), 8, 30)

    Empty boards bottom at K=8 (the floor). The 0.5 weighting on neutral
    planets reflects that they're potential rather than active conflict —
    they raise entropy less than an already-launched fleet.
    """
    raw = getattr(world, "obs_raw", None)
    if raw is None:
        return K_MIN
    fleets_raw = (
        raw.get("fleets", []) if isinstance(raw, dict) else getattr(raw, "fleets", [])
    )
    fleets_in_flight = len(fleets_raw)
    contested = 0
    for p in world.planets_by_id.values():
        if p.owner == -1:
            contested += 1
    entropy = fleets_in_flight + 0.5 * contested
    K = int(round(K_MIN + 0.5 * entropy))
    return max(K_MIN, min(K, K_MAX))


def truncate_K_to_comet_boundary(K: int, step: int) -> int:
    """Shorten K so rollout doesn't cross a comet spawn boundary.

    Spawn boundaries are at steps in `COMET_SPAWN_STEPS`. `env_from_obs`
    is bit-exact within an inter-boundary segment; crossing a boundary
    re-rolls the env's comet RNG, which diverges from the real game. We
    cap K so the clone's `step + K` stays strictly below the next
    boundary. Floor at 1 — we always apply at least our chosen action.
    """
    for boundary in COMET_SPAWN_STEPS:
        if boundary > step:
            allowed = boundary - step - 1
            return max(1, min(K, allowed))
    return K
