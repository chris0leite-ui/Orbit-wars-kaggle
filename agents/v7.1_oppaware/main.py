"""v7.1_oppaware — v7_minimax + opp-action-aware arrival_size.

v7 already simulates the opponent's predicted v3 action during maximin
enumeration. v7.1 closes the loop: it threads the opponent's predicted
fleet launches into the WorldModel used by `arrival_size` when sizing
OUR fleets. The bounce-margin telemetry
(audit/2026-05-12-opening-variants-and-bounce-telemetry.md) showed
that **14 %** of bounces occur at margin ≥ +6 — we sent more than the
visible defender count but still bounced — because `WorldModel.ships_at`
is blind to opponent fleets launched *this turn* (obs is end-of-prev-turn).

The fix is a 2-step turn-order patch:
1. Compute the opponent's predicted action FIRST (via v3 on swapped obs).
2. Append each predicted launch to a copy of `obs.fleets` as a virtual
   in-flight fleet, then build our v3 action from the enriched obs.
   `WorldModel.from_world` ray-casts these virtual fleets and adds them
   to the target-arrival ledger, so `arrival_size` over-sizes our fleet
   when an opponent fleet will reach the same target.

The maximin rollout is UNCHANGED — `env_from_obs` reads the original
obs (not the enriched copy) so the rollout env applies opp's action
exactly once (via env.step), not double-counted as a pre-existing fleet.

Safety:
- If opp model is wrong (e.g. opp doesn't actually launch what v3 would),
  we over-size our fleet. Over-sizing wastes ships at the source but
  doesn't cause bounces — the failure mode is strictly less harmful
  than the under-sizing we're fixing.
- Falls back to v3 in 4P (v7's existing limitation).
- 2P self-play parity preserved by σ-equivariance (sym_hypot,
  SCORE_ROUND=6 in planner) — the enriched obs is symmetric under
  player swap, so two v7.1 instances in self-play see mirror-symmetric
  enriched obs → mirror actions → draws.
"""

from __future__ import annotations

import math
import time

from lib.intent import World, realize
from lib.lookahead import env_from_obs, score_joint_action_symmetric
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel


# Same budget params as v7_minimax — opp-action enrichment adds ~30-50ms
# (one extra _v3 call) but stays within v7's already-validated guard.
N_CANDS = 2
M_OPPS = 2
K_INIT = 3
K_FALLBACK = 2
DOWNSHIFT_MS = 300.0
HARD_DEADLINE_MS = 750.0


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _v3_agent_impl(obs):
    """v3_snipe's agent function, inlined for bundle compatibility.

    Mirrors `agents/v3_snipe/main.py:agent` exactly. Same code path as
    v7_minimax's inlined _v3 to keep bundle bit-parity.
    """
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    missions = (
        propose_snipe_missions(world, model)
        + propose_reinforce_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)


def _v3():
    return _v3_agent_impl


def _detect_num_players(planets) -> int:
    return len({p[1] for p in planets if p[1] != -1})


def _drop_smallest(action: list) -> list:
    """Return `action` with its smallest-ship launch removed."""
    if not action:
        return []
    if len(action) == 1:
        return []
    min_idx = 0
    min_ships = int(action[0][2])
    for i, la in enumerate(action[1:], start=1):
        if int(la[2]) < min_ships:
            min_ships = int(la[2])
            min_idx = i
    return [la for i, la in enumerate(action) if i != min_idx]


def _swap_obs_player(obs, opp_id: int):
    """Return a shallow-copied obs with `player` set to opp_id."""
    if isinstance(obs, dict):
        obs2 = dict(obs)
        obs2["player"] = opp_id
        return obs2
    keys = (
        "player", "planets", "fleets", "angular_velocity",
        "initial_planets", "comet_planet_ids", "comets",
        "step", "next_fleet_id", "remainingOverageTime",
    )
    obs2 = {}
    for k in keys:
        v = getattr(obs, k, None)
        if v is not None:
            obs2[k] = v
    obs2["player"] = opp_id
    return obs2


def _enrich_obs_with_opp_launches(obs, opp_action: list, opp_id: int):
    """Append opp's predicted launches to obs.fleets as virtual fleets.

    The hypothetical fleet spawns at the source planet's outer edge
    in the launch direction (matching the env's actual spawn rule:
    fleet appears just outside src.radius). `WorldModel.from_world`
    will ray-cast these new fleets and include them in the
    target-arrival ledger so `arrival_size` accounts for them when
    sizing our launches.

    Returns a new obs dict (shallow copy with a fresh `fleets` list);
    `obs` is not mutated.

    Fleet schema (from comp-context.md and lib/world_model.py):
        [id, owner, x, y, angle, from_planet_id, ships]
    """
    if not isinstance(obs, dict):
        # The agent runtime always passes dict obs; defensive guard.
        return obs
    new_obs = dict(obs)
    fleets = list(obs.get("fleets", []))
    # Allocate virtual fleet IDs starting from next_fleet_id (or the
    # current max + 1 as a fallback). Real fleet IDs from the env are
    # always positive ints; collision is theoretically possible but
    # WorldModel only consumes the (owner, ships, target_eta) tuple —
    # the id is opaque to the arrival ledger.
    next_id = int(obs.get("next_fleet_id", 0) or 0)
    if next_id <= 0 and fleets:
        next_id = max(int(f[0]) for f in fleets) + 1
    if next_id <= 0:
        next_id = 1_000_000  # safe high value if obs has no fleets

    planets_by_id = {p[0]: p for p in obs.get("planets", [])}
    for launch in opp_action:
        if len(launch) < 3:
            continue
        try:
            src_id = int(launch[0])
            angle = float(launch[1])
            ships = int(launch[2])
        except (TypeError, ValueError):
            continue
        if ships <= 0:
            continue
        src = planets_by_id.get(src_id)
        if src is None:
            continue
        sx, sy = float(src[2]), float(src[3])
        srad = float(src[4])
        # Spawn position: outer edge of src in launch direction
        # (env spec: "fleet appears just outside the planet's radius
        # in the given direction").
        spawn_x = sx + (srad + 0.1) * math.cos(angle)
        spawn_y = sy + (srad + 0.1) * math.sin(angle)
        new_fleet = [next_id, opp_id, spawn_x, spawn_y, angle, src_id, ships]
        fleets.append(new_fleet)
        next_id += 1

    new_obs["fleets"] = fleets
    new_obs["next_fleet_id"] = next_id
    return new_obs


def _our_candidates(obs) -> list[list]:
    """N=2 candidates derived from v3 on the (already-enriched) obs."""
    c1 = _v3()(obs)
    c2 = _drop_smallest(c1)
    seen = set()
    out = []
    for c in (c1, c2):
        k = repr(c)
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
        if len(out) >= N_CANDS:
            break
    return out


def _opp_candidates(obs, opp_id: int) -> list[list]:
    """M=2 opp models: v3-from-opp-POV + drop-smallest."""
    swapped = _swap_obs_player(obs, opp_id)
    try:
        o1 = _v3()(swapped)
    except Exception:
        return [[]]
    o2 = _drop_smallest(o1)
    seen = set()
    out = []
    for o in (o1, o2):
        k = repr(o)
        if k in seen:
            continue
        seen.add(k)
        out.append(o)
        if len(out) >= M_OPPS:
            break
    return out


def _maximin_pick(matrix: list[list[float]], unfilled: list[list[bool]]) -> int:
    best_i = 0
    best_worst = float("-inf")
    n = len(matrix)
    if n == 0:
        return 0
    m = len(matrix[0]) if matrix[0] else 0
    for i in range(n):
        evaluated = [matrix[i][j] for j in range(m) if not unfilled[i][j]]
        if not evaluated:
            worst = float("-inf")
        else:
            worst = min(evaluated)
        if worst > best_worst:
            best_worst = worst
            best_i = i
    return best_i


def agent(obs):
    my_id = int(_obs_get(obs, "player", 0))
    planets = _obs_get(obs, "planets", []) or []

    # 4P fallback — minimax doesn't extend cleanly to n>2 zero-sum.
    if _detect_num_players(planets) != 2:
        return _v3()(obs)

    opp_id = 1 - my_id

    # Generate opp candidates FIRST (so we can use O[0] to enrich obs).
    O = _opp_candidates(obs, opp_id)
    if len(O) == 0:
        O = [[]]

    # Build enriched obs: append opp's most-likely (= O[0]) launches as
    # virtual fleets so arrival_size accounts for them when sizing
    # OUR fleets. We use O[0] (v3 incumbent — most aggressive opp model)
    # because over-sizing on the aggressive prediction is safer than
    # under-sizing on the conservative one.
    enriched_obs = _enrich_obs_with_opp_launches(obs, O[0], opp_id)

    # Generate OUR candidates from enriched obs (key v7.1 change).
    C = _our_candidates(enriched_obs)
    if len(C) <= 1:
        return C[0] if C else []

    # Rollout env uses ORIGINAL obs — env.step applies opp's action
    # once; including O[0] in obs.fleets here would double-count it.
    try:
        env = env_from_obs(obs)
    except Exception:
        return C[0]

    N = len(C)
    M = len(O)
    P: list[list[float]] = [[0.0] * M for _ in range(N)]
    unfilled: list[list[bool]] = [[True] * M for _ in range(N)]

    t0 = time.monotonic()
    K = K_INIT

    for i in range(N):
        for j in range(M):
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            if i > 0 and elapsed_ms > HARD_DEADLINE_MS:
                break
            if i > 0 and elapsed_ms > DOWNSHIFT_MS and K == K_INIT:
                K = K_FALLBACK
            try:
                P[i][j] = score_joint_action_symmetric(
                    env, C[i], O[j], K=K, policy=_v3(),
                )
                unfilled[i][j] = False
            except Exception:
                P[i][j] = float("-inf")
                unfilled[i][j] = False
        else:
            continue
        break

    i_star = _maximin_pick(P, unfilled)
    return C[i_star]
