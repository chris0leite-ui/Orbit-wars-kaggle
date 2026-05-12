"""v7_minimax — K-step maximin agent (real game theory at the action level).

At each turn:
  1. Generate N candidate actions for us
       C[0] = v3 incumbent
       C[1] = v3 incumbent with smallest-ship launch dropped (defensive variant)
  2. Generate M candidate actions for opp
       O[0] = v3 from opp's POV (run v3 on swapped-player obs); greedy opp
       O[1] = O[0] with smallest-ship launch dropped; less-aggressive opp
  3. For each (our_i, opp_j) pair, run Sim<K=5/3 adaptive> with v3 as
     rollout policy and compute (our_ships - opp_ships) at horizon
  4. Pick i* = argmax_i min_j P[i,j]  — maximin against the opp class
     tie-break preferring lower index (v3 incumbent first)

This is von Neumann minimax at the action-evaluation level. Within the
modeled opp-policy class {O0, O1}, the chosen action's K-step ship-delta
is no worse than worst[i*] = max_i min_j P[i,j] — not strictly dominated
by another C candidate.

Key engineering decisions (from the plan):
- Rollout policy = v3 (NOT roi). v5_psp's failure proved ROI rollout
  produces policy-mismatch noise that breaks minimax. v3 is slower
  (~30-50ms/call) but in-class.
- N=2, M=2, K=3 (adaptive K=5 if first sim finishes under 200ms).
- Budget guard: bail remaining sims past 700ms wallclock; row 0 (v3)
  always evaluated in full first so its worst is honest.
- 4P fallback: defer to v3 (no Nash guarantee in n≥3).
- σ-equivariance: preserved via the existing lib/planner patches. Two
  v7 instances in self-play should produce identical payoff matrices
  (up to player relabeling) → mirror picks → draws.

The σ-equivariance commits (6c12b9f sym_hypot, 7b60938 σ-equiv tie-break,
24bae06 score rounding) stay — they're orthogonal to minimax and
correct in their own right.
"""

from __future__ import annotations

import time

from lib.intent import World, realize
from lib.lookahead import env_from_obs, score_joint_action_symmetric
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel



# Maximin parameters — see plan
# After self-play gate failure (1/8 draws) we switched to
# score_joint_action_symmetric to cancel env seat-bias. That doubles
# per-cell rollout cost, so K had to drop from 5→3→2 to fit actTimeout.
N_CANDS = 2
M_OPPS = 2
K_INIT = 3            # was 5; symmetric score is 2× cost
K_FALLBACK = 2        # was 3
DOWNSHIFT_MS = 300.0    # downshift K if elapsed > this after first sim
HARD_DEADLINE_MS = 750.0  # bail remaining sims past this


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _v3_agent_impl(obs):
    """v3_snipe's agent function, inlined for bundle compatibility.

    Mirrors `agents/v3_snipe/main.py:agent` exactly. We can't `importlib`
    the v3_snipe file at runtime in the Kaggle-bundled environment
    because only this file + lib/* gets uploaded. Calling the lib
    primitives directly is the bundle-friendly path.
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
    """Compatibility wrapper — returns the inlined v3 callable."""
    return _v3_agent_impl


def _detect_num_players(planets) -> int:
    return len({p[1] for p in planets if p[1] != -1})


def _drop_smallest(action: list) -> list:
    """Return `action` with its smallest-ship launch removed.

    If action is empty or has only one launch, returns the empty list
    (the maximum drop). Ties broken by removing the EARLIEST launch
    among smallest, which is σ-deterministic given upstream ordering.
    """
    if not action:
        return []
    if len(action) == 1:
        return []
    # Each launch = [src_id, angle, ships]; pick min by ships, then by index.
    min_idx = 0
    min_ships = int(action[0][2])
    for i, la in enumerate(action[1:], start=1):
        if int(la[2]) < min_ships:
            min_ships = int(la[2])
            min_idx = i
    return [la for i, la in enumerate(action) if i != min_idx]


def _swap_obs_player(obs, opp_id: int):
    """Return a shallow-copied obs with `player` set to opp_id.

    Used to invoke v3 from the opponent's POV without mutating the
    real obs (which the agent runtime expects unmodified).
    """
    if isinstance(obs, dict):
        obs2 = dict(obs)
        obs2["player"] = opp_id
        return obs2
    # Fall back: build a minimal dict view that v3 / World.from_obs accepts.
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


def _our_candidates(obs) -> list[list]:
    """N=2 candidates: v3 incumbent + drop-smallest variant. Dedup by repr."""
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
    """M=2 opp models: v3-from-opp-POV + drop-smallest. Dedup by repr."""
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
    """Argmax over rows of (min over evaluated columns).

    Rows where all columns are unfilled are treated as having worst = -inf
    so they never win unless they're the only row. Tie-break: prefer
    lower row index (= v3 incumbent first).
    """
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

    # Generate candidates
    C = _our_candidates(obs)
    if len(C) <= 1:
        return C[0] if C else []
    O = _opp_candidates(obs, opp_id)

    # Defensive: env rebuild may fail in edge cases
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

    # Evaluate row 0 (v3 incumbent) FIRST and in FULL — its worst-case
    # must be honest because it's the conservative fallback at tie-break.
    # Then evaluate row 1+ in O0 → O1 order (most aggressive opp first,
    # so if we bail mid-row the row's worst-against-aggression is honest).
    for i in range(N):
        for j in range(M):
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            if i > 0 and elapsed_ms > HARD_DEADLINE_MS:
                # only abort after row 0 is complete
                break
            if i > 0 and elapsed_ms > DOWNSHIFT_MS and K == K_INIT:
                K = K_FALLBACK
            try:
                # score_joint_action_symmetric: averages over seat-flipped
                # rollouts to cancel env's documented P1-favoring tie-break
                # asymmetry. Returns ship-delta from OUR POV (seat-invariant).
                P[i][j] = score_joint_action_symmetric(
                    env, C[i], O[j], K=K, policy=_v3(),
                )
                unfilled[i][j] = False
            except Exception:
                P[i][j] = float("-inf")
                unfilled[i][j] = False
        else:
            continue
        break  # exited inner loop via budget bail

    i_star = _maximin_pick(P, unfilled)
    return C[i_star]
