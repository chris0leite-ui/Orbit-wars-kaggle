"""main.py — our Orbit Wars agent (kaggle-submittable as-is).

v4 — exact-physics chooser. Per turn, per candidate (src, tgt, ships):

  1. Build a fast_sim Snapshot from the obs (parity-tested byte-exact
     with kaggle env; `lib/fast_sim.py`).
  2. Apply my candidate action via `fs_step` for turn 0, with all
     other seats IDLE.
  3. Continue idle stepping for `arrival + 2` turns so my fleet
     reaches the target (or is destroyed) and combat resolves
     EXACTLY (no analytic approximation of orbital motion / fleet-
     in-flight collisions / multi-fleet arrivals).
  4. Score = favor(leaf_state, me) − favor(obs, me). Uses v1 F1+F2
     formula (AUC 0.945 on saved snapshots).

Why this replaces v3 (analytic _apply_action) and v2 (fwd-sim rollout
with favor-greedy opp):

The PI's framing: we have two proven foundations — the parity-tested
physics engine (`fast_sim`, 62/62 parity tests) AND the validated
F1+F2 favor formula. Stop reinventing approximations.

v3 used `_apply_action` (an analytic mutation of the obs):
  - ignored orbital motion (planets rotate during fleet travel)
  - ignored existing in-flight fleets
  - ignored multi-fleet combat arrivals same turn
  - flat-added production × arrival to every owned planet
v3 result: 79.2 % vs nearest (regressed from 100 %), 0/24 vs v7_0.

v2 used 10-turn rollout with favor-greedy opp:
  - speculative opp model caused cascade losses
  - leaf converged to baseline-minus-combat-cost for ALL candidates
  - rollout overrode positive analytic Δfavor predictions
v2 result: 100 % vs nearest, 0/24 vs v7_0.

This version: physics simulator handles arrival window EXACTLY; opp
is IDLE during my arrival (no speculative counter-recapture model);
v1 favor formula scores the leaf (proven AUC). The Δ tells me what
this action accomplishes against a stationary baseline; opp's actual
response is delegated to the next turn's decision.

F3 was reverted in favor.py — it was a closed-form *approximation*
of what the simulator computes exactly. With the simulator, F3 is
redundant and (per AUC 0.912 vs 0.945) actively miscalibrated.

Submit with:  ./submit.sh "message"
Eval with:    python eval.py --vs nearest -n 24
              python eval.py --vs v7_0 -n 24
"""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from favor import favor
from lib.fast_sim import from_obs, step as fs_step, clone as fs_clone

# --- comp constants (from data/README.md) ----------------------------------
EPISODE_STEPS = 500
MAX_SPEED = 6.0
SUN_X = 50.0
SUN_Y = 50.0
SUN_R = 10.0

# --- chooser config --------------------------------------------------------
NUM_TARGETS_PER_SOURCE = 8     # K nearest non-owned planets considered
MIN_FLEET_SIZE = 2             # 1-ship fleets move at speed 1; rarely useful
SIM_SETTLE_TURNS = 2           # extra idle turns after arrival to let combat resolve


# ---------------------------------------------------------------------------
# Comp-spec math
# ---------------------------------------------------------------------------


def _speed(ships: int) -> float:
    """Fleet speed as a function of size (comp spec)."""
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5


def _crosses_sun(sx: float, sy: float, tx: float, ty: float) -> bool:
    """True if the straight line from (sx,sy) to (tx,ty) passes within SUN_R of the sun."""
    dx = tx - sx
    dy = ty - sy
    L2 = dx * dx + dy * dy
    if L2 == 0.0:
        return math.hypot(sx - SUN_X, sy - SUN_Y) < SUN_R
    t = ((SUN_X - sx) * dx + (SUN_Y - sy) * dy) / L2
    t = max(0.0, min(1.0, t))
    cx = sx + t * dx
    cy = sy + t * dy
    return math.hypot(cx - SUN_X, cy - SUN_Y) < SUN_R


# ---------------------------------------------------------------------------
# Δfavor scoring (analytic; no forward sim needed)
# ---------------------------------------------------------------------------


def _arrival_turns(src: Planet, tgt: Planet, ships: int) -> int:
    """Estimated turns for a fleet of `ships` to fly from src to tgt
    (centre-to-centre minus both radii, at `_speed(ships)`).
    Used to set the sim horizon per candidate.
    """
    dist = max(0.0, math.hypot(src.x - tgt.x, src.y - tgt.y) - src.radius - tgt.radius)
    return max(1, math.ceil(dist / _speed(ships)))


def score_action(
    src: Planet,
    tgt: Planet,
    ships: int,
    step: int,
    me: int,
    snap_base=None,
    num_seats: int = 2,
    favor_before: float | None = None,
) -> float:
    """Δfavor for launching `ships` from src toward tgt — computed
    via the parity-tested physics simulator (`lib/fast_sim`), not an
    analytic approximation.

    Steps:
      1. Clone `snap_base` (the agent's current Snapshot).
      2. Apply my candidate as turn-0 action; all opp seats IDLE.
      3. Step idly for `arrival + SIM_SETTLE_TURNS` more turns so
         my fleet arrives, combat resolves, and accumulated
         production is recorded exactly. (Orbital motion, sun
         crossing, multi-fleet collisions all handled by the engine.)
      4. Return favor(leaf, me) − favor_before.

    Idle opp during arrival is intentional: this measures what THIS
    action accomplishes against a stationary world. Opp's actual
    response is captured by next-turn re-evaluation (the action is
    re-decided every turn from the live obs). Cascading speculative
    opp policies (favor-greedy in v2-rollout) introduced noise that
    overrode genuine analytic-positive captures (audit notes in v3
    commit message).

    `snap_base`, `num_seats`, and `favor_before` are computed once
    per agent turn and passed in to avoid per-candidate rebuilding.
    """
    if ships < MIN_FLEET_SIZE or ships > src.ships or src.id == tgt.id:
        return float("-inf")
    if tgt.owner == me:
        return 0.0
    if snap_base is None or favor_before is None:
        raise ValueError("score_action requires snap_base and favor_before")

    arrival = _arrival_turns(src, tgt, ships)
    angle = math.atan2(tgt.y - src.y, tgt.x - src.x)

    snap = fs_clone(snap_base)
    actions = [[] for _ in range(num_seats)]
    actions[me] = [[src.id, angle, ships]]
    snap = fs_step(snap, actions, in_place=True)

    idle_actions = [[] for _ in range(num_seats)]
    for _ in range(arrival - 1 + SIM_SETTLE_TURNS):
        if snap.fake_env.done:
            break
        snap = fs_step(snap, idle_actions, in_place=True)

    return favor(snap.state[me].observation, me) - favor_before


# ---------------------------------------------------------------------------
# Per-planet chooser
# ---------------------------------------------------------------------------


def _incoming_threat(src: Planet, fleets: list[Fleet], me: int) -> int:
    """Sum of enemy fleet ships aimed approximately at this source.

    Heuristic: a fleet is "aimed at src" when its heading is within
    ~17 degrees of the bearing from the fleet to src. Cheap, doesn't
    project full trajectories. Misses fleets that will arrive via
    orbital sweep; over-counts fleets that will pass nearby. Good
    enough for a defensive ship-reserve heuristic.
    """
    threat = 0
    for f in fleets:
        if f.owner == me or f.owner == -1:
            continue
        dx = src.x - f.x
        dy = src.y - f.y
        if dx == 0 and dy == 0:
            threat += f.ships
            continue
        bearing = math.atan2(dy, dx)
        # Angular distance between fleet heading and bearing-to-src.
        d_ang = math.atan2(math.sin(f.angle - bearing), math.cos(f.angle - bearing))
        if abs(d_ang) < 0.3:  # ~17 degrees
            threat += f.ships
    return threat


def _capture_size_guess(src: Planet, tgt: Planet) -> int:
    """Approximate min ships needed to capture tgt from src.

    For neutrals: tgt.ships + 1.
    For enemies: tgt.ships + tgt.production × estimated_arrival_turns + 1
                 (arrival estimated with a guess of capture_size; one
                 iteration suffices in practice).
    """
    if tgt.owner == -1:
        return int(tgt.ships) + 1
    dist = max(
        0.0,
        math.hypot(src.x - tgt.x, src.y - tgt.y) - src.radius - tgt.radius,
    )
    guess = int(tgt.ships) + 1
    arrival = math.ceil(dist / _speed(guess)) if guess > 0 else 1
    return int(tgt.ships) + int(tgt.production) * arrival + 1


def _num_seats(planets: list[Planet], fleets: list[Fleet]) -> int:
    """Detect 2P vs 4P from the obs (orbit-wars supports only those)."""
    max_owner = -1
    for p in planets:
        if p.owner > max_owner:
            max_owner = p.owner
    for f in fleets:
        if f.owner > max_owner:
            max_owner = f.owner
    return 4 if max_owner >= 2 else 2


def _enumerate_candidates(
    my_planets: list[Planet],
    targets: list[Planet],
    fleets: list[Fleet],
    step: int,
    me: int,
    snap_base,
    num_seats: int,
    favor_before: float,
) -> list[tuple[float, Planet, Planet, int]]:
    """All (Δfavor, src, tgt, ships) triples with Δfavor > 0.

    Δfavor is the simulator-based delta: clone snap_base, apply my
    candidate at turn 0, idle through arrival, compute v1 favor at
    leaf, subtract favor_before. Defense: reserve garrison against
    incoming enemy fleets aimed at the source.
    """
    out: list[tuple[float, Planet, Planet, int]] = []
    for src in my_planets:
        if src.ships < MIN_FLEET_SIZE:
            continue
        threat = _incoming_threat(src, fleets, me)
        launch_budget = max(0, src.ships - threat)
        if launch_budget < MIN_FLEET_SIZE:
            continue
        ranked = sorted(
            targets,
            key=lambda t: math.hypot(src.x - t.x, src.y - t.y),
        )[:NUM_TARGETS_PER_SOURCE]
        for tgt in ranked:
            capture_size = _capture_size_guess(src, tgt)
            if capture_size < MIN_FLEET_SIZE or capture_size > launch_budget:
                continue
            # Multi-size enumeration: let the formula pick the best size
            # per (src, tgt). min-cap is the cheapest capture; 2× provides
            # post-capture defensibility against opp counter; launch_budget
            # is overwhelming force (useful when opp likely contests).
            sizes = {capture_size}
            if 2 * capture_size <= launch_budget:
                sizes.add(2 * capture_size)
            if launch_budget >= capture_size + 5:
                sizes.add(launch_budget)
            for ships in sizes:
                s = score_action(
                    src, tgt, ships, step, me,
                    snap_base=snap_base, num_seats=num_seats,
                    favor_before=favor_before,
                )
                if s > 0.0:
                    out.append((s, src, tgt, ships))
    out.sort(key=lambda c: -c[0])
    return out


# ---------------------------------------------------------------------------
# Public agent
# ---------------------------------------------------------------------------


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    step = int(obs.get("step", 0)) if isinstance(obs, dict) else int(getattr(obs, "step", 0))
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not my_planets or not targets:
        return []

    # Build snap_base once per turn; per-candidate clone is cheap.
    num_seats = _num_seats(planets, fleets)
    snap_base = from_obs(obs, num_seats=num_seats)
    favor_before = favor(obs, player)
    candidates = _enumerate_candidates(
        my_planets, targets, fleets, step, player,
        snap_base, num_seats, favor_before,
    )
    if not candidates:
        return []   # no positive-Δfavor candidate; hold

    # Greedy non-dogpile: each source emits at most one fleet; each target
    # is the destination of at most one fleet this turn.
    used_srcs: set[int] = set()
    used_tgts: set[int] = set()
    moves: list[list] = []
    for _delta, src, tgt, ships in candidates:
        if src.id in used_srcs or tgt.id in used_tgts:
            continue
        used_srcs.add(src.id)
        used_tgts.add(tgt.id)
        angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
        moves.append([src.id, angle, ships])
    return moves
