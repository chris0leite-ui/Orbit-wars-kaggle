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
MIN_HORIZON = 15               # floor for sim horizon — must cover incoming threats
                               # (~time for fast cross-board fleet)
MAX_HORIZON = 50               # baseline_favors cache depth (≥ any candidate horizon)


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


def _build_idle_baseline(snap_base, me: int, num_seats: int, horizon: int) -> list[float]:
    """Run an idle-everyone sim from snap_base for `horizon` turns,
    recording favor(me) at every step. Returns a list of length
    horizon + 1: index 0 is "now," index k is "k idle turns ahead."

    This is the correct baseline for marginal-Δfavor scoring: a
    candidate that takes A turns to arrive must be compared against
    "what would my favor be if I just sat for A turns?", NOT against
    favor at the current state. Production naturally accrues during
    those A turns whether I act or not.
    """
    snap = fs_clone(snap_base)
    me_obs = snap.state[me].observation
    out: list[float] = [favor(me_obs, me)]
    idle_actions = [[] for _ in range(num_seats)]
    for _ in range(horizon):
        if snap.fake_env.done:
            out.append(out[-1])
            continue
        snap = fs_step(snap, idle_actions, in_place=True)
        out.append(favor(snap.state[me].observation, me))
    return out


def score_action(
    src: Planet,
    tgt: Planet,
    ships: int,
    step: int,
    me: int,
    snap_base=None,
    num_seats: int = 2,
    baseline_favors: list[float] | None = None,
) -> float:
    """Marginal Δfavor for launching `ships` from src toward tgt —
    computed via the parity-tested physics simulator.

    Compares `favor(leaf_after_action_at_horizon_H)` against
    `favor(leaf_after_idle_at_same_horizon_H)`, where H = arrival +
    SIM_SETTLE_TURNS. This isolates the ACTION'S contribution from
    the natural favor growth that would occur during the same H
    turns even without acting.

    PRIOR BUG (this commit fixes): comparing to favor_before
    (i.e. the current state) inflated every candidate by the
    growth-over-H-turns. The Δ "value" was essentially the agent's
    own production accumulated during the sim, not the action's
    effect. Discovered turn 2 of seed 1003: launching a redundant
    fleet at P10 (already targeted) scored +2873 because both leaf
    states have identical 143 my-ships and identical favor; the
    +2873 was just my production over 24 sim turns, attributed to
    the action.
    """
    if ships < MIN_FLEET_SIZE or ships > src.ships or src.id == tgt.id:
        return float("-inf")
    if tgt.owner == me:
        return 0.0
    if snap_base is None or baseline_favors is None:
        raise ValueError("score_action requires snap_base and baseline_favors")

    arrival = _arrival_turns(src, tgt, ships)
    # Floor the horizon at MIN_HORIZON so the sim runs long enough to
    # see any incoming enemy fleet resolve at the source planet. We
    # dropped the `_incoming_threat` defensive reserve; without it, a
    # short offensive-arrival sim would miss the threat landing at my
    # source planet a few turns later. MIN_HORIZON ≈ time for a fast
    # cross-board fleet at typical comp distances.
    horizon = max(arrival + SIM_SETTLE_TURNS, MIN_HORIZON)
    if horizon >= len(baseline_favors):
        horizon = len(baseline_favors) - 1
    angle = math.atan2(tgt.y - src.y, tgt.x - src.x)

    snap = fs_clone(snap_base)
    actions = [[] for _ in range(num_seats)]
    actions[me] = [[src.id, angle, ships]]
    snap = fs_step(snap, actions, in_place=True)

    idle_actions = [[] for _ in range(num_seats)]
    for _ in range(horizon - 1):
        if snap.fake_env.done:
            break
        snap = fs_step(snap, idle_actions, in_place=True)

    return favor(snap.state[me].observation, me) - baseline_favors[horizon]


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
    baseline_favors: list[float],
) -> list[tuple[float, Planet, Planet, int]]:
    """All (Δfavor, src, tgt, ships) triples with Δfavor > 0.

    Δfavor is the marginal contribution of THIS action vs idle for
    the same arrival horizon (see score_action's docstring for the
    prior bug fix). Defense: reserve garrison against incoming
    enemy fleets aimed at the source.
    """
    out: list[tuple[float, Planet, Planet, int]] = []
    for src in my_planets:
        if src.ships < MIN_FLEET_SIZE:
            continue
        # Trust the simulator: the simulator's leaf will show defeat
        # if the threat overruns my source. The _incoming_threat
        # heuristic was over-cautious — at seed 1003 turn 20 it
        # blocked a +2320-favor capture from P8 because of a 10-ship
        # threat that P8's own production would have repelled. See
        # the v6 plan in /root/.claude/plans/do-it-jolly-sprout.md.
        launch_budget = src.ships
        ranked = sorted(
            targets,
            key=lambda t: math.hypot(src.x - t.x, src.y - t.y),
        )[:NUM_TARGETS_PER_SOURCE]
        for tgt in ranked:
            capture_size = _capture_size_guess(src, tgt)
            if capture_size < MIN_FLEET_SIZE or capture_size > launch_budget:
                continue
            # Multi-size enumeration: let the formula pick the best
            # size per (src, tgt). The `+5` gate previously kept the
            # launch_budget candidate out when budget was close to
            # capture_size; that prevented the chooser from trying
            # the faster-speed full-send option (+6 favor at turn 0
            # of seed 1003). Drop it; trust the simulator.
            sizes = {capture_size}
            if 2 * capture_size <= launch_budget:
                sizes.add(2 * capture_size)
            if launch_budget > capture_size:
                sizes.add(launch_budget)
            for ships in sizes:
                s = score_action(
                    src, tgt, ships, step, me,
                    snap_base=snap_base, num_seats=num_seats,
                    baseline_favors=baseline_favors,
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

    # Build snap_base + idle baseline once per turn; per-candidate
    # clone is cheap. MAX_HORIZON is the deepest sim any candidate
    # will request (arrival can be ~25 turns for a slow long-distance
    # fleet; SIM_SETTLE_TURNS extra to let combat resolve).
    num_seats = _num_seats(planets, fleets)
    snap_base = from_obs(obs, num_seats=num_seats)
    baseline_favors = _build_idle_baseline(snap_base, player, num_seats, MAX_HORIZON)
    candidates = _enumerate_candidates(
        my_planets, targets, fleets, step, player,
        snap_base, num_seats, baseline_favors,
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
