"""main.py — our Orbit Wars agent (kaggle-submittable as-is).

v2 — multi-step rollout chooser. Per turn:

  1. Enumerate candidate (src, tgt, ships) triples (8 nearest targets
     per owned source, min-capture sizing, threat-aware defensive
     reserve). Pre-filter via analytic Δfavor (`score_action`).
  2. For the top-K candidates by analytic score, run an N-turn
     forward rollout via `lib.fast_sim` with `nearest_sniper` policy
     for the opponent (and for our other sources beyond the candidate).
     Score = favor(leaf_state, me) − favor(current_state, me).
  3. Greedy non-dogpile match: walk candidates rollout-score-desc;
     each source used at most once, each target hit at most once.

Why rollout: the favor function predicts winners at AUC 0.945 on
saved states but greedy 1-step Δfavor cannot beat v7_0 (0 / 24).
Greedy on V is suboptimal when V ≠ V*; multi-step lookahead lets the
chooser see consequences (e.g., enemy recapturing a weakly-held
new planet) that 1-step misses.

Submit with:  ./submit.sh "message describing the change"
Eval with:    python eval.py --vs nearest -n 24
              python eval.py --vs v7_0 -n 24
              python eval.py --panel -n 24

Favor function: `favor.py`. Simulator: `lib/fast_sim.py` (parity-tested
vs the kaggle env; see `tests/test_fast_sim_parity.py`).
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

# --- favor weights (mirrors favor.FavorConfig defaults) --------------------
ALPHA = 1.0   # weight on ship-lead Δ
BETA = 1.0    # weight on (production lead × turns_remaining) Δ

# --- chooser config --------------------------------------------------------
NUM_TARGETS_PER_SOURCE = 8     # K nearest non-owned planets considered
MIN_FLEET_SIZE = 2             # 1-ship fleets move at speed 1; rarely useful
ROLLOUT_TURNS = 10             # depth of forward simulation per candidate
ROLLOUT_TOP_K = 20             # top-K analytic-scored candidates that get rolled


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


def score_action(
    src: Planet,
    tgt: Planet,
    ships: int,
    step: int,
    me: int,
) -> float:
    """Analytic Δfavor for launching `ships` from src toward tgt.

    Outcomes (per data/README.md "Combat"):
      - reinforce own planet:       ΔF1=0,   ΔF2=0
      - sun-crossing path:          ΔF1=−ships  (all destroyed)
      - capture neutral (K > G):    ΔF1=−G,  ΔF2=+T.prod × horizon_after
      - capture enemy (K > G):      ΔF1=0    (symmetric), ΔF2=+2·T.prod × horizon_after
      - fail / tie (K ≤ G):         ΔF1=−ships, ΔF2=0
    """
    if ships < MIN_FLEET_SIZE or ships > src.ships or src.id == tgt.id:
        return float("-inf")

    # Reinforcing own planet — net neutral for v0 (ships still ours).
    if tgt.owner == me:
        return 0.0

    # Distance the fleet actually travels (centre-to-centre minus both radii).
    dist = max(
        0.0,
        math.hypot(src.x - tgt.x, src.y - tgt.y) - src.radius - tgt.radius,
    )
    speed = _speed(ships)
    arrival_turns = math.ceil(dist / speed) if speed > 0 else 1

    # Sun crossing destroys the fleet.
    if _crosses_sun(src.x, src.y, tgt.x, tgt.y):
        return ALPHA * (-ships)

    # Garrison at arrival.
    if tgt.owner == -1:                    # neutral, doesn't grow
        garrison_at_arrival = tgt.ships
    else:                                  # enemy, grows by production
        garrison_at_arrival = tgt.ships + tgt.production * arrival_turns

    horizon_after = max(0, EPISODE_STEPS - step - arrival_turns)

    if ships > garrison_at_arrival:
        # Capture.
        if tgt.owner == -1:
            d_f1 = -garrison_at_arrival           # ships destroyed in combat
            d_f2 = tgt.production * horizon_after
        else:
            d_f1 = 0                              # we lose G, enemy loses G — symmetric
            d_f2 = 2 * tgt.production * horizon_after
    else:
        # Fail or tie: entire fleet destroyed; no ownership change.
        d_f1 = -ships
        d_f2 = 0

    return ALPHA * d_f1 + BETA * d_f2


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


def _enumerate_candidates(
    my_planets: list[Planet],
    targets: list[Planet],
    fleets: list[Fleet],
    step: int,
    me: int,
) -> list[tuple[float, Planet, Planet, int]]:
    """All (score, src, tgt, ships) triples with score > 0.

    Defense: if an enemy fleet is aimed at the source, reserve enough
    garrison to repel it. Ship budget for launches = max(0, src.ships −
    incoming_threat). This avoids the failure mode where we strip a
    source of garrison to capture a neutral and then lose the source
    to an arriving enemy fleet.
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
            ships = capture_size  # min-capture; matches nearest's expansion velocity
            s = score_action(src, tgt, ships, step, me)
            if s > 0.0:
                out.append((s, src, tgt, ships))
    out.sort(key=lambda c: -c[0])
    return out


# ---------------------------------------------------------------------------
# Rollout scoring (uses lib.fast_sim, parity-tested vs the kaggle env)
# ---------------------------------------------------------------------------


def _nearest_sniper_moves(obs_struct, player: int) -> list[list]:
    """Re-implementation of `baselines/nearest.py` operating on a
    Snapshot's observation (Struct or dict). Used as the rollout
    policy for both the opponent and our own non-candidate sources.
    """
    raw_planets = (
        obs_struct.planets if hasattr(obs_struct, "planets")
        else obs_struct.get("planets", [])
    )
    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    if not my_planets or not targets:
        return []
    moves = []
    for mine in my_planets:
        nearest = min(targets, key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y))
        ships_needed = nearest.ships + 1
        if mine.ships >= ships_needed:
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])
    return moves


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


def _rollout_score(
    snap_base,
    our_action: list | None,
    me: int,
    num_seats: int,
    n_turns: int,
) -> float:
    """Run an N-turn rollout starting from `snap_base`, applying
    `our_action` on the first turn for seat `me` and `nearest_sniper`
    for everyone else (and for seat `me` on all subsequent turns).
    Return favor(leaf_state, me).

    Holds opponent + our-other-sources to a fixed nearest-sniper
    policy so candidate scores are comparable. Pessimistic vs our
    actual chooser, but correct for ranking.
    """
    snap = fs_clone(snap_base)

    # Turn 0: candidate action for me; nearest for everyone else.
    actions = []
    for seat in range(num_seats):
        if seat == me:
            actions.append([our_action] if our_action else [])
        else:
            actions.append(_nearest_sniper_moves(snap.state[seat].observation, seat))
    snap = fs_step(snap, actions, in_place=True)

    # Turns 1..n_turns-1: everyone plays nearest.
    for _ in range(n_turns - 1):
        if snap.fake_env.done:
            break
        actions = [
            _nearest_sniper_moves(snap.state[seat].observation, seat)
            for seat in range(num_seats)
        ]
        snap = fs_step(snap, actions, in_place=True)

    return favor(snap.state[me].observation, me)


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

    # Pre-filter candidates with cheap analytic Δfavor (existing v1 logic).
    analytic = _enumerate_candidates(my_planets, targets, fleets, step, player)
    if not analytic:
        return []  # no positive-analytic-score candidate; hold

    # Cap to the top-K analytic-score candidates so rollout cost stays
    # under budget. ~20 candidates × 10 turns × ~1 ms = ~200 ms / turn.
    analytic = analytic[:ROLLOUT_TOP_K]

    # Build the Snapshot once for this turn; clone-per-candidate is cheap.
    num_seats = _num_seats(planets, fleets)
    snap_base = from_obs(obs, num_seats=num_seats)

    # Baseline rollout: do nothing this turn; everyone plays nearest going
    # forward. Candidates must beat this baseline to be worth launching.
    baseline_favor = _rollout_score(snap_base, None, player, num_seats, ROLLOUT_TURNS)

    # Rollout-score each candidate.
    rolled: list[tuple[float, Planet, Planet, int]] = []
    for _analytic_s, src, tgt, ships in analytic:
        angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
        action = [src.id, angle, ships]
        leaf = _rollout_score(snap_base, action, player, num_seats, ROLLOUT_TURNS)
        delta = leaf - baseline_favor
        if delta > 0:
            rolled.append((delta, src, tgt, ships))

    rolled.sort(key=lambda c: -c[0])

    # Greedy non-dogpile: each source emits at most one fleet; each target
    # is the destination of at most one fleet this turn.
    used_srcs: set[int] = set()
    used_tgts: set[int] = set()
    moves: list[list] = []
    for _delta, src, tgt, ships in rolled:
        if src.id in used_srcs or tgt.id in used_tgts:
            continue
        used_srcs.add(src.id)
        used_tgts.add(tgt.id)
        angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
        moves.append([src.id, angle, ships])
    return moves
