"""main.py — our Orbit Wars agent (kaggle-submittable as-is).

v3 — formula-consistent analytic chooser. Per turn, per (src, tgt, ships):

  1. Enumerate candidate (src, tgt, ships) triples (8 nearest targets
     per owned source, min-capture sizing, threat-aware defensive
     reserve).
  2. For each candidate, apply the action analytically and compute
     Δfavor = favor(obs_after_action) − favor(obs). Uses the SAME
     F3-aware favor formula end-to-end — pre-filter, scoring, and
     leaf-eval are now one consistent function.
  3. Greedy non-dogpile match: walk candidates Δfavor-desc; each
     source emits at most one fleet; each target hit at most once.

Why this replaces v2 (10-turn fast_sim rollout chooser):

Diagnostic dump at seed 1003 turn 30 (where my agent under-launches
vs v7_0) showed three different formulas being applied to the same
Δfavor question:
  - pre-filter score_action used v1 F2 (prod × full_horizon)
  - leaf-eval favor() used v2 F2* (prod × hold_time)
  - rollout cascade of favor-greedy actions then overrode both with
    pure simulation noise (every launch leaf_favor = baseline − 2672,
    regardless of which candidate; the 10-turn fwd-sim converged to
    similar states because favor-greedy plays the same way no matter
    the turn-0 candidate).

Result: 5 candidates passed v1 pre-filter; 1 was genuinely positive
under the v2 formula (P12→P20 ×26 at +1864); rollout rejected ALL 5
because the cascade-of-favor-greedy-launches drained ships in turns
1-9 of every simulation. Hence the under-launch behaviour observed
in 3 seeds vs v7_0 — and the 0/96 across 5 favor-axis variants.

The fix: trust the formula. F3 already encodes expected hold-time
multi-turn; F2*'s prod × hold_time term already accounts for capture
duration; horizon already shrinks as the game progresses. Layering a
10-turn favor-greedy simulator on top is double-counting that gets
overridden by simulation noise.

Submit with:  ./submit.sh "message describing the change"
Eval with:    python eval.py --vs nearest -n 24
              python eval.py --vs v7_0 -n 24

Favor function: `favor.py`. (lib/fast_sim is no longer imported here;
the rollout layer was removed.)
"""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from favor import favor

# --- comp constants (from data/README.md) ----------------------------------
EPISODE_STEPS = 500
MAX_SPEED = 6.0
SUN_X = 50.0
SUN_Y = 50.0
SUN_R = 10.0

# --- chooser config --------------------------------------------------------
NUM_TARGETS_PER_SOURCE = 8     # K nearest non-owned planets considered
MIN_FLEET_SIZE = 2             # 1-ship fleets move at speed 1; rarely useful


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


def _apply_action(obs, src: Planet, tgt: Planet, ships: int, arrival: int, me: int) -> dict:
    """Build the post-action observation analytically: src loses `ships`
    in garrison and gains arrival-turns of production; tgt becomes me
    (with surplus garrison) if capture succeeds, or keeps its owner
    (with reduced garrison) if the fleet is defeated; all other owned
    planets accumulate arrival-turns of production.

    Sun-crossing destroys the fleet — src still loses the ships, tgt is
    untouched.
    """
    raw_planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    sun_kill = _crosses_sun(src.x, src.y, tgt.x, tgt.y)

    if tgt.owner == -1:
        garrison_at_arrival = int(tgt.ships)
    else:
        garrison_at_arrival = int(tgt.ships) + int(tgt.production) * arrival

    captured = (not sun_kill) and (ships > garrison_at_arrival)

    new_planets = []
    for p in raw_planets:
        p = list(p)
        if p[0] == src.id:
            p[5] = int(p[5]) - ships + int(p[6]) * arrival
        elif p[0] == tgt.id:
            if sun_kill:
                if int(p[1]) >= 0:
                    p[5] = int(p[5]) + int(p[6]) * arrival
            elif captured:
                p[1] = me
                p[5] = ships - garrison_at_arrival
            else:
                # Fleet defeated. Garrison soaks `ships`; owner unchanged.
                p[5] = max(0, garrison_at_arrival - ships)
        else:
            if int(p[1]) >= 0:
                p[5] = int(p[5]) + int(p[6]) * arrival
        new_planets.append(p)

    return {
        "planets": new_planets,
        "fleets": obs["fleets"] if isinstance(obs, dict) else obs.fleets,
        "comets": obs["comets"] if isinstance(obs, dict) else getattr(obs, "comets", []),
        "comet_planet_ids": (
            obs["comet_planet_ids"] if isinstance(obs, dict)
            else getattr(obs, "comet_planet_ids", [])
        ),
        "step": int(obs["step"] if isinstance(obs, dict) else getattr(obs, "step", 0)) + arrival,
    }


def score_action(
    src: Planet,
    tgt: Planet,
    ships: int,
    step: int,
    me: int,
    obs=None,
    favor_before: float | None = None,
) -> float:
    """Δfavor for launching `ships` from src toward tgt.

    Computed by applying the action analytically and returning
    favor(post_action_obs, me) − favor(obs, me). Uses the F3-aware
    favor end-to-end — same formula governs candidate ranking and
    leaf evaluation (the v2-rollout chooser had three different
    formulas in three places; root cause of 0/96 vs v7_0).

    `obs` is REQUIRED. Optional `favor_before` lets callers cache the
    pre-action favor across the candidate enumeration (called once
    per planet-state instead of per candidate).
    """
    if ships < MIN_FLEET_SIZE or ships > src.ships or src.id == tgt.id:
        return float("-inf")

    if tgt.owner == me:
        return 0.0   # reinforce own planet — neutral

    if obs is None:
        raise ValueError("score_action requires obs for analytic Δfavor")

    dist = max(
        0.0,
        math.hypot(src.x - tgt.x, src.y - tgt.y) - src.radius - tgt.radius,
    )
    speed = _speed(ships)
    arrival = max(1, math.ceil(dist / speed))

    post_obs = _apply_action(obs, src, tgt, ships, arrival, me)
    if favor_before is None:
        favor_before = favor(obs, me)
    return favor(post_obs, me) - favor_before


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
    obs,
    favor_before: float | None = None,
) -> list[tuple[float, Planet, Planet, int]]:
    """All (Δfavor, src, tgt, ships) triples with Δfavor > 0.

    Δfavor is the v2 F3-aware analytic delta — same formula as
    `favor()` itself. Defense: if an enemy fleet is aimed at the
    source, reserve enough garrison to repel it.
    """
    if favor_before is None:
        favor_before = favor(obs, me)
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
            s = score_action(src, tgt, ships, step, me, obs=obs, favor_before=favor_before)
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

    # Score every candidate with the v2 F3-aware analytic Δfavor (the
    # SAME function favor() uses at the leaf). No second-pass rollout —
    # the formula already encodes hold-time and horizon multi-turn.
    favor_before = favor(obs, player)
    candidates = _enumerate_candidates(
        my_planets, targets, fleets, step, player, obs, favor_before=favor_before
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
