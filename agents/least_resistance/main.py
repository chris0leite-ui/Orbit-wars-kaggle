"""least_resistance — simulation-driven forward-expansion agent for Orbit Wars.

Plain-English strategy
----------------------
Be smart by *simulating*, not by hand-tuned weights. Every turn the agent:

  1. Lists the sensible coordinated moves it could make — capture a planet
     (from one source, or several ganging up when one isn't enough), or
     stream idle ships forward toward the front. Candidates are ordered by
     the "path of least resistance to production": the most production per
     turn of travel first, with ties broken toward whatever shortens our
     distance to the nearest opponent. This ordering is the strategy's
     *flavour* — which moves we try first.

  2. Decides which of those moves to actually make by FORWARD SIMULATION.
     For each candidate, it plays the move, then rolls the whole game
     forward a dozen-plus turns with both sides following a fast greedy
     policy, and measures the result as "our ships minus theirs" at the
     horizon. A launch is committed only if it improves that simulated
     outcome. We keep adding launches (a coordinated multi-fleet plan)
     until nothing further helps.

Everything the earlier hand-tuned version needed a knob for now falls out
of the simulation for free:
  - How many ships to keep back on a planet? — Draining a planet that the
    rollout shows the opponent then captures lowers our score, so the
    simulation keeps exactly the reserve that's worth keeping.
  - Gang up or solo? — A partial wave that doesn't capture shows no gain;
    a coordinated wave that does shows a gain and gets committed as a unit.
  - Attack the enemy or grab a neutral? — Whichever simulates to the larger
    ship/production advantage over the horizon.
  - "Minimize ETA to the closest opponent" — forward moves enable more
    future expansion inside the rollout, so they score higher; the only
    explicit nod to it is the tie-break ordering.

This is exactly how our strongest agents decide (candidate enumeration +
fast-sim rollout + `delta_us_minus_them` leaf, per `lib/v7_search.py` and
the baseline trajectory chooser). The leaf value is the self-calibrating
"more production = more ships" objective — no strategy weights to tune.

Physics + machinery reused
--------------------------
  - lib.fast_sim    accurate, fast forward simulator (from_obs / step / clone)
                    + the `delta_us_minus_them` scoring head
  - lib.opp_model   `lite_greedy_policy` — the cheap (~0.02 ms) rollout policy
  - lib.aim         orbit-aware lead intercept (aim_orbiting / aim_comet)
  - lib.fleet/orbit/geometry  speed, ETA, moving-planet prediction, plus a
                    cheap `path_clears_sun` pre-filter (full path safety -
                    off-board / wrong-planet / undersized waves - is left to
                    the forward simulation, which is the exact ground truth)
  - lib.world_model comet path / lifetime helpers

The only parameters are compute bounds (simulation horizon, candidate cap,
wallclock budget) — not strategy tuning. They are set conservatively so the
agent stays well inside the 1 s/turn limit.
"""

from __future__ import annotations

import math
import os
import time

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from lib.geometry import dist, path_clears_sun
from lib.fleet import speed as fleet_speed
from lib.orbit import is_orbiting
from lib.aim import aim_orbiting, aim_comet, estimate_eta
from lib.intent import World
from lib.world_model import comet_remaining_lifetime, _comet_paths_by_id
from lib.fast_sim import from_obs, clone, step, delta_us_minus_them
from lib.opp_model import lite_greedy_policy


# --------------------------------------------------------------------------
# Compute bounds (NOT strategy weights). Horizon = how far we simulate the
# consequence of a move; the rest cap per-turn rollout cost. Defaults keep
# us well inside the 1 s/turn budget (a K=14 rollout is ~5 ms; ~24 candidate
# rollouts ≈ 120 ms). Overridable by env var only for benchmarking.
# --------------------------------------------------------------------------
def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _wallclock_ms() -> float:
    """Per-turn rollout budget, read at CALL time.

    Honours `ORBIT_WARS_PARITY_WALLCLOCK_MS` first: the bundle parity gate
    sets it to a huge value so the greedy loop never bails mid-list, making
    the agent a pure function of `obs` (timing cannot change the output).
    """
    override = os.environ.get("ORBIT_WARS_PARITY_WALLCLOCK_MS")
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    return _f("LR_WALLCLOCK_MS", 700.0)


SIM_HORIZON = _i("LR_SIM_HORIZON", 14)        # turns of forward simulation
MAX_CANDIDATES = _i("LR_MAX_CANDIDATES", 24)  # most candidate moves we simulate
VALUE_EPS = _f("LR_VALUE_EPS", 1.0)           # min simulated ship gain to commit
STAGING_MIN_EXCESS = _i("LR_STAGING_MIN_EXCESS", 8)  # don't bother simulating tiny stages
FRONTIER_REF_SHIPS = _f("LR_FRONTIER_REF_SHIPS", 30.0)  # reference for the ETA metric
RANK_HINT_SHIPS = 20                          # nominal fleet size for ranking-pass aim


# --------------------------------------------------------------------------
# Obs parsing.
# --------------------------------------------------------------------------
def _as_dict(obs) -> dict:
    if isinstance(obs, dict):
        return obs
    return {
        "player": getattr(obs, "player", 0),
        "step": getattr(obs, "step", 0),
        "planets": list(getattr(obs, "planets", []) or []),
        "fleets": list(getattr(obs, "fleets", []) or []),
        "comets": list(getattr(obs, "comets", []) or []),
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", []) or []),
        "angular_velocity": float(getattr(obs, "angular_velocity", 0.0) or 0.0),
    }


def _num_seats(planets, fleets) -> int:
    max_owner = -1
    for p in planets:
        if int(p.owner) > max_owner:
            max_owner = int(p.owner)
    for f in fleets:
        if int(f.owner) > max_owner:
            max_owner = int(f.owner)
    return 4 if max_owner >= 2 else 2


# --------------------------------------------------------------------------
# Aim with the correct accurate physics per body type.
# --------------------------------------------------------------------------
def _plan_shot(src, tgt, world, comet_paths, omega, ships):
    """Return (aim_angle, eta_turns, arrival_xy) or None if no intercept."""
    s_xy = (float(src.x), float(src.y))
    t_tuple = [int(tgt.id), int(tgt.owner), float(tgt.x), float(tgt.y),
               float(tgt.radius), float(tgt.ships), float(tgt.production)]
    ships = max(1, int(ships))

    if int(tgt.id) in world.comet_ids:
        entry = comet_paths.get(int(tgt.id))
        if entry is None:
            return None
        path, path_index = entry
        res = aim_comet(s_xy, float(src.radius), t_tuple, float(tgt.radius),
                        ships, path, path_index)
    elif omega != 0.0 and is_orbiting(t_tuple):
        res = aim_orbiting(s_xy, float(src.radius), t_tuple, float(tgt.radius),
                           ships, omega)
    else:
        eta_f = estimate_eta(s_xy, float(src.radius), (float(tgt.x), float(tgt.y)),
                             float(tgt.radius), ships)
        if eta_f is None:
            return None
        angle = math.atan2(float(tgt.y) - float(src.y),
                           float(tgt.x) - float(src.x))
        res = (angle, (float(tgt.x), float(tgt.y)), eta_f)

    if res is None:
        return None
    angle, arrival_xy, eta_f = res
    return angle, max(1, int(math.ceil(eta_f))), arrival_xy


def _sun_clear(src, arrival_xy) -> bool:
    """Cheap geometry pre-filter: does the straight path from `src` to the
    intercept point stay clear of the sun?

    This is the only physics gate we apply up front. Off-board overshoot,
    collisions with the wrong planet, and undersized waves are all left to
    the forward simulation to judge — a fleet that dies or bounces simply
    shows up as a lower horizon ship-delta, so the rollout rejects it. The
    simulation is the exact ground truth (it runs the env's interpreter), so
    leaning on it is both simpler and more accurate than a separate ray-cast.
    """
    return path_clears_sun((float(src.x), float(src.y)), arrival_xy)


# --------------------------------------------------------------------------
# The agent.
# --------------------------------------------------------------------------
def agent(obs, configuration=None):
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    raw_planets = obs_d.get("planets", []) or []
    if not raw_planets:
        return []
    raw_fleets = obs_d.get("fleets", []) or []

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if int(p.owner) == me]
    targets = [p for p in planets if int(p.owner) != me]
    if not my_planets or not targets:
        return []

    omega = float(obs_d.get("angular_velocity", 0.0) or 0.0)
    num_seats = _num_seats(planets, fleets)
    world = World.from_obs(obs_d)
    comet_paths = _comet_paths_by_id(world) if world.comet_ids else {}

    # Forward simulator + the leaf value our strong agents use.
    snap = from_obs(obs, configuration, num_seats=num_seats)
    pol = lite_greedy_policy

    def value(launches) -> float:
        """Our-minus-their ships at the horizon if we play `launches` this
        turn, then both sides follow the fast greedy policy."""
        s = clone(snap)
        first = [list(launches) if i == me else pol(s.state[i].observation)
                 for i in range(num_seats)]
        step(s, first, in_place=True)
        for _ in range(SIM_HORIZON - 1):
            if s.fake_env.done:
                break
            acts = [pol(s.state[i].observation) for i in range(num_seats)]
            step(s, acts, in_place=True)
        return delta_us_minus_them(s, me)

    # ----------------------------------------------------------------------
    # Candidate moves. Each is a coordinated launch-set (1 fleet, or several
    # ganging up on one target). Built with the accurate physics so we only
    # ever propose shots that actually reach their target.
    # ----------------------------------------------------------------------
    opp_xy = [(float(p.x), float(p.y)) for p in planets
              if int(p.owner) != me and int(p.owner) != -1]
    ref_speed = max(1e-6, fleet_speed(FRONTIER_REF_SHIPS))

    def frontier_eta(xy):
        if not opp_xy:
            return 0.0
        return min(dist(xy, o) for o in opp_xy) / ref_speed

    available = {int(p.id): int(p.ships) for p in my_planets}
    candidates = []   # each: {"launches": [[src,angle,ships],...], "srcs": {sid:ships},
                      #        "rank": prod/eta, "front": eta_to_opp}

    for tgt in targets:
        tid = int(tgt.id)
        is_enemy = int(tgt.owner) != -1
        is_comet = tid in world.comet_ids
        prod = float(tgt.production)

        shots = []   # (eta, capture_size, sid, src, angle)
        for src in my_planets:
            shot = _plan_shot(src, tgt, world, comet_paths, omega, RANK_HINT_SHIPS)
            if shot is None:
                continue
            angle, eta, arr = shot
            if not _sun_clear(src, arr):
                continue
            if is_comet:
                life = comet_remaining_lifetime(tid, world)
                if life is not None and life <= eta:
                    continue
            # Cheap capture-size estimate (neutrals don't grow; enemies do).
            # The rollout reveals undersizing (a bounce lowers our score), so
            # this only needs to be in the right ballpark.
            defenders = prod * eta + tgt.ships if is_enemy else tgt.ships
            size = int(math.ceil(defenders)) + 1
            shots.append((eta, size, int(src.id), src, angle))
        if not shots:
            continue

        shots.sort(key=lambda x: x[0])     # nearest (smallest ETA) first
        rank = prod / max(1.0, shots[0][0])
        front = frontier_eta((float(tgt.x), float(tgt.y)))

        # Solo capture from the cheapest affordable source.
        solo = None
        for (eta, size, sid, src, angle) in shots:
            if available[sid] >= size:
                solo = {"launches": [[sid, float(angle), size]],
                        "srcs": {sid: size}, "rank": rank, "front": front}
                break
        if solo is not None:
            candidates.append(solo)
            continue

        # No single source can afford it — gang up the nearest sources.
        need = shots[0][1]
        launches, srcs, acc = [], {}, 0
        for (eta, size, sid, src, angle) in shots:
            if sid in srcs:
                continue
            take = min(available[sid], need - acc)
            if take <= 0:
                continue
            launches.append([sid, float(angle), take])
            srcs[sid] = take
            acc += take
            if acc >= need:
                break
        if acc >= need and launches:
            candidates.append({"launches": launches, "srcs": srcs,
                               "rank": rank, "front": front})

    # Forward-staging moves: stream a rear planet's idle excess toward the
    # friendly planet nearest the front. The rollout keeps these only if the
    # repositioned ships earn their keep within the horizon.
    if opp_xy and len(my_planets) >= 2:
        fe = {int(p.id): frontier_eta((float(p.x), float(p.y))) for p in my_planets}
        for src in my_planets:
            sid = int(src.id)
            if sid in world.comet_ids:
                continue
            excess = available[sid]
            if excess < STAGING_MIN_EXCESS:
                continue
            best, best_f = None, fe[sid]
            for q in my_planets:
                qid = int(q.id)
                if qid == sid or qid in world.comet_ids:
                    continue
                if fe[qid] < best_f:
                    best_f, best = fe[qid], q
            if best is None:
                continue
            shot = _plan_shot(src, best, world, comet_paths, omega, excess)
            if shot is None:
                continue
            angle, _eta, arr = shot
            if not _sun_clear(src, arr):
                continue
            candidates.append({"launches": [[sid, float(angle), excess]],
                               "srcs": {sid: excess}, "rank": 0.0,
                               "front": best_f})

    if not candidates:
        return []

    # Order by the path of least resistance to production (most production
    # per turn of travel), tie-broken toward the front. This decides only
    # which moves we *try first*; the rollout decides which we *keep*.
    candidates.sort(key=lambda c: (-c["rank"], c["front"]))
    candidates = candidates[:MAX_CANDIDATES]

    # ----------------------------------------------------------------------
    # Greedy plan construction by simulated value. Add a candidate only if it
    # improves the horizon ship-delta over the plan so far; stop when nothing
    # helps (or the wallclock budget runs out).
    # ----------------------------------------------------------------------
    committed = []
    avail = dict(available)
    current = value([])           # value of doing nothing this turn
    budget_ms = _wallclock_ms()
    t_start = time.perf_counter()

    for c in candidates:
        if (time.perf_counter() - t_start) * 1000.0 > budget_ms:
            break
        if any(avail.get(s, 0) < sz for s, sz in c["srcs"].items()):
            continue
        trial = committed + [list(l) for l in c["launches"]]
        v = value(trial)
        if v > current + VALUE_EPS:
            committed = trial
            current = v
            for s, sz in c["srcs"].items():
                avail[s] = avail.get(s, 0) - sz

    return committed
