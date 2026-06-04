"""protoflow — action-space value field (PROBE, not a submission).

PI reframe (2026-06-04): the field is NOT a spatial potential over planet
positions — it lives in the ACTION SPACE. The thing we score and follow a
gradient through is the set of moves we can actually make from the current
state: which launch trajectories are feasible right now, and which captures
matter most based on production and location. A position-pull sends senseless
tiny/far fleets because it does not know which actions are feasible or worth
it; an action field can never contain such a move, because it is not feasible
or not value-positive.

Build:
  1. FIELD = the feasible-action set. Reuse the champion's `propose()` — it
     returns every move we can make this turn, already trajectory-feasible and
     already sized so each one is a real capture (no 2-ship bouncers, no
     sun-crossers, no hopeless far shots past the reach ceiling). That kills the
     waste of the spatial prototype by construction. A same-arrival cohort
     SUPPLEMENT covers defended targets no single planet can solo-take (combat
     rule 1 sums same-owner same-turn arrivals — the teamwork-coalition regime).
  2. HEIGHT = value per action in ONE currency: ships added to the final
     zero-sum margin = mult * production(target) * (500 - arrival_turn) *
     winnability. mult=2 for an enemy planet (gain the production stream AND
     deny it), else 1. (500 - arrival) is the production we collect by holding
     the target to game end -- and because a smaller fleet is slower (speed
     law) it arrives later, so slowness is priced automatically and needs no
     near-bias term. winnability is the engine's floored reach-race class.
  3. GRADIENT-FOLLOW with a HOLD gate: per (source, target) a fire-now action
     is admitted only if its value is >= the best wait-then-mass candidate for
     that pair; if waiting dominates the ships are held. Then greedily take the
     highest-value admitted actions under a per-source budget (a planet keeps a
     defensive reserve), and a same-arrival cohort supplement merges legs only
     when that is what wins a still-uncaptured target.

Probe lens (PI): do the champion's good properties — no wasted ships, no two
small fleets, no shots so far the opponent reacts easily — EMERGE NATURALLY
here? `scripts/protoflow_probe.py` reads the per-turn trace below and reports
tiny-fleet / far-shot / idle / convergence rates plus winrate vs light-greedy.

Imports lib/* and agents.baseline.* directly (fine for local A/B; NOT bundled).
"""
from __future__ import annotations

import math
from collections import defaultdict

from lib.intent import World
from lib.world_model import WorldModel, predict_arrival_contest
from lib.kinematic_table import KinematicTable
from lib.aim import aim_orbiting
from lib.fleet import travel_time
from lib.trajectory import predict_fleet_fate
from agents.baseline.proposer import (
    propose, capture_size, capture_floor_arrival, aim_and_eta,
)

EPISODE_STEPS = 500
WIN_FLOOR = 0.10       # winnability never reaches zero -> the field never freezes
RACE_SCALE = 6.0       # turns; steepness of the graded reach-race confidence
REACH_CEIL = 26        # feasibility: no capture whose (wait+eta) exceeds this (no overreach)
# Per-turn probability that a launched plan survives one turn of opponent action
# without being invalidated. A fleet in the air for tau turns survives to arrival
# with probability SURVIVAL_PER_TURN**tau -- this is how the field prices the
# world drifting (entropy) while the fleet flies. Closer to 1 = more patient/
# trusting of long flights; lower = sharper near-preference. Calibrated on S4/S5.
SURVIVAL_PER_TURN = 0.97
THREAT_ANGLE = 0.45    # rad (~26 deg); enemy fleet "incoming" if heading within this of the bearing
THREAT_RANGE = 60.0    # board units; only nearby enemy fleets count as a committed threat
MAX_CONVERGE_TARGETS = 8   # bound the cohort supplement scan

# Per-game trace for the probe runner. Each launch is a dict (see bottom).
_TRACE: list[dict] = []
# Last turn's ranked action field, for synthetic-situation calibration.
# Each entry: {"src","tgt","ships","ttc","imp","tgt_owner","prod"}.
_LAST_FIELD: list[dict] = []


def reset_trace() -> None:
    _TRACE.clear()


def get_trace() -> list[dict]:
    return list(_TRACE)


def get_last_field() -> list[dict]:
    return list(_LAST_FIELD)


def _sigmoid(x: float) -> float:
    if x < -60.0:
        return 0.0
    if x > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _parse_fleets(obs):
    raw = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
    return [tuple(f) for f in raw]  # (id, owner, x, y, angle, from_planet_id, ships)


def _incoming_threat(planet, enemy_fleets) -> float:
    """Sum of enemy-fleet ships currently heading at `planet` from within range."""
    total = 0.0
    for f in enemy_fleets:
        fx, fy, fang, fships = f[2], f[3], f[4], f[6]
        d = math.hypot(planet.x - fx, planet.y - fy)
        if d > THREAT_RANGE:
            continue
        bearing = math.atan2(planet.y - fy, planet.x - fx)
        diff = abs(((bearing - fang + math.pi) % (2.0 * math.pi)) - math.pi)
        if diff < THREAT_ANGLE:
            total += fships
    return total


def agent(obs, configuration=None):
    world = World.from_obs(obs)
    me = world.my_id
    step = world.step
    omega = world.omega
    remain = max(1, EPISODE_STEPS - step)

    world._kt = KinematicTable()
    world._kt.begin_turn(world)
    model = WorldModel.from_world(world)

    planets = list(world.planets_by_id.values())
    my_planets = [p for p in planets if int(p.owner) == me]
    enemy_planets = [p for p in planets if int(p.owner) not in (-1, me)]
    enemy_fleets = [f for f in _parse_fleets(obs) if f[1] not in (-1, me)]
    if not my_planets:
        _TRACE.append({"step": step, "launches": [], "idle": True,
                       "sources": 0, "my_planets": 0, "my_ships": 0})
        return []

    # OFFENSE field targets only: enemy planets + neutrals. Own planets are NOT
    # offense targets -- defending them is a separate concern handled by the
    # per-source reserve below (`spare` already withholds ships to cover committed
    # incoming threat). Putting own planets in the offense ranking made the agent
    # trickle 3 ships between its own planets (the friendly-reinforce pollution
    # seen in the synthetic states).
    target_pool = enemy_planets + [p for p in planets if int(p.owner) == -1]

    # --- per-source budget: keep enough to cover committed incoming threat ---
    threat = {p.id: _incoming_threat(p, enemy_fleets) for p in my_planets}
    spare = {p.id: max(0, int(p.ships) - int(math.ceil(threat[p.id]))) for p in my_planets}
    planet_by_id = {int(p.id): p for p in planets}

    # --- single currency: ships added to the final zero-sum margin -------------
    # value(target, arrival_turn) = mult * production * (500 - arrival) * winnability
    #   production * (500 - arrival)  = the production stream we collect by holding
    #                                   the target for the rest of the game.
    #   mult = 2 for an enemy planet (we gain the stream AND deny it to the enemy;
    #          this is the principled origin of the old "enemy = 2x" weight), else 1.
    #   winnability = floored reach-race probability from the engine's arrival-
    #          contest classifier (never 0, so the field never freezes).
    # Slowness needs NO separate penalty: a smaller fleet is slower (speed law),
    # so it arrives later -> larger `arrival` -> smaller (500 - arrival) -> less
    # value. The old explicit near-bias term is dropped (it would double-count).
    def winnability(t, arrival_turn):
        # Probability the action delivers the value we priced. Independent failure
        # modes multiply:  floor + (1-floor) * time_survival * race_conf * path_clear
        ac = predict_arrival_contest(model, world, int(t.id), int(arrival_turn), me)
        # race_conf: graded lead over the opponent's earliest contest turn. A wide
        # lead -> ~1; arriving at/after the contest -> ~0. This rewards capturing
        # EARLY (before the garrison grows / before a counter), and folds in target
        # contestedness ("where we aim") via opp_earliest_contest_tick. None = the
        # opponent cannot contest at all (uncontestable -> full confidence).
        opp = ac.opp_earliest_contest_tick
        if opp is None:
            race_conf = 1.0
        else:
            race_conf = _sigmoid((float(opp) - float(arrival_turn)) / RACE_SCALE)
        # time_survival: the world drifts while the fleet is in the air; longer
        # flight = more divergence from this prediction (the entropy term).
        time_survival = SURVIVAL_PER_TURN ** float(arrival_turn)
        # path_clear: interception risk from enemy mass near the flight corridor.
        # DEFERRED (next factor) -- defined here at 1.0 so the structure is in place.
        path_clear = 1.0
        return WIN_FLOOR + (1.0 - WIN_FLOOR) * time_survival * race_conf * path_clear

    def value(t, arrival_turn):
        owner = int(t.owner)
        mult = 2.0 if owner not in (-1, me) else 1.0
        # Remaining production turns we collect AFTER arrival: (turns left in the
        # game) minus the flight time. Uses `remain` (= 500 - step), not the bare
        # 500, so late-game captures are not over-valued.
        horizon = max(0.0, float(remain) - float(arrival_turn))
        return mult * int(t.production) * horizon * winnability(t, arrival_turn)

    # ============================================================
    # FIELD (1): the champion's feasible solo-capture action set.
    # propose -> (cheap_delta, src, tgt, ships, angle, eta, horizon, wait_N)
    # ============================================================
    prerank = propose(my_planets, target_pool, world, model, me, omega,
                      baseline_len=max(2, remain))
    # Score each feasible action in the single currency; drop overreach (feasibility).
    actions = []
    for _cheap, src, tgt, ships, angle, eta, _hz, wait in prerank:
        ttc = float(wait) + float(eta)
        if ttc > REACH_CEIL:
            continue
        arrive = int(math.ceil(ttc))
        actions.append({
            "src": src, "tgt": tgt, "ships": int(ships), "angle": float(angle),
            "eta": float(eta), "wait": int(wait), "arrive": arrive,
            "value": value(tgt, arrive),
        })
    actions.sort(key=lambda a: -a["value"])
    _LAST_FIELD.clear()
    for a in actions:
        _LAST_FIELD.append({
            "src": int(a["src"].id), "tgt": int(a["tgt"].id), "ships": a["ships"],
            "ttc": round(float(a["wait"]) + float(a["eta"]), 1),
            "imp": round(a["value"], 1),   # wire key kept "imp" so the harness reads it
            "win": round(winnability(a["tgt"], a["arrive"]), 3),
            "wait": int(a["wait"]),
            "tgt_owner": int(a["tgt"].owner), "prod": int(a["tgt"].production),
        })

    # ============================================================
    # HOLD vs FIRE: per (src, tgt) admit the fire-now action only if it is worth
    # at least as much (in currency) as the best wait-then-mass candidate for the
    # same pair. If waiting strictly dominates we emit nothing for that pair -> the
    # ships are HELD (stay in spare to accumulate or fund a better target). This is
    # where "slow launches are improbable" comes from: a slow fire-now fleet that
    # loses the race (winnability=floor) or lands too late is dominated by a later,
    # faster, massed fleet -- yet a free-neutral grab has no wait rival (propose
    # yields none for an armed source vs a bankable target) and so fires now.
    # ============================================================
    floor_cache: dict[tuple[int, int], int] = {}

    def capture_floor(s, t):
        key = (int(s.id), int(t.id))
        f = floor_cache.get(key)
        if f is None:
            f = capture_floor_arrival(s, t, model, omega, me, world)
            floor_cache[key] = f
        return f

    best_fire: dict[tuple[int, int], dict] = {}
    best_wait_val: dict[tuple[int, int], float] = {}
    for a in actions:
        sid, tid = int(a["src"].id), int(a["tgt"].id)
        key = (sid, tid)
        if a["wait"] == 0:
            # A fire-now action is a real capture ONLY if the source can put a
            # floor-clearing fleet on the target this turn; a sub-floor fleet
            # reaches the planet but loses the combat (it bounces), so it is not a
            # capture. We do not trust propose's deduped ship count here -- we
            # size the emitted fleet to the floor ourselves (below), so the gate
            # is "can the source afford the floor", not "did propose pick >=floor".
            # This is what makes a slow/small launch improbable: a source that
            # cannot field a capturing fleet simply has no fire-now action; the
            # target waits to mass (its wait candidate) or falls to a cohort.
            floor = capture_floor(a["src"], a["tgt"])
            if spare.get(sid, 0) < floor:
                continue
            a["floor"] = floor
            cur = best_fire.get(key)
            if cur is None or a["value"] > cur["value"]:
                best_fire[key] = a
        elif a["value"] > best_wait_val.get(key, float("-inf")):
            best_wait_val[key] = a["value"]
    admitted = [
        a for key, a in best_fire.items()
        if a["value"] >= best_wait_val.get(key, float("-inf"))
    ]
    admitted.sort(key=lambda a: -a["value"])

    moves: list[list] = []
    launches: list[dict] = []
    committed_tgt: set[int] = set()

    def emit(s, t, angle, send, eta, arrive, kind, floor):
        if send <= 0:
            return False
        if predict_fleet_fate(s, t, angle, int(send), world).outcome != "target":
            return False  # path blocked (sun / wrong planet / oob)
        moves.append([int(s.id), float(angle), int(send)])
        launches.append({
            "src": int(s.id), "tgt": int(t.id), "ships": int(send),
            "eta": round(eta, 1), "arrive_turn": int(arrive),
            "dist": round(math.hypot(t.x - s.x, t.y - s.y), 1),
            "tgt_owner": int(t.owner), "kind": kind, "floor": int(floor),
        })
        spare[int(s.id)] -= int(send)
        return True

    # GRADIENT-FOLLOW: take the best admitted (fire-now-wins) actions under budget.
    # Size each fleet to clear the arrival floor (re-aiming for the real count, so
    # the fleet actually captures); surplus above the floor is left in spare for
    # the next-best target rather than padding one capture (full-drain sizing is a
    # later lever). propose's deduped count is only a lower bound here.
    for a in admitted:
        tid, sid = int(a["tgt"].id), int(a["src"].id)
        if tid in committed_tgt:
            continue  # target already won this turn
        floor = int(a["floor"])
        if spare.get(sid, 0) < floor:
            continue  # can't field a capturing fleet without overdrawing the reserve
        send = max(int(a["ships"]), floor)
        send = min(send, spare[sid])
        angle, eta = aim_and_eta(a["src"], a["tgt"], send, omega, world=world)
        arrive = int(math.ceil(float(eta)))
        if emit(a["src"], a["tgt"], angle, send, eta, arrive,
                "off" if int(a["tgt"].owner) != me else "def", floor):
            committed_tgt.add(tid)

    # ============================================================
    # FIELD (2): convergence supplement. Defended targets that NO single planet
    # could solo-fund never appear in propose; assemble a same-arrival cohort
    # from remaining spare (combat rule 1 sums the legs). This is the only place
    # multi-stream alignment can arise, and only when it actually wins a target.
    # ============================================================
    # A target that appeared in the field but was HELD (fire-now lost to its wait
    # rival) or simply went unfunded is still eligible here -- the only exclusion
    # is a target we already won this turn. (Pre-HOLD this used a `solo_tids` set
    # built from every field entry, which shadowed held targets out of cohorts.)
    extra = [t for t in target_pool
             if int(t.owner) != me and int(t.id) not in committed_tgt]
    # rank the supplement scan by a cheap value proxy so we bound the cost
    extra.sort(key=lambda t: -int(t.production))
    for t in extra[:MAX_CONVERGE_TARGETS]:
        legs = []  # (src, angle, eta)
        for s in my_planets:
            if spare.get(int(s.id), 0) <= 0:
                continue
            res = aim_orbiting((s.x, s.y), s.radius, t, t.radius,
                               max(1, spare[int(s.id)]), omega)
            if res is None:
                continue
            ang, _arr_xy, eta = res
            if eta > REACH_CEIL:
                continue
            legs.append((s, ang, eta))
        if len(legs) < 2:
            continue  # not a convergence opportunity
        by_turn: dict[int, list] = defaultdict(list)
        for s, ang, eta in legs:
            by_turn[int(math.ceil(eta))].append((s, ang, eta))
        for arrive in sorted(by_turn):
            cohort = by_turn[arrive]
            floor = capture_size(cohort[0][0], t, model, omega, me, world)
            if sum(spare[int(s.id)] for s, _, _ in cohort) < floor:
                continue
            cohort.sort(key=lambda c: c[2])  # nearest-first
            need = floor
            for s, ang, eta in cohort:
                if need <= 0:
                    break
                send = min(spare[int(s.id)], need)
                if emit(s, t, ang, send, eta, arrive, "conv", floor):
                    need -= send
            committed_tgt.add(int(t.id))
            break

    _TRACE.append({
        "step": step,
        "launches": launches,
        "idle": len(launches) == 0,
        "sources": len(my_planets),
        "my_planets": len(my_planets),
        "my_ships": int(sum(p.ships for p in my_planets)),
    })
    return moves
