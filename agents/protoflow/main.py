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
  1. FIELD = our OWN feasible-capture set. We enumerate candidates ourselves
     using HARD feasibility only (the trajectory clears the sun / reaches the
     target). We do NOT call the champion's `propose()`: its default-ON
     defensive filters (hold-feasibility / source-survival) DELETE contested
     captures before the field can judge them (verified: a 40-ship home next to
     a capturable enemy planet yields zero propose candidates with the filters
     on). A same-arrival cohort SUPPLEMENT covers defended targets no single
     planet can solo-take (combat rule 1 sums same-owner same-turn arrivals).
  2. HEIGHT = TWO-SIDED value in ONE currency: the margin swing in ships =
     mult * production * (turns_left - arrival) * winnability, where mult = 2
     when NOT acting cedes the target to the opponent (an enemy planet, OR a
     neutral the opponent will take -- we gain the stream AND deny it) and 1 for
     a neutral that would stay neutral. This is what lets the field price the
     cost of inaction: holding while the opponent expands is now negative-sum,
     so the agent acts under pressure instead of going inert.
  3. GRADIENT-FOLLOW with a HOLD gate: per (source, target) a fire-now action is
     admitted only if its value is >= the best wait-then-mass candidate for that
     pair; else the ships are held. A separate DEFENSE pass reinforces genuinely
     threatened own planets first. Then greedily take the highest-value admitted
     actions under a per-source reserve, plus the cohort supplement.

Probe lens (PI): does force concentration / sound expansion EMERGE NATURALLY?
`scripts/protoflow_probe.py` reads the per-turn trace and reports tiny-fleet /
far-shot / idle / convergence rates plus winrate vs light-greedy and Producer.

Imports lib/* and agents.baseline.* directly (fine for local A/B; NOT bundled).
"""
from __future__ import annotations

import math
from collections import defaultdict

from lib.intent import World
from lib.world_model import WorldModel, WAVE_LOOKAHEAD, predict_arrival_contest
from lib.kinematic_table import KinematicTable
from lib.aim import aim_orbiting
from lib.trajectory import predict_fleet_fate
from agents.baseline.proposer import (
    capture_floor_arrival, aim_and_eta, nearest_k,
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
# A neutral the opponent can contest within (our_arrival + CONTEST_MARGIN) is
# treated as one they would take if we did nothing -> doubled value (gain + deny).
CONTEST_MARGIN = 2
MIN_FLEET_SIZE = 2          # no sub-2-ship launches
NUM_TARGETS_PER_SOURCE = 8  # nearest-k targets per source (bounds the generator)
WAIT_HORIZONS = (2, 4, 8)   # accumulate-then-strike windows for the HOLD comparison
MAX_CONVERGE_TARGETS = 8    # bound the cohort supplement scan

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
    if not my_planets:
        _TRACE.append({"step": step, "launches": [], "idle": True,
                       "sources": 0, "my_planets": 0, "my_ships": 0})
        return []
    enemy_planets = [p for p in planets if int(p.owner) not in (-1, me)]
    neutrals = [p for p in planets if int(p.owner) == -1]
    # OFFENSE targets: enemy planets + neutrals. Own planets are defended by the
    # separate defense pass below, not ranked as offense (which used to make the
    # agent trickle ships between its own planets).
    target_pool = enemy_planets + neutrals

    # --- shared per-turn memo of the opponent's earliest threat to a planet -----
    _tte_cache: dict[int, object] = {}

    def tte(pid: int):
        if pid not in _tte_cache:
            _tte_cache[pid] = model.time_to_enemy_threat(int(pid), me, world)
        return _tte_cache[pid]

    # --- per-source reserve: withhold ONLY the ships needed to survive a threat
    # that can ARRIVE before we could react, crediting our own production. This
    # replaces the old half-board cone that counted distant fleets and stranded
    # ships everywhere (a top cause of the inert 0/12 vs the Producer).
    def reserve(p) -> int:
        t = tte(int(p.id))
        if t is None:
            return 0
        force = sum(sh for (eta_arr, owner, sh) in model.ledger.get(int(p.id), [])
                    if owner != me and eta_arr <= t + WAVE_LOOKAHEAD)
        hold = float(p.ships) + float(p.production) * float(t)
        return max(0, int(math.ceil(force - hold + 1)))

    spare = {int(p.id): max(0, int(p.ships) - reserve(p)) for p in my_planets}

    floor_cache: dict[tuple[int, int], int] = {}

    def capture_floor(s, t):
        key = (int(s.id), int(t.id))
        f = floor_cache.get(key)
        if f is None:
            f = capture_floor_arrival(s, t, model, omega, me, world)
            floor_cache[key] = f
        return f

    # --- winnability: probability the action delivers the value we priced.
    # Independent failure modes multiply: floor + (1-floor)*time_survival*race_conf.
    def winnability(t, arrival_turn):
        ac = predict_arrival_contest(model, world, int(t.id), int(arrival_turn), me)
        # race_conf: graded lead over the opponent's earliest contest turn. Wide
        # lead -> ~1; arriving at/after contest -> ~0. Rewards capturing EARLY
        # (before the garrison grows / a counter lands); folds in target
        # contestedness ("where we aim"). None = uncontestable -> full confidence.
        opp = ac.opp_earliest_contest_tick
        if opp is None:
            race_conf = 1.0
        else:
            race_conf = _sigmoid((float(opp) - float(arrival_turn)) / RACE_SCALE)
        # time_survival: the world drifts while the fleet is in the air (entropy).
        time_survival = SURVIVAL_PER_TURN ** float(arrival_turn)
        path_clear = 1.0  # DEFERRED next factor: interception along the corridor.
        return WIN_FLOOR + (1.0 - WIN_FLOOR) * time_survival * race_conf * path_clear

    # --- two-sided value: the margin SWING in ships. Doubled when NOT acting
    # cedes the target to the opponent (an enemy planet, OR a neutral the opponent
    # will take) -- we gain the production stream AND deny it. This is what prices
    # the cost of inaction, so the agent acts under pressure instead of going inert.
    def counterfactual_owner(t, arrive):
        o = int(t.owner)
        if o not in (-1, me):
            return "opp"                      # enemy planet: opp keeps it
        ac = predict_arrival_contest(model, world, int(t.id), int(arrive), me)
        po = ac.predicted_owner
        if po is not None and int(po) not in (-1, me):
            return "opp"                      # predicted to be enemy at arrival
        th = tte(int(t.id))
        if th is not None and th <= arrive + CONTEST_MARGIN:
            return "opp_likely"               # contested neutral the opp will take
        return "neutral"

    def value(t, arrive):
        stream = int(t.production) * max(0.0, float(remain) - float(arrive))
        mult = 2.0 if counterfactual_owner(t, arrive) in ("opp", "opp_likely") else 1.0
        return mult * stream * winnability(t, arrive)

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
            "eta": round(float(eta), 1), "arrive_turn": int(arrive),
            "dist": round(math.hypot(t.x - s.x, t.y - s.y), 1),
            "tgt_owner": int(t.owner), "kind": kind, "floor": int(floor),
        })
        spare[int(s.id)] -= int(send)
        return True

    # ============================================================
    # DEFENSE pass (BEFORE offense consumes the budget). Reinforce a planet only
    # against a COMMITTED in-flight attack (the ledger) it cannot self-cover --
    # NOT speculative threats (which would re-create the friendly-reinforce trickle).
    # Sized to the shortfall, from the nearest surplus source that can arrive in time.
    # ============================================================
    def committed_threat(p):
        incoming = [(int(eta_arr), float(sh))
                    for (eta_arr, owner, sh) in model.ledger.get(int(p.id), [])
                    if owner != me]
        if not incoming:
            return 0, None
        deadline = min(e for e, _ in incoming)
        force = sum(sh for e, sh in incoming if e <= deadline + WAVE_LOOKAHEAD)
        hold = float(p.ships) + float(p.production) * float(deadline)
        return max(0, int(math.ceil(force - hold + 1))), deadline

    for p in my_planets:
        need, deadline = committed_threat(p)
        if need <= 0:
            continue
        srcs = sorted(
            (s for s in my_planets
             if int(s.id) != int(p.id) and spare.get(int(s.id), 0) > 0),
            key=lambda s: math.hypot(s.x - p.x, s.y - p.y),
        )
        for s in srcs:
            if need <= 0:
                break
            send = min(spare[int(s.id)], need)
            if send < MIN_FLEET_SIZE:
                continue
            ang, eta = aim_and_eta(s, p, send, omega, world=world)
            if deadline is not None and eta > deadline:
                continue  # arrives after the attack lands (same-turn sums, so OK)
            if emit(s, p, ang, send, eta, int(eta), "def", need):
                need -= send

    # ============================================================
    # FIELD (1): OWN candidate generator -- HARD feasibility only (trajectory
    # clears the sun / reaches the target). We do NOT call the champion's
    # propose(): its default-ON hold-feasibility / source-survival filters delete
    # contested captures before the field can judge them. The two-sided value is
    # the sole judge.
    # ============================================================
    actions: list[dict] = []

    def add_candidate(src, tgt, ships, angle, eta, wait):
        arrive = int(wait) + int(eta)
        if arrive > REACH_CEIL or int(ships) < MIN_FLEET_SIZE:
            return
        # Only fire-now candidates can actually be launched, so only they need the
        # (costly) trajectory ray-cast; wait candidates exist purely for the HOLD
        # comparison and are never emitted.
        if wait == 0 and predict_fleet_fate(
                src, tgt, angle, int(ships), world).outcome != "target":
            return
        actions.append({
            "src": src, "tgt": tgt, "ships": int(ships), "angle": float(angle),
            "eta": float(eta), "wait": int(wait), "arrive": arrive,
            "floor": int(ships), "value": value(tgt, arrive),
        })

    for src in my_planets:
        if int(src.ships) < MIN_FLEET_SIZE:
            continue
        for tgt in nearest_k(target_pool, src, NUM_TARGETS_PER_SOURCE):
            if int(tgt.id) == int(src.id):
                continue
            floor = capture_floor(src, tgt)
            # fire-now: a floor-clearing fleet launched this turn.
            ang, eta = aim_and_eta(src, tgt, floor, omega, world=world)
            add_candidate(src, tgt, floor, ang, eta, 0)
            # wait-then-mass: accumulate K turns, then fire. Only the HOLD gate
            # consumes these (compare fire-now vs waiting); never emitted.
            for K in WAIT_HORIZONS:
                if int(src.ships) + int(src.production) * K < floor:
                    continue
                wang, weta = aim_and_eta(src, tgt, floor, omega, wait_N=K, world=world)
                add_candidate(src, tgt, floor, wang, weta, K)

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
    # at least as much as the best wait-then-mass candidate for that pair; else the
    # ships are held. A fire-now action also requires the source can afford the
    # floor (a sub-floor fleet bounces, so it is not a capture).
    # ============================================================
    best_fire: dict[tuple[int, int], dict] = {}
    best_wait_val: dict[tuple[int, int], float] = {}
    for a in actions:
        sid, tid = int(a["src"].id), int(a["tgt"].id)
        key = (sid, tid)
        if a["wait"] == 0:
            if spare.get(sid, 0) < a["floor"]:
                continue
            cur = best_fire.get(key)
            if cur is None or a["value"] > cur["value"]:
                best_fire[key] = a
        elif a["value"] > best_wait_val.get(key, float("-inf")):
            best_wait_val[key] = a["value"]
    admitted = [a for key, a in best_fire.items()
                if a["value"] >= best_wait_val.get(key, float("-inf"))]
    admitted.sort(key=lambda a: -a["value"])

    # GRADIENT-FOLLOW: take the best admitted fire-now actions under the budget.
    for a in admitted:
        tid, sid = int(a["tgt"].id), int(a["src"].id)
        if tid in committed_tgt:
            continue  # target already won this turn
        floor = int(a["floor"])
        if spare.get(sid, 0) < floor:
            continue
        send = min(spare[sid], floor)
        ang, eta = aim_and_eta(a["src"], a["tgt"], send, omega, world=world)
        if emit(a["src"], a["tgt"], ang, send, eta, int(eta), "off", floor):
            committed_tgt.add(tid)

    # ============================================================
    # FIELD (2): convergence supplement. A defended target no single planet can
    # solo-fund: assemble a same-arrival cohort from remaining spare (combat rule 1
    # sums the legs). Only fires when it actually wins the target.
    # ============================================================
    extra = [t for t in target_pool
             if int(t.owner) != me and int(t.id) not in committed_tgt]
    extra.sort(key=lambda t: -int(t.production))  # cheap value proxy to bound cost
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
            # Same combat floor as the solo path (was capture_size, which sizes at
            # a different eta and thinner margin -> tie-and-die under combat rule 4).
            floor = capture_floor(cohort[0][0], t)
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
