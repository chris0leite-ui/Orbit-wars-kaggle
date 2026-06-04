"""protoflow — coalition-as-unit value field (PROBE, not a submission).

PI reframe (2026-06-04, coalition pass): the scored UNIT is no longer a single
(source -> target) launch but a (target, arrival-turn) CELL. The agent assembles
the multi-source wave that delivers a HOLDABLE force -- one that survives the
opponent's COMBINED counter -- landing on a single turn (combat rule 1 sums
same-owner same-turn arrivals). Solo captures, multi-source waves, and defensive
reinforcement all become the same operation.

WHY: holdability is a THRESHOLD -- a fleet below it realizes zero value (it bounces
or is retaken), at it captures and holds. That makes value-realization
super-additive: two half-fleets that separately hold nothing are jointly worth a
planet. An additive, per-source, greedy field provably cannot prefer concentration,
so it disperses and dies to a concentrated opponent (0/12 vs the Producer, ground
down by 60-80 ship waves we answered with 5-8). Making the coalition the scored
unit makes concentration the ONLY way value is realized -- it emerges, not tuned in.

The stateless wave (no remembered schedule): pin each cell to an absolute landing
turn; every source fires exactly when its launch-now fleet would land on it (the
binding far source leaves first; nearer sources are RESERVED -- they wait -- so they
do not disperse to a nearer solo). The FRIENDLY in-flight ledger is the memory: it
credits legs already en route, shrinking the remaining required force, so the wave
self-assembles across turns. Temporal consistency falls out of the geometry.

Build:
  1. CELLS = (target, arrival) over enemy+neutral targets (offense) and threatened
     own planets (defense). Each cell's REQUIRED FORCE is holdable against the
     COMBINED counter (sum over every enemy planet that can join the recapture
     wave -- not just the nearest one).
  2. VALUE = TWO-SIDED margin swing in ships = mult * production * (turns_left -
     arrival) * winnability (mult=2 when not acting cedes the target to the
     opponent). Prices the cell; unchanged from the single-currency field.
  3. ASSEMBLE: rank cells by value; for each, fund the wave from sources timed to a
     common landing turn under one global ship budget (offense and defense compete
     together, so defense is no longer starved). Then a positional REGROUP pass
     marches leftover idle ships toward the frontier.

Imports lib/* and agents.baseline.* directly (fine for local A/B; NOT bundled).
"""
from __future__ import annotations

import math

from lib.intent import World
from lib.world_model import WorldModel, WAVE_LOOKAHEAD, predict_arrival_contest
from lib.kinematic_table import KinematicTable
from lib.trajectory import predict_fleet_fate
from lib.fleet import speed as fleet_speed
from agents.baseline.proposer import aim_and_eta, nearest_k, HOLD_SAFETY_MARGIN

EPISODE_STEPS = 500
WIN_FLOOR = 0.10       # winnability never reaches zero -> the field never freezes
RACE_SCALE = 6.0       # turns; steepness of the graded reach-race confidence
REACH_CEIL = 26        # feasibility: a source whose launch-now arrival exceeds this can't join a wave
# Per-turn probability that a launched plan survives one turn of opponent action
# without being invalidated. A fleet in the air for tau turns survives to arrival
# with probability SURVIVAL_PER_TURN**tau -- how the field prices world drift.
SURVIVAL_PER_TURN = 0.97
# A neutral the opponent can contest within (our_arrival + CONTEST_MARGIN) is
# treated as one they would take if we did nothing -> doubled value (gain + deny).
CONTEST_MARGIN = 2
MIN_FLEET_SIZE = 2          # no sub-2-ship launches
SOURCES_PER_TARGET = 8      # nearest-k own planets considered per target (bounds cost)
CAPTURE_MARGIN = 2          # slack over the defender for integer combat resolution
COUNTER_WINDOW = WAVE_LOOKAHEAD  # turns after arrival within which enemies can join the counter
ARRIVAL_PROBE = 16          # representative fleet size for estimating a source's launch-now arrival
# WAIT GATE (opportunity cost on accumulation): a source fires a cell only if firing
# now beats HOLDING to fund a strictly-better cell it could afford in a few turns as
# its production accrues. WAIT_HORIZON bounds how long it may hold (so it can't freeze
# on a far dream); WAIT_VALUE_MARGIN is how much better the held cell must be before we
# forgo a sure capture (avoids thrash on near-ties).
WAIT_HORIZON = 4            # max turns of accumulation a source may wait for a better cell
WAIT_VALUE_MARGIN = 1.10    # the held cell must beat the fire-now cell by this factor
# REGROUP: a positional pass that marches idle rear ships up the enemy-pressure
# gradient toward the frontier, so force concentrates forward for future strikes.
REGROUP_PRESSURE_HORIZON = 14   # turns; decay reach for the enemy-pressure signal
REGROUP_MAX_ETA = 12            # don't send ships on long regroup flights
REGROUP_MIN_SHIPS = 4           # small fleets are slow; don't dribble a regroup
REGROUP_GAP_MIN = 1.0           # ship-mass; destination must be materially more forward
REGROUP_ETA_PENALTY = 0.1       # prefer near forward hops over distant ones

# Per-game trace for the probe runner. Each launch is a dict (see bottom).
_TRACE: list[dict] = []
# Last turn's ranked (target, arrival) field, for synthetic-situation calibration.
# Each entry: {"src","tgt","ships","ttc","imp","win","tgt_owner","prod"}.
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
    target_pool = enemy_planets + neutrals

    # --- shared per-turn memo of the opponent's earliest threat to a planet -----
    _tte_cache: dict[int, object] = {}

    def tte(pid: int):
        if pid not in _tte_cache:
            _tte_cache[pid] = model.time_to_enemy_threat(int(pid), me, world)
        return _tte_cache[pid]

    # --- per-source reserve: withhold ONLY the ships needed to survive a threat
    # that can ARRIVE before we could react, crediting our own production.
    def reserve(p) -> int:
        t = tte(int(p.id))
        if t is None:
            return 0
        force = sum(sh for (eta_arr, owner, sh) in model.ledger.get(int(p.id), [])
                    if owner != me and eta_arr <= t + WAVE_LOOKAHEAD)
        hold = float(p.ships) + float(p.production) * float(t)
        return max(0, int(math.ceil(force - hold + 1)))

    spare = {int(p.id): max(0, int(p.ships) - reserve(p)) for p in my_planets}

    # --- launch-now arrival: the turn a fleet launched THIS turn from s would reach
    # t (a representative probe size; the actual leg is re-aimed at its real count in
    # emit). This is what lets sources at different distances align on one landing
    # turn -- the far source fires now, nearer sources wait until their own
    # launch-now arrival equals the shared turn.
    _arr_cache: dict[tuple[int, int], int] = {}

    def arr(s, t) -> int:
        key = (int(s.id), int(t.id))
        a = _arr_cache.get(key)
        if a is None:
            _ang, eta = aim_and_eta(s, t, ARRIVAL_PROBE, omega, world=world)
            a = int(math.ceil(float(eta)))
            _arr_cache[key] = a
        return a

    # --- winnability: probability the action delivers the value we priced.
    def winnability(t, arrival_turn):
        ac = predict_arrival_contest(model, world, int(t.id), int(arrival_turn), me)
        opp = ac.opp_earliest_contest_tick
        if opp is None:
            race_conf = 1.0
        else:
            race_conf = _sigmoid((float(opp) - float(arrival_turn)) / RACE_SCALE)
        time_survival = SURVIVAL_PER_TURN ** float(arrival_turn)
        path_clear = 1.0  # DEFERRED next factor: interception along the corridor.
        return WIN_FLOOR + (1.0 - WIN_FLOOR) * time_survival * race_conf * path_clear

    # --- two-sided value: the margin SWING in ships. Doubled when NOT acting cedes
    # the target to the opponent. Prices the cost of inaction.
    def counterfactual_owner(t, arrive):
        o = int(t.owner)
        if o not in (-1, me):
            return "opp"
        ac = predict_arrival_contest(model, world, int(t.id), int(arrive), me)
        po = ac.predicted_owner
        if po is not None and int(po) not in (-1, me):
            return "opp"
        th = tte(int(t.id))
        if th is not None and th <= arrive + CONTEST_MARGIN:
            return "opp_likely"
        return "neutral"

    def value(t, arrive):
        stream = int(t.production) * max(0.0, float(remain) - float(arrive))
        mult = 2.0 if counterfactual_owner(t, arrive) in ("opp", "opp_likely") else 1.0
        return mult * stream * winnability(t, arrive)

    # --- COMBINED counter: the opponent's real recapture wave is the SUM over every
    # enemy planet that can reach t within COUNTER_WINDOW turns after we capture (not
    # just the nearest one -- that single-planet under-estimate is what let the
    # Producer out-mass us). In-flight enemy arriving AFTER our capture is added too;
    # enemy arriving by A is already in ships_at(t, A) (the defender), so no double
    # count.
    def combined_counter(t, A):
        total = 0.0
        t_min = None
        tx, ty, tr = float(t.x), float(t.y), float(t.radius)
        for e in enemy_planets:
            flight = math.hypot(float(e.x) - tx, float(e.y) - ty) - float(e.radius) - tr
            sp = fleet_speed(int(e.ships))
            tau = 1 if flight <= 0.0 else (int(math.ceil(flight / sp)) if sp > 0 else 999)
            if tau <= COUNTER_WINDOW:
                total += float(e.ships) + float(e.production) * float(A + tau)
                t_min = tau if t_min is None else min(t_min, tau)
        for (eta_arr, owner, sh) in model.ledger.get(int(t.id), []):
            if owner != me and A < eta_arr <= A + COUNTER_WINDOW:
                total += float(sh)
                tau = max(1, int(eta_arr) - int(A))
                t_min = tau if t_min is None else min(t_min, tau)
        return total, (t_min if t_min is not None else 0)

    # --- required force at a cell: the flip floor at arrival A, up-sized so the
    # post-capture garrison survives the COMBINED counter (same inequality as the
    # champion's hold_need, but summed). Returns the flip floor when no counter is in
    # range, so safe rear expansion stays cheap.
    def required_force(t, A):
        base = model.ships_at(int(t.id), int(A))
        defender = int(math.ceil(base)) if base is not None else int(t.ships)
        floor = max(MIN_FLEET_SIZE, defender + CAPTURE_MARGIN)
        counter, t_op = combined_counter(t, A)
        if counter <= 0.0:
            return floor
        g_min = math.floor((counter - 1.0) / HOLD_SAFETY_MARGIN) + 1
        ships_needed = (g_min - int(t.production) * int(t_op)) + defender
        return max(floor, int(math.ceil(ships_needed)))

    def friendly_inflight(t, A):
        return sum(sh for (eta_arr, owner, sh) in model.ledger.get(int(t.id), [])
                   if owner == me and abs(int(eta_arr) - int(A)) <= 1)

    moves: list[list] = []
    launches: list[dict] = []

    def emit(s, t, angle, send, arrive, kind, req):
        if send <= 0:
            return False
        if predict_fleet_fate(s, t, angle, int(send), world).outcome != "target":
            return False  # path blocked (sun / wrong planet / oob)
        moves.append([int(s.id), float(angle), int(send)])
        launches.append({
            "src": int(s.id), "tgt": int(t.id), "ships": int(send),
            "eta": round(float(arrive), 1), "arrive_turn": int(arrive),
            "dist": round(math.hypot(t.x - s.x, t.y - s.y), 1),
            "tgt_owner": int(t.owner), "kind": kind, "floor": int(req),
        })
        spare[int(s.id)] -= int(send)
        return True

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

    # ============================================================
    # BUILD CELLS. Offense: for each enemy/neutral target, find the soonest landing
    # turn at which a holdable wave can be funded from the nearest sources (the
    # binding far source sets the turn). Defense: each threatened own planet is a
    # cell whose required force is the incoming wave.
    # ============================================================
    _LAST_FIELD.clear()
    cells: list[dict] = []
    # Per-source ASPIRATION: the best wait-discounted value a source could realize by
    # HOLDING this turn to fund a target it can't quite afford yet. Built below from the
    # not-yet-fundable targets; consulted by the offense wait gate during assembly.
    aspiration: dict[int, float] = {int(p.id): 0.0 for p in my_planets}

    for t in target_pool:
        srcs = []  # (source, launch_now_arrival)
        for s in nearest_k(my_planets, t, SOURCES_PER_TARGET):
            if int(s.id) == int(t.id):
                continue
            a = arr(s, t)
            if a > REACH_CEIL:
                continue
            srcs.append((s, a))
        if not srcs:
            continue
        srcs.sort(key=lambda sa: sa[1])

        # Field introspection: record the value/winnability curve over the candidate
        # arrivals (plus a small forward sweep) so calibration can read the field.
        amin = srcs[0][1]
        field_arrivals = sorted({a for _, a in srcs} | {amin, amin + 4, amin + 8})
        for A in field_arrivals:
            _LAST_FIELD.append({
                "src": int(srcs[0][0].id), "tgt": int(t.id),
                "ships": int(required_force(t, A)), "ttc": int(A),
                "imp": round(value(t, A), 1), "win": round(winnability(t, A), 3),
                "tgt_owner": int(t.owner), "prod": int(t.production),
            })

        # Soonest holdable landing turn: extend the prefix of nearest sources until
        # their spare (plus friendly already in flight) can fund the required force.
        chosen = None
        for i in range(len(srcs)):
            A = srcs[i][1]
            R = required_force(t, A)
            avail = friendly_inflight(t, A) + sum(spare[int(sj.id)] for sj, _ in srcs[:i + 1])
            if avail >= R:
                chosen = (A, [sa for sa in srcs[:i + 1]], R)
                break
        if chosen is None:
            # Can't fund a holdable wave THIS turn. If a short accumulation would
            # afford it, record its wait-discounted value as each nearby source's
            # aspiration -- the opportunity cost they weigh before dribbling a lesser,
            # affordable-now cell. value() already discounts the later landing turn, so
            # v_wait is on the same scale as any fundable cell's value.
            A0 = srcs[0][1]
            R0 = required_force(t, A0)
            shortfall = R0 - (friendly_inflight(t, A0)
                              + sum(spare[int(sj.id)] for sj, _ in srcs))
            prod_rate = sum(int(sj.production) for sj, _ in srcs)
            if shortfall > 0 and prod_rate > 0:
                tau = int(math.ceil(shortfall / prod_rate))
                if tau <= WAIT_HORIZON:
                    v_wait = value(t, A0 + tau)
                    for sj, _ in srcs:
                        if spare[int(sj.id)] > 0:
                            aspiration[int(sj.id)] = max(aspiration[int(sj.id)], v_wait)
            continue  # HOLD and accumulate
        A_rel, chosen_srcs, R = chosen
        if R - friendly_inflight(t, A_rel) <= 0:
            continue  # friendly fleets already en route cover it -> don't double-send
        cells.append({"kind": "off", "t": t, "A": int(A_rel), "R": int(R),
                      "srcs": chosen_srcs, "value": value(t, A_rel)})

    for p in my_planets:
        need, deadline = committed_threat(p)
        if need <= 0 or deadline is None:
            continue
        d_srcs = []
        for s in nearest_k(my_planets, p, SOURCES_PER_TARGET):
            if int(s.id) == int(p.id):
                continue
            a = arr(s, p)
            if a <= int(deadline):  # arriving by the deadline reinforces in time
                d_srcs.append((s, a))
        if not d_srcs:
            continue
        # Losing a planet is a 2x swing (our stream lost + the enemy gains it).
        cells.append({"kind": "def", "t": p, "A": int(deadline), "R": int(need),
                      "srcs": d_srcs, "value": 2.0 * int(p.production) * float(remain)})

    # ============================================================
    # ASSEMBLE. Rank cells by value; fund each wave under one global budget. Offense
    # waves fire the boundary sources (launch-now arrival == landing turn) and
    # RESERVE the nearer ones (they wait this turn so they don't disperse); the
    # friendly ledger carries the wave across turns. Defense fires soonest-first
    # (arriving early just sits in the garrison).
    # ============================================================
    cells.sort(key=lambda c: -c["value"])
    committed_tgt: set[int] = set()
    reserved: set[int] = set()

    for c in cells:
        t = c["t"]
        tid = int(t.id)
        if tid in committed_tgt:
            continue
        elig = [(s, a) for (s, a) in c["srcs"]
                if int(s.id) not in reserved and spare.get(int(s.id), 0) > 0]
        if not elig:
            continue

        if c["kind"] == "off":
            A_rel = c["A"]
            need = c["R"] - friendly_inflight(t, A_rel)
            if need <= 0:
                committed_tgt.add(tid)
                continue
            if friendly_inflight(t, A_rel) + sum(spare[int(s.id)] for s, _ in elig) < c["R"]:
                continue  # a higher cell took our sources -> no longer fundable
            # Fire the boundary (launch-now arrival == landing turn) now; those legs
            # land together on the shared turn. Reserve the nearer ones to wait.
            for s, a in sorted(elig, key=lambda x: -x[1]):
                if a == A_rel and need > 0:
                    # WAIT GATE: if this source could fund a strictly-better cell by
                    # holding a few turns, don't dribble it into this lesser one --
                    # reserve it (it accumulates and fires the better cell once afforded).
                    if aspiration.get(int(s.id), 0.0) > c["value"] * WAIT_VALUE_MARGIN:
                        reserved.add(int(s.id))
                        continue
                    send = min(spare[int(s.id)], need)
                    ang, _eta = aim_and_eta(s, t, send, omega, world=world)
                    if emit(s, t, ang, send, A_rel, "wave", c["R"]):
                        need -= send
            if need > 0:
                for s, a in elig:
                    if a < A_rel:
                        reserved.add(int(s.id))  # waits this turn (anti-dispersion)
            committed_tgt.add(tid)

        else:  # defense: get force there by the deadline, soonest-first
            need = c["R"]
            for s, a in sorted(elig, key=lambda x: x[1]):
                if need <= 0:
                    break
                if int(s.id) in reserved:
                    continue
                send = min(spare[int(s.id)], need)
                if send < MIN_FLEET_SIZE:
                    continue
                ang, _eta = aim_and_eta(s, t, send, omega, world=world)
                if emit(s, t, ang, send, a, "def", c["R"]):
                    need -= send
            committed_tgt.add(tid)

    # ============================================================
    # REGROUP: march leftover idle ships up the enemy-pressure gradient toward the
    # frontier (movement WITHOUT capture), so force concentrates forward for future
    # waves instead of stranding in the rear. Mirrors the Producer's marshalling.
    # ============================================================
    if enemy_planets:
        def enemy_pressure(p):
            tot = 0.0
            for e in enemy_planets:
                reach = fleet_speed(int(e.ships)) * REGROUP_PRESSURE_HORIZON
                if reach <= 0.0:
                    continue
                decay = 1.0 - math.hypot(e.x - p.x, e.y - p.y) / reach
                if decay > 0.0:
                    tot += float(e.ships) * decay
            return tot

        pressure = {int(p.id): enemy_pressure(p) for p in my_planets}
        srcs = sorted(
            (p for p in my_planets
             if int(p.id) not in reserved and spare.get(int(p.id), 0) >= REGROUP_MIN_SHIPS),
            key=lambda p: -spare[int(p.id)],
        )
        for s in srcs:
            leftover = spare.get(int(s.id), 0)
            if leftover < REGROUP_MIN_SHIPS:
                continue
            best = None  # (score, dest, angle, eta)
            for d in my_planets:
                if int(d.id) == int(s.id):
                    continue
                gap = pressure[int(d.id)] - pressure[int(s.id)]
                if gap <= REGROUP_GAP_MIN:
                    continue  # only move TOWARD the front (directional -> no oscillation)
                if committed_threat(d)[0] > 0:
                    continue  # don't stage into a falling planet (defense's job)
                ang, eta = aim_and_eta(s, d, leftover, omega, world=world)
                if eta > REGROUP_MAX_ETA:
                    continue  # near forward hops only -- stay reactive
                score = gap - REGROUP_ETA_PENALTY * eta
                if best is None or score > best[0]:
                    best = (score, d, ang, int(math.ceil(float(eta))))
            if best is not None:
                _score, d, ang, eta = best
                emit(s, d, ang, leftover, eta, "regroup", 0)

    _TRACE.append({
        "step": step,
        "launches": launches,
        "idle": len(launches) == 0,
        "sources": len(my_planets),
        "my_planets": len(my_planets),
        "my_ships": int(sum(p.ships for p in my_planets)),
    })
    return moves
