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
# its production accrues. WAITING IS A FIRST-CLASS ACTION: a "wait" cell, valued by the
# same value() at its accumulation-arrival, competes in the SAME ranked list as fire-now
# cells; if it wins it holds back ONLY the ships its future wave needs (surplus still
# fires now). WAIT_HORIZON bounds how long we may accumulate -- set to the value window so
# the wait reasons over the SAME future as value (not a short 4-turn slice, which made us
# grab cheap low-production planets instead of saving up for defended high-production ones).
# WAIT_VALUE_MARGIN discounts the wait's value so we only hold when the future cell is
# clearly better (anti-thrash on near-ties).
WAIT_HORIZON = 18           # max turns of accumulation a wait may span (= VALUE_HORIZON)
WAIT_VALUE_MARGIN = 1.10    # the held cell must beat the fire-now cell by this factor
# When True, a threatened planet's hold requirement also covers the enemy's FREE
# recapture follow-up (not just the launched wave) -- the opponent model applied
# symmetrically to defense. A/B (n=4 vs Producer): ON vs OFF is a wash on win/loss
# (both lose to the Producer, both 8/8 vs light-greedy), but ON gives cleaner thrash
# (reinforce-then-lose 2.5 vs 3.8) -- the "no bleed" goal -- so it is the default. Kept
# as a knob: it raises the hold bar and can over-abandon a massing opponent's frontier.
DEFENSE_FOLLOWUP = True
# FARSIGHTED VALUE (value-to-go expansion potential). The myopic field scored a planet
# by its OWN production stream in isolation -- blind to the snowball (a capture frees
# ships that take more planets). The Producer beats us by scoring the production a move
# UNLOCKS over an ~18-turn window. We make that a field property: each planet's value is
# its windowed production PLUS the travel-discounted production it can spring to, summed
# over a few expansion hops (a truncated value-to-go). No rollout, no opponent tree --
# recomputed each turn from the board.
EXPANSION_POTENTIAL = True   # A/B knob: False -> exact pre-farsight value (own stream)
VALUE_HORIZON = 18           # tempo window in turns (matches the Producer's planning H)
VALUE_DISCOUNT = 0.9         # per-turn discount d; d**VALUE_HORIZON ~ 0.15
POTENTIAL_HOPS = 3           # sequential captures looked ahead (~18 turns / ~6 turns/hop)
EXPANSION_COUPLING = 0.6     # weight on each successive hop (keeps the series bounded)
ENEMY_REACH_DISCOUNT = 0.5   # enemy production is harder springboard mass than neutral
# SPRINGBOARD SATURATION FIX. Summing the springboard over EVERY reachable planet made it
# grow with the COUNT of neighbours, so on a dense board it saturated to a near-uniform
# "how central am I" number (88-98% of every planet's value) that drowned own production --
# we ranked by centrality, not economy, and undervalued big planets. Credit only the BEST
# few unlocks per hop instead: quality, not count; bounded and density-stable. A stepping
# stone that opens a big planet still scores; loitering next to many small planets does not.
SPRINGBOARD_TOPK = 2         # best-K unlocks credited per hop; 0 -> today's unbounded sum (A/B)
# OFFENSIVE PRESSURE (two-sided potential). An enemy planet is worth not just its own
# production (denied) but the productive REGION it anchors for the opponent -- the snowball
# that collapses when we take it. We compute the SAME potential from the opponent's seat
# (opp_phi) and add the region we destroy to an enemy target's value. This is the Producer's
# competitive objective ("my production minus theirs") as a field property: the most valuable
# enemy target becomes their keystone hub, and the coalition machinery concentrates force on
# it (a spear) rather than scattering into neutrals. Replaces the crude x2-for-enemy denial.
OFFENSIVE_PRESSURE = True     # A/B knob: False -> today's x2-enemy denial, no opp_phi term
OFFENSE_WEIGHT = 1.0         # weight on the opponent-region-loss term for enemy targets
# CONCENTRATED SALVO. Combat rule 1 sums only SAME-TURN same-owner arrivals, but the old
# assembly fired a boundary leg now and deferred nearer legs to later turns -- estimating
# arrivals with a fixed probe size while real fleet speed depends on ship count. So legs
# landed on different turns, fought the garrison piecemeal, and BOUNCED (15/19 targets needed
# multiple attempts; we sent 35 at a floor of 47, 16 at 22). Instead deliver a capture as a
# SINGLE-TURN synchronized salvo at real leg sizes: fire only co-arriving sources whose summed
# mass meets the requirement, all launched now -- or wait. Never emit a sub-threshold leg.
CONCENTRATED_SALVO = True     # A/B knob: False -> old cross-turn boundary/defer (piecemeal)
# SELF-PROTECTIVE RESERVE. reserve() withheld ships only against enemy fleets ALREADY in
# flight, blind to the standing threat of a nearby enemy PLANET. So draining a frontier planet
# to grab a neutral looked free -- the enemy then launched and took our emptied planet. Price
# the standing threat too (the same combined_counter we size attacks against, from p's seat),
# so a planet never drains below what keeps it alive.
RESERVE_THREAT = True         # A/B knob: False -> today's in-flight-only reserve
# FLIP-vs-HOLD TIERS. The combined-counter hold requirement was a HARD GATE: a contested
# planet we could flip but not HOLD built no cell at all -> we froze (launches collapsed
# to ~65/game vs the Producer, idle 0.56, wiped to one planet). The fix: offer each target
# a second, CHEAP "flip" cell (sized just to flip, like the Producer's capture_floor),
# priced by its EXPECTED TENURE -- the own production we bank before the counter retakes it,
# with no springboard. The full-value HOLD cell always outranks the flip for the same
# target, so we still concentrate where we can afford to hold; the flip only wins where the
# hold is unaffordable, banking production instead of freezing. Holdability becomes a choice
# the field makes, not a gate it enforces.
FLIP_TIER = True             # A/B knob: False -> today's hold-only gate (no flip cell)
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
    # Tempo window in turns, capped by what's left of the game. Read by the phi
    # potential below AND by value()'s tenure discount, so it lives at turn scope.
    win_turns = min(VALUE_HORIZON, int(remain))

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

    # --- TWO-SIDED VALUE-TO-GO. A potential phi[t] = the windowed production of holding t
    # PLUS the travel-discounted production it can spring to over a few capture-hops (a
    # truncated value-to-go). Computed from a SEAT's perspective via `getmass` (which planets
    # are capturable NEW mass for that seat). `phi` is our gain; `opp_phi` is the opponent's
    # region we collapse by taking an enemy planet. Recomputed each turn; no rollout, no tree.
    W = sum(VALUE_DISCOUNT ** tau for tau in range(win_turns + 1)) if EXPANSION_POTENTIAL else 0.0

    def compute_potential(getmass: dict[int, float]) -> dict[int, float]:
        rep_speed = fleet_speed(ARRIVAL_PROBE)
        ids = [int(p.id) for p in planets]
        prize = {int(p.id): float(p.production) * W for p in planets}
        # reach[a] = [(b, d**travel(a->b) * getmass(b)), ...] -- travel-discounted edges.
        reach: dict[int, list[float]] = {aid: [] for aid in ids}
        for a in planets:
            ax, ay, aid = float(a.x), float(a.y), int(a.id)
            for b in planets:
                bid = int(b.id)
                if bid == aid or getmass[bid] <= 0.0:
                    continue
                travel = math.hypot(float(b.x) - ax, float(b.y) - ay) / rep_speed \
                    if rep_speed > 0 else 999.0
                w = (VALUE_DISCOUNT ** travel) * getmass[bid]
                if w > 1e-3:
                    reach[aid].append((bid, w))
        # phi = prize + sum_{hop=1..HOPS} COUPLING^hop * (best-K of R^hop prize). Crediting
        # only the top-K unlocks per hop keeps the springboard quality-driven and bounded
        # (it can't grow with the raw neighbour count), so own production stays the lead term.
        phi = dict(prize)
        term = dict(prize)
        for _ in range(POTENTIAL_HOPS):
            nxt = {}
            for aid in ids:
                contribs = [w * term[bid] for (bid, w) in reach[aid]]
                if SPRINGBOARD_TOPK > 0 and len(contribs) > SPRINGBOARD_TOPK:
                    contribs = sorted(contribs, reverse=True)[:SPRINGBOARD_TOPK]
                nxt[aid] = EXPANSION_COUPLING * sum(contribs)
            for aid in ids:
                phi[aid] += nxt[aid]
            term = nxt
        return phi

    phi: dict[int, float] = {}
    opp_phi: dict[int, float] = {}
    if EXPANSION_POTENTIAL:
        # Our seat: neutrals are NEW mass (1.0), our planets already ours (0), enemy mass is
        # harder to chain through (discounted).
        getmass_me = {int(p.id): (1.0 if int(p.owner) == -1
                                  else (0.0 if int(p.owner) == me else ENEMY_REACH_DISCOUNT))
                      for p in planets}
        phi = compute_potential(getmass_me)
        if OFFENSIVE_PRESSURE:
            # Opponent's seat (mirror): neutrals NEW mass for them (1.0), OUR planets are
            # their springboard mass (discounted), their own planets already theirs (0). In
            # 2P this is the single opponent; in 4P it lumps all opponents (follow-up).
            getmass_opp = {int(p.id): (1.0 if int(p.owner) == -1
                                       else (ENEMY_REACH_DISCOUNT if int(p.owner) == me else 0.0))
                           for p in planets}
            opp_phi = compute_potential(getmass_opp)

    # --- shared per-turn memo of the opponent's earliest threat to a planet -----
    _tte_cache: dict[int, object] = {}

    def tte(pid: int):
        if pid not in _tte_cache:
            _tte_cache[pid] = model.time_to_enemy_threat(int(pid), me, world)
        return _tte_cache[pid]

    # --- enemy FREE force: the mirror of our own reserve(), from the enemy's seat. An enemy
    # planet must hold back enough to survive OUR reachable force; only the remainder is free
    # to counter our captures or attack us. Defined BEFORE reserve so reserve can price the
    # standing threat. time_to_enemy_threat is perspective-symmetric (passing e's own id asks
    # "soonest WE could threaten e").
    _efree_cache: dict[int, float] = {}

    def enemy_free(e) -> float:
        key = int(e.id)
        v = _efree_cache.get(key)
        if v is not None:
            return v
        t_us = model.time_to_enemy_threat(int(e.id), int(e.owner), world)
        if t_us is None:
            reserve_e = 0.0  # we pose no threat -> all of e's force is free to counter
        else:
            horizon = int(t_us) + WAVE_LOOKAHEAD
            ex, ey, er = float(e.x), float(e.y), float(e.radius)
            ours = sum(sh for (eta_arr, owner, sh) in model.ledger.get(int(e.id), [])
                       if owner == me)
            for f in my_planets:
                flight = math.hypot(float(f.x) - ex, float(f.y) - ey) - float(f.radius) - er
                sp = fleet_speed(int(f.ships))
                tau = 1 if flight <= 0.0 else (int(math.ceil(flight / sp)) if sp > 0 else 999)
                if tau <= horizon:
                    ours += float(f.ships)
            hold = float(e.ships) + float(e.production) * float(t_us)
            reserve_e = max(0.0, ours - hold)
        v = max(0.0, float(e.ships) - reserve_e)
        _efree_cache[key] = v
        return v

    # --- COMBINED counter: the opponent's real wave on a target -- the SUM over every enemy
    # planet that can reach it within COUNTER_WINDOW of its FREE force (net of what it must
    # hold against us), plus in-flight enemy, floored at the nearest enemy's full garrison.
    # Used both to size our captures (required_force) AND, on our own planet, to price the
    # standing threat in reserve.
    def combined_counter(t, A):
        total = 0.0
        t_min = None
        tx, ty, tr = float(t.x), float(t.y), float(t.radius)
        nearest_full = None
        nearest_tau = None
        for e in enemy_planets:
            flight = math.hypot(float(e.x) - tx, float(e.y) - ty) - float(e.radius) - tr
            sp = fleet_speed(int(e.ships))
            tau = 1 if flight <= 0.0 else (int(math.ceil(flight / sp)) if sp > 0 else 999)
            if tau <= COUNTER_WINDOW:
                total += enemy_free(e) + float(e.production) * float(A + tau)
                cand_full = float(e.ships) + float(e.production) * float(A + tau)
                if nearest_full is None or tau < nearest_tau:
                    nearest_full, nearest_tau = cand_full, tau
                t_min = tau if t_min is None else min(t_min, tau)
        for (eta_arr, owner, sh) in model.ledger.get(int(t.id), []):
            if owner != me and A < eta_arr <= A + COUNTER_WINDOW:
                total += float(sh)
                tau = max(1, int(eta_arr) - int(A))
                t_min = tau if t_min is None else min(t_min, tau)
        if nearest_full is not None:
            total = max(total, nearest_full)
        return total, (t_min if t_min is not None else 0)

    # --- per-source reserve: withhold the ships needed to survive a threat that can ARRIVE
    # before we could react, crediting our own production. The threat is enemy fleets already
    # in flight AND -- when RESERVE_THREAT -- the standing FREE force a nearby enemy PLANET can
    # launch at p (combined_counter evaluated on p, the same quantity we size our own attacks
    # against). Without the standing term we drain frontier planets and lose them next turn.
    def reserve(p) -> int:
        t = tte(int(p.id))
        if t is None and not RESERVE_THREAT:
            return 0
        inflight = sum(sh for (eta_arr, owner, sh) in model.ledger.get(int(p.id), [])
                       if owner != me and t is not None and eta_arr <= t + WAVE_LOOKAHEAD)
        if RESERVE_THREAT:
            standing, t_op = combined_counter(p, 0)
            threat = max(float(inflight), standing)
            if threat <= 0.0:
                return 0
            grow_t = t_op if t_op > 0 else (t if t is not None else 0)
            grow = float(p.production) * float(grow_t)
            # If we can't survive even by holding EVERYTHING, reserving is futile -- those
            # ships should deploy (bank + deny) rather than sit and be wiped (the doomed
            # no-bleed logic). Otherwise keep enough RETAINED ships that, with production
            # growth, p survives -- NOT crediting the garrison we're about to allocate.
            if threat > float(p.ships) + grow:
                return 0
            return max(0, min(int(p.ships), int(math.ceil(threat - grow))))
        hold = float(p.ships) + float(p.production) * float(t)
        return max(0, int(math.ceil(float(inflight) - hold + 1)))

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

    # Real-size launch-now arrival: a fleet's speed RISES with ship count, so a small leg is
    # slow. Estimating arrival at the ACTUAL send size (not a fixed probe) is what lets the
    # salvo know which legs truly co-arrive and combine (combat rule 1 sums same-TURN arrivals).
    _arr_sized_cache: dict[tuple[int, int, int], int] = {}

    def arr_sized(s, t, ships) -> int:
        ships = max(MIN_FLEET_SIZE, int(ships))
        key = (int(s.id), int(t.id), ships)
        a = _arr_sized_cache.get(key)
        if a is None:
            _ang, eta = aim_and_eta(s, t, ships, omega, world=world)
            a = int(math.ceil(float(eta)))
            _arr_sized_cache[key] = a
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

    def value(t, arrive, tenure=None):
        # tenure=None -> we expect to HOLD the target (full value-to-go). tenure=k -> we
        # expect to lose it to the counter in ~k turns (a flip): realize only the target's
        # OWN production over those k turns -- no springboard, since we don't keep it long
        # enough to chain off it. mult (deny) and winnability (realizability) are unchanged.
        if arrive >= remain:  # capture would land after the game ends
            return 0.0
        mult = 2.0 if counterfactual_owner(t, arrive) in ("opp", "opp_likely") else 1.0
        is_enemy = int(t.owner) not in (-1, me)
        if EXPANSION_POTENTIAL:
            if tenure is None:
                # farsighted: the prize is the reachable productive region (phi). Arrival
                # timing is carried by winnability (0.97**arrive + the race).
                base = phi.get(int(t.id), 0.0)
                # OFFENSIVE PRESSURE: taking an enemy planet also COLLAPSES the region it
                # anchors for the opponent (opp_phi). Add that loss explicitly and drop the
                # crude x2-for-enemy denial (the doubling is now the real region term).
                if OFFENSIVE_PRESSURE and is_enemy:
                    mult = 1.0
                    base = base + OFFENSE_WEIGHT * opp_phi.get(int(t.id), 0.0)
            else:
                # flip: own production discounted over the tenure window only.
                ten = max(0, min(int(tenure), win_turns))
                W_ten = sum(VALUE_DISCOUNT ** tau for tau in range(ten + 1))
                base = float(t.production) * W_ten
            return mult * winnability(t, arrive) * base
        stream_turns = max(0.0, float(remain) - float(arrive))
        if tenure is not None:
            stream_turns = min(stream_turns, float(tenure))
        stream = int(t.production) * stream_turns
        return mult * stream * winnability(t, arrive)

    # --- required force at a cell: the flip floor at arrival A, up-sized so the
    # post-capture garrison survives the COMBINED counter (same inequality as the
    # champion's hold_need, but summed). Returns the flip floor when no counter is in
    # range, so safe rear expansion stays cheap.
    def flip_floor(t, A):
        # bare force to FLIP the target at arrival A (the Producer's capture_floor):
        # the defender at arrival + integer-combat margin. Holding is NOT priced in.
        base = model.ships_at(int(t.id), int(A))
        defender = int(math.ceil(base)) if base is not None else int(t.ships)
        return max(MIN_FLEET_SIZE, defender + CAPTURE_MARGIN), defender

    def required_force(t, A):
        floor, defender = flip_floor(t, A)
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
        base = float(force) - hold + 1.0  # reinforcement to WIN the landing wave
        # FOLLOW-UP (opponent model, symmetric with offense): after we win the wave,
        # our garrison must survive the enemy's FREE recapture force launched from
        # their PLANETS. Only planet launches -- the in-flight wave is already in
        # `force`, so excluding it here avoids a double count. Same hold inequality
        # as required_force. A planet facing a wave AND an unanswerable follow-up is
        # genuinely doomed; the bigger `need` makes the fundability gate abandon it
        # rather than bleed ships in.
        follow = 0.0
        px, py, pr = float(p.x), float(p.y), float(p.radius)
        for e in enemy_planets:
            flight = math.hypot(float(e.x) - px, float(e.y) - py) - float(e.radius) - pr
            sp = fleet_speed(int(e.ships))
            tau = 1 if flight <= 0.0 else (int(math.ceil(flight / sp)) if sp > 0 else 999)
            if tau <= COUNTER_WINDOW:
                follow += enemy_free(e) + float(e.production) * float(int(deadline) + tau)
        need = base + (follow / HOLD_SAFETY_MARGIN if follow > 0.0 and DEFENSE_FOLLOWUP else 0.0)
        return max(0, int(math.ceil(need))), deadline

    # ============================================================
    # BUILD CELLS. Offense: for each enemy/neutral target, find the soonest landing
    # turn at which a holdable wave can be funded from the nearest sources (the
    # binding far source sets the turn). Defense: each threatened own planet is a
    # cell whose required force is the incoming wave.
    # ============================================================
    _LAST_FIELD.clear()
    cells: list[dict] = []

    def salvo_select(t, threshold_fn):
        # The soonest landing turn at which a SYNCHRONIZED salvo captures: among sources
        # launching now (arrival estimated at their REAL send size = current spare), the
        # co-arriving group on that turn, plus our fleets already in flight arriving then,
        # must sum to >= the threshold. Returns (A, [(src, arrival)], R) or None. Only same-
        # turn arrivals combine (combat rule 1), so this never relies on un-synced legs.
        cand = []
        for s in nearest_k(my_planets, t, SOURCES_PER_TARGET):
            if int(s.id) == int(t.id):
                continue
            sp = spare.get(int(s.id), 0)
            if sp <= 0:
                continue
            a = arr_sized(s, t, sp)
            if a <= REACH_CEIL:
                cand.append((s, a, sp))
        if not cand:
            return None
        for A in sorted({a for _, a, _ in cand}):
            co = [(s, a, sp) for (s, a, sp) in cand if a == A]
            deliver = friendly_inflight(t, A) + sum(sp for _, _, sp in co)
            R = int(threshold_fn(t, A))
            if deliver >= R:
                return (A, [(s, a) for (s, a, _sp) in co], R)
        return None

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

        A0 = srcs[0][1]
        counter0, t_op0 = combined_counter(t, A0)

        if CONCENTRATED_SALVO:
            # SALVO: a capture is delivered as a SINGLE-TURN synchronized salvo at real leg
            # sizes -- only co-arriving sources whose summed mass meets the requirement, all
            # launched now -- or we wait. salvo_select finds the soonest such landing turn.
            hold = salvo_select(t, required_force)
            if hold is not None:
                A_rel, co_srcs, R = hold
                if R - friendly_inflight(t, A_rel) > 0:
                    cells.append({"kind": "off", "t": t, "A": int(A_rel), "R": int(R),
                                  "srcs": co_srcs, "value": value(t, A_rel)})
            else:
                # No salvo meets the hold this turn. WAIT (accumulate at the strongest source)
                # for an UNCONTESTED target -- a contested target's hold cost balloons while we
                # wait, so the flip salvo below deploys now instead.
                R0 = required_force(t, A0)
                shortfall = R0 - (friendly_inflight(t, A0)
                                  + sum(spare[int(sj.id)] for sj, _ in srcs))
                prod_rate = sum(int(sj.production) for sj, _ in srcs)
                if counter0 <= 0.0 and shortfall > 0 and prod_rate > 0:
                    tau = int(math.ceil(shortfall / prod_rate))
                    if tau <= WAIT_HORIZON:
                        cells.append({"kind": "wait", "t": t, "A": int(A0 + tau), "R": int(R0),
                                      "srcs": [sa for sa in srcs],
                                      "value": value(t, A0 + tau) / WAIT_VALUE_MARGIN})
            # FLIP salvo (contested only): a decisive flip-floor salvo, expected-tenure value.
            if FLIP_TIER and counter0 > 0.0:
                fl = salvo_select(t, lambda tt, AA: flip_floor(tt, AA)[0])
                if fl is not None:
                    A_f, f_srcs, Rf = fl
                    if Rf - friendly_inflight(t, A_f) > 0:
                        cells.append({"kind": "off", "t": t, "A": int(A_f), "R": int(Rf),
                                      "srcs": f_srcs, "value": value(t, A_f, tenure=t_op0)})
        else:
            # --- OLD cross-turn assembly (A/B baseline): prefix of nearest sources, boundary
            # fires now and nearer legs defer to later turns (the piecemeal-bounce path).
            chosen = None
            for i in range(len(srcs)):
                A = srcs[i][1]
                R = required_force(t, A)
                avail = friendly_inflight(t, A) + sum(spare[int(sj.id)] for sj, _ in srcs[:i + 1])
                if avail >= R:
                    chosen = (A, [sa for sa in srcs[:i + 1]], R)
                    break
            if chosen is None:
                R0 = required_force(t, A0)
                shortfall = R0 - (friendly_inflight(t, A0)
                                  + sum(spare[int(sj.id)] for sj, _ in srcs))
                prod_rate = sum(int(sj.production) for sj, _ in srcs)
                if counter0 <= 0.0 and shortfall > 0 and prod_rate > 0:
                    tau = int(math.ceil(shortfall / prod_rate))
                    if tau <= WAIT_HORIZON:
                        cells.append({"kind": "wait", "t": t, "A": int(A0 + tau), "R": int(R0),
                                      "srcs": [sa for sa in srcs],
                                      "value": value(t, A0 + tau) / WAIT_VALUE_MARGIN})
            else:
                A_rel, chosen_srcs, R = chosen
                if R - friendly_inflight(t, A_rel) > 0:
                    cells.append({"kind": "off", "t": t, "A": int(A_rel), "R": int(R),
                                  "srcs": chosen_srcs, "value": value(t, A_rel)})
            if FLIP_TIER and counter0 > 0.0:
                fchosen = None
                for i in range(len(srcs)):
                    A = srcs[i][1]
                    Rf = flip_floor(t, A)[0]
                    avail = (friendly_inflight(t, A)
                             + sum(spare[int(sj.id)] for sj, _ in srcs[:i + 1]))
                    if avail >= Rf:
                        fchosen = (A, [sa for sa in srcs[:i + 1]], Rf)
                        break
                if fchosen is not None:
                    A_f, f_srcs, Rf = fchosen
                    if Rf - friendly_inflight(t, A_f) > 0:
                        cells.append({"kind": "off", "t": t, "A": int(A_f), "R": int(Rf),
                                      "srcs": f_srcs, "value": value(t, A_f, tenure=t_op0)})

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
        # FUNDABILITY GATE: a doomed planet -- one whose reachable reinforcement can't
        # meet the shortfall by the deadline -- builds NO cell, exactly as an offensive
        # target that can't fund a holdable wave this turn builds none. Those ships are
        # freed for offense/consolidation instead of bleeding into a planet that falls
        # anyway (kills the thrash; lets offense win budget slots -> initiative).
        if sum(spare[int(s.id)] for s, _ in d_srcs) < need:
            continue
        # Same currency as offense: value(p, deadline). counterfactual_owner already
        # returns "opp_likely" for a threatened own planet (tte <= deadline) -> mult=2;
        # winnability discounts a shaky hold. No more special-case 2*prod*remain.
        cells.append({"kind": "def", "t": p, "A": int(deadline), "R": int(need),
                      "srcs": d_srcs, "value": value(p, int(deadline))})

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
    # HELD: ships a WAIT cell reserves this turn -- NOT emitted, so they accumulate on the
    # planet for a future wave. avail_spare nets them out so no fire-now cell can spend them.
    held: dict[int, int] = {}

    def avail_spare(sid: int) -> int:
        return max(0, spare.get(sid, 0) - held.get(sid, 0))

    for c in cells:
        t = c["t"]
        tid = int(t.id)
        if tid in committed_tgt:
            continue

        if c["kind"] == "wait":
            # Hold back ONLY the ships this future wave needs (soonest sources first), so
            # they accumulate instead of dribbling into a lower-value fire-now cell. Surplus
            # beyond the need stays available -- the "fire cheap now AND mass for the big
            # one" line. Emit nothing this turn.
            need = c["R"] - friendly_inflight(t, c["A"])
            for s, a in sorted(c["srcs"], key=lambda x: x[1]):
                if need <= 0:
                    break
                take = min(avail_spare(int(s.id)), need)
                if take > 0:
                    held[int(s.id)] = held.get(int(s.id), 0) + take
                    need -= take
            committed_tgt.add(tid)
            continue

        elig = [(s, a) for (s, a) in c["srcs"]
                if int(s.id) not in reserved and avail_spare(int(s.id)) > 0]
        if not elig:
            continue

        if c["kind"] == "off":
            A_rel = c["A"]
            need = c["R"] - friendly_inflight(t, A_rel)
            if need <= 0:
                committed_tgt.add(tid)
                continue
            if friendly_inflight(t, A_rel) + sum(avail_spare(int(s.id)) for s, _ in elig) < c["R"]:
                continue  # a higher cell took our sources -> no longer fundable
            if CONCENTRATED_SALVO:
                # Fire ALL co-arriving sources NOW (largest first) -- they were selected to
                # land on the same turn at real size, so they SUM into one decisive mass and
                # capture on the first attempt. No sub-threshold lone leg (the fundability
                # check above guarantees the salvo meets the requirement).
                for s, a in sorted(elig, key=lambda x: -avail_spare(int(x[0].id))):
                    if need <= 0:
                        break
                    send = min(avail_spare(int(s.id)), need)
                    ang, _eta = aim_and_eta(s, t, send, omega, world=world)
                    if emit(s, t, ang, send, A_rel, "wave", c["R"]):
                        need -= send
            else:
                # OLD: fire the boundary leg now, reserve nearer ones for later turns.
                for s, a in sorted(elig, key=lambda x: -x[1]):
                    if a == A_rel and need > 0:
                        send = min(avail_spare(int(s.id)), need)
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
                send = min(avail_spare(int(s.id)), need)
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
             if int(p.id) not in reserved and avail_spare(int(p.id)) >= REGROUP_MIN_SHIPS),
            key=lambda p: -avail_spare(int(p.id)),
        )
        for s in srcs:
            leftover = avail_spare(int(s.id))  # don't regroup ships a wait cell is holding
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
