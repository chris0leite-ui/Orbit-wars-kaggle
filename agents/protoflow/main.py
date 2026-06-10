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
import types

from lib.intent import World
from lib.world_model import (
    WorldModel, WAVE_LOOKAHEAD, predict_arrival_contest, simulate_planet_timeline,
)
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
IGNORE_COMETS = True          # don't target comets (fleeting, moving) -- skip them as targets/mass
# VALUE HELD PRODUCTION (two-sided value, completed). The field priced only ACQUIRING targets;
# an owned planet was worth 0 (not a target, springboard mass 0), so losing it cost 0 and
# reinforcing it gained 0 -- the field was blind to attacks (4 def launches/game, 14/19 losses
# were planets we drained to <=5 ships and abandoned). Price the region we HOLD at risk: each
# threatened owned planet gets a first-class PROTECT cell valued by value(p) (the region it
# anchors, x2 deny), competing in the SAME ranked list as offense. It holds the planet's own
# ships before a lower-value capture can drain them (drain-cost emerges) and pulls anticipatory
# reinforcement from the STANDING threat (not just in-flight). Replaces the reactive def cell +
# the value-blind reserve threshold; defense falls out of the objective, not a patch.
VALUE_HELD = True             # A/B knob: False -> today's reactive def + threshold reserve
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
# SIMULATE-AND-SCORE EVALUATOR (probe, phase 1). The closed-form value field (phi/opp_phi
# springboard potential + winnability sigmoids + counterfactual_owner deny multiplier) is a
# hand-derived stand-in for the real objective, with ~12 unvalidated shape parameters. The
# Producer beats us 0/12 with NO value field: it rolls the real board forward ~18 turns and
# scores each candidate by the change it causes in (my projected production - opponents'),
# then fires the best few. This flag swaps ONLY the evaluator: a cell's value becomes the
# marginal competitive flow-diff -- inject the cell's real sized arrival into the target's
# per-planet timeline and diff (my-opp) projected ownership against the do-nothing baseline,
# integrated to the horizon. The two-sided denial (enemy->me = +2/turn, neutral->me = +1,
# hold-vs-lose = +2) and the in-flight race fall out of the projection; no springboard term
# (parallel expansion is meant to come from the assembler firing many cells, not a potential).
# The coalition assembler and ALL sizing mechanics (required_force/flip_floor/combined_counter/
# salvo_select/reserve) are UNCHANGED. Default False so existing calibration stays green; the A/B
# and the new SV checks flip it via proto.SIMULATE_VALUE = True.
SIMULATE_VALUE = False
# DRAIN COST (the emergent cost side of the flow-diff). The simulation evaluator above prices a
# move's PRIZE (the target's margin swing) but not its COST (the source planets it empties). So we
# expanded then COLLAPSED (peak ~8 planets vs the Producer, then wiped to 0) -- we drained planets
# we then lost to fund the next capture. Complete the marginal flow-diff: an offense cell's value
# becomes gain(target) MINUS the projected production we lose when its sources are drained -- each
# source re-rolled with its ships removed, charged 2*production per turn it now falls. "Don't gut a
# planet you'll lose" then EMERGES as a value fact (the cell ranks below a safer-funded capture or
# a hold), not a safe-drain cap. Exact per-planet: the do-nothing rollout has no launches, so
# planets don't interact. Applies to offense cells only (wait launches nothing; protect/def HOLD
# ships). No-op unless SIMULATE_VALUE is also on. Default False; the A/B flips it via proto attr.
SIMVALUE_DRAIN_COST = False
# ANTICIPATORY DRAIN (the Producer's safe_drain discipline, expressed as value). The drain cost above
# re-rolls a drained source against only the IN-FLIGHT ledger -- so a frontier source with a strong
# enemy NEARBY but not yet launched merely grows in that baseline, and draining it costs ZERO. We gut
# exposed frontier planets for free, then lose them (the seed-0 collapse). The Producer never does:
# its safe_drain reserves the garrison a planet needs to survive on every held turn. We give the same
# discipline emergently -- inject the source's anticipatory standing+in-flight counter (the SAME
# combined_counter the PROTECT branch sizes against) into both the full and the drained re-roll, so a
# drain that drops a source below holdable against the enemy it can SEE is charged its projected loss.
# A safe rear source (no enemy in range -> counter 0) still drains free. No-op unless SIMULATE_VALUE
# and SIMVALUE_DRAIN_COST are also on. Default False; the A/B flips it via proto attr.
SIMVALUE_DRAIN_ANTICIPATORY = False
# FLOWDIFF (the single-currency rebuild; supersedes SIMULATE_VALUE + SIMVALUE_DRAIN_COST when on).
# Diagnosis (panel-measured): the sim/analytic values price what a move GAINS (production-margin
# swing) but never the ships it SPENDS -- attrition is charged nowhere, so value can't go negative,
# something always looks worth doing, and we over-launch into neutral walls and collapse. The fix is
# ONE currency: signed terminal wealth (my garrison positive, enemy negative, neutral zero) at the
# projection horizon, injected-minus-baseline, MINUS the newly-sent ships at par. Relocation is free
# (survivors credit back), attrition is charged exactly (the dead are missing from the terminal),
# denial counts (an enemy terminal removed), and a pyrrhic capture goes NEGATIVE -> the do-nothing
# alternative (score 0) wins -> discipline emerges. Subsumes the drain cost (the source-side falls
# are the only extra) and winnability's in-flight race (combat in the rollout decides). Takes
# precedence over SIMULATE_VALUE in cell_value dispatch. Default False; A/B via proto attr.
FLOWDIFF_VALUE = False
# FLOWDIFF TAIL (ownership continuation value). Terminal wealth read at the window edge is myopic:
# a neutral whose garrison exceeds its in-window repayment is refused even with 400 turns left, so
# we under-expand while the Producer gobbles the board (seed-0 trace: planet count 5-7 vs 8-13 from
# turn 61). The do-nothing model actually says the owner at the readout KEEPS the planet to game
# end; we credit that continuation stream discounted per turn by VALUE_DISCOUNT -- the discount
# prices the retake the rollout cannot see (a soft ~9-production-turn cap at 0.9), not a hard
# window cliff. Attrition stays charged, so a truly pyrrhic buy stays negative. No-op unless
# FLOWDIFF_VALUE is on. Default False; A/B via proto attr.
FLOWDIFF_TAIL = False
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
    # Comets are short-lived moving bodies (production 1, removed when they leave the board).
    # For now we do NOT target them: capturing one is fleeting value and it drags fleets off
    # course. Exclude them from the target pool and from the springboard mass below.
    comet_ids = world.comet_ids if IGNORE_COMETS else frozenset()
    target_pool = [p for p in (enemy_planets + neutrals) if int(p.id) not in comet_ids]

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
        getmass_me = {int(p.id): (0.0 if int(p.id) in comet_ids
                                  else (1.0 if int(p.owner) == -1
                                        else (0.0 if int(p.owner) == me else ENEMY_REACH_DISCOUNT)))
                      for p in planets}
        phi = compute_potential(getmass_me)
        if OFFENSIVE_PRESSURE:
            # Opponent's seat (mirror): neutrals NEW mass for them (1.0), OUR planets are
            # their springboard mass (discounted), their own planets already theirs (0). In
            # 2P this is the single opponent; in 4P it lumps all opponents (follow-up).
            getmass_opp = {int(p.id): (0.0 if int(p.id) in comet_ids
                                       else (1.0 if int(p.owner) == -1
                                             else (ENEMY_REACH_DISCOUNT if int(p.owner) == me else 0.0)))
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
        if VALUE_HELD:
            return 0  # defensive holding is now a value-ranked PROTECT cell, not a pre-pass cap
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

    # --- SIMULATION value (SIMULATE_VALUE): the marginal competitive flow-diff for
    # capturing/holding `t` landing at `arrive`. We inject the cell's REAL sized arrival into
    # the target's per-planet timeline and diff (my - opponents') projected ownership against
    # the do-nothing baseline (model.timelines), integrated to the horizon. Combat resolution
    # inside the timeline decides whether the capture actually holds against IN-FLIGHT enemy
    # fleets (subsumes winnability's race); the race against enemy PLANET launches is already
    # baked into the sized force via combined_counter (a mechanic). The +1/0/-1 owner sign makes
    # the two-sided denial emerge: enemy->me swings +2/turn, neutral->me +1, hold-vs-lose +2.
    def _margin_owner(o):
        if o == me:
            return 1
        if o == -1 or o is None:
            return 0
        return -1  # an enemy holds it

    def sim_value(t, arrive, tenure=None):
        arrive = int(arrive)
        if arrive >= remain:           # capture lands after the game ends
            return 0.0
        H = int(win_turns)
        if arrive > H:                 # arrival past the projection window -> no measurable swing
            return 0.0
        tid = int(t.id)
        ledger_t = model.ledger.get(tid, [])
        if int(t.owner) == me:
            # PROTECT evaluation. The counterfactual of NOT holding `t` is it facing its threat
            # UNREINFORCED. The real in-flight ledger alone misses a STANDING enemy that has not
            # launched yet, so the baseline models the combined standing+in-flight counter as the
            # attack (the same quantity required_force sizes against) -- this is what makes
            # anticipatory defense ("see the attack coming") emerge instead of only reacting to
            # fleets already in the air. injected = that same attack PLUS our reinforcement.
            threat, t_thr = combined_counter(t, 0)
            if threat <= 0.0 or not enemy_planets:
                return 0.0
            atk_owner = int(min(enemy_planets,
                                key=lambda e: math.hypot(float(e.x) - float(t.x),
                                                         float(e.y) - float(t.y))).owner)
            base_arr = ledger_t + [(int(t_thr), atk_owner, int(math.ceil(threat)))]
            inj_arr = base_arr + [(arrive, me, int(required_force(t, arrive)))]
            base_line = simulate_planet_timeline(t, base_arr, horizon=H)
        else:
            # OFFENSE evaluation. Baseline = the target's natural timeline (do nothing). Inject the
            # SAME force this cell would deliver, so combat decides if the capture actually holds
            # against in-flight enemy fleets (subsumes winnability's in-flight race). Flip cells
            # inject the bare flip floor; hold cells the full combined-counter-holdable force.
            R_eff = int(flip_floor(t, arrive)[0]) if tenure is not None else int(required_force(t, arrive))
            inj_arr = ledger_t + [(arrive, me, R_eff)]
            base_line = model.timelines.get(tid)
            if base_line is None:
                return 0.0
        injected = simulate_planet_timeline(t, inj_arr, horizon=H)
        # A flip we only hold ~tenure turns realizes only its tenure-window swing (the projection
        # window replaces today's tenure discount series).
        upper = H if tenure is None else min(H, arrive + int(tenure))
        swing = 0
        for turn in range(arrive, upper + 1):
            swing += _margin_owner(injected["owner_at"][turn]) - _margin_owner(base_line["owner_at"][turn])
        return float(t.production) * float(swing)

    # --- FLOWDIFF value (FLOWDIFF_VALUE): the exact single-currency evaluator. Score a candidate
    # by the SIGNED TERMINAL WEALTH it creates: garrison at the horizon, mine positive / enemy
    # negative / neutral zero, injected minus baseline, MINUS the newly-sent ships at par. Ships
    # that merely relocate cancel (they reappear in the terminal garrison); ships that die in
    # combat are missing from it -> attrition is finally priced, and a pyrrhic capture goes
    # negative. Special cases become value facts: reinforcing a planet that survives anyway nets
    # exactly 0 (parked ships = home ships); saving a faller nets the saved garrison plus the
    # denied enemy terminal; feeding a doomed planet nets negative (no-bleed emerges).
    def _signed_terminal(tl, turn):
        return float(_margin_owner(tl["owner_at"][turn])) * float(tl["ships_at"][turn])

    def flow_value(t, arrive, tenure=None):
        arrive = int(arrive)
        if arrive >= remain:           # capture lands after the game ends
            return 0.0
        H = int(win_turns)
        if arrive > H:                 # arrival past the projection window -> no measurable swing
            return 0.0
        tid = int(t.id)
        ledger_t = model.ledger.get(tid, [])
        if int(t.owner) == me:
            # PROTECT: same anticipatory construction as sim_value (the standing+in-flight
            # counter is the baseline attack); only the integral changes to terminal wealth.
            threat, t_thr = combined_counter(t, 0)
            if threat <= 0.0 or not enemy_planets:
                return 0.0
            atk_owner = int(min(enemy_planets,
                                key=lambda e: math.hypot(float(e.x) - float(t.x),
                                                         float(e.y) - float(t.y))).owner)
            base_arr = ledger_t + [(int(t_thr), atk_owner, int(math.ceil(threat)))]
            q_new = int(required_force(t, arrive))
            inj_arr = base_arr + [(arrive, me, q_new)]
            base_line = simulate_planet_timeline(t, base_arr, horizon=H)
        else:
            # OFFENSE: inject only the ships we'd NEWLY send (the ledger baseline already
            # carries the friendly in-flight portion of R; injecting full R would credit
            # phantom survivors in the terminal).
            R_eff = int(flip_floor(t, arrive)[0]) if tenure is not None else int(required_force(t, arrive))
            q_new = max(0, R_eff - friendly_inflight(t, arrive))
            if q_new <= 0:
                return 0.0             # in-flight already covers it; nothing new to price
            inj_arr = ledger_t + [(arrive, me, q_new)]
            base_line = model.timelines.get(tid)
            if base_line is None:
                return 0.0
        injected = simulate_planet_timeline(t, inj_arr, horizon=H)
        # A flip we only hold ~tenure turns banks its wealth at the expected-retake turn, not H
        # (the rollout can't see the standing-counter retake, so an uncapped flip looks permanent).
        upper = H if tenure is None else min(H, arrive + int(tenure))
        score = (_signed_terminal(injected, upper) - _signed_terminal(base_line, upper)) - float(q_new)
        if FLOWDIFF_TAIL and tenure is None:
            # Ownership continuation beyond the readout: the owner at `upper` keeps producing to
            # game end under do-nothing; credit it discounted per turn (retake-uncertainty), in
            # the same ship units. Flips are excluded -- their tenure cap IS their continuation.
            own_swing = (_margin_owner(injected["owner_at"][upper])
                         - _margin_owner(base_line["owner_at"][upper]))
            tail = max(0, int(remain) - int(upper))
            if own_swing != 0 and tail > 0:
                d = VALUE_DISCOUNT
                score += float(own_swing) * float(t.production) * (d * (1.0 - d ** tail) / (1.0 - d))
        return score

    def cell_value(t, arrive, tenure=None):
        # Dispatch the cell's RANKING value: flowdiff > simulation evaluator > analytic field.
        if FLOWDIFF_VALUE:
            return flow_value(t, arrive, tenure)
        return sim_value(t, arrive, tenure) if SIMULATE_VALUE else value(t, arrive, tenure)

    # --- DRAIN COST: the cost side of the marginal flow-diff. Removing q ships from a source NOW
    # may make it fall to an in-flight enemy wave it would otherwise have survived. We re-roll the
    # source with its garrison reduced by q and charge the production it loses (2*prod per turn it
    # now falls, by the same owner-sign as the gain). The do-nothing rollout has no launches, so a
    # source's future is independent of the rest of the board -> this is exact per-planet.
    def source_loss(s, q):
        q = int(q)
        if q <= 0:
            return 0.0
        sid = int(s.id)
        H = int(win_turns)
        arr = model.ledger.get(sid, [])
        base_tl = model.timelines.get(sid)
        # ANTICIPATORY DRAIN: the in-flight ledger alone misses a STANDING enemy that has not launched
        # yet, so a drained frontier source merely grows in the baseline and the gut costs nothing.
        # Mirror the PROTECT branch (combined_counter): inject the source's standing+in-flight counter
        # into BOTH the full and drained re-rolls, so a drain that drops the source below holdable
        # against the enemy it can SEE is charged. Both sides must face the SAME counter, so the full
        # baseline is re-rolled here (not model.timelines, which is counter-free).
        if SIMVALUE_DRAIN_ANTICIPATORY and enemy_planets:
            threat, t_thr = combined_counter(s, 0)
            if threat > 0.0:
                atk_owner = int(min(enemy_planets,
                                    key=lambda e: math.hypot(float(e.x) - float(s.x),
                                                             float(e.y) - float(s.y))).owner)
                arr = arr + [(int(t_thr), atk_owner, int(math.ceil(threat)))]
                base_tl = simulate_planet_timeline(
                    types.SimpleNamespace(owner=s.owner, ships=float(s.ships), production=s.production),
                    arr, horizon=H)
        if base_tl is None:
            return 0.0
        shim = types.SimpleNamespace(owner=s.owner, ships=max(0.0, float(s.ships) - q),
                                     production=s.production)
        drained = simulate_planet_timeline(shim, arr, horizon=H)
        loss = 0
        for turn in range(1, H + 1):
            loss += _margin_owner(base_tl["owner_at"][turn]) - _margin_owner(drained["owner_at"][turn])
        return float(s.production) * float(loss)

    # FLOWDIFF source side: flow_value already charges the sent ships at PAR (relocation). The
    # only EXTRA cost is a source that now FALLS to an in-flight wave it would have survived:
    # the terminal diff of the drained re-roll beyond the par prediction, in the same ship units.
    # A safe source drains at exactly par -> extra 0.
    def source_loss_flow(s, q):
        q = int(q)
        if q <= 0:
            return 0.0
        sid = int(s.id)
        base_tl = model.timelines.get(sid)
        if base_tl is None:
            return 0.0
        H = int(win_turns)
        shim = types.SimpleNamespace(owner=s.owner, ships=max(0.0, float(s.ships) - q),
                                     production=s.production)
        drained = simulate_planet_timeline(shim, model.ledger.get(sid, []), horizon=H)
        extra = (_signed_terminal(base_tl, H) - _signed_terminal(drained, H)) - float(q)
        return max(0.0, extra)

    def drain_cost(t, srcs, R, A):
        # Total source-side cost of this offense cell. Allocate the needed force across the
        # sources largest-first by spare (the assembler's funding order). Under FLOWDIFF the
        # per-source charge is the beyond-par fall loss (ship units, always on); under the sim
        # evaluator it is the gated production-margin source_loss.
        if FLOWDIFF_VALUE:
            loss_fn = source_loss_flow
        elif SIMULATE_VALUE and SIMVALUE_DRAIN_COST:
            loss_fn = source_loss
        else:
            return 0.0
        need = int(R) - friendly_inflight(t, A)
        if need <= 0:
            return 0.0
        cost = 0.0
        rem = need
        for s in sorted((s for s, _ in srcs), key=lambda s: -spare.get(int(s.id), 0)):
            if rem <= 0:
                break
            q = min(int(spare.get(int(s.id), 0)), rem)
            if q <= 0:
                continue
            rem -= q
            cost += loss_fn(s, q)
        return cost

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
                "imp": round(cell_value(t, A), 1), "win": round(winnability(t, A), 3),
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
                                  "srcs": co_srcs,
                                  "value": cell_value(t, A_rel) - drain_cost(t, co_srcs, R, A_rel)})
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
                                      "value": cell_value(t, A0 + tau) / WAIT_VALUE_MARGIN})
            # FLIP salvo (contested only): a decisive flip-floor salvo, expected-tenure value.
            if FLIP_TIER and counter0 > 0.0:
                fl = salvo_select(t, lambda tt, AA: flip_floor(tt, AA)[0])
                if fl is not None:
                    A_f, f_srcs, Rf = fl
                    if Rf - friendly_inflight(t, A_f) > 0:
                        cells.append({"kind": "off", "t": t, "A": int(A_f), "R": int(Rf),
                                      "srcs": f_srcs,
                                      "value": cell_value(t, A_f, tenure=t_op0) - drain_cost(t, f_srcs, Rf, A_f)})
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
                                      "value": cell_value(t, A0 + tau) / WAIT_VALUE_MARGIN})
            else:
                A_rel, chosen_srcs, R = chosen
                if R - friendly_inflight(t, A_rel) > 0:
                    cells.append({"kind": "off", "t": t, "A": int(A_rel), "R": int(R),
                                  "srcs": chosen_srcs,
                                  "value": cell_value(t, A_rel) - drain_cost(t, chosen_srcs, R, A_rel)})
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
                                      "srcs": f_srcs,
                                      "value": cell_value(t, A_f, tenure=t_op0) - drain_cost(t, f_srcs, Rf, A_f)})

    for p in my_planets:
        if VALUE_HELD:
            # PROTECT cell: price the region this owned planet anchors against its STANDING
            # threat (anticipatory, not just in-flight). It holds the planet's OWN ships (so a
            # lower-value capture can't drain them) and pulls ally reinforcement, valued by the
            # SAME value() as offense -- defense competes head-to-head, no separate subsystem.
            threat, t_op = combined_counter(p, 0)
            if threat <= 0.0:
                continue
            grow = float(p.production) * float(t_op)
            need_hold = int(math.ceil(threat - grow))
            if need_hold <= 0:
                continue  # production growth alone holds it -> nothing at risk this turn
            d_srcs = [(p, 0)]  # p holds its own ships (arrival 0)
            for s in nearest_k(my_planets, p, SOURCES_PER_TARGET):
                if int(s.id) == int(p.id):
                    continue
                a = arr(s, p)
                if a <= int(t_op):  # an ally arriving by the threat reinforces in time
                    d_srcs.append((s, a))
            cells.append({"kind": "protect", "t": p, "A": int(t_op), "R": int(need_hold),
                          "srcs": d_srcs, "value": cell_value(p, int(t_op))})
        else:
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
            # FUNDABILITY GATE: a doomed planet builds NO cell -> ships freed for offense.
            if sum(spare[int(s.id)] for s, _ in d_srcs) < need:
                continue
            cells.append({"kind": "def", "t": p, "A": int(deadline), "R": int(need),
                          "srcs": d_srcs, "value": cell_value(p, int(deadline))})

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

        if c["kind"] == "protect":
            # Keep an owned planet alive: HOLD its own ships (so a lower-value capture can't
            # drain them) then pull ally reinforcement arriving by the threat. NO-BLEED: if we
            # can't actually meet the hold requirement, hold nothing -- free the ships for
            # offense rather than feed a planet that falls anyway.
            pid = tid
            need = c["R"] - friendly_inflight(t, c["A"])
            if need <= 0:
                committed_tgt.add(pid)
                continue
            total_avail = sum(avail_spare(int(s.id)) for s, _ in c["srcs"]
                              if int(s.id) == pid or int(s.id) not in reserved)
            if total_avail < need:
                committed_tgt.add(pid)  # unsavable -> no bleed
                continue
            hold_own = min(avail_spare(pid), need)
            if hold_own > 0:
                held[pid] = held.get(pid, 0) + hold_own
                need -= hold_own
            for s, a in sorted((sa for sa in c["srcs"] if int(sa[0].id) != pid),
                               key=lambda x: x[1]):
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
            committed_tgt.add(pid)
            continue

        elig = [(s, a) for (s, a) in c["srcs"]
                if int(s.id) not in reserved and avail_spare(int(s.id)) > 0]
        if not elig:
            continue

        if c["kind"] == "off":
            # A move whose net competitive flow-diff is non-positive is worse than holding the
            # ships -- once the drain cost is priced in, firing it would gut a source for less
            # than it costs. HOLDING is the value-0 alternative, so skip it (the cost VETOes a
            # self-defeating drain, not just lowers its rank). Gated with the drain-cost mechanism.
            if (FLOWDIFF_VALUE or SIMVALUE_DRAIN_COST) and c["value"] <= 0.0:
                committed_tgt.add(tid)
                continue
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
