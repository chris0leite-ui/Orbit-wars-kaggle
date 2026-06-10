"""protoflow_calib — synthetic-situation calibration for the action-space field.

PI method (2026-06-04): play simple opponents AND hand-built synthetic states to
learn how to define/calibrate the field. A synthetic state isolates ONE decision
so we can read the field's valuations directly instead of inferring them from
noisy games. Each scenario prints the ranked action field (importance per move)
and what the agent emits, then checks a desired property.

Key physics under test: fleet speed RISES with ship count
  speed = 1 + 5*(ln(ships)/ln(1000))**1.5
so a small fleet is a SLOW fleet. A well-defined field should make slow/small
launches low-value on their own, so dribbling 2-3 ships becomes improbable and
"wait, mass a faster fleet, strike" wins where it should.

Run:  python scripts/protoflow_calib.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agents.protoflow.main as proto
from lib.fleet import speed as fleet_speed


def make_obs(planets, fleets=None, player=0, step=20, omega=0.0):
    """planets: list of [id, owner, x, y, radius, ships, production].
    fleets:  list of [id, owner, x, y, angle, from_planet_id, ships]."""
    # Sanity guard: a planet may never sit inside the sun (radius 10 at the centre) -- such a
    # board is physically impossible and silently breaks trajectories (fleets to/from it cross
    # the sun and die), so any scenario built that way is invalid. Catch it loudly.
    for p in planets:
        if math.hypot(float(p[2]) - 50.0, float(p[3]) - 50.0) < 10.0:
            raise ValueError(f"planet {p[0]} at ({p[2]},{p[3]}) is inside the sun (r=10 @ (50,50))")
    # Harness hygiene: lib.world_model._contest_cache is keyed by planet id and only cleared
    # when the STEP changes. Our scenarios reuse ids 0,1,2 at the same default step, so without
    # this reset a prior board's contest predictions leak into the next. (Real games are immune:
    # fresh process per episode, ids stable.) Reset it so each synthetic board starts clean.
    import lib.world_model as _wm
    _wm._contest_cache.clear()
    _wm._contest_cache_step = None
    return {
        "player": player,
        "planets": [list(p) for p in planets],
        "fleets": [list(f) for f in (fleets or [])],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "comets": [],
        "step": step,
        "remainingOverageTime": 60.0,
    }


def radius(prod):
    return 1.0 + math.log(prod)


def show(name, obs, want):
    proto.reset_trace()
    moves = proto.agent(obs)
    field = proto.get_last_field()
    print(f"\n=== {name} ===")
    print(f"  want: {want}")
    print("  field (ranked by importance):")
    for f in field[:8]:
        own = {-1: "neutral"}.get(f["tgt_owner"], f"P{f['tgt_owner']}")
        spd = fleet_speed(f["ships"])
        print(f"    src{f['src']:>2} -> tgt{f['tgt']:>2} [{own:>7} prod={f['prod']}]  "
              f"ships={f['ships']:>3} speed={spd:.2f}  ttc={f['ttc']:>4}  "
              f"win={f.get('win', '?')}  imp={f['imp']}")
    if not field:
        print("    (field empty)")
    print(f"  EMITTED: {moves if moves else '(hold)'}")
    return moves, field


def _check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main():
    # NOTE: the sun is at (50,50) with radius 10 and destroys any fleet that
    # crosses it. All planets below sit well clear of the sun AND have a clear
    # line of sight to their targets, so trajectories are not silently rejected.

    # S1 — HOLD vs FIRE (the real dribble test). A single home that CANNOT solo-
    # capture a defended target right now (needs ~26, has 10), with no own planet
    # nearby to create reinforcement noise. The only field entries are wait-then-
    # mass candidates (no affordable fire-now capture), so the agent should HOLD
    # and accumulate -- NOT dribble a doomed 10-ship fleet that bounces. A distant
    # enemy means the eventual massed strike still wins the race (worth waiting).
    home = [0, 0, 15.0, 15.0, radius(3), 10, 3]        # 10 ships, prod 3
    defended = [1, -1, 38.0, 28.0, radius(3), 25, 3]   # garrison 25, dist ~26; floor ~26 > 10
    far_enemy = [2, 1, 85.0, 85.0, radius(3), 20, 3]   # far -> the held mass still wins the race
    moves, field = show("S1 hold-vs-fire (home has 10, target needs ~26)",
         make_obs([home, defended, far_enemy]),
         "no affordable fire-now capture -> HOLD and accumulate (no dribble)")
    # The field may still SHOW a sub-floor fire-now entry; the test is that the
    # agent does not FIRE it (it holds and accumulates a capturing fleet instead).
    _check("S1", not moves, f"emitted={moves or '(hold)'} (want hold to accumulate)")
    # This same target appears at several arrival times (fire-now + wait variants),
    # so it doubles as the winnability-vs-arrival check: later arrival must give
    # strictly lower winnability (the entropy discount).
    by_ttc = sorted((f["ttc"], f["win"]) for f in field if f["tgt"] == 1)
    monotone = all(b[1] < a[1] + 1e-9 for a, b in zip(by_ttc, by_ttc[1:]))
    _check("S1-win", len(by_ttc) >= 2 and monotone,
           f"(ttc,win) by ascending time -> {by_ttc}")

    # S1b — same target, home has accumulated a real strike force. Now a decisive
    # fast capture is affordable and high-value -> it should fire.
    home_big = [0, 0, 15.0, 15.0, radius(3), 30, 3]    # 30 ships, can solo now
    moves, field = show("S1b same target, home has 30 ships (decisive fleet ready)",
         make_obs([home_big, defended, far_enemy]),
         "now a decisive fast capture is affordable and should be emitted")
    _check("S1b", bool(moves), f"emitted={moves or '(hold)'}")

    # S2 — CONVERGENCE NEEDED. A defended neutral that NO single planet can take
    # alone, but two planets arriving the same turn can (combat sums them). The
    # two sources are placed symmetrically so both legs share an arrival turn.
    a = [0, 0, 15.0, 30.0, radius(3), 18, 3]
    b = [1, 0, 15.0, 8.0, radius(3), 18, 3]
    defended = [2, -1, 40.0, 19.0, radius(4), 26, 4]   # floor ~27; neither solo (18), both (36) yes
    enemy = [3, 1, 88.0, 80.0, radius(3), 20, 3]
    moves, field = show("S2 convergence-needed (two 18-ship planets vs a 26-garrison target)",
         make_obs([a, b, defended, enemy]),
         "a 2-source same-arrival cohort should form (combat-rule-1 summation)")
    legs_to_def = [m for m in moves if int(m[0]) in (0, 1)]
    _check("S2", len(legs_to_def) >= 2,
           f"emitted={moves or '(hold)'} (want >=2 legs converging on target 2)")

    # S3 — OVERREACH. A juicy target so far the flight exceeds the reach ceiling
    # AND the enemy adjacent wins the race. The field should not send it.
    home3 = [0, 0, 10.0, 80.0, radius(3), 40, 3]
    juicy_far = [1, -1, 95.0, 80.0, radius(5), 5, 5]   # dist 85 (> reach ceiling), enemy adjacent
    enemy_adj = [2, 1, 90.0, 80.0, radius(5), 40, 5]
    moves, field = show("S3 overreach (juicy target far away, enemy adjacent)",
         make_obs([home3, juicy_far, enemy_adj]),
         "should NOT send the far losing-race shot")
    _check("S3", not moves, f"emitted={moves or '(hold)'} (want hold)")

    # S4 — WINNABILITY: near-sure vs far-contested. One home with ample ships; a
    # near uncontested neutral and a far neutral the enemy can reach soon. The
    # near target's winnability should clearly exceed the far contested one, and
    # the agent should prefer the near capture.
    home4 = [0, 0, 15.0, 15.0, radius(3), 40, 3]
    near_safe = [1, -1, 30.0, 25.0, radius(2), 8, 2]    # dist ~18, no enemy near
    far_cont = [2, -1, 60.0, 80.0, radius(3), 8, 3]     # dist ~80; enemy adjacent
    enemy4 = [3, 1, 66.0, 84.0, radius(3), 40, 3]
    moves, field = show("S4 winnability (near-sure vs far-contested)",
         make_obs([home4, near_safe, far_cont, enemy4]),
         "near.win >> far.win; prefer the near capture, not the far gamble")
    win_near = next((f["win"] for f in field if f["tgt"] == 1), None)
    win_far = next((f["win"] for f in field if f["tgt"] == 2), None)
    near_fired = any(int(m[0]) == 0 for m in moves) and not any(
        lc["tgt"] == 2 for lc in proto.get_trace()[-1]["launches"])
    _check("S4", win_near is not None and (win_far is None or win_near > win_far)
           and near_fired,
           f"win_near={win_near} win_far={win_far} emitted={moves or '(hold)'}")

    # S5 — COST OF INACTION (the direct inertia test). A neutral that an enemy is
    # racing for (an in-flight enemy fleet inbound to it), plus our home that can
    # win the race. The two-sided value treats the neutral as a 2x swing (we gain it
    # AND deny it), so the agent FIRES rather than sitting idle. The enemy PLANET is
    # placed out of immediate counter range, so the denial is genuinely HOLDABLE --
    # cost-of-inaction must fire a denial it can keep, not donate one it can't.
    home5 = [0, 0, 15.0, 20.0, radius(3), 40, 3]
    contested_neutral = [1, -1, 33.0, 25.0, radius(2), 6, 3]   # we can reach in ~9
    enemy5 = [2, 1, 85.0, 80.0, radius(3), 40, 3]             # far -> can't immediately counter
    enemy_fleet = [0, 1, 45.0, 30.0, -2.747, 2, 20]            # 20 ships inbound to the neutral
    moves, field = show("S5 cost-of-inaction (enemy racing for a neutral we can win)",
         make_obs([home5, contested_neutral, enemy5], fleets=[enemy_fleet]),
         "the contested neutral is a 2x swing -> FIRE now, do not sit idle")
    fired_at_neutral = any(lc["tgt"] == 1 for lc in proto.get_trace()[-1]["launches"])
    mult2 = next((f for f in field if f["tgt"] == 1 and f["imp"] >= 2 * 3 * (500 - 10) * 0.4), None)
    _check("S5", fired_at_neutral, f"emitted={moves or '(hold)'} (want a launch at the contested neutral)")

    # S6 — DEFENSE on COMMITTED threat. An owned planet with a real enemy fleet
    # inbound it cannot self-cover, plus a nearby surplus ally. A "def" reinforce
    # should be emitted, sized to the shortfall. (A far/speculative enemy must NOT
    # trigger reinforcement -- that was the friendly-reinforce trickle.)
    threatened = [0, 0, 30.0, 50.0, radius(3), 8, 3]           # only 8 ships
    ally = [1, 0, 18.0, 50.0, radius(3), 40, 3]                # nearby surplus
    enemy6 = [2, 1, 80.0, 50.0, radius(3), 40, 3]
    atk_fleet = [0, 1, 45.0, 50.0, 3.1416, 2, 30]             # 30 ships inbound to the threatened planet
    moves, field = show("S6 defense on committed threat (reinforce the threatened planet)",
         make_obs([threatened, ally, enemy6], fleets=[atk_fleet]),
         "a 'def' reinforce from the ally should be emitted, sized to the shortfall")
    def_launch = [lc for lc in proto.get_trace()[-1]["launches"] if lc["kind"] == "def" and lc["tgt"] == 0]
    _check("S6", bool(def_launch), f"def launches={def_launch} emitted={moves or '(hold)'}")

    # S7 — HOLDABLE up-size. The SAME capturable neutral, taken in two worlds. With
    # a strong enemy planet sitting CLOSER to it than our home (so the enemy can
    # counter-recapture before we can defend), the capture must be sized to SURVIVE
    # that counter -- strictly more than the bare flip floor. With no such enemy
    # there is no counter, so the same neutral is taken at the bare floor. This is
    # the force concentration that makes a capture hold. (The counter enemy has only
    # production 1, so it is a far lower-value target than the production-3 neutral
    # and does not outrank it -- the neutral is the capture under test.)
    home7 = [0, 0, 15.0, 30.0, radius(3), 90, 3]               # ample ships
    neutral7 = [1, -1, 33.0, 30.0, radius(2), 6, 3]            # the capture under test
    counter_enemy = [2, 1, 41.0, 30.0, radius(1), 40, 1]       # closer to the neutral; fast counter
    # Baseline world FIRST: no enemy, so the neutral is taken at the bare flip floor.
    moves, field = show("S7b no up-size (same neutral, no counter -> bare floor)",
         make_obs([home7, neutral7]),
         "with no enemy to counter, the same neutral is taken at the bare flip floor")
    flat_l = next((lc for lc in proto.get_trace()[-1]["launches"] if lc["tgt"] == 1), None)
    _check("S7b", flat_l is not None, f"launch={flat_l} (want a floor-sized capture)")
    # Counter world: a closer strong enemy can recapture -> the SAME neutral must be
    # taken with strictly more ships (sized to survive the counter, not just to flip).
    moves, field = show("S7a holdable up-size (capture next to a strong enemy > floor)",
         make_obs([home7, neutral7, counter_enemy]),
         "the SAME neutral is now sized ABOVE the bare floor (to hold the counter)")
    con_l = next((lc for lc in proto.get_trace()[-1]["launches"] if lc["tgt"] == 1), None)
    con_up = con_l is not None and flat_l is not None and con_l["ships"] > flat_l["ships"]
    _check("S7a", con_up,
           f"counter-world ships={con_l['ships'] if con_l else None} > "
           f"floor-world ships={flat_l['ships'] if flat_l else None}")

    # S8 — REGROUP forward. Two own planets -- a rear stockpile and a forward planet
    # nearer the enemy -- and one enemy too strong/far to capture or threaten (so
    # offense and defense emit nothing). The idle rear ships should march FORWARD to
    # the higher-pressure planet, not sit, and not flow backward.
    rear = [0, 0, 15.0, 15.0, radius(3), 40, 3]                # full of idle ships
    forward = [1, 0, 40.0, 22.0, radius(3), 5, 3]              # closer to the enemy
    big_enemy = [2, 1, 82.0, 30.0, radius(5), 200, 5]          # uncapturable + far (no threat)
    moves, field = show("S8 regroup forward (rear idle ships march to the frontier planet)",
         make_obs([rear, forward, big_enemy]),
         "a 'regroup' launch from the rear planet to the forward (higher-pressure) one")
    reg = [lc for lc in proto.get_trace()[-1]["launches"] if lc["kind"] == "regroup"]
    reg_fwd = any(lc["src"] == 0 and lc["tgt"] == 1 for lc in reg)
    reg_back = any(lc["src"] == 1 and lc["tgt"] == 0 for lc in reg)
    _check("S8", reg_fwd and not reg_back,
           f"regroup launches={reg} (want rear0 -> forward1, not backward)")

    # S9 — COMBINED-counter sizing over FREE force. The SAME capturable neutral in
    # two worlds. With ONE enemy able to counter, the holdable size is some value;
    # adding a SECOND counter enemy must raise it. Under the finite-force opponent
    # model the counter sums each enemy's FREE force (net of what it must hold against
    # us), floored at the nearest enemy's full garrison. The counter enemies are made
    # STRONG (60 ships) so they retain free force even while our home pressures them
    # -- a weak enemy near the captured neutral would be fully pinned (free->0) and the
    # "two counters sum" property would be invisible. Both are production-1 (low value)
    # so the production-3 neutral stays the capture under test.
    home9 = [0, 0, 15.0, 40.0, radius(3), 80, 3]
    neutral9 = [1, -1, 33.0, 40.0, radius(3), 6, 3]            # the capture under test
    enemyA = [2, 1, 41.0, 40.0, radius(1), 60, 1]             # strong counter from the east
    enemyB = [3, 1, 33.0, 28.0, radius(1), 60, 1]             # strong counter from the north
    # We read the REQUIRED FORCE off the field (the sizing under test), not the funded launch:
    # with self-protective reserve a home flanked by two strong enemies holds ships back, so
    # the funded launch reflects reserve, not the combined-counter sizing. req_for reads the
    # required ships the field computed for the cell.
    def req_for9(field, tgt):
        es = [f["ships"] for f in field if f["tgt"] == tgt]
        return max(es) if es else None
    moves, field = show("S9a one counter (single strong enemy in recapture range)",
         make_obs([home9, neutral9, enemyA]),
         "the neutral is sized to hold against ONE free counter")
    one_req = req_for9(field, 1)
    _check("S9a", one_req is not None, f"required force={one_req} (want a holdable sizing)")
    moves, field = show("S9b two counters (combined free recapture wave is larger)",
         make_obs([home9, neutral9, enemyA, enemyB]),
         "with TWO free counters in range, the SAME neutral needs strictly more ships")
    two_req = req_for9(field, 1)
    combined = two_req is not None and one_req is not None and two_req > one_req
    _check("S9b", combined,
           f"two-counter req={two_req} > one-counter req={one_req}")

    # S10 — NO SUB-THRESHOLD BOUNCE on a multi-distance target (salvo model). A defended
    # target NO single planet can fund, reachable by a FAR and a NEAR source at DIFFERENT
    # distances (so they cannot co-arrive), plus a cheap solo neutral. The OLD design fired a
    # lone far boundary leg now and deferred the near leg -- the legs landed on different turns
    # and BOUNCED. Under the salvo model we must NOT send any leg below the floor at the wave
    # target (it can't be synchronized this turn); we either take a decisive solo or wait.
    far_src = [0, 0, 20.0, 20.0, radius(3), 20, 3]            # far from the target
    near_src = [1, 0, 55.0, 32.0, radius(3), 20, 3]           # near target AND near the solo
    wave_tgt = [2, -1, 70.0, 20.0, radius(4), 30, 4]          # defended; no single source funds it
    solo = [3, -1, 60.0, 42.0, radius(2), 5, 2]              # cheap decisive solo
    moves, field = show("S10 no sub-threshold bounce (multi-distance target can't synchronize)",
         make_obs([far_src, near_src, wave_tgt, solo]),
         "no lone sub-threshold leg at the wave target; a decisive solo is fine, bouncing is not")
    l10 = proto.get_trace()[-1]["launches"]
    floor_tgt2 = max(proto.MIN_FLEET_SIZE, 30 + proto.CAPTURE_MARGIN)
    bounce = any(lc["tgt"] == 2 and lc["ships"] < floor_tgt2 for lc in l10)
    _check("S10", not bounce,
           f"launches={l10} (want NO sub-threshold leg <{floor_tgt2} at wave target 2)")

    # S11 — WAIT GATE (the opening-tempo dribble, reproduced). One small home in ship
    # range of BOTH a NEAR high-production neutral it cannot quite afford this turn
    # (needs ~11, has 8) and a FAR low-production neutral it CAN afford now. The old
    # field dribbled the cheap far capture (replay step 6: 7 ships at a floor-7 neutral
    # 62 away). The wait gate must instead HOLD -- the near high-prod capture is one
    # turn of accumulation away and worth far more. Then, once the home has accrued the
    # ships, it must FIRE at the near neutral (Rule 38: reproduce the failing state).
    home11 = [0, 0, 20.0, 30.0, radius(3), 8, 3]
    near_hi = [1, -1, 33.0, 28.0, radius(5), 9, 5]   # near + high prod; floor ~11 > home's 8
    far_lo = [2, -1, 20.0, 52.0, radius(2), 5, 2]    # far + low prod; floor ~7, affordable NOW
    moves, field = show("S11a wait gate: hold for near high-prod, don't dribble far",
         make_obs([home11, near_hi, far_lo]),
         "HOLD this turn -- do NOT fire the cheap far neutral")
    l11a = proto.get_trace()[-1]["launches"]
    _check("S11a", len(l11a) == 0,
           f"emitted={moves if moves else '(hold)'} (want hold; no far-neutral dribble)")

    home11_rich = [0, 0, 20.0, 30.0, radius(3), 12, 3]   # accumulated -> can now afford near
    moves, field = show("S11b after accumulation: fire the near high-prod neutral",
         make_obs([home11_rich, near_hi, far_lo]),
         "fire at the NEAR high-prod neutral (target 1)")
    l11b = proto.get_trace()[-1]["launches"]
    fires_near = any(lc["src"] == 0 and lc["tgt"] == 1 for lc in l11b)
    _check("S11b", fires_near,
           f"launches={l11b} (want a launch at near high-prod neutral 1)")

    # S12 — FREE-FORCE PINNING (the over-conservatism fix, Rule 38). The SAME neutral
    # with TWO strong counter enemies, in two worlds that differ ONLY by our home's
    # size. The required force to hold a capture is the COMBINED counter; under the
    # finite-force model an enemy under our pressure must keep ships home to survive us,
    # so its FREE counter shrinks. A bigger home pressures the enemies harder -> their
    # free force drops -> the combined counter falls to the nearest-full floor -> the
    # SAME neutral is sized with STRICTLY FEWER ships. The old "sum of every enemy's
    # full garrison" model computed the same large counter in both worlds (no free-force
    # concept) and over-sized always; this reproduces and fixes that. We read the
    # required force directly off the field (independent of our sourcing).
    def req_for(field, tgt):
        es = [f["ships"] for f in field if f["tgt"] == tgt]
        return max(es) if es else None
    enemyA12 = [2, 1, 41.0, 40.0, radius(1), 50, 1]
    enemyB12 = [3, 1, 33.0, 28.0, radius(1), 50, 1]
    neutral12 = [1, -1, 33.0, 40.0, radius(3), 6, 3]
    home_weak = [0, 0, 15.0, 40.0, radius(3), 40, 3]   # too small to pressure -> enemies free
    _m, field_weak = show("S12a small home: enemies free -> large combined counter",
         make_obs([home_weak, neutral12, enemyA12, enemyB12]),
         "required force is high (both enemies contribute full free counter)")
    req_weak = req_for(field_weak, 1)
    home_strong = [0, 0, 15.0, 40.0, radius(3), 95, 3]  # pressures both enemies -> free shrinks
    _m, field_strong = show("S12b strong home: enemies pinned -> smaller combined counter",
         make_obs([home_strong, neutral12, enemyA12, enemyB12]),
         "required force DROPS (our pressure cuts the enemies' free counter)")
    req_strong = req_for(field_strong, 1)
    pinned_cheaper = (req_weak is not None and req_strong is not None
                      and req_strong < req_weak)
    _check("S12", pinned_cheaper,
           f"required force pinned={req_strong} < unpressured={req_weak} "
           f"(pressure shrinks the enemy's free counter)")

    # S13 — DOOMED DEFENSE: no bleed (Rule 38). An owned planet under an overwhelming
    # committed wave it cannot be saved from (the only reachable ally can field far less
    # than the shortfall by the deadline). Defense is now scored in the SAME currency as
    # offense and GATED by fundability, so an unholdable planet builds NO cell -> we send
    # NOTHING into it (the old hardcoded 2*prod*remain made it max-priority and we bled
    # ships into a planet that falls anyway -- the ~38% thrash). The doomed planet's ally
    # keeps its ships for offense/consolidation instead.
    doomed = [0, 0, 30.0, 50.0, radius(2), 5, 2]              # tiny garrison
    far_ally = [1, 0, 18.0, 50.0, radius(3), 12, 3]           # only 12 ships -- can't cover the wave
    enemy13 = [2, 1, 85.0, 50.0, radius(3), 40, 3]            # far -> no free follow-up in window
    big_wave = [0, 1, 40.0, 50.0, 3.1416, 2, 90]             # 90 ships inbound -> shortfall ~80 >> 12
    moves, field = show("S13 doomed defense (overwhelming wave, ally can't cover)",
         make_obs([doomed, far_ally, enemy13], fleets=[big_wave]),
         "no 'def' launch into the doomed planet -- ships are not bled in")
    def13 = [lc for lc in proto.get_trace()[-1]["launches"]
             if lc["kind"] == "def" and lc["tgt"] == 0]
    _check("S13", not def13,
           f"def launches to doomed planet={def13} (want none -- no bleed)")

    # S14 — EXPANSION POTENTIAL (the snowball, Rule 38). Two neutrals of EQUAL production
    # and EQUAL distance from home (hence equal winnability), both affordable: one
    # ISOLATED, one ADJACENT to a cluster of other capturable neutrals. The farsighted
    # field values the cluster-adjacent one far higher (capturing it springs to more
    # production) and fires it. The old isolated-stream field valued them equally -- the
    # EXPANSION_POTENTIAL=False cross-check confirms the lift comes from the potential,
    # not the geometry.
    def imp_for(field, tgt):
        es = [f["imp"] for f in field if f["tgt"] == tgt]
        return max(es) if es else None
    home14 = [0, 0, 15.0, 15.0, radius(3), 14, 3]
    iso = [1, -1, 15.0, 40.0, radius(3), 6, 3]          # isolated neutral (north), dist 25
    clus = [2, -1, 40.0, 15.0, radius(3), 6, 3]         # cluster-adjacent neutral (east), dist 25
    c1 = [3, -1, 58.0, 15.0, radius(3), 6, 3]           # cluster around `clus`, away from home/iso
    c2 = [4, -1, 50.0, 6.0, radius(3), 6, 3]
    c3 = [5, -1, 40.0, 4.0, radius(3), 6, 3]
    board14 = [home14, iso, clus, c1, c2, c3]
    moves, field = show("S14 expansion potential (cluster-adjacent beats isolated)",
         make_obs(board14),
         "the cluster-adjacent neutral (2) outscores the isolated one (1) and is fired")
    imp_iso, imp_clus = imp_for(field, 1), imp_for(field, 2)
    # The cluster-adjacent planets are 2,3,4,5 (clus + the cluster members c1/c2/c3, which
    # are themselves cluster-adjacent); the isolated one is 1. We require the cluster value
    # to beat the isolated value AND that we fire at SOME cluster planet, not the isolated.
    fired_clus = any(lc["tgt"] in (2, 3, 4, 5) for lc in proto.get_trace()[-1]["launches"])
    fired_iso = any(lc["tgt"] == 1 for lc in proto.get_trace()[-1]["launches"])
    _check("S14", imp_clus is not None and imp_iso is not None
           and imp_clus > imp_iso and fired_clus and not fired_iso,
           f"cluster imp={imp_clus} > isolated imp={imp_iso}; fired cluster={fired_clus}, iso={fired_iso}")
    # baseline cross-check: with the potential OFF, equal prod + equal distance -> tie.
    proto.EXPANSION_POTENTIAL = False
    _m, field_base = show("S14b baseline tie (potential OFF -> equal value)",
         make_obs(board14),
         "without the potential the two neutrals tie (lift comes from phi, not geometry)")
    proto.EXPANSION_POTENTIAL = True
    bi, bc = imp_for(field_base, 1), imp_for(field_base, 2)
    tie = (bi is not None and bc is not None
           and abs(bi - bc) <= 0.01 * max(bi, bc))
    _check("S14b", tie, f"baseline isolated={bi} ~= cluster={bc} (tie within 1%)")

    # S15 — FLIP TIER: break the freeze on an UNHOLDABLE contested capture (Rule 38).
    # Home has ample ships and faces no in-flight threat (reserve ~ 0 -> spare full), but a
    # strong enemy sits right next to a contested neutral, so the COMBINED-counter hold
    # force is far more than we can fund. Today's hold-only gate builds NO cell for that
    # neutral and we freeze -- no launch (this is the 65-launches/game-vs-Producer failure
    # in miniature). The flip tier offers a cheap flip-floor capture priced by its expected
    # tenure, so we DEPLOY and bank production instead of sitting idle.
    # Home can FLIP (floor ~8) but is far short of the HOLD force (~72) -- and far enough
    # short that even the max wait-gate accumulation (4 turns * prod 3 = 12 ships) can't
    # close the gap, so the agent does not merely "wait then hold": it freezes outright.
    home15 = [0, 0, 15.0, 40.0, radius(3), 50, 3]            # can flip, can't hold even after waiting
    neutral15 = [1, -1, 33.0, 40.0, radius(3), 6, 3]         # contested capture under test
    strong_enemy15 = [2, 1, 41.0, 40.0, radius(1), 99, 1]    # adjacent -> huge combined counter
    board15 = [home15, neutral15, strong_enemy15]
    # FLIP OFF first: reproduce the freeze (no launch at the unholdable neutral).
    proto.FLIP_TIER = False
    show("S15a-off freeze (unholdable contested neutral, hold-only gate)",
         make_obs(board15),
         "with the flip tier OFF we cannot fund the hold -> NO launch (the freeze)")
    off_at_neutral = [lc for lc in proto.get_trace()[-1]["launches"] if lc["tgt"] == 1]
    proto.FLIP_TIER = True
    _check("S15a-off", not off_at_neutral,
           f"launches at neutral with flip OFF={off_at_neutral} (want none -- frozen)")
    # FLIP ON: the same neutral now gets a cheap flip-sized capture (below the hold size).
    _m, field_on = show("S15a flip tier deploys (cheap flip capture, banks production)",
         make_obs(board15),
         "with the flip tier ON we fire a cheap flip-floor capture at the neutral")
    flip_l = next((lc for lc in proto.get_trace()[-1]["launches"] if lc["tgt"] == 1), None)
    hold_req = req_for(field_on, 1)  # the displayed hold size (combined counter)
    _check("S15a", flip_l is not None and hold_req is not None and flip_l["ships"] < hold_req,
           f"flip launch ships={flip_l['ships'] if flip_l else None} < hold size={hold_req} "
           f"(deploy a flip we can fund, not the unfundable hold)")

    # S15b — CONCENTRATION preserved: where the hold IS affordable, we pay for it. The flip
    # tier must not cannibalize a holdable capture into a cheap flip. Ample home + a
    # contested neutral with a counter enemy it CAN out-fund -> the launched capture must
    # exceed the bare flip floor (the full-value hold cell outranks the flip and fires).
    home15b = [0, 0, 15.0, 30.0, radius(3), 90, 3]
    neutral15b = [1, -1, 33.0, 30.0, radius(2), 6, 3]
    counter15b = [2, 1, 41.0, 30.0, radius(1), 40, 1]
    show("S15b concentration preserved (holdable -> pay the hold, not the flip)",
         make_obs([home15b, neutral15b, counter15b]),
         "the holdable neutral is taken ABOVE the bare flip floor (hold outranks flip)")
    l15b = next((lc for lc in proto.get_trace()[-1]["launches"] if lc["tgt"] == 1), None)
    bare = max(proto.MIN_FLEET_SIZE, 6 + proto.CAPTURE_MARGIN)  # flip floor for a 6-garrison neutral
    _check("S15b", l15b is not None and l15b["ships"] > bare,
           f"launched ships={l15b['ships'] if l15b else None} > flip floor={bare} (paid for the hold)")

    # S16 — PRODUCTION LEADS on a dense board (Rule 38). A big isolated planet (prod 5) vs a
    # small planet (prod 1) embedded in a dense cluster of other small neutrals, at equal
    # distance from home. With the springboard summed over ALL neighbours (SPRINGBOARD_TOPK=0,
    # the old behaviour) the clustered small planet's connectivity outscores the big one -- we
    # chase centrality. With the bounded top-K springboard, own production leads and the big
    # planet wins. We read values off the field (fundability-independent) and toggle the knob.
    home16 = [0, 0, 8.0, 50.0, radius(3), 40, 3]
    big16 = [1, -1, 33.0, 50.0, radius(5), 6, 5]    # isolated, high production
    small16 = [2, -1, 8.0, 78.0, radius(1), 6, 1]   # low production, but in a dense cluster
    cl16 = [[3, -1, 20.0, 80.0, radius(1), 6, 1], [4, -1, 2.0, 88.0, radius(1), 6, 1],
            [5, -1, 18.0, 70.0, radius(1), 6, 1], [6, -1, 2.0, 68.0, radius(1), 6, 1],
            [7, -1, 22.0, 72.0, radius(1), 6, 1], [8, -1, 12.0, 90.0, radius(1), 6, 1],
            [9, -1, 28.0, 84.0, radius(1), 6, 1]]
    board16 = [home16, big16, small16] + cl16
    proto.SPRINGBOARD_TOPK = 0
    _m, f16_sum = show("S16-sum dense board, springboard summed (clustered small wins -- the bug)",
         make_obs(board16),
         "with the unbounded sum the clustered small planet outscores the big one")
    big_sum, small_sum = imp_for(f16_sum, 1), imp_for(f16_sum, 2)
    proto.SPRINGBOARD_TOPK = 2
    _m, f16 = show("S16 dense board, bounded springboard (big planet leads)",
         make_obs(board16),
         "with the bounded top-K the big planet outscores the clustered small one")
    big_b, small_b = imp_for(f16, 1), imp_for(f16, 2)
    _check("S16",
           None not in (big_sum, small_sum, big_b, small_b)
           and small_sum > big_sum and big_b > small_b,
           f"summed: small={small_sum} > big={big_sum}; bounded: big={big_b} > small={small_b}")

    # S16b — STEPPING-STONE still rewarded under bounding. Two equal small planets at equal
    # distance from home: A unlocks a BIG planet one hop beyond; B unlocks only a small one.
    # The bounded springboard must still prefer the stepping stone that opens the big planet.
    homeS = [0, 0, 15.0, 50.0, radius(3), 14, 3]
    A16 = [1, -1, 30.0, 30.0, radius(1), 6, 1]
    big_beyond = [2, -1, 42.0, 22.0, radius(5), 6, 5]
    B16 = [3, -1, 30.0, 70.0, radius(1), 6, 1]
    small_beyond = [4, -1, 42.0, 78.0, radius(1), 6, 1]
    _m, f16b = show("S16b stepping-stone to big still beats stepping-stone to small",
         make_obs([homeS, A16, big_beyond, B16, small_beyond]),
         "the stepping stone that unlocks a BIG planet outvalues the one that unlocks a small")
    vA, vB = imp_for(f16b, 1), imp_for(f16b, 3)
    _check("S16b", vA is not None and vB is not None and vA > vB,
           f"stepping-to-big A={vA} > stepping-to-small B={vB}")

    # S17 — GO FOR THE HUB (offensive pressure, Rule 38). Two enemy planets of equal
    # production at equal distance from home: a HUB adjacent to a cluster of OUR planets (the
    # region the opponent can press into -- opp_phi credits our planets, our own phi ignores
    # them) and an isolated OUTPOST. With OFFENSIVE_PRESSURE on, taking the hub collapses more
    # of the opponent's reachable region, so the offense term lifts the hub far more than the
    # outpost. (Off=False already favours the hub a little via the reach race; the test is the
    # offense-induced LIFT, which is large for the hub and negligible for the outpost.)
    homeH = [0, 0, 15.0, 15.0, radius(3), 60, 3]
    hub = [1, 1, 15.0, 45.0, radius(3), 10, 3]      # enemy adjacent to our cluster (their reach into us)
    outpost = [2, 1, 45.0, 15.0, radius(3), 10, 3]  # isolated enemy, equal distance from home
    mine_near_hub = [[3, 0, 5.0, 55.0, radius(3), 20, 3], [4, 0, 8.0, 33.0, radius(3), 20, 3],
                     [5, 0, 26.0, 52.0, radius(3), 20, 3]]  # all clear of the sun
    boardH = [homeH, hub, outpost] + mine_near_hub

    def hub_out_vals():
        proto.reset_trace(); proto.agent(make_obs(boardH))
        f = proto.get_last_field()
        return imp_for(f, 1), imp_for(f, 2)

    proto.OFFENSIVE_PRESSURE = True
    hub_on, out_on = hub_out_vals()
    proto.OFFENSIVE_PRESSURE = False
    hub_off, out_off = hub_out_vals()
    proto.OFFENSIVE_PRESSURE = True
    _m, _f = show("S17 offensive pressure (collapse the opponent's hub, not a lone outpost)",
         make_obs(boardH),
         "offense lifts the region-anchoring hub far more than the isolated outpost")
    s17_ok = (None not in (hub_on, out_on, hub_off, out_off)
              and hub_on > out_on and (hub_on - hub_off) > (out_on - out_off))
    _check("S17", s17_ok,
           f"hub_on={hub_on} > out_on={out_on}; hub lift={hub_on-hub_off:.1f} "
           f">> outpost lift={out_on-out_off:.1f}")

    # S18 — SAVE UP for a defended big planet over a long wait (Rule 38; the opening bug).
    # Home produces 1/turn with ~10 ships (like the real home). A big production-5 planet is
    # defended (garrison 20), so flipping it needs ~12 turns of accumulation; a cheap
    # production-1 planet is affordable now; no enemy is in counter range. With the old
    # 4-turn wait cap we dribble into the cheap planet (what lost us the opening vs the
    # Producer); with the wait reasoning over the value window we HOLD and save up for the
    # big one. Waiting is a first-class value choice, not a fixed cap.
    home18 = [0, 0, 15.0, 15.0, radius(1), 10, 1]
    big18 = [1, -1, 40.0, 15.0, radius(5), 20, 5]    # high production, defended -> long save-up
    cheap18 = [2, -1, 15.0, 35.0, radius(1), 5, 1]   # low production, affordable now
    board18 = [home18, big18, cheap18]
    proto.WAIT_HORIZON = 4
    show("S18-short wait cap 4 -> dribble the cheap planet (the opening bug)",
         make_obs(board18),
         "with a 4-turn cap we cannot save up, so we fire the cheap production-1")
    short_cheap = any(lc["tgt"] == 2 for lc in proto.get_trace()[-1]["launches"])
    proto.WAIT_HORIZON = 18
    show("S18 wait over the value window -> HOLD and save up for the big planet",
         make_obs(board18),
         "with the value-horizon wait we hold for the production-5, no cheap dribble")
    long_launches = proto.get_trace()[-1]["launches"]
    held_for_big = not any(lc["tgt"] == 2 for lc in long_launches)
    _check("S18", short_cheap and held_for_big,
           f"cap-4 fired cheap={short_cheap}; value-horizon held (no cheap dribble)={held_for_big} "
           f"(launches={long_launches or '(hold)'})")

    # S19 — NO SUB-THRESHOLD BOUNCE (Rule 38; the multi-attempt waste). A defended target that
    # no single source can fund, reachable by a FAR and a NEAR source at DIFFERENT distances
    # (so they can't co-arrive this turn) plus a cheap solo. With CONCENTRATED_SALVO off the
    # old assembly fires a lone far leg below the floor (it bounces; we re-attack later -- the
    # 15/19 multi-attempt waste). With it on we never emit a sub-threshold leg at that target.
    far19 = [0, 0, 20.0, 20.0, radius(3), 20, 3]
    near19 = [1, 0, 55.0, 32.0, radius(3), 20, 3]
    wave19 = [2, -1, 70.0, 20.0, radius(4), 30, 4]   # floor ~32; no single source funds it
    solo19 = [3, -1, 60.0, 42.0, radius(2), 5, 2]
    board19 = [far19, near19, wave19, solo19]
    floor19 = max(proto.MIN_FLEET_SIZE, 30 + proto.CAPTURE_MARGIN)
    proto.CONCENTRATED_SALVO = False
    show("S19-old cross-turn assembly -> sub-threshold leg bounces off the wave target",
         make_obs(board19), "old: a lone far leg lands below the floor and bounces")
    sub_off = [lc for lc in proto.get_trace()[-1]["launches"]
               if lc["tgt"] == 2 and lc["ships"] < floor19]
    proto.CONCENTRATED_SALVO = True
    show("S19 salvo -> no sub-threshold leg at the wave target",
         make_obs(board19), "salvo: never emit a leg below the floor at an un-synchronizable target")
    sub_on = [lc for lc in proto.get_trace()[-1]["launches"]
              if lc["tgt"] == 2 and lc["ships"] < floor19]
    _check("S19", bool(sub_off) and not sub_on,
           f"old emitted sub-threshold leg={sub_off}; salvo emitted none={not sub_on}")

    # S20 / S21 — DON'T DRAIN A VALUABLE THREATENED PLANET (Rule 38; the core leak: 14/19
    # losses were planets we drained to <=5 ships). A frontier planet next to a strong enemy it
    # CAN survive by holding, plus a tempting neutral. With all protection OFF we drain the
    # frontier to grab the neutral and leave it open; with VALUE_HELD on its PROTECT cell holds
    # its own ships (the region at stake outvalues the marginal capture) and we do not drain it.
    frontier20 = [0, 0, 30.0, 20.0, radius(3), 30, 3]
    enemy20 = [1, 1, 47.0, 20.0, radius(3), 40, 1]    # strong, but frontier+growth can survive it
    neutral20 = [2, -1, 15.0, 20.0, radius(2), 6, 2]  # tempting capture on the far side
    board20 = [frontier20, enemy20, neutral20]
    proto.VALUE_HELD = False
    proto.RESERVE_THREAT = False
    show("S21-off drain the frontier to grab the neutral (self-exposure)",
         make_obs(board20), "with no held-value, draining the frontier looks free")
    drain_off = any(lc["src"] == 0 and lc["tgt"] == 2 for lc in proto.get_trace()[-1]["launches"])
    proto.RESERVE_THREAT = True
    proto.VALUE_HELD = True
    show("S21 hold the frontier (its protect cell outvalues the marginal capture)",
         make_obs(board20), "valuing the held region, the frontier is not drained open")
    drain_on = any(lc["src"] == 0 and lc["tgt"] == 2 for lc in proto.get_trace()[-1]["launches"])
    _check("S21", drain_off and not drain_on,
           f"off drained the frontier={drain_off}; held-value protected it={not drain_on}")

    # S22 — ANTICIPATORY REINFORCEMENT (Rule 38; "see the attack coming"). A threatened planet
    # facing a STANDING enemy threat with NO in-flight fleet yet, plus a surplus ally. The old
    # reactive defense (committed_threat) does nothing -- there is no in-flight fleet to react
    # to. Valuing the held region, we reinforce/hold from the standing threat BEFORE it lands.
    threatened22 = [0, 0, 30.0, 80.0, radius(3), 6, 3]   # weak garrison, no in-flight attacker yet
    ally22 = [1, 0, 25.0, 80.0, radius(3), 40, 3]         # close surplus (can reinforce in time)
    enemy22 = [2, 1, 46.0, 80.0, radius(3), 40, 1]        # standing threat, has NOT launched
    board22 = [threatened22, ally22, enemy22]
    proto.VALUE_HELD = False
    show("S22-off reactive defense does nothing (no in-flight fleet to react to)",
         make_obs(board22), "the old def path waits for a launched fleet -- it never comes in time")
    react_def = [lc for lc in proto.get_trace()[-1]["launches"] if lc["kind"] == "def" and lc["tgt"] == 0]
    proto.VALUE_HELD = True
    show("S22 anticipatory protect from the standing threat",
         make_obs(board22), "valuing the held planet, we reinforce/hold before the attack lands")
    l22 = proto.get_trace()[-1]["launches"]
    protect_act = any(lc["tgt"] == 0 and lc["kind"] == "def" for lc in l22)
    _check("S22", (not react_def) and protect_act,
           f"reactive def fired={bool(react_def)} (want none); anticipatory reinforce={protect_act}")

    # =====================================================================================
    # SIMULATE_VALUE — the simulation-based evaluator (forward-projection competitive flow-diff)
    # replaces the analytic phi/winnability field. These checks (Rule 38) reproduce the states
    # the analytic field MISVALUES and assert the simulated value gets them right. Each runs with
    # proto.SIMULATE_VALUE = True; the flag is restored to False after the block so the rest of
    # the default suite is unaffected. (The springboard checks S14/S16/S17 are NOT re-asserted
    # here -- the simulation evaluator deliberately has no springboard, so cluster-adjacency gives
    # no lift; that is the intended difference, not a regression.)
    # =====================================================================================
    def _imp(field, tgt):
        vals = [f["imp"] for f in field if f["tgt"] == tgt]
        return max(vals) if vals else None

    proto.SIMULATE_VALUE = True

    # SV1 — value scales with production (basic ordering). Two affordable, uncontested neutrals,
    # production 5 vs 1; the integrated production swing must rank the big one above the small.
    home_sv = [0, 0, 15.0, 50.0, radius(3), 60, 3]
    big_sv = [1, -1, 40.0, 30.0, radius(5), 6, 5]
    small_sv = [2, -1, 40.0, 70.0, radius(1), 6, 1]
    moves, field = show("SV1 simulated value scales with production (big prod5 > small prod1)",
         make_obs([home_sv, big_sv, small_sv]),
         "the integrated production swing ranks the production-5 neutral above the production-1")
    imp_big, imp_small = _imp(field, 1), _imp(field, 2)
    fired_big = any(lc["tgt"] == 1 for lc in proto.get_trace()[-1]["launches"])
    _check("SV1", imp_big is not None and imp_small is not None and imp_big > imp_small and fired_big,
           f"imp(big prod5)={imp_big} > imp(small prod1)={imp_small}; fired big={fired_big}")

    # SV2 — two-sided denial emerges from the owner SIGN (no counterfactual multiplier). Capturing
    # an ENEMY planet flips a held owner from -1 to +1 (a swing of 2 per turn); capturing a NEUTRAL
    # of the same production flips 0 to +1 (a swing of 1). So an equal-production, equidistant enemy
    # target must score about TWICE the neutral -- the deny doubling falls out of the integral, not
    # a bolted-on x2. A single, non-adjacent enemy keeps the sizing comparable for both.
    home_d = [0, 0, 15.0, 50.0, radius(3), 90, 3]
    neutral_d = [1, -1, 38.0, 35.0, radius(3), 6, 3]
    enemy_d = [2, 1, 38.0, 65.0, radius(3), 6, 3]   # same production, ~same distance as the neutral
    moves, field = show("SV2 two-sided denial from the owner sign (enemy ~2x an equal neutral)",
         make_obs([home_d, neutral_d, enemy_d]),
         "an equal-production enemy target scores ~2x the neutral (deny doubling emerges)")
    imp_neutral_d, imp_enemy_d = _imp(field, 1), _imp(field, 2)
    _check("SV2", imp_neutral_d is not None and imp_enemy_d is not None
           and imp_enemy_d > 1.5 * imp_neutral_d,
           f"imp(enemy prod3)={imp_enemy_d} ~2x imp(neutral prod3)={imp_neutral_d} (denial from sign)")

    # SV3 — protecting a valuable planet about to fall scores high enough to act. The S22 board (a
    # threatened planet under a STANDING enemy threat, no in-flight fleet yet, plus a surplus
    # ally). The simulation's protect baseline models that standing threat -> the planet falls
    # unreinforced -> the held region is a +2/turn swing -> we reinforce anticipatorily.
    moves, field = show("SV3 anticipatory protect under the simulation evaluator",
         make_obs(board22), "the simulated protect value (planet falls unreinforced) drives a hold/reinforce")
    l_sv3 = proto.get_trace()[-1]["launches"]
    protect_sv3 = any(lc["tgt"] == 0 and lc["kind"] == "def" for lc in l_sv3)
    _check("SV3", protect_sv3,
           f"anticipatory reinforce under simulation={protect_sv3} (sees the standing attack)")

    # SV4 — a neutral the projection ALREADY gives us is worth ~0 (marginal value, no double-pay).
    # A neutral with our own larger fleet already inbound that captures it regardless: the baseline
    # timeline already shows it as ours, so any cell on it adds ~no swing.
    home_f = [0, 0, 15.0, 50.0, radius(3), 60, 3]
    free_n = [1, -1, 40.0, 50.0, radius(3), 6, 3]
    own_fleet = [0, 0, 30.0, 50.0, 0.0, 0, 40]   # 40 of OUR ships already inbound, capture it soon
    moves, field = show("SV4 a neutral we already get scores ~0 (marginal value)",
         make_obs([home_f, free_n], fleets=[own_fleet]),
         "baseline already shows the neutral as ours -> the cell adds ~no competitive swing")
    imp_free = _imp(field, 1)
    _check("SV4", imp_free is not None and imp_free <= 3.0,
           f"imp(already-ours neutral)={imp_free} ~= 0 (no double-paying for an inevitable capture)")

    # SD1 — DON'T DRAIN A PLANET INTO A LOSS (the cost side of the flow-diff; the peak-then-collapse
    # in miniature). A frontier planet that survives an in-flight enemy wave ONLY if it keeps its
    # ships -- so undrained it is safe and builds NO protect cell (this isolates the drain COST from
    # the protect mechanic) -- plus a tempting neutral reachable only by draining the frontier below
    # survival. With the drain cost OFF the offense cell that drains the frontier keeps full value
    # and the drain fires; with it ON that cell's value is cut by the frontier's projected loss, so
    # we do NOT gut the frontier to grab the neutral. (SIMULATE_VALUE stays on for both arms.)
    proto.SIMULATE_VALUE = True
    frontier_sd = [0, 0, 30.0, 50.0, radius(3), 30, 3]    # survives the wave iff it keeps its ships
    neutral_sd = [1, -1, 30.0, 74.0, radius(2), 25, 2]    # defended -> capturing it drains ~27 ships off the frontier
    enemy_sd = [2, 1, 85.0, 15.0, radius(3), 40, 1]       # distant -- not a standing counter to the frontier
    wave_sd = [0, 1, 18.0, 50.0, 0.0, 2, 20]              # 20 ships in-flight at the frontier (survivable if held, fatal if drained)
    board_sd = [frontier_sd, neutral_sd, enemy_sd]
    proto.SIMVALUE_DRAIN_COST = False
    show("SD1-off drain the frontier to grab the neutral (no drain cost)",
         make_obs(board_sd, fleets=[wave_sd]), "without the cost, draining the frontier looks free")
    drain_off_sd = any(lc["src"] == 0 and lc["tgt"] == 1 for lc in proto.get_trace()[-1]["launches"])
    proto.SIMVALUE_DRAIN_COST = True
    show("SD1 hold the frontier (the drain cost outweighs the marginal capture)",
         make_obs(board_sd, fleets=[wave_sd]), "pricing the lost frontier, we do not drain it open")
    drain_on_sd = any(lc["src"] == 0 and lc["tgt"] == 1 for lc in proto.get_trace()[-1]["launches"])
    _check("SD1", drain_off_sd and not drain_on_sd,
           f"off drained the frontier={drain_off_sd}; drain-cost held it={not drain_on_sd}")

    # SA1 — DON'T DRAIN INTO A LOSS AGAINST A STANDING ENEMY (the safe_drain discipline, emergent).
    # The CONTRAST with SD1: here the threat is a strong enemy PLANET sitting near the frontier with
    # NO fleet in the air. The in-flight-only drain cost (SD1's mechanism) sees nothing -- the drained
    # source merely grows in the do-nothing baseline -> the gut looks free and fires. The ANTICIPATORY
    # drain cost injects the source's standing counter (combined_counter, the PROTECT quantity) into
    # the re-roll, so the drained frontier falls and the gut is charged its loss -> we hold instead.
    # (SIMULATE_VALUE + SIMVALUE_DRAIN_COST stay on for both arms; only SIMVALUE_DRAIN_ANTICIPATORY
    # toggles.)
    proto.SIMULATE_VALUE = True
    proto.SIMVALUE_DRAIN_COST = True
    frontier_sa = [0, 0, 30.0, 50.0, radius(3), 30, 3]    # survives the standing counter iff it keeps its ships
    neutral_sa = [1, -1, 30.0, 74.0, radius(2), 25, 4]    # tempting (prod 4); capturing it drains ~27 off the frontier
    enemy_sa = [2, 1, 30.0, 20.0, radius(3), 30, 1]       # STRONG and CLOSE -- a standing counter, but NO fleet in the air
    board_sa = [frontier_sa, neutral_sa, enemy_sa]
    proto.SIMVALUE_DRAIN_ANTICIPATORY = False
    show("SA1-off drain the frontier to grab the neutral (in-flight-only drain cost)",
         make_obs(board_sa), "the standing enemy has not launched -> the gut looks free")
    drain_off_sa = any(lc["src"] == 0 and lc["tgt"] == 1 for lc in proto.get_trace()[-1]["launches"])
    proto.SIMVALUE_DRAIN_ANTICIPATORY = True
    show("SA1 hold the frontier (the anticipated standing counter prices the gut)",
         make_obs(board_sa), "reserving against the enemy it can SEE, we do not drain the frontier open")
    drain_on_sa = any(lc["src"] == 0 and lc["tgt"] == 1 for lc in proto.get_trace()[-1]["launches"])
    _check("SA1", drain_off_sa and not drain_on_sa,
           f"off drained the frontier={drain_off_sa}; anticipatory held it={not drain_on_sa}")
    proto.SIMVALUE_DRAIN_ANTICIPATORY = False

    # SD1 (in-flight drain) and SV1 must stay green with the anticipatory flag ON (it generalizes,
    # not replaces, the in-flight cost).
    proto.SIMVALUE_DRAIN_ANTICIPATORY = True
    show("SD1+anticipatory re-check (in-flight drain still held)",
         make_obs(board_sd, fleets=[wave_sd]), "the in-flight drain cost still fires with anticipatory on")
    drain_on_sd_ant = any(lc["src"] == 0 and lc["tgt"] == 1 for lc in proto.get_trace()[-1]["launches"])
    _check("SD1+anticipatory", not drain_on_sd_ant,
           f"in-flight drain still held with anticipatory on={not drain_on_sd_ant}")
    proto.SIMVALUE_DRAIN_ANTICIPATORY = False

    # FW1 — DON'T BUY A PYRRHIC CAPTURE (the root cause of the over-launch collapse, in
    # miniature). A defended neutral whose garrison costs more ships to kill than its production
    # repays within the window. The sim evaluator prices only the ownership swing (always
    # positive, attrition charged nowhere) -> it fires. The flowdiff evaluator charges the dead
    # ships against the terminal wealth -> the capture is NET NEGATIVE -> holding (score 0) wins.
    proto.SIMULATE_VALUE = True
    proto.SIMVALUE_DRAIN_COST = True
    home_fw = [0, 0, 30.0, 50.0, radius(3), 30, 3]
    pyrrhic_fw = [1, -1, 30.0, 74.0, radius(1), 26, 1]   # 26 defenders for a prod-1 stream: never repays in-window
    board_fw = [home_fw, pyrrhic_fw]
    proto.FLOWDIFF_VALUE = False
    show("FW1-off buy the pyrrhic neutral (sim evaluator: swing only, attrition free)",
         make_obs(board_fw), "ownership swing is positive, the 26 dead ships are priced nowhere")
    fired_off_fw = any(lc["src"] == 0 and lc["tgt"] == 1 for lc in proto.get_trace()[-1]["launches"])
    proto.FLOWDIFF_VALUE = True
    show("FW1 hold (flowdiff: the dead ships outweigh the in-window stream)",
         make_obs(board_fw), "terminal wealth of the capture is negative -> do nothing scores higher")
    fired_on_fw = any(lc["src"] == 0 and lc["tgt"] == 1 for lc in proto.get_trace()[-1]["launches"])
    _check("FW1", fired_off_fw and not fired_on_fw,
           f"sim bought the pyrrhic neutral={fired_off_fw}; flowdiff held={not fired_on_fw}")

    # FW2 — A PROFITABLE CAPTURE STILL FIRES (the timidity guard): a cheap rich neutral repays
    # its tiny garrison many times over within the window -> flowdiff must still take it.
    rich_fw = [1, -1, 30.0, 74.0, radius(5), 5, 5]
    show("FW2 take the rich neutral (flowdiff: stream repays the spend in-window)",
         make_obs([home_fw, rich_fw]), "5 defenders for a prod-5 stream is strongly net positive")
    fired_rich_fw = any(lc["src"] == 0 and lc["tgt"] == 1 for lc in proto.get_trace()[-1]["launches"])
    _check("FW2", fired_rich_fw, f"flowdiff captured the rich neutral={fired_rich_fw}")
    proto.FLOWDIFF_VALUE = False

    # SV1-SV4 must remain green with the drain cost ON (they involve no over-draining -> cost ~0).
    proto.SIMVALUE_DRAIN_COST = True
    _, field = show("SV1+cost re-check (drain cost on; big still > small)",
         make_obs([home_sv, big_sv, small_sv]), "the drain cost is ~0 here -> ordering unchanged")
    imp_big2, imp_small2 = _imp(field, 1), _imp(field, 2)
    _check("SV1+cost", imp_big2 is not None and imp_small2 is not None and imp_big2 > imp_small2,
           f"imp(big)={imp_big2} > imp(small)={imp_small2} with drain cost on")
    proto.SIMVALUE_DRAIN_COST = False

    proto.SIMULATE_VALUE = False


if __name__ == "__main__":
    main()
