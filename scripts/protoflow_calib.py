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
    moves, field = show("S9a one counter (single strong enemy in recapture range)",
         make_obs([home9, neutral9, enemyA]),
         "the neutral is sized to hold against ONE free counter")
    one_l = next((lc for lc in proto.get_trace()[-1]["launches"] if lc["tgt"] == 1), None)
    _check("S9a", one_l is not None, f"launch={one_l} (want a holdable capture)")
    moves, field = show("S9b two counters (combined free recapture wave is larger)",
         make_obs([home9, neutral9, enemyA, enemyB]),
         "with TWO free counters in range, the SAME neutral needs strictly more ships")
    two_l = next((lc for lc in proto.get_trace()[-1]["launches"] if lc["tgt"] == 1), None)
    combined = two_l is not None and one_l is not None and two_l["ships"] > one_l["ships"]
    _check("S9b", combined,
           f"two-counter ships={two_l['ships'] if two_l else None} > "
           f"one-counter ships={one_l['ships'] if one_l else None}")

    # S10 — WAVE boundary + anti-dispersion RESERVATION. A defended target that NO
    # single planet can fund, reachable by a FAR source and a NEAR source; plus a
    # cheap solo neutral the near source could grab alone. This turn the FAR source
    # must fire at the wave target (it is the binding leg that must leave now to land
    # on the shared turn) and the NEAR source must be RESERVED -- emit nothing,
    # waiting for the wave instead of defecting to the easy solo.
    far_src = [0, 0, 20.0, 20.0, radius(3), 20, 3]            # far from the target
    near_src = [1, 0, 55.0, 32.0, radius(3), 20, 3]           # near target AND near the solo (off the far->tgt line)
    wave_tgt = [2, -1, 70.0, 20.0, radius(4), 30, 4]          # defended; needs both sources
    solo = [3, -1, 60.0, 42.0, radius(2), 5, 2]              # tempting easy solo for near_src
    moves, field = show("S10 wave boundary + reservation (far fires, near waits)",
         make_obs([far_src, near_src, wave_tgt, solo]),
         "far source fires at the wave target; near source reserved (no solo defect)")
    l10 = proto.get_trace()[-1]["launches"]
    far_fires = any(lc["src"] == 0 and lc["tgt"] == 2 and lc["kind"] == "wave" for lc in l10)
    near_silent = not any(lc["src"] == 1 for lc in l10)
    _check("S10", far_fires and near_silent,
           f"launches={l10} (want far0 -> wave_tgt2, near1 silent/reserved)")

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


if __name__ == "__main__":
    main()
