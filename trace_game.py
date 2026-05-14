"""trace_game.py — full play-by-play of a single game.

Plays my agent vs v7_0 and, for every fleet launched (by either side),
traces:
  - turn launched
  - src planet, target inferred from angle/distance, ships sent
  - estimated arrival turn
  - actual outcome: did the target's owner flip to the launcher, or
    did the fleet vanish, or did the garrison absorb it
  - extra context: planet's owner ARRIVED IS the launch was decided on

Used to spot WEIRD things: fleets that miss orbital targets, fleets
killed by sun, fleets that arrive at a planet already captured by
someone else, etc.

Usage:  python trace_game.py [seed] [my_slot]
"""
from __future__ import annotations
import math, sys
from kaggle_environments import make


def _planet_owner_at(steps, turn, pid):
    """Return owner of planet pid at given turn (-1=neutral)."""
    if turn >= len(steps):
        turn = len(steps) - 1
    for p in steps[turn][0].observation.planets:
        if int(p[0]) == pid:
            return int(p[1])
    return None


def _planet_ships_at(steps, turn, pid):
    if turn >= len(steps):
        turn = len(steps) - 1
    for p in steps[turn][0].observation.planets:
        if int(p[0]) == pid:
            return int(p[5])
    return None


def _infer_target(action_at_launch_obs, src_id, angle):
    """Match angle to the nearest planet ID along that bearing."""
    src_pos = None
    for p in action_at_launch_obs["planets"]:
        if int(p[0]) == src_id:
            src_pos = (float(p[2]), float(p[3]))
            break
    if src_pos is None:
        return None
    sx, sy = src_pos
    best, best_score = None, None
    for p in action_at_launch_obs["planets"]:
        pid = int(p[0])
        if pid == src_id:
            continue
        px, py = float(p[2]), float(p[3])
        bearing = math.atan2(py - sy, px - sx)
        d_ang = abs(math.atan2(math.sin(angle - bearing), math.cos(angle - bearing)))
        if d_ang < 0.2:                                  # within ~11 degrees
            dist = math.hypot(px - sx, py - sy)
            score = d_ang * 10 + dist * 0.01            # prefer close + on-bearing
            if best_score is None or score < best_score:
                best_score, best = score, p
    if best is None:
        return None
    return int(best[0]), float(best[2]), float(best[3])


def _speed(ships):
    if ships <= 1:
        return 1.0
    return 1.0 + 5.0 * (math.log(max(ships, 2)) / math.log(1000.0)) ** 1.5


def trace(seed=1003, my_slot=0, only_my=False):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    agents = ["baselines/v7_0.py", "baselines/v7_0.py"]
    agents[my_slot] = "main.py"
    env.run(agents)
    steps = env.steps
    n = len(steps)
    print(f"seed={seed} my_slot={my_slot}  total_turns={n}  result={[s.reward for s in steps[-1]]}")
    print()

    events = []
    for t in range(n):
        for player in (my_slot, 1 - my_slot):
            if only_my and player != my_slot:
                continue
            action = steps[t][player].action
            if not action:
                continue
            obs = steps[t][player].observation  # state BEFORE this turn's resolution
            for mv in action:
                src_id, angle, ships = int(mv[0]), float(mv[1]), int(mv[2])
                tgt = _infer_target(dict(obs), src_id, angle)
                # arrival turn
                src_pos = None; src_r = 1.0
                for p in obs.planets:
                    if int(p[0]) == src_id:
                        src_pos = (float(p[2]), float(p[3]))
                        src_r = float(p[4])
                        break
                if src_pos is None or tgt is None:
                    arrival_t = None
                    tgt_id = None
                    dist = None
                else:
                    sx, sy = src_pos
                    tgt_id, tx, ty = tgt
                    dist = math.hypot(tx - sx, ty - sy)
                    spd = _speed(ships)
                    arrival_t = t + math.ceil(max(0.0, dist - src_r - 1.0) / spd)
                # owner state pre-arrival and post-arrival
                pre_owner = tgt[0] if False else None
                if tgt_id is not None:
                    tgt_owner_at_launch = _planet_owner_at(steps, t, tgt_id)
                    arrival_lookback = arrival_t if arrival_t else t
                    owner_arrival = _planet_owner_at(steps, arrival_lookback, tgt_id)
                    owner_post = _planet_owner_at(steps, arrival_lookback + 2, tgt_id)
                    tgt_ships_launch = _planet_ships_at(steps, t, tgt_id)
                    tgt_ships_arrival = _planet_ships_at(steps, arrival_lookback, tgt_id)
                else:
                    tgt_owner_at_launch = owner_arrival = owner_post = None
                    tgt_ships_launch = tgt_ships_arrival = None
                # Verdict
                if tgt_id is None:
                    verdict = "UNTARGETED?"
                elif owner_arrival == player:
                    verdict = f"CAPTURED at t={arrival_t}"
                elif owner_post == player:
                    verdict = f"CAPTURED soon after t={arrival_t}"
                elif tgt_owner_at_launch != owner_arrival:
                    verdict = f"target changed owner {tgt_owner_at_launch}→{owner_arrival} before/at arrival"
                else:
                    verdict = "MISSED or DEFEATED"
                tag = "ME " if player == my_slot else "OPP"
                events.append((t, tag, src_id, tgt_id, ships, arrival_t, dist, verdict,
                               tgt_owner_at_launch, tgt_ships_launch, tgt_ships_arrival))

    # Print all events
    print(f"{'turn':>4} {'side':>4} {'src':>3}→{'tgt':>3} {'×ships':>7} {'arr@':>5} {'dist':>5} "
          f"{'tgt_own':>8} {'tgt_s_launch':>12} {'tgt_s_arr':>10}  verdict")
    print("-" * 130)
    for ev in events:
        t, tag, src, tgt, ships, arr, dist, verdict, town, tsl, tsa = ev
        print(f"{t:>4} {tag:>4} {src:>3}→{tgt!s:>3} {ships:>7} {arr!s:>5} {dist or 0:>5.1f} "
              f"{town!s:>8} {tsl!s:>12} {tsa!s:>10}  {verdict}")

    # Summary stats
    my_events = [e for e in events if e[1] == "ME "]
    opp_events = [e for e in events if e[1] == "OPP"]
    my_caps = sum(1 for e in my_events if "CAPTURED" in e[7])
    opp_caps = sum(1 for e in opp_events if "CAPTURED" in e[7])
    print(f"\nME : {len(my_events)} launches, {my_caps} captures ({my_caps/max(1,len(my_events))*100:.0f}%)")
    print(f"OPP: {len(opp_events)} launches, {opp_caps} captures ({opp_caps/max(1,len(opp_events))*100:.0f}%)")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1003
    my_slot = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    trace(seed, my_slot)
