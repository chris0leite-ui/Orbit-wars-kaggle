"""
Where does mirror symmetry break in baseline self-play?

Setup. Per the "How to Play Orbit Wars" page:
  - Planets and comets are placed with 4-fold mirror symmetry around
    the centre (CENTER = (50, 50) on a 100x100 board).
  - In a 2-player game, players start on diagonally-opposite home
    planets (Q1 vs Q4), so the layout under 180-degree rotation maps
    P0's view onto P1's view exactly.
The Kaggle-provided baseline (`data/main.py`) is deterministic. So a
genuinely seat-symmetric system (map + obs + agent + simulator)
should make baseline-vs-baseline evolve as a perfect mirror forever,
and every game would be a draw at step 500.

This probe wraps the baseline so we can log its outputs per turn from
each seat, runs one game at seed=1, and walks the game tick-by-tick
to find the first turn where the two seats' actions are not
mirror-equivalent. ("Mirror-equivalent" = for every P0 move from
planet A at angle theta with n ships, P1 has a move from A's
180-degree mirror at angle theta+pi with the same n.)

At the first break, it dumps the relevant distance computations at
full IEEE-754 precision so the cause is visible.

Requires: kaggle-environments + data/main.py in the current directory.
"""

import importlib.util
import math
from kaggle_environments import make

BOARD = 100.0
SEED = 1

spec = importlib.util.spec_from_file_location("baseline", "data/main.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

p_actions = [[], []]


def wrap(pid):
    def f(obs):
        moves = baseline.agent(obs)
        p_actions[pid].append([list(m) for m in moves])
        return moves
    return f


env = make("orbit_wars", configuration={"seed": SEED}, debug=False)
env.run([wrap(0), wrap(1)])
print(f"Game ran {len(env.steps)} turns at seed={SEED}.\n")


def ang_diff(a, b):
    return abs(((a - b + math.pi) % (2 * math.pi)) - math.pi)


def actions_mirror(moves_a, moves_b, planets):
    if len(moves_a) != len(moves_b):
        return False
    pos = {p[0]: (p[2], p[3]) for p in planets}
    used = set()
    for src_a, ang_a, n_a in moves_a:
        xa, ya = pos[src_a]
        mx, my = BOARD - xa, BOARD - ya
        ok = False
        for j, (src_b, ang_b, n_b) in enumerate(moves_b):
            if j in used:
                continue
            xb, yb = pos[src_b]
            if (abs(xb - mx) < 1e-9 and abs(yb - my) < 1e-9
                    and n_b == n_a
                    and ang_diff(ang_b, ang_a + math.pi) < 1e-9):
                used.add(j)
                ok = True
                break
        if not ok:
            return False
    return True


for t in range(min(len(p_actions[0]), len(p_actions[1]))):
    planets = list(env.steps[t][0].observation.planets)
    p0_obs = list(env.steps[t][0].observation.planets)
    p1_obs = list(env.steps[t][1].observation.planets)

    if not actions_mirror(p_actions[0][t], p_actions[1][t], planets):
        print(f"Turn {t}: action mirror breaks.")
        print(f"  obs.planets identical between seats: {p0_obs == p1_obs}")
        print(f"  P0 moves: {p_actions[0][t]}")
        print(f"  P1 moves: {p_actions[1][t]}")

        # Diagnose: for each of P0's own planets, dump its distance to
        # every target at full precision, then do the same for P1's
        # mirror home, and put them side by side.
        homes_p0 = [p for p in planets if p[1] == 0]
        homes_p1 = [p for p in planets if p[1] == 1]
        pos = {p[0]: (p[2], p[3]) for p in planets}

        # Match P0 homes to their mirror P1 homes by position.
        for h0 in homes_p0:
            mx, my = BOARD - h0[2], BOARD - h0[3]
            h1 = next((p for p in homes_p1
                       if abs(p[2] - mx) < 1e-9 and abs(p[3] - my) < 1e-9),
                      None)
            if h1 is None:
                continue
            targets_p0 = [p for p in planets if p[1] != 0]
            d0 = sorted(
                (math.sqrt((h0[2] - p[2]) ** 2 + (h0[3] - p[3]) ** 2), p[0])
                for p in targets_p0
            )[:4]
            targets_p1 = [p for p in planets if p[1] != 1]
            d1 = sorted(
                (math.sqrt((h1[2] - p[2]) ** 2 + (h1[3] - p[3]) ** 2), p[0])
                for p in targets_p1
            )[:4]
            print()
            print(f"  P0 home pid={h0[0]} at ({h0[2]:.9f}, {h0[3]:.9f}):"
                  " nearest targets")
            for d, tid in d0:
                print(f"    pid={tid:3d}  dist={d!r}")
            print(f"  P1 home pid={h1[0]} (mirror) at"
                  f" ({h1[2]:.9f}, {h1[3]:.9f}): nearest targets")
            for d, tid in d1:
                print(f"    pid={tid:3d}  dist={d!r}")
            if d0[0][0] == d0[1][0]:
                print("  Note: P0's top-2 distances are bit-equal floats.")
            print("  By mirror, expected: P0's distance to pid X ==")
            print("  P1's distance to pid mirror(X) at bit level.")
            print("  If those are NOT bit-equal, floating-point computes")
            print("  the supposedly-mirror distances slightly differently,")
            print("  and the agent's strict `<` resolves the near-tie to a")
            print("  DIFFERENT planet on each seat -- breaking the mirror.")
        break
