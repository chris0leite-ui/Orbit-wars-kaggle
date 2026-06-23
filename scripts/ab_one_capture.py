"""One-capture-per-round (LR_ONE_CAPTURE) A/B vs Producer V2.

OFF = pure proven offense stack (LR_ONE_CAPTURE=0).
ON  = + one high-confidence capture per round (LR_ONE_CAPTURE=1); regroup uncapped.

Independence: one fresh distinct seed per game; seat rotated ACROSS seeds, never
within. OFF and ON share each seed+seat -> paired diff. Also logs mean focal
launches/active-turn so we can confirm the cap actually bites (Rule 38).

    python scripts/ab_one_capture.py [N] [P]    # N seeds (default 32), P=2 or 4
"""
from __future__ import annotations
import importlib.util, os, random, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agents" / "producer"))   # orbit_lite for V2 + focal
from kaggle_environments import make  # noqa: E402

EXT = REPO / "audit" / "external" / "agents"
LR = str(REPO / "agents" / "least_resistance" / "main.py")
V2 = str(EXT / "slawekbiel_the-producer-v2" / "main.py")
ROMAN = str(EXT / "romantamrazov_orbit-star-wars-lb-max-1224" / "main.py")
KONBU = str(EXT / "konbu17_orbit-wars-rule-base-ml-shot-validator-hybrid" / "main.py")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
P = int(sys.argv[2]) if len(sys.argv) > 2 else 2
SEEDS = random.Random(20260623).sample(range(1, 2_000_000_000), N)


def _load(path):
    spec = importlib.util.spec_from_file_location("_ab_%d" % abs(hash(path)), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, "agent")


FOCAL = _load(LR)
OPPS = {2: [V2], 4: [V2, ROMAN, KONBU]}[P]
OPP_AGENTS = [_load(p) for p in OPPS]


def _setenv(on):
    # pure proven offense base both sides; toggle only the one-capture cap
    os.environ["LR_ONE_CAPTURE"] = "1" if on else "0"


def play(seed, focal_seat, on):
    _setenv(on)
    seats = [None] * P
    seats[focal_seat] = FOCAL
    j = 0
    for i in range(P):
        if i != focal_seat:
            seats[i] = OPP_AGENTS[j % len(OPP_AGENTS)]
            j += 1
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(seats)
    rew = [s.reward for s in env.steps[-1]]
    fr = rew[focal_seat]
    others = [rew[i] for i in range(P) if i != focal_seat]
    win = 1 if (fr is not None and all(fr >= (o if o is not None else -1e9) for o in others)
                and fr > max([o for o in others if o is not None], default=-1e9)) else 0
    # focal launches per active turn
    fl = [len(st[focal_seat].action) for st in env.steps
          if st[focal_seat].action]
    lpt = (sum(fl) / len(fl)) if fl else 0.0
    return win, lpt


def main():
    tally = {"OFF": 0, "ON": 0}
    lpt = {"OFF": [], "ON": []}
    log = REPO / "audit" / ("ab_one_capture_%dp.log" % P)
    f = open(log, "w")
    def out(s):
        print(s); f.write(s + "\n"); f.flush()
    out("# one-capture A/B vs %s | %dP | n=%d | seat rotated across seeds"
        % ("V2" if P == 2 else "V2+Roman+konbu", P, N))
    t0 = time.time()
    for k, seed in enumerate(SEEDS):
        seat = k % P
        for on, key in ((False, "OFF"), (True, "ON")):
            w, l = play(seed, seat, on)
            tally[key] += w
            lpt[key].append(l)
        if (k + 1) % 4 == 0 or k + 1 == N:
            out("  [%d/%d] OFF %d/%d  ON %d/%d  | launches/turn OFF %.1f ON %.1f | %.0fs"
                % (k + 1, N, tally["OFF"], k + 1, tally["ON"], k + 1,
                   sum(lpt["OFF"]) / len(lpt["OFF"]), sum(lpt["ON"]) / len(lpt["ON"]),
                   time.time() - t0))
    out("# RESULT %dP n=%d: OFF %d/%d (%.0f%%)  ON %d/%d (%.0f%%) | "
        "launches/turn OFF %.2f ON %.2f"
        % (P, N, tally["OFF"], N, 100 * tally["OFF"] / N,
           tally["ON"], N, 100 * tally["ON"] / N,
           sum(lpt["OFF"]) / len(lpt["OFF"]), sum(lpt["ON"]) / len(lpt["ON"])))
    f.close()


if __name__ == "__main__":
    main()
