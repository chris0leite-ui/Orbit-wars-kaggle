"""2P head-to-head: current agent vs REAL public agents, with idle-verification.

Critical fix: agents have different arities (def agent(obs) vs agent(obs, cfg)).
We let the env call each with the correct arity and read launches straight from
the game record, so a 1-arg agent is never mis-called into idling. Every game
logs BOTH players' launch counts; a game where either side launches 0 is flagged
INVALID (not a real game).

Producer V2 (and Roman "I'm Stronger") run on OUR orbit_lite (verified), so no
module conflict with least_resistance.

Focal = least_resistance (all levers OFF = shipped). Opponents incl. Producer V2.
2P, seeds x both seats, 250-step cap. Checkpointed every 2 games.

    python scripts/verify_public_2p.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agents" / "producer"))  # our orbit_lite (works for V2 too)
from kaggle_environments import make  # noqa: E402

BRANCH = "claude/dreamy-fermi-8unqi5"
STEPS = 250
SEEDS = [76670184, 1492346051, 768065184]
EXT = REPO / "audit" / "external" / "agents"
FOCAL = ("least_resistance", str(REPO / "agents" / "least_resistance" / "main.py"))
OPPONENTS = [
    ("producerV2", str(EXT / "slawekbiel_the-producer-v2" / "main.py")),
    ("konbu17",    str(EXT / "konbu17_orbit-wars-rule-base-ml-shot-validator-hybrid" / "main.py")),
    ("roman1224",  str(EXT / "romantamrazov_orbit-star-wars-lb-max-1224" / "main.py")),
    ("ykhnkf1100", str(EXT / "ykhnkf_distance-prioritized-agent-lb-max-score-1100" / "main.py")),
]
LOG = REPO / "audit" / ("public2p-verify-%s.log"
                        % datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))


def _load(path):
    spec = importlib.util.spec_from_file_location("_h_%d" % abs(hash(path)), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, "agent", None) or getattr(mod, "act")


def _current_agent_env():
    for k in ("LR_LEADER_RELATIVE_4P", "LR_VALUE_COMMIT", "LR_ANYTIME"):
        os.environ[k] = "0"
    os.environ["LR_ENEMY_BOOST"] = "1.0"
    os.environ["LR_ROLLOUT_DEPTH"] = "0"


def _arity_call(fn):
    """Wrap fn so the env can call it, preserving fn's true arity."""
    argc = fn.__code__.co_argcount if hasattr(fn, "__code__") else 2

    def w1(obs):
        return fn(obs)

    def w2(obs, cfg=None):
        return fn(obs, cfg)
    return w1 if argc == 1 else w2


def _run_game(opp_path, seed, focal_seat):
    _current_agent_env()
    focal = _load(FOCAL[1])
    opp = _load(opp_path)
    seats = [None, None]
    seats[focal_seat] = _arity_call(focal)
    seats[1 - focal_seat] = _arity_call(opp)
    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": STEPS}, debug=False)
    env.run(seats)
    rew = [s.reward for s in env.steps[-1]]
    fr = rew[focal_seat]
    valid = [r for r in rew if r is not None]
    win = fr is not None and bool(valid) and fr == max(valid)
    f_launch = sum(len(st[focal_seat].action) for st in env.steps if st[focal_seat].action)
    o_launch = sum(len(st[1 - focal_seat].action) for st in env.steps if st[1 - focal_seat].action)
    return win, f_launch, o_launch, len(env.steps), rew


def _log(s):
    with open(LOG, "a") as f:
        f.write(s + "\n")
    print(s, flush=True)


def _checkpoint(msg):
    for cmd in (["git", "add", str(LOG)],
                ["git", "commit", "-q", "-m", msg],
                ["git", "push", "-q", "origin", BRANCH]):
        subprocess.run(cmd, cwd=str(REPO), check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    _log("# 2P head-to-head | focal=%s (levers OFF) | steps<=%d | seeds=%s"
         % (FOCAL[0], STEPS, SEEDS))
    n = 0
    for oname, opath in OPPONENTS:
        wins = invalid = 0
        games = 0
        for seed in SEEDS:
            for seat in (0, 1):
                win, fl, ol, nsteps, rew = _run_game(opath, seed, seat)
                games += 1
                n += 1
                idle = (fl == 0) or (ol == 0)
                if idle:
                    invalid += 1
                else:
                    wins += int(win)
                tag = "INVALID-IDLE" if idle else ("WIN " if win else "loss")
                _log("vs %-11s seed=%-11d seat=%d %s steps=%d "
                     "focal_launch=%d opp_launch=%d rew=%s"
                     % (oname, seed, seat, tag, nsteps, fl, ol, rew))
                if n % 2 == 0:
                    _checkpoint("public2p checkpoint: %d games" % n)
        valid = games - invalid
        _log("# %s: focal %d/%d wins on VALID games (%d invalid/idle of %d)"
             % (oname, wins, valid, invalid, games))
    _checkpoint("public2p complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
