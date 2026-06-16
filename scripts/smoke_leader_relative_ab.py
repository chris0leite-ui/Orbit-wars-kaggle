"""Smoke A/B — least_resistance 4P objective: gap-to-field vs gap-to-strongest.

OFF = current (score = my ships - SUM of all opponents).
ON  = LR_LEADER_RELATIVE_4P (score = my ships - the STRONGEST opponent).

4 games per variant (focal rotates through every seat), capped at 250 steps,
focal = least_resistance, background = three copies of a fixed baseline, same
seed for both variants. Per-turn budgets forced huge so each variant is a
deterministic function of the board (difference = objective, not timing).

DIRECTIONAL ONLY (n=4). This is a triage go/no-go (Rule 45 exception: no lift
claim), NOT a submit gate.

    python scripts/smoke_leader_relative_ab.py [seed]
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Determinism: never bail the greedy loop or the 2-ply loop on wallclock, so
# both variants are pure functions of obs. Set BEFORE importing any agent.
os.environ["ORBIT_WARS_PARITY_WALLCLOCK_MS"] = "100000000"
os.environ["LR_TWOPLY_MS"] = "100000000"

from kaggle_environments import make  # noqa: E402

# 4 distinct real-game seeds; focal plays each seat exactly once (seed i -> seat i).
SEEDS = [76670184, 1492346051, 768065184, 641308308]
if len(sys.argv) > 1:
    SEEDS = ([int(x) for x in sys.argv[1:]] + SEEDS)[:4]
STEPS = 250
LR = str(REPO / "agents" / "least_resistance" / "main.py")
_BG_ALIASES = {
    "v7_0": str(REPO / "submissions" / "v7_0_drop_one.py"),
    "producer": str(REPO / "agents" / "producer" / "main.py"),
}
_bg_sel = os.environ.get("SMOKE_BG", "v7_0")
BG = _BG_ALIASES.get(_bg_sel, _bg_sel)
BG_NAME = _bg_sel if _bg_sel in _BG_ALIASES else Path(_bg_sel).stem
PLAYERS = int(os.environ.get("SMOKE_PLAYERS", "4"))   # 2 or 4


def _load_agent(path: str):
    spec = importlib.util.spec_from_file_location(
        f"_ab_{Path(path).stem}_{id(object())}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _run_game(seed: int, focal_seat: int, players: int):
    focal = _load_agent(LR)
    bgs = [_load_agent(BG) for _ in range(players - 1)]
    turn_ms: list[float] = []

    def timed(obs, cfg=None, _f=focal, _s=turn_ms):
        t0 = time.perf_counter()
        try:
            return _f(obs, cfg)
        finally:
            _s.append((time.perf_counter() - t0) * 1000.0)

    seats = [None] * players
    seats[focal_seat] = timed
    j = 0
    for i in range(players):
        if i == focal_seat:
            continue
        seats[i] = bgs[j]
        j += 1

    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": STEPS},
               debug=False)
    env.run(seats)

    final = env.steps[-1]
    rewards = [s.reward for s in final]
    fr = rewards[focal_seat]
    valid = [r for r in rewards if r is not None]
    win = fr is not None and valid and fr == max(valid)

    launches = sum(len(st[focal_seat].action) for st in env.steps
                   if st[focal_seat].action)
    return win, turn_ms, len(env.steps), launches, rewards


def _run_batch(on: bool):
    # ON = the fundamental stack: win-equity objective (4P-gated, inert in 2P)
    # + value-ordered commitment (both modes). The enemy-boost/anytime hacks
    # stay OFF -- value-commit is their principled replacement.
    os.environ["LR_LEADER_RELATIVE_4P"] = "1" if on else "0"
    os.environ["LR_VALUE_COMMIT"] = "1" if on else "0"
    os.environ["LR_ENEMY_BOOST"] = "1.0"
    os.environ["LR_ANYTIME"] = "0"
    wins = 0
    games = 0
    all_ms: list[float] = []
    total_launches = 0
    lines = []
    for seed in SEEDS:
        for seat in range(PLAYERS):
            win, ms, nsteps, launches, rewards = _run_game(seed, seat, PLAYERS)
            wins += int(win)
            games += 1
            all_ms += ms
            total_launches += launches
            mx = max(ms) if ms else 0.0
            lines.append(
                f"    seed={seed:<11} seat={seat}  {'WIN ' if win else 'loss'}  "
                f"steps={nsteps}  focal_launches={launches:>4}  "
                f"max_turn_ms={mx:5.0f}  rewards={rewards}")
    return wins, games, total_launches, (max(all_ms) if all_ms else 0.0), lines


def main() -> int:
    n = PLAYERS * len(SEEDS)
    print(f"{PLAYERS}P  seeds={SEEDS}  each seat each seed  steps<= {STEPS}  "
          f"focal=least_resistance  bg={BG_NAME} x{PLAYERS - 1}   "
          f"(n={n} per variant, directional)\n")
    summary = []
    for on, label in ((False, "OFF  current (all levers off)"),
                      (True,  "ON   leader-relative + value-ordered commit")):
        wins, games, launches, mx, lines = _run_batch(on)
        print(f"== {label} ==")
        print("\n".join(lines))
        print(f"  -> first-place {wins}/{games}   total focal launches "
              f"{launches}   max turn ms {mx:.0f}\n")
        summary.append((label, wins, games, launches, mx))
    print("SUMMARY")
    for label, wins, games, launches, mx in summary:
        print(f"  {label:<44}  first={wins}/{games}  launches={launches:>4}  "
              f"max_ms={mx:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
