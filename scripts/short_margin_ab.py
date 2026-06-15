"""short_margin_ab.py — fast A/B over SHORT (step-capped) games by score margin.

Why this exists
---------------
The inverse-producer line is iterated in short 200-step games (PI directive).
A 200-step game rarely terminates (the engine runs to 500 or one-player-left),
so binary win/loss from a full game is the wrong instrument and `fast.py`
(full-length, reward-only) doesn't apply. This harness:

  * caps the episode at --steps (default 200) via the env's episodeSteps,
  * scores each truncated game by the COMPETITION metric directly from the
    step-N observation: total ships on owned planets + ships in owned fleets
    (data/comp-context.md `scoring.final_score`), and
  * reports BOTH the binary outcome (margin sign, == the env's truncation
    reward) AND the continuous margin, which is a far finer iteration signal
    at small n than 4 coin-flips.

Seat bias is real in this game, so every seed is played at BOTH seats
(focal as P0 and as P1); a "seed" therefore contributes 2 games.

Usage
-----
    python scripts/short_margin_ab.py FOCAL OPP [--steps 200] [--seeds 0 1 2 3]
                                      [--workers 8]

FOCAL / OPP are paths to bundled `.py` agents (def agent(obs, ...)).
"""
from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_agent(path: str):
    """Load `agent` from a .py file under a collision-safe module name.

    NB: never reuse a stdlib module name (e.g. 'stat') as the module name —
    kaggle_environments eagerly imports sibling envs and a shadowed stdlib
    module breaks their import. We namespace with a fixed prefix + id().
    """
    p = Path(path)
    mod_name = f"_smab_{p.stem}_{abs(hash(str(p.resolve())))}"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    if not hasattr(m, "agent"):
        raise AttributeError(f"{path} has no top-level `agent`")
    return m.agent


def _score(obs, pid: int) -> float:
    """Competition score for player `pid` from a single observation:
    ships on owned planets + ships in owned fleets."""
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    fleets = obs["fleets"] if isinstance(obs, dict) else obs.fleets
    s = sum(p[5] for p in planets if p[1] == pid)
    s += sum(f[6] for f in fleets if f[1] == pid)
    return float(s)


@dataclass
class GameMargin:
    seed: int
    focal_is_p0: bool
    n_steps: int
    focal_score: float
    opp_score: float
    terminated_early: bool   # True if the game ended before the step cap
    max_turn_ms: float

    @property
    def margin(self) -> float:
        return self.focal_score - self.opp_score

    @property
    def focal_won(self) -> bool:
        return self.margin > 0


def _timed(agent, sink):
    argc = agent.__code__.co_argcount if hasattr(agent, "__code__") else 2

    def wrapped(observation, configuration):
        t0 = time.perf_counter()
        try:
            return agent(*(observation, configuration)[:argc])
        finally:
            sink.append((time.perf_counter() - t0) * 1000.0)
    return wrapped


def play_short(args) -> GameMargin:
    seed, focal_path, opp_path, focal_is_p0, steps = args
    from kaggle_environments import make

    focal = _load_agent(focal_path)
    opp = _load_agent(opp_path)
    fms: list[float] = []
    focal_t = _timed(focal, fms)
    if focal_is_p0:
        p0, p1, focal_pid = focal_t, opp, 0
    else:
        p0, p1, focal_pid = opp, focal_t, 1
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": int(steps)}, debug=False)
    env.run([p0, p1])
    last = env.steps[-1]
    st0 = last[0]
    obs0 = st0["observation"] if isinstance(st0, dict) else st0.observation
    n_steps = len(env.steps)
    return GameMargin(
        seed=seed,
        focal_is_p0=focal_is_p0,
        n_steps=n_steps,
        focal_score=_score(obs0, focal_pid),
        opp_score=_score(obs0, 1 - focal_pid),
        terminated_early=(n_steps < int(steps)),
        max_turn_ms=max(fms) if fms else 0.0,
    )


def run(focal_path: str, opp_path: str, seeds, steps: int, workers: int):
    tasks = []
    for s in seeds:
        tasks.append((s, focal_path, opp_path, True, steps))    # focal P0
        tasks.append((s, focal_path, opp_path, False, steps))   # focal P1
    results: list[GameMargin] = []
    t0 = time.perf_counter()
    if workers <= 1:
        for t in tasks:
            results.append(play_short(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(play_short, t) for t in tasks]
            for f in as_completed(futs):
                results.append(f.result())
    elapsed = time.perf_counter() - t0
    results.sort(key=lambda r: (r.seed, not r.focal_is_p0))
    return results, elapsed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("focal")
    ap.add_argument("opp")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--label", default=None, help="optional label for the focal agent")
    a = ap.parse_args(argv)

    focal_label = a.label or Path(a.focal).stem
    opp_label = Path(a.opp).stem
    print(f"== short-margin A/B  focal={focal_label}  vs  opp={opp_label} ==")
    print(f"   {len(a.seeds)} seeds x 2 seats = {2*len(a.seeds)} games, "
          f"{a.steps}-step cap, {a.workers} workers")
    results, elapsed = run(a.focal, a.opp, a.seeds, a.steps, a.workers)

    print(f"\n   {'seed':>10s} {'seat':>5s} {'focal':>7s} {'opp':>7s} "
          f"{'margin':>7s} {'won':>4s} {'steps':>6s} {'maxms':>6s}")
    for r in results:
        seat = "P0" if r.focal_is_p0 else "P1"
        early = "*" if r.terminated_early else " "
        print(f"   {r.seed:>10d} {seat:>5s} {r.focal_score:>7.0f} {r.opp_score:>7.0f} "
              f"{r.margin:>+7.0f} {('Y' if r.focal_won else '.'):>4s} "
              f"{r.n_steps:>5d}{early} {r.max_turn_ms:>6.0f}")

    wins = sum(1 for r in results if r.focal_won)
    n = len(results)
    margins = [r.margin for r in results]
    mean_margin = statistics.mean(margins)
    med_margin = statistics.median(margins)
    early = sum(1 for r in results if r.terminated_early)
    pmax = max((r.max_turn_ms for r in results), default=0.0)
    print(f"\n   focal wins {wins}/{n}  ({100*wins/n:.0f}%)")
    print(f"   margin  mean={mean_margin:+.1f}  median={med_margin:+.1f}  "
          f"min={min(margins):+.0f}  max={max(margins):+.0f}")
    print(f"   {early}/{n} games terminated before the {a.steps}-step cap;  "
          f"focal max turn {pmax:.0f} ms;  elapsed {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
