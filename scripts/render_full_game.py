"""render_full_game.py — play one full game and emit a watchable HTML replay.

Plays P0 vs P1 to natural termination (or --steps), writes the kaggle HTML
replay to --out, and prints a score-over-time trajectory (competition score =
ships on owned planets + ships in owned fleets) so you can see who is ahead
across the whole game without opening the file.

    python scripts/render_full_game.py P0.py P1.py --seed 7 --steps 500 \
        --out /tmp/game.html --p0-label producer --p1-label anti_producer
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load(path: str, tag: str):
    p = Path(path)
    name = f"_rfg_{tag}_{p.stem}"
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m.agent


def _score(obs, pid: int) -> float:
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    fleets = obs["fleets"] if isinstance(obs, dict) else obs.fleets
    return float(sum(p[5] for p in planets if p[1] == pid)
                 + sum(f[6] for f in fleets if f[1] == pid))


def _planets(obs, pid: int) -> int:
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    return sum(1 for p in planets if p[1] == pid)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("p0")
    ap.add_argument("p1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", default="/tmp/inv/game.html")
    ap.add_argument("--p0-label", default="P0")
    ap.add_argument("--p1-label", default="P1")
    a = ap.parse_args(argv)

    from kaggle_environments import make
    a0 = _load(a.p0, "p0")
    a1 = _load(a.p1, "p1")
    env = make("orbit_wars", configuration={"seed": a.seed, "episodeSteps": a.steps}, debug=False)
    env.run([a0, a1])
    n = len(env.steps)

    # score trajectory from each step's player-0 observation (full board view)
    def obs_at(t):
        st = env.steps[t][0]
        return st["observation"] if isinstance(st, dict) else st.observation

    print(f"== {a.p0_label} (P0) vs {a.p1_label} (P1)  seed={a.seed}  steps={n} ==")
    print(f"\n   {'step':>4s} {a.p0_label[:10]:>10s} {a.p1_label[:10]:>10s} "
          f"{'margin(P1-P0)':>13s} {'planetsP0':>9s} {'planetsP1':>9s}")
    for t in range(0, n, max(1, n // 20)):
        o = obs_at(t)
        s0, s1 = _score(o, 0), _score(o, 1)
        print(f"   {t:>4d} {s0:>10.0f} {s1:>10.0f} {s1 - s0:>+13.0f} "
              f"{_planets(o,0):>9d} {_planets(o,1):>9d}")
    # final
    of = obs_at(n - 1)
    s0, s1 = _score(of, 0), _score(of, 1)
    winner = a.p0_label if s0 > s1 else (a.p1_label if s1 > s0 else "DRAW")
    early = " (one player eliminated)" if n < a.steps else " (reached step cap)"
    print(f"\n   FINAL @step {n-1}: {a.p0_label}={s0:.0f}  {a.p1_label}={s1:.0f}  "
          f"-> winner: {winner}{early}")

    html = env.render(mode="html")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(html)
    print(f"\n   wrote replay -> {a.out}  ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
