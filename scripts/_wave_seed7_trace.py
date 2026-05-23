"""Turn-by-turn trace of seed 7 — show how the game evolves alongside
wave fires, capture events, planet ownership flips, ship balance."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ["BASELINE_CONVERGENCE_WAVE"] = "1"
import agents.baseline.main as bm
bm.CONVERGENCE_WAVE_ENABLED = True

from kaggle_environments import make
import importlib.util
import math

WAVE_LOG: dict[int, list] = {}  # step -> list of (src, ships, tgt_resolved)
_orig_wave = bm.emit_convergence_wave


def _logged_wave(moves, planets, my_id, world, model):
    moves_in = list(moves)
    out = _orig_wave(moves_in, planets, my_id, world, model)
    extras = out[len(moves_in):]
    if extras:
        enemy = [p for p in planets if int(p.owner) != my_id and int(p.owner) != -1]
        decoded = []
        for src_id, angle, ships in extras:
            src = next(p for p in planets if int(p.id) == int(src_id))
            best_tgt = None; best_cos = -2.0
            for t in enemy:
                dx = float(t.x) - float(src.x); dy = float(t.y) - float(src.y)
                norm = math.hypot(dx, dy)
                if norm == 0: continue
                cos = (math.cos(angle) * dx + math.sin(angle) * dy) / norm
                if cos > best_cos:
                    best_cos = cos; best_tgt = t
            decoded.append({
                "src": int(src_id),
                "ships": int(ships),
                "tgt": int(best_tgt.id) if best_tgt else None,
            })
        WAVE_LOG.setdefault(int(world.step), []).extend(decoded)
    return out


bm.emit_convergence_wave = _logged_wave


def load(path):
    p = Path(path)
    spec = importlib.util.spec_from_file_location(f"_opp_{p.stem}_{id(p)}", path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m.agent


def planet_state_at(env, step):
    if step >= len(env.steps):
        return None
    obs = env.steps[step][0].observation
    return {int(p[0]): (int(p[1]), int(p[5])) for p in obs["planets"]}


def fleets_at(env, step):
    if step >= len(env.steps):
        return []
    obs = env.steps[step][0].observation
    return obs.get("fleets", [])


def run_trace(seed):
    WAVE_LOG.clear()
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    opp = load("submissions/baseline_full.py")
    env.run([bm.agent, opp])
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    outcome = "p0_win" if r0 > r1 else "p1_win" if r1 > r0 else "draw"
    n_steps = len(env.steps)

    # Compute ownership-flip events across all steps.
    flips: list[tuple[int, int, int, int]] = []  # (step, pid, old, new)
    prev = planet_state_at(env, 0)
    for s in range(1, n_steps):
        cur = planet_state_at(env, s)
        if cur is None or prev is None: break
        for pid, (own, ships) in cur.items():
            if pid in prev and prev[pid][0] != own:
                flips.append((s, pid, prev[pid][0], own))
        prev = cur

    # Snapshot summary every 20 steps + every wave-fire turn + every flip step.
    interesting = set(range(0, n_steps, 20))
    for s, *_ in flips: interesting.add(s)
    for s in WAVE_LOG: interesting.add(s); interesting.add(min(s + 5, n_steps - 1))
    interesting = sorted(s for s in interesting if 0 <= s < n_steps)

    print(f"\n=== seed={seed} chooser-gate-wave vs baseline_full ===")
    print(f"  outcome: {outcome}  n_steps={n_steps}  rewards P0={r0} P1={r1}")
    print(f"  total wave-fire turns: {len(WAVE_LOG)}  (launches: "
          f"{sum(len(v) for v in WAVE_LOG.values())})")
    print(f"  ownership flips: {len(flips)}")
    print()
    print(f"  step | p0_planets/ships | p1_planets/ships | flips@step | wave@step")
    print(f"  " + "-" * 78)
    for s in interesting:
        st = planet_state_at(env, s)
        if not st: continue
        p0p = sum(1 for o, sh in st.values() if o == 0)
        p0s = sum(sh for o, sh in st.values() if o == 0)
        p1p = sum(1 for o, sh in st.values() if o == 1)
        p1s = sum(sh for o, sh in st.values() if o == 1)
        flips_here = [f for f in flips if f[0] == s]
        flip_str = ""
        if flips_here:
            flip_str = "  flips: " + ", ".join(
                f"P{old}→P{new} on planet#{pid}" for (_, pid, old, new) in flips_here
            )
        wave_here = WAVE_LOG.get(s, [])
        wave_str = ""
        if wave_here:
            wave_str = "  WAVE: " + " | ".join(
                f"src{w['src']}→tgt{w['tgt']} ({w['ships']}s)" for w in wave_here
            )
        print(f"  {s:>4} | {p0p:>2}p / {p0s:>4}s        | "
              f"{p1p:>2}p / {p1s:>4}s        {flip_str}{wave_str}")


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [7]
    for s in seeds:
        run_trace(s)
