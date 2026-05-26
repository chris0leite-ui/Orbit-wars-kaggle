"""Simulated-annealing solver for the solo-mode (no-opponent) game.

PI 2026-05-26: in solo, the game IS a deterministic scheduling problem.
Given a starting state and an action plan, terminal ship count is a
deterministic function. Run ROI to get an initial plan, then perturb
with simulated annealing to find a near-optimal plan. The gap between
ROI's score and SA's best score is the *ceiling diagnostic* — it tells
us how much room there is to improve any agent in solo.

Pipeline:
  1. Record ROI's emissions per turn via `env.run([roi, noop])`.
  2. Score the plan via `fast_sim` (byte-exact, ~50 ms per 200-step game).
  3. Parity-check that the env.run terminal ships matches the fast_sim
     replay score (otherwise the SA loop optimises a different objective).
  4. SA loop: perturb plan, re-score, accept/reject by Metropolis.
  5. Report per-seed (and per-archetype if --archetype-panel) gap %.

Perturbations (uniform random, one per iteration):
  - remove a random emission
  - modify an emission's ship count (±30%)
  - shift an emission's turn (±2)
  - perturb an emission's aim angle (±0.2 rad)

Out of scope for v1: ADDING new emissions at empty turns. The initial
ROI plan provides the emission *shape*; SA explores local neighbourhoods.
If the ROI shape is far from optimal, add-emission perturbations are a
follow-up.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make  # noqa: E402

from lib.fast_sim import from_obs as fs_from_obs  # noqa: E402
from lib.fast_sim import rollout as fs_rollout
from lib.fast_sim import ship_totals


def _load_agent(path):
    spec = spec_from_file_location("a", str(path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def _get_step(obs):
    if isinstance(obs, dict):
        return int(obs.get("step", 0))
    return int(getattr(obs, "step", 0))


def record_initial_plan(seed: int, steps: int, agent_path: Path):
    """Run focal-vs-noop via env.run, log every focal emission.

    Returns (emissions_list, env_terminal_ships, n_steps).
    emissions: list of (turn, [src, angle, ships]).
    """
    agent_fn = _load_agent(agent_path)
    emissions: list[tuple[int, list]] = []

    def recorder(obs):
        t = _get_step(obs)
        acts = agent_fn(obs)
        for a in acts:
            emissions.append((t, [int(a[0]), float(a[1]), int(a[2])]))
        return acts

    def noop(obs):
        return []

    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": steps},
               debug=False)
    env.reset(num_agents=2)
    env.run([recorder, noop])
    final = env.steps[-1]
    obs0 = final[0]["observation"] if isinstance(final[0], dict) else final[0].observation
    od = obs0 if isinstance(obs0, dict) else dict(obs0)
    planets = od.get("planets") or []
    fleets = od.get("fleets") or []
    p0_ships = sum(float(p[5]) for p in planets if int(p[1]) == 0) + \
               sum(float(f[6]) for f in fleets if int(f[1]) == 0)
    return emissions, p0_ships, len(env.steps)


def score_plan(emissions, seed: int, steps: int) -> float:
    """Replay the plan via fast_sim, return P0 terminal ships."""
    plan_by_turn: dict[int, list[list]] = {}
    for t, action in emissions:
        plan_by_turn.setdefault(int(t), []).append(list(action))

    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": steps},
               debug=False)
    env.reset(num_agents=2)
    obs0 = env.steps[0][0]["observation"] if isinstance(env.steps[0][0], dict) else env.steps[0][0].observation
    snap = fs_from_obs(obs0, env.configuration,
                       episode_seed=seed, num_seats=2)

    def replay(obs):
        t = _get_step(obs)
        return [list(a) for a in plan_by_turn.get(t, [])]

    def noop(obs):
        return []

    snap = fs_rollout(snap, K=steps, policies=[replay, noop], in_place=False)
    return ship_totals(snap).get(0, 0.0)


def perturb(plan: list[tuple[int, list]], rng: random.Random) -> list[tuple[int, list]]:
    """One uniform random local edit."""
    if not plan:
        return list(plan)
    op = rng.choice(["remove", "ships", "shift", "angle"])
    new_plan = list(plan)
    idx = rng.randrange(len(new_plan))
    if op == "remove":
        new_plan.pop(idx)
    elif op == "ships":
        t, action = new_plan[idx]
        src, ang, ships = action
        new_ships = max(1, int(ships * rng.uniform(0.7, 1.3)))
        new_plan[idx] = (t, [src, ang, new_ships])
    elif op == "shift":
        t, action = new_plan[idx]
        new_t = max(0, t + rng.choice([-2, -1, 1, 2]))
        new_plan[idx] = (new_t, action)
    elif op == "angle":
        t, action = new_plan[idx]
        src, ang, ships = action
        new_ang = ang + rng.uniform(-0.2, 0.2)
        new_plan[idx] = (t, [src, float(new_ang), ships])
    return new_plan


def simulated_anneal(initial_plan, seed, steps, n_iterations,
                     t0, cooling, rng):
    """SA loop: returns (best_plan, best_score, history)."""
    score = score_plan(initial_plan, seed, steps)
    best_plan, best_score = list(initial_plan), score
    current_plan, current_score = list(initial_plan), score
    history: list[tuple[int, float, float]] = []  # (iter, current, best)

    temp = t0
    for i in range(n_iterations):
        new_plan = perturb(current_plan, rng)
        new_score = score_plan(new_plan, seed, steps)
        delta = new_score - current_score
        if delta > 0 or rng.random() < math.exp(delta / max(1e-9, temp)):
            current_plan = new_plan
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_plan = list(current_plan)
        temp *= cooling
        if i % 50 == 0 or i == n_iterations - 1:
            history.append((i, current_score, best_score))
    return best_plan, best_score, history


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--agent", default="agents/simple/roi.py",
                    help="Initial-plan agent (default: simple/roi)")
    ap.add_argument("--t0", type=float, default=200.0,
                    help="SA initial temperature")
    ap.add_argument("--cooling", type=float, default=0.995)
    ap.add_argument("--rng-seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    log = (lambda *a, **k: None) if args.quiet else (lambda *a, **k: print(*a, file=sys.stderr, **k))
    rng = random.Random(args.rng_seed)

    log(f"=== seed={args.seed} steps={args.steps} iter={args.iterations} ===")

    t0 = time.perf_counter()
    plan, env_score, env_n_steps = record_initial_plan(
        args.seed, args.steps, REPO / args.agent)
    t_record = time.perf_counter() - t0
    log(f"recorded {len(plan)} emissions in {t_record:.1f}s "
        f"(env.run terminal ships = {env_score:.0f}, n_steps={env_n_steps})")

    t1 = time.perf_counter()
    fs_score = score_plan(plan, args.seed, args.steps)
    t_score = time.perf_counter() - t1
    log(f"fast_sim replay score = {fs_score:.0f} ({t_score*1000:.0f} ms)")

    parity_ok = abs(fs_score - env_score) < 1.0
    parity_str = "PASS" if parity_ok else f"FAIL (Δ={fs_score - env_score:+.1f})"
    log(f"parity check: {parity_str}")
    if not parity_ok:
        log("WARN: env.run and fast_sim disagree; SA will optimise the fast_sim score.")

    sa_t0 = time.perf_counter()
    best_plan, best_score, history = simulated_anneal(
        plan, args.seed, args.steps, args.iterations,
        args.t0, args.cooling, rng)
    sa_elapsed = time.perf_counter() - sa_t0

    initial_score = fs_score
    gap_abs = best_score - initial_score
    gap_pct = 100.0 * gap_abs / max(1.0, initial_score)

    print(f"\nseed={args.seed}  iter={args.iterations}  SA wall={sa_elapsed:.1f}s")
    print(f"  initial (ROI):   {initial_score:>8.0f} ships  ({len(plan)} emissions)")
    print(f"  best (SA):       {best_score:>8.0f} ships  ({len(best_plan)} emissions)")
    print(f"  gap:             {gap_abs:>+8.0f}  ({gap_pct:+.1f}%)")
    if history:
        log("\nSA history (iter, current, best):")
        for h in history[::max(1, len(history) // 10)]:
            log(f"  iter={h[0]:>4d}  current={h[1]:>7.0f}  best={h[2]:>7.0f}")

    print(json.dumps({
        "seed": args.seed,
        "steps": args.steps,
        "iterations": args.iterations,
        "agent": args.agent,
        "initial_score": initial_score,
        "best_score": best_score,
        "env_score": env_score,
        "parity_ok": parity_ok,
        "gap_abs": gap_abs,
        "gap_pct": gap_pct,
        "initial_n_emissions": len(plan),
        "best_n_emissions": len(best_plan),
        "record_s": t_record,
        "score_s": t_score,
        "sa_wall_s": sa_elapsed,
    }), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
