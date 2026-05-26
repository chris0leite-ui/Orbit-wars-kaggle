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
from lib.fast_sim import rollout as fs_rollout  # noqa: F401  (re-export)
from lib.fast_sim import ship_totals  # noqa: F401  (re-export)
from lib.sa_core import (  # noqa: E402
    perturb,
    score_plan_from_snap,
    simulated_anneal_online as _sa_online,
    _get_step as _sa_get_step,  # noqa: F401  (kept for callers that imported it)
)


def _load_agent(path):
    spec = spec_from_file_location("a", str(path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def _get_step(obs):
    if isinstance(obs, dict):
        return int(obs.get("step", 0))
    return int(getattr(obs, "step", 0))


def record_initial_plan(seed: int, steps: int, agent_path: Path,
                        opp_path: Path | None = None):
    """Run focal-vs-opp via env.run, log every focal emission.

    `opp_path` defaults to agents/simple/noop.py (solo bench). For online
    MPC we pass `agents/simple/roi.py` so the recorded plan is one that
    arose against the actual opponent model we'll be optimising against.
    Backward-compatible: existing callers leave opp_path=None.

    Returns (emissions_list, env_terminal_ships, n_steps, initial_planets).
    """
    agent_fn = _load_agent(agent_path)
    if opp_path is None:
        opp_path = REPO / "agents" / "simple" / "noop.py"
    opp_fn = _load_agent(opp_path)
    emissions: list[tuple[int, list]] = []

    def recorder(obs):
        t = _get_step(obs)
        acts = agent_fn(obs)
        for a in acts:
            emissions.append((t, [int(a[0]), float(a[1]), int(a[2])]))
        return acts

    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": steps},
               debug=False)
    env.reset(num_agents=2)
    # Capture initial planets BEFORE env.run so we have them for SA's
    # add-emission perturbation (we need to know which planet IDs / positions exist).
    obs0 = env.steps[0][0]["observation"] if isinstance(env.steps[0][0], dict) else env.steps[0][0].observation
    od0 = obs0 if isinstance(obs0, dict) else dict(obs0)
    initial_planets = [list(p) for p in (od0.get("planets") or [])]
    env.run([recorder, opp_fn])
    final = env.steps[-1]
    obs_f = final[0]["observation"] if isinstance(final[0], dict) else final[0].observation
    odf = obs_f if isinstance(obs_f, dict) else dict(obs_f)
    planets = odf.get("planets") or []
    fleets = odf.get("fleets") or []
    p0_ships = sum(float(p[5]) for p in planets if int(p[1]) == 0) + \
               sum(float(f[6]) for f in fleets if int(f[1]) == 0)
    return emissions, p0_ships, len(env.steps), initial_planets


def _build_solo_snap0(seed: int, steps: int):
    """Build a turn-0 snapshot for the solo (vs noop) game on `seed`."""
    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": steps},
               debug=False)
    env.reset(num_agents=2)
    obs0 = env.steps[0][0]["observation"] if isinstance(env.steps[0][0], dict) else env.steps[0][0].observation
    return fs_from_obs(obs0, env.configuration,
                       episode_seed=seed, num_seats=2)


def score_plan(emissions, seed: int, steps: int) -> float:
    """Solo-mode score wrapper: build a turn-0 snap, delegate to sa_core.

    Backward-compatible signature. Used by callers that don't yet hold
    a snap (e.g. the panel runner doing one-shot scoring per seed).
    """
    snap = _build_solo_snap0(seed, steps)
    return score_plan_from_snap(emissions, snap,
                                 opp_policy=None, max_steps=steps)


def simulated_anneal(initial_plan, seed, steps, n_iterations,
                     t0, cooling, rng, initial_planets=None):
    """Solo SA wrapper. Builds snap0 ONCE, delegates to sa_core.

    The old implementation called `score_plan` (which built a new env +
    snap each iteration). Switching to the snap-based path of sa_core
    keeps the math identical (fs_rollout(in_place=False) clones, so each
    iter starts from the same state) while removing the per-iter env
    construction overhead.
    """
    snap0 = _build_solo_snap0(seed, steps)
    return _sa_online(initial_plan, snap0, max_steps=steps,
                      opp_policy=None, n_iter=n_iterations,
                      t0=t0, cooling=cooling, rng=rng,
                      start_step=0, initial_planets=initial_planets)


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
    plan, env_score, env_n_steps, initial_planets = record_initial_plan(
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
        args.t0, args.cooling, rng,
        initial_planets=initial_planets)
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
