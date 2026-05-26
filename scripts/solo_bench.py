"""Solo benchmark — focal agent vs no-op opponent, measure terminal ships.

PI 2026-05-26: the strategic objective is terminal ship count. The
first stress test is a no-opponent game at a fixed horizon, measuring
how many ships each agent produces. Simpler strategies are the
baseline; a value head that encodes the integral form should at least
tie production-greedy here.

CLI:
    python scripts/solo_bench.py
        --agents agents/baseline_integral agents/simple/production ...
        --seeds 32
        --steps 250
        --workers 8
        [--integral-t-end 250]   # overrides INTEGRAL_T_END in each worker

For each (agent, seed) pair, runs `agent` as P0 and `agents/simple/noop.py`
as P1 in `kaggle_environments.make("orbit_wars", ...)`, subprocess-isolated
so each agent reads its own env vars at import time.

Output: one row per agent with mean / std / 95% CI (normal-approx) of
terminal ship count and mean planet count.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOOP_PATH = REPO / "agents" / "simple" / "noop.py"
PANEL_PATH = REPO / "data" / "seed_panel_128.json"


def _load_archetype_seeds(per_archetype: int, rng_seed: int
                          ) -> list[tuple[int, str]]:
    """Return `[(seed, archetype_name), ...]` — `per_archetype` random
    picks per archetype, deterministic given `rng_seed`.

    Each archetype in `data/seed_panel_128.json` has exactly 4 seeds;
    we draw `per_archetype` of them without replacement (4 max).
    """
    with PANEL_PATH.open() as f:
        panel = json.load(f)
    by_arc: dict[str, list[int]] = defaultdict(list)
    for entry in panel["panel"]:
        by_arc[entry["archetype"]].append(int(entry["seed"]))
    rng = random.Random(rng_seed)
    out: list[tuple[int, str]] = []
    for arc in sorted(by_arc.keys()):
        pool = list(by_arc[arc])
        rng.shuffle(pool)
        for s in pool[:per_archetype]:
            out.append((s, arc))
    return out


def _resolve_agent_path(spec: str) -> tuple[str, str]:
    """Map a CLI spec to (label, absolute path).

    Tries these in order:
      - spec as absolute path
      - spec relative to REPO (e.g. `agents/baseline_integral`, `agents/simple/foo.py`)
      - spec relative to REPO + `.py` (e.g. `agents/simple/foo`)
      - `agents/` prefix (e.g. `simple/production` → `agents/simple/production`)
      - `agents/<spec>.py` (e.g. `simple/production` → `agents/simple/production.py`)
      - `agents/simple/<spec>.py` (short name, e.g. `production`)
    Directories must contain `main.py`.
    """
    candidates = [
        Path(spec),
        REPO / spec,
        REPO / f"{spec}.py",
        REPO / "agents" / spec,
        REPO / "agents" / f"{spec}.py",
        REPO / "agents" / "simple" / f"{spec}.py",
    ]
    for p in candidates:
        if p.is_dir():
            mp = p / "main.py"
            if mp.is_file():
                return (spec, str(mp.resolve()))
        elif p.is_file():
            return (spec, str(p.resolve()))
    raise SystemExit(f"cannot resolve agent: {spec}")


def _worker(args: tuple[str, str, int, str, int, dict]) -> dict:
    """Run one (agent, seed) game in a subprocess; return terminal stats."""
    label, agent_path, seed, archetype, episode_steps, extra_env = args
    code = (
        "import json, sys, time;"
        "sys.path.insert(0, %r);"
        "from kaggle_environments import make;"
        "env = make('orbit_wars', configuration={'seed': %d, 'episodeSteps': %d}, debug=False);"
        "t0 = time.perf_counter();"
        "env.run([%r, %r]);"
        "wall = time.perf_counter() - t0;"
        "final = env.steps[-1];"
        "obs = final[0]['observation'] if isinstance(final[0], dict) else final[0].observation;"
        "od = obs if isinstance(obs, dict) else dict(obs);"
        "planets = od.get('planets') or [];"
        "fleets  = od.get('fleets')  or [];"
        "p0_planet_ships = sum(float(p[5]) for p in planets if int(p[1]) == 0);"
        "p0_fleet_ships  = sum(float(f[6]) for f in fleets  if int(f[1]) == 0);"
        "p0_planets = sum(1 for p in planets if int(p[1]) == 0);"
        "step = int(od.get('step', 0));"
        "print(json.dumps({"
        "    'ships_planets': p0_planet_ships,"
        "    'ships_fleets':  p0_fleet_ships,"
        "    'n_planets': p0_planets,"
        "    'step': step,"
        "    'n_steps': len(env.steps),"
        "    'wall': wall,"
        "}))"
    ) % (str(REPO), int(seed), int(episode_steps),
         str(agent_path), str(NOOP_PATH))
    env = {**os.environ, **{k: str(v) for k, v in extra_env.items()}}
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired as e:
        return {"label": label, "seed": seed, "archetype": archetype,
                "outcome": "timeout",
                "wall_total": time.perf_counter() - t0,
                "stderr": f"timed out after {e.timeout}s"}
    out_lines = (proc.stdout or "").strip().splitlines()
    json_line = next((l for l in reversed(out_lines) if l.startswith("{")), "")
    if not json_line:
        return {"label": label, "seed": seed, "archetype": archetype,
                "outcome": "error",
                "wall_total": time.perf_counter() - t0,
                "stderr": (proc.stderr or "")[:600]}
    data = json.loads(json_line)
    return {
        "label": label,
        "seed": seed,
        "archetype": archetype,
        "outcome": "ok",
        "ships": data["ships_planets"] + data["ships_fleets"],
        "ships_planets": data["ships_planets"],
        "ships_fleets": data["ships_fleets"],
        "n_planets": data["n_planets"],
        "step": data["step"],
        "n_steps": data["n_steps"],
        "wall": data["wall"],
        "wall_total": time.perf_counter() - t0,
    }


def _summarize(rows: list[dict]) -> dict:
    """Aggregate per-agent stats."""
    ok = [r for r in rows if r.get("outcome") == "ok"]
    n = len(ok)
    if n == 0:
        return {"n": 0, "errors": len(rows)}
    ships = [r["ships"] for r in ok]
    planets = [r["n_planets"] for r in ok]
    walls = [r["wall"] for r in ok]
    mean = statistics.mean(ships)
    std = statistics.stdev(ships) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    half = 1.96 * se
    return {
        "n": n,
        "mean_ships": mean,
        "std_ships": std,
        "ci95_lo": mean - half,
        "ci95_hi": mean + half,
        "mean_planets": statistics.mean(planets),
        "wall_s": sum(walls),
        "errors": len(rows) - n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents", nargs="+", required=True,
                    help="Agent specs (paths, short names, or dirs)")
    ap.add_argument("--seeds", type=int, default=32,
                    help="Seeds 0..N-1 (ignored if --archetype-panel set)")
    ap.add_argument("--archetype-panel", action="store_true",
                    help="Draw seeds from data/seed_panel_128.json — one (or "
                         "--per-archetype) random seed per archetype.")
    ap.add_argument("--per-archetype", type=int, default=1,
                    help="Seeds per archetype when --archetype-panel set "
                         "(max 4, the panel pool per bucket).")
    ap.add_argument("--rng-seed", type=int, default=42,
                    help="RNG seed for archetype seed-pick (reproducible).")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--integral-t-end", type=int, default=None,
                    help="Override INTEGRAL_T_END in worker env (default: leave unset)")
    args = ap.parse_args()

    resolved = [_resolve_agent_path(s) for s in args.agents]
    extra_env: dict[str, str] = {}
    if args.integral_t_end is not None:
        extra_env["INTEGRAL_T_END"] = str(args.integral_t_end)

    if args.archetype_panel:
        seed_arc_pairs = _load_archetype_seeds(
            per_archetype=args.per_archetype, rng_seed=args.rng_seed)
    else:
        seed_arc_pairs = [(seed, "") for seed in range(args.seeds)]

    tasks = [
        (label, path, seed, arc, args.steps, extra_env)
        for (label, path) in resolved
        for (seed, arc) in seed_arc_pairs
    ]

    n_seeds_per_agent = len(seed_arc_pairs)
    panel_desc = (f"archetype-panel (per_archetype={args.per_archetype}, "
                  f"rng_seed={args.rng_seed})" if args.archetype_panel
                  else f"seeds 0..{args.seeds - 1}")
    print(f"[solo_bench] {len(resolved)} agent(s) × {n_seeds_per_agent} "
          f"seed(s)  [{panel_desc}]  = {len(tasks)} games, "
          f"{args.steps} steps, {args.workers} worker(s)", file=sys.stderr)
    if extra_env:
        print(f"[solo_bench] worker env: {extra_env}", file=sys.stderr)

    results_by_agent: dict[str, list[dict]] = {label: [] for label, _ in resolved}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            row = fut.result()
            results_by_agent[row["label"]].append(row)
            done += 1
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                elapsed = time.perf_counter() - t0
                print(f"[solo_bench]  {done}/{len(tasks)} games "
                      f"in {elapsed:.1f}s", file=sys.stderr)

    elapsed = time.perf_counter() - t0
    print(f"[solo_bench] DONE in {elapsed:.1f}s\n", file=sys.stderr)

    # Stable order: as specified on the CLI.
    summaries = [(label, _summarize(results_by_agent[label])) for label, _ in resolved]

    # Sort report by mean_ships descending (so the winner is top), but keep
    # error / empty rows at the bottom.
    summaries.sort(
        key=lambda kv: (-(kv[1].get("mean_ships") or -math.inf),),
    )

    header = (
        f"{'agent':<40s} {'n':>3s} {'mean_ships':>11s} {'std':>7s} "
        f"{'ci95_lo':>9s} {'ci95_hi':>9s} {'planets':>8s} {'wall_s':>8s} "
        f"{'err':>4s}"
    )
    print(header)
    print("-" * len(header))
    for label, s in summaries:
        if s["n"] == 0:
            print(f"{label:<40s} {0:>3d} {'-':>11s} {'-':>7s} "
                  f"{'-':>9s} {'-':>9s} {'-':>8s} {'-':>8s} "
                  f"{s.get('errors', 0):>4d}")
            continue
        print(f"{label:<40s} {s['n']:>3d} {s['mean_ships']:>11.1f} "
              f"{s['std_ships']:>7.1f} {s['ci95_lo']:>9.1f} {s['ci95_hi']:>9.1f} "
              f"{s['mean_planets']:>8.2f} {s['wall_s']:>8.1f} "
              f"{s.get('errors', 0):>4d}")

    if args.archetype_panel:
        # Per-archetype breakdown: rows = archetypes (in panel order),
        # columns = agents (in CLI order). Diff column shown if exactly
        # two agents (the canonical A/B case).
        arc_order: list[str] = []
        seen: set[str] = set()
        for (_, arc) in seed_arc_pairs:
            if arc not in seen:
                arc_order.append(arc); seen.add(arc)
        cli_order = [label for label, _ in resolved]
        print("\nper-archetype mean terminal ships "
              f"(n={args.per_archetype} game(s) per cell):")
        head_cols = "  ".join(f"{lab[:22]:>22s}" for lab in cli_order)
        diff_col = f"  {'diff[1-2]':>10s}" if len(cli_order) == 2 else ""
        print(f"{'archetype':<42s}  {head_cols}{diff_col}")
        print("-" * (44 + len(head_cols) + len(diff_col)))
        for arc in arc_order:
            cells: list[str] = []
            means: list[float] = []
            for lab in cli_order:
                vals = [r["ships"] for r in results_by_agent[lab]
                        if r.get("archetype") == arc and r.get("outcome") == "ok"]
                if vals:
                    m = statistics.mean(vals)
                    means.append(m)
                    cells.append(f"{m:>22.1f}")
                else:
                    means.append(float("nan"))
                    cells.append(f"{'-':>22s}")
            row = "  ".join(cells)
            if len(cli_order) == 2 and not any(math.isnan(m) for m in means):
                diff = means[0] - means[1]
                row += f"  {diff:>+10.1f}"
            print(f"{arc:<42s}  {row}")

    # JSON dump for downstream analysis. To stderr to keep stdout = table.
    print("\n[solo_bench] raw JSON:", file=sys.stderr)
    print(json.dumps([
        {"label": label, **s, "raw": results_by_agent[label]}
        for label, s in summaries
    ]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
