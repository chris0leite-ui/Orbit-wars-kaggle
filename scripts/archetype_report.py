"""Per-archetype strategy-conformance report for an agent.

For each of the 32 panel archetypes, runs the agent (self-play, P0 view)
on one representative seed, extracts behavioural metrics, and tallies
how many ``lib.archetype_strategy.EXPECTED_BEHAVIOR`` rules it violates.
Prints a sorted table — high-divergence rows are the next tuning
targets.

Run:
  python scripts/archetype_report.py <agent>             # all 32, ~3 min on 8 workers
  python scripts/archetype_report.py <agent> --seeds 2   # 2 seeds per archetype (slower)
  python scripts/archetype_report.py <agent> --workers 1 # serial (for debugging)

`<agent>` resolves via the same logic as ``fast.py`` (file path,
agents/<name>/ dir, or builtin like ``random``).
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.archetype_strategy import (
    ARCHETYPES,
    EXPECTED_BEHAVIOR,
    KNOWN_REGRESSIONS,
    check,
)
from lib.fingerprint import FEATURE_NAMES, fingerprint
from lib.seed_panel import SEED_PANEL_BY_ARCHETYPE

PREFIX_TURNS = 100
EPISODE_STEPS = 200


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_agent(spec: str) -> str:
    """Return the path string passed to env.run().

    fast.py has its own resolver; we replicate the file/dir/builtin
    path logic minimally here to avoid pulling in fast.py at import.
    """
    if spec in {"random", "starter"}:
        return spec
    p = Path(spec)
    if p.is_file():
        return str(p.resolve())
    if p.is_dir():
        main_py = p / "main.py"
        if main_py.is_file():
            return str(main_py.resolve())
    # treat as agents/<spec>/main.py
    candidate = REPO / "agents" / spec / "main.py"
    if candidate.is_file():
        return str(candidate.resolve())
    raise FileNotFoundError(f"cannot resolve agent spec: {spec!r}")


def _run_one(args: tuple) -> tuple:
    """Worker: returns (archetype, seed, metrics, violations, elapsed_s)."""
    archetype, seed, agent_path = args
    from kaggle_environments import make

    tournament = _load_module("tournament", REPO / "scripts" / "tournament.py")
    extended = _load_module("extended_features", REPO / "scripts" / "extended_features.py")

    t0 = time.perf_counter()
    env = make(
        "orbit_wars",
        configuration={"seed": seed, "episodeSteps": EPISODE_STEPS},
        debug=False,
    )
    env.run([agent_path, agent_path])
    replay = tournament._build_replay(env, seed, "focal", "focal")
    n = min(PREFIX_TURNS, len(replay["steps"]))
    fp = fingerprint(replay, player_id=0, prefix_turns=n)
    ext = extended.replay_extended({"steps": env.steps}, 0)
    metrics: dict[str, float] = {name: float(fp[i]) for i, name in enumerate(FEATURE_NAMES)}
    for k, v in ext.items():
        metrics[k] = float(v)
    violations = check(archetype, metrics)
    return archetype, seed, metrics, violations, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", help="path / agents/<name> / 'random' / 'starter'")
    ap.add_argument("--seeds", type=int, default=1,
                    help="seeds per archetype (max = 4)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dump-metrics", type=Path, default=None,
                    help="Write a JSON file mapping archetype -> mean metrics. "
                         "Useful for calibrating EXPECTED_BEHAVIOR thresholds.")
    args = ap.parse_args()

    if args.seeds < 1 or args.seeds > 4:
        ap.error("--seeds must be in [1, 4]")
    agent_path = _resolve_agent(args.agent)
    print(f"agent: {agent_path}")
    print(f"archetypes: {len(ARCHETYPES)}, seeds/archetype: {args.seeds}, workers: {args.workers}")

    tasks = []
    for arch in ARCHETYPES:
        seeds = SEED_PANEL_BY_ARCHETYPE[arch][: args.seeds]
        for seed in seeds:
            tasks.append((arch, seed, agent_path))

    t0 = time.perf_counter()
    results: dict[str, list[tuple[int, list[str], dict]]] = {a: [] for a in ARCHETYPES}
    if args.workers <= 1:
        for task in tasks:
            arch, seed, metrics, violations, _ = _run_one(task)
            results[arch].append((seed, violations, metrics))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_one, t): t for t in tasks}
            for fut in as_completed(futs):
                arch, seed, metrics, violations, _ = fut.result()
                results[arch].append((seed, violations, metrics))
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.1f}s\n")

    # Score = mean violation count per archetype across its seeds
    rows: list[tuple[str, float, int, list[str]]] = []
    for arch in ARCHETYPES:
        per_seed = results[arch]
        n_violations = [len(v) for _, v, _ in per_seed]
        score = statistics.fmean(n_violations) if n_violations else 0.0
        unique_violations: list[str] = []
        seen: set[str] = set()
        for _, v_list, _ in per_seed:
            for v in v_list:
                metric_name = v.split("=", 1)[0]
                if metric_name not in seen:
                    seen.add(metric_name)
                    unique_violations.append(v)
        rows.append((arch, score, len(per_seed), unique_violations))

    if args.dump_metrics is not None:
        import json
        dump: dict[str, dict[str, float]] = {}
        for arch in ARCHETYPES:
            per_seed = results[arch]
            if not per_seed:
                dump[arch] = {}
                continue
            # Average each metric across seeds
            keys = set()
            for _, _, m in per_seed:
                keys.update(m.keys())
            dump[arch] = {
                k: statistics.fmean(m.get(k, 0.0) for _, _, m in per_seed)
                for k in keys
            }
        args.dump_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.dump_metrics.write_text(json.dumps(dump, indent=2, sort_keys=True))
        print(f"dumped per-archetype metrics to {args.dump_metrics}\n")

    rows.sort(key=lambda r: -r[1])  # high divergence first

    print(f"{'archetype':<55s}  {'score':>5s}  {'n':>3s}  flag")
    print("-" * 80)
    n_conforming = 0
    n_known = 0
    for arch, score, n, violations in rows:
        flag = ""
        if arch in KNOWN_REGRESSIONS:
            flag = "KNOWN-REGRESSION"
            n_known += int(score > 0)
        elif score == 0:
            flag = "OK"
            n_conforming += 1
        elif score >= 3:
            flag = "DIVERGENT"
        else:
            flag = "minor"
        print(f"{arch:<55s}  {score:>5.1f}  {n:>3d}  {flag}")
        if violations and score > 0:
            for v in violations[:3]:
                print(f"    - {v}")
            if len(violations) > 3:
                print(f"    - (+{len(violations)-3} more metric(s))")

    n_unknown_div = sum(1 for r in rows if r[1] > 0 and r[0] not in KNOWN_REGRESSIONS)
    print()
    print(f"summary: {n_conforming}/{len(rows)} conforming, "
          f"{n_known}/{len(KNOWN_REGRESSIONS)} known regressions detected, "
          f"{n_unknown_div} UNEXPECTED divergent archetypes")
    return 0 if n_unknown_div == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
