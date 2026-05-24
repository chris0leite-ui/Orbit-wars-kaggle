"""scripts/elim_matrix.py — focal × opponent win-rate matrix, subprocess-isolated.

Each cell runs N seeds × 2 seats. One game per Python subprocess (mirrors
clean_ab.py to dodge env-var pollution + module caching). Opponents may be
file paths OR kaggle builtin names ('random', 'starter').

Usage:
    python scripts/elim_matrix.py \
        --focals submissions/baseline_joint_aggr_consolidated_orbitfix.py \
        --opps random starter agents/simple/nearest.py agents/simple/roi.py \
               submissions/v3.5.1.py submissions/v4_planner.py \
        --seeds 4 --workers 8 --out audit/elim-sweep-2026-05-24/matrix.json
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _spec_for_env(spec: str) -> str:
    # Accept builtin names verbatim; resolve file paths to absolute.
    if spec in ("random", "starter"):
        return spec
    p = Path(spec).resolve()
    if not p.is_file():
        raise FileNotFoundError(spec)
    return str(p)


def _worker_play(args):
    seed, focal_spec, opp_spec, focal_is_p0 = args
    p0, p1 = (focal_spec, opp_spec) if focal_is_p0 else (opp_spec, focal_spec)
    code = (
        "import json, sys, time;"
        f"sys.path.insert(0, {str(REPO)!r});"
        "from kaggle_environments import make;"
        f"env = make('orbit_wars', configuration={{'seed': {seed}}}, debug=False);"
        "t0 = time.perf_counter();"
        f"env.run([{p0!r}, {p1!r}]);"
        "wall = time.perf_counter() - t0;"
        "final = env.steps[-1];"
        "r0 = final[0]['reward']; r1 = final[1]['reward'];"
        "print(json.dumps({'r0': r0, 'r1': r1, 'n_steps': len(env.steps), 'wall': wall}))"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ},
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        return {"seed": seed, "focal_is_p0": focal_is_p0, "outcome": "timeout"}
    out = (proc.stdout or "").strip().splitlines()
    line = next((l for l in reversed(out) if l.startswith("{")), "")
    if not line:
        return {"seed": seed, "focal_is_p0": focal_is_p0, "outcome": "error",
                "stderr": (proc.stderr or "")[:200]}
    try:
        data = json.loads(line)
    except Exception as e:
        return {"seed": seed, "focal_is_p0": focal_is_p0, "outcome": "error",
                "stderr": f"json: {e}"}
    r0, r1 = data["r0"], data["r1"]
    if r0 is None or r1 is None:
        outcome = "error"
    elif r0 > r1:
        outcome = "p0_win"
    elif r1 > r0:
        outcome = "p1_win"
    else:
        outcome = "draw"
    focal_won = (focal_is_p0 and outcome == "p0_win") or (not focal_is_p0 and outcome == "p1_win")
    return {"seed": seed, "focal_is_p0": focal_is_p0, "outcome": outcome,
            "focal_won": bool(focal_won), "n_steps": data["n_steps"], "wall": data["wall"]}


def wilson(wins: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--focals", nargs="+", required=True)
    ap.add_argument("--opps", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    focals = [_spec_for_env(f) for f in args.focals]
    opps = [_spec_for_env(o) for o in args.opps]

    tasks = []
    for f in focals:
        for o in opps:
            if f == o:
                continue
            for s in range(args.seeds):
                tasks.append((s, f, o, True))
                tasks.append((s, f, o, False))

    label = lambda s: s if s in ("random", "starter") else Path(s).stem
    print(f"== elim matrix  focals={len(focals)}  opps={len(opps)}  "
          f"seeds={args.seeds} (x2 seats)  total_games={len(tasks)}  workers={args.workers} ==")
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker_play, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"outcome": "error", "stderr": f"{type(e).__name__}: {e}"}
            r["focal"] = t[1]
            r["opp"] = t[2]
            results.append(r)
            done += 1
            if done % 16 == 0 or done == len(tasks):
                print(f"   progress {done}/{len(tasks)}  elapsed={time.perf_counter()-t0:.0f}s")

    # Aggregate per cell.
    cells = {}
    for r in results:
        key = (r["focal"], r["opp"])
        cells.setdefault(key, []).append(r)

    print()
    print(f"{'focal':<45} {'opponent':<32} {'wins/n':>9} {'rate':>7} {'Wlo':>6} {'Whi':>6} {'errs':>5}")
    print("-" * 115)
    rows = []
    for (f, o), rs in sorted(cells.items()):
        errs = sum(1 for r in rs if r["outcome"] in ("error", "timeout"))
        valid = [r for r in rs if r["outcome"] not in ("error", "timeout")]
        wins = sum(1 for r in valid if r.get("focal_won"))
        n = len(valid)
        if n:
            lo, hi = wilson(wins, n)
            rate = wins / n
        else:
            lo, hi, rate = 0.0, 1.0, 0.0
        rows.append({"focal": label(f), "opp": label(o), "wins": wins, "n": n,
                     "rate": rate, "wlo": lo, "whi": hi, "errs": errs})
        flag = " ✘" if (n >= 8 and rate < 0.95) else ""
        print(f"{label(f):<45} {label(o):<32} {wins:>4}/{n:<4} {100*rate:>6.1f}% "
              f"{lo:>5.2f} {hi:>5.2f} {errs:>4}{flag}")

    elapsed = time.perf_counter() - t0
    print(f"\nelapsed {elapsed:.0f}s")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"rows": rows, "elapsed": elapsed}, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
