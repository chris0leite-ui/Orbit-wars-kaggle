"""Wave instrumentation probe: run N seeds in fresh subprocesses with
wave-ON + telemetry-ON, dump per-fire records + ownership trace, then
aggregate the diagnostic signals described in the V3 plan.

Usage:
  python scripts/_wave_instrument_probe.py --seeds 5199 2083 3493 1649 \
    --out audit/wave_probe/v2
  python scripts/_wave_instrument_probe.py --seeds 5199 2083 \
    --out audit/wave_probe/v3 --leaf-validate

Diagnostics aggregated across seeds:
  - single-source-dominance: % of actual_fires with n_sources_in_prefix == 1
  - achieved-margin histogram: % of accepted captures with margin in
    (1.05, 1.15] vs > 1.15
  - source-cannibalization: % of wave-launched sources captured by opp
    within +10 steps after the fire
  - latency: p50 / p95 / max wall_ms
  - outcome per seed (focal as P0 only)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


RUNNER = textwrap.dedent("""
    import json
    import os
    import sys
    sys.path.insert(0, {repo!r})

    os.environ["BASELINE_CONVERGENCE_WAVE"] = "1"
    os.environ["BASELINE_WAVE_INSTRUMENT"] = "1"
    if {leaf_validate!r}:
        os.environ["BASELINE_WAVE_LEAF_VALIDATE"] = "1"

    import importlib.util
    from kaggle_environments import make
    import agents.baseline.main as bm
    bm.CONVERGENCE_WAVE_ENABLED = True
    bm.WAVE_INSTRUMENT = True
    bm.WAVE_LEAF_VALIDATE = {leaf_validate!r}

    opp_path = "submissions/baseline_full.py"
    spec = importlib.util.spec_from_file_location("_opp", opp_path)
    m = importlib.util.module_from_spec(spec); sys.modules["_opp"] = m
    spec.loader.exec_module(m); opp = m.agent

    env = make("orbit_wars", configuration={{"seed": {seed}}}, debug=False)
    env.run([bm.agent, opp])
    final = env.steps[-1]
    r0 = final[0].reward; r1 = final[1].reward
    outcome = "P0" if r0 > r1 else ("P1" if r1 > r0 else "DRAW")

    out_dir = {out!r}
    os.makedirs(out_dir, exist_ok=True)
    bm.dump_wave_telemetry(os.path.join(out_dir, "telemetry_{seed}.json"))
    bm.dump_ownership_trace(os.path.join(out_dir, "ownership_{seed}.json"))
    print(json.dumps({{
        "seed": {seed},
        "outcome": outcome,
        "n_steps": len(env.steps),
        "n_telemetry_records": len(bm._WAVE_TELEMETRY),
        "n_ownership_records": len(bm._OWNERSHIP_TRACE),
    }}), flush=True)
""")


def run_one(seed: int, out_dir: str, leaf_validate: bool) -> dict | None:
    code = RUNNER.format(
        repo=str(REPO), seed=int(seed), out=out_dir,
        leaf_validate=leaf_validate,
    )
    res = subprocess.run(
        [sys.executable, "-u", "-c", code],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(f"  [seed {seed}] FAILED rc={res.returncode}", flush=True)
        print(res.stderr[-1200:], flush=True)
        return None
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rec = json.loads(line)
                return rec
            except json.JSONDecodeError:
                continue
    return None


def cannibalized(ownership: list[dict], step: int, src_id: int,
                 me: int, k: int = 10) -> bool:
    """Did opp capture this source planet within +k steps?"""
    after = [r for r in ownership if step < r["step"] <= step + k]
    for snap in after:
        own = snap["owners"]
        cur = own.get(str(src_id))
        if cur is None:
            cur = own.get(src_id)
        if cur is not None and int(cur) != int(me):
            return True
    return False


def aggregate(out_dir: str, results: list[dict]) -> None:
    fires: list[dict] = []
    cannibal_count = 0
    cannibal_total_sources = 0
    n_singletons = 0
    margins_low: list[float] = []
    margins_high: list[float] = []
    walls: list[float] = []
    gate_deltas: list[float] = []
    gate_rejected = 0
    gate_accepted = 0

    for r in results:
        if r is None:
            continue
        seed = r["seed"]
        tel_path = Path(out_dir) / f"telemetry_{seed}.json"
        own_path = Path(out_dir) / f"ownership_{seed}.json"
        if not tel_path.exists():
            continue
        tel = json.loads(tel_path.read_text())
        own = json.loads(own_path.read_text()) if own_path.exists() else []
        for rec in tel:
            walls.append(rec.get("wall_ms") or 0.0)
            if rec.get("gate_delta") is not None:
                gate_deltas.append(float(rec["gate_delta"]))
            if rec.get("gate_rejected_reason"):
                gate_rejected += 1
            if not rec.get("actual_fire"):
                continue
            fires.append(rec)
            gate_accepted += 1 if rec.get("gate_delta") is not None else 0
            n_prefix = int(rec.get("n_sources_in_prefix", 0))
            if n_prefix == 1:
                n_singletons += 1
            tgt = rec.get("chosen_tgt_id")
            sub = next((s for s in rec.get("per_target", [])
                        if s.get("tgt_id") == tgt), None)
            if sub and "achieved_margin" in sub:
                m = float(sub["achieved_margin"])
                if 1.05 < m <= 1.15:
                    margins_low.append(m)
                elif m > 1.15:
                    margins_high.append(m)
            # Cannibalization scan using logged prefix source IDs.
            step = int(rec.get("step", 0))
            me = int(rec.get("me", 0))
            for src_id in rec.get("prefix_src_ids", []):
                cannibal_total_sources += 1
                if cannibalized(own, step, int(src_id), me, k=10):
                    cannibal_count += 1

    n_fires = len(fires)
    print("=" * 72)
    print(f"  aggregate over {len(results)} seeds, {n_fires} fires")
    print("=" * 72)
    print(f"  single-source dominance: "
          f"{n_singletons}/{n_fires} = "
          f"{100*n_singletons/max(1, n_fires):.1f}%")
    print(f"  achieved-margin (1.05, 1.15]: {len(margins_low)} fires"
          f"  (>1.15: {len(margins_high)})")
    if margins_low or margins_high:
        all_m = margins_low + margins_high
        print(f"  margin p50={statistics.median(all_m):.3f}"
              f"  max={max(all_m):.3f}")
    if cannibal_total_sources > 0:
        print(f"  source-cannibalization (within +10 steps): "
              f"{cannibal_count}/{cannibal_total_sources} = "
              f"{100*cannibal_count/cannibal_total_sources:.1f}%")
    if gate_deltas:
        gd_sorted = sorted(gate_deltas)
        print(f"  leaf-Δ gate: n={len(gate_deltas)}  "
              f"rejected={gate_rejected}  "
              f"fired={gate_accepted}  "
              f"Δ p50={gd_sorted[len(gd_sorted)//2]:.4f}  "
              f"min={min(gate_deltas):.4f}  max={max(gate_deltas):.4f}")
    if walls:
        walls_sorted = sorted(walls)
        p50 = walls_sorted[len(walls_sorted) // 2]
        p95 = walls_sorted[int(0.95 * len(walls_sorted))]
        print(f"  wave wall_ms: p50={p50:.2f}  p95={p95:.2f}  "
              f"max={max(walls):.2f}  n={len(walls)}")
    for r in results:
        if r is None:
            continue
        print(f"  seed {r['seed']}: outcome={r['outcome']:>4}  "
              f"steps={r['n_steps']:>3}  "
              f"telemetry={r['n_telemetry_records']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--leaf-validate", action="store_true")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    results: list[dict | None] = []
    for s in args.seeds:
        print(f"  [seed {s}] running...", flush=True)
        rec = run_one(int(s), args.out, args.leaf_validate)
        results.append(rec)
        if rec:
            print(f"  [seed {s}] {rec['outcome']}  "
                  f"steps={rec['n_steps']}  "
                  f"telemetry={rec['n_telemetry_records']}",
                  flush=True)
    aggregate(args.out, results)


if __name__ == "__main__":
    main()
