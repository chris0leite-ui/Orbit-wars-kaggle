"""Single-state probe: reconstruct one Kaggle replay step's obs, run the
baseline agent under three env-var configurations, compare returned move
lists.

Goal: verify whether the B.3 head composite (`BASELINE_VH_LAMBDA=1.0`)
re-enables launches that the bare distilled-Tier-2 opp model
(`BASELINE_VH_LAMBDA=0`) is suppressing in equilibrium-stuck states.

Each configuration is run in a fresh subprocess because `_value_head.py`
reads `BASELINE_VH_LAMBDA` at module-load time (line 44).

Usage:
    python scripts/probe_dead_zone_state.py \\
        audit/2026-05-31-distill-postmortem/episode-78324838-replay.json \\
        --step 120 --seat 0

Writes a JSON summary to `audit/2026-05-31-distill-postmortem/probe-results.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CONFIGS = [
    ("bare_dist", {
        "BASELINE_OPP_TIER": "2",
        "BASELINE_VH_LAMBDA": "0.0",
        "BASELINE_OPP_FILTER_THRESHOLD": "0.15",
        "BASELINE_PV_ETA": "1",
        "KINEMATIC_TABLE_ENABLED": "0",
    }),
    ("composite_l1", {
        "BASELINE_OPP_TIER": "2",
        "BASELINE_VH_LAMBDA": "1.0",
        "BASELINE_OPP_FILTER_THRESHOLD": "0.15",
        "BASELINE_PV_ETA": "1",
        "KINEMATIC_TABLE_ENABLED": "0",
    }),
    ("composite_l2", {
        "BASELINE_OPP_TIER": "2",
        "BASELINE_VH_LAMBDA": "2.0",
        "BASELINE_OPP_FILTER_THRESHOLD": "0.15",
        "BASELINE_PV_ETA": "1",
        "KINEMATIC_TABLE_ENABLED": "0",
    }),
]


WORKER = r"""
import os, sys, pickle, time, json

sys.path.insert(0, %(repo)r)
from agents.baseline.main import agent

with open(%(state_path)r, 'rb') as f:
    state = pickle.load(f)
obs = state['obs']
configuration = state['configuration']

t0 = time.perf_counter()
moves = agent(obs, configuration)
dt_ms = (time.perf_counter() - t0) * 1000.0

print(json.dumps({
    'moves': [list(m) for m in (moves or [])],
    'n_moves': len(moves or []),
    'wallclock_ms': dt_ms,
}))
"""


def extract_state(replay_path: Path, step: int, seat: int) -> dict:
    with open(replay_path) as f:
        ep = json.load(f)
    steps = ep['steps']
    if step >= len(steps):
        raise ValueError(f"step {step} out of range (n_steps={len(steps)})")
    s = steps[step]
    if seat >= len(s):
        raise ValueError(f"seat {seat} out of range (n_seats={len(s)})")
    obs = s[seat]['observation']
    # Force seat ID into obs (Kaggle replay format puts it in obs.player
    # already, but some replays omit it — make explicit).
    obs = dict(obs)
    obs['player'] = seat
    configuration = ep.get('configuration', {}) or {}
    return {'obs': obs, 'configuration': configuration}


def run_config(name: str, env_overrides: dict, state_path: Path) -> dict:
    env = dict(os.environ)
    env.update(env_overrides)
    src = WORKER % {'repo': str(REPO), 'state_path': str(state_path)}
    proc = subprocess.run(
        [sys.executable, '-c', src],
        env=env,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        return {'name': name, 'error': proc.stderr[-2000:], 'env': env_overrides}
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {'name': name, 'error': f'parse failed: {e}; stdout: {proc.stdout[-500:]}'}
    out['name'] = name
    out['env'] = env_overrides
    return out


def summarize(state: dict, results: list[dict], replay_path: Path,
              step: int, seat: int) -> dict:
    obs = state['obs']
    planets = obs.get('planets', [])
    fleets = obs.get('fleets', [])
    me = seat
    my_planets = [p for p in planets if int(p[1]) == me]
    opp_planets = [p for p in planets if int(p[1]) >= 0 and int(p[1]) != me]
    my_garrison = sum(float(p[6]) for p in my_planets)
    opp_garrison = sum(float(p[6]) for p in opp_planets)
    my_inflight = sum(float(f[6]) for f in fleets if int(f[1]) == me)
    opp_inflight = sum(float(f[6]) for f in fleets if int(f[1]) >= 0 and int(f[1]) != me)
    return {
        'state': {
            'replay': str(replay_path),
            'step': step,
            'seat': seat,
            'my_planets': len(my_planets),
            'opp_planets': len(opp_planets),
            'my_garrison': my_garrison,
            'opp_garrison': opp_garrison,
            'my_inflight': my_inflight,
            'opp_inflight': opp_inflight,
        },
        'results': results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('replay', type=Path)
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--seat', type=int, default=0)
    ap.add_argument('--out', type=Path,
                    default=REPO / 'audit/2026-05-31-distill-postmortem/probe-results.json')
    args = ap.parse_args()

    state = extract_state(args.replay, args.step, args.seat)
    with tempfile.NamedTemporaryFile('wb', suffix='.pkl', delete=False) as f:
        pickle.dump(state, f)
        state_path = Path(f.name)
    try:
        results = []
        for name, env_overrides in CONFIGS:
            print(f'--- running {name} ---', file=sys.stderr)
            r = run_config(name, env_overrides, state_path)
            results.append(r)
            if 'error' in r:
                print(f'  ERROR: {r["error"][:300]}', file=sys.stderr)
            else:
                print(f'  n_moves={r["n_moves"]} wallclock={r["wallclock_ms"]:.0f}ms', file=sys.stderr)
                for m in r['moves'][:5]:
                    print(f'    src={int(m[0])} angle={m[1]:.3f} ships={int(m[2])}', file=sys.stderr)
    finally:
        state_path.unlink(missing_ok=True)

    summary = summarize(state, results, args.replay, args.step, args.seat)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nState: own={summary["state"]["my_planets"]} garr={summary["state"]["my_garrison"]:.0f} '
          f'inflight={summary["state"]["my_inflight"]:.0f}  |  '
          f'opp={summary["state"]["opp_planets"]} garr={summary["state"]["opp_garrison"]:.0f} '
          f'inflight={summary["state"]["opp_inflight"]:.0f}', file=sys.stderr)
    print(f'\nWrote {args.out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
