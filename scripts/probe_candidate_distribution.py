"""Dump the per-candidate leaf_delta distribution from one game state,
classified by target ownership (own/threatened-own/neutral/enemy).

Goal: show evidence for what the chooser sees. For the state at
ep 78367540 step 100 (where the user observed "start strong then stop
attacking"), this prints every candidate the proposer surfaced, its
leaf_delta score, and what target owner each candidate targets.

Reuses BASELINE_PRERANK_TRACE — the existing JSONL trace hook in
agents/baseline/_trace_hook.py:trace_prerank.

Usage:
    python scripts/probe_candidate_distribution.py \\
        audit/live-episodes/53239342/episode-78367540-replay.json \\
        --step 100 --seat 0
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


WORKER = r"""
import os, sys, pickle, json

sys.path.insert(0, %(repo)r)
os.environ['BASELINE_PRERANK_TRACE'] = %(trace_path)r
from agents.baseline.main import agent

with open(%(state_path)r, 'rb') as f:
    state = pickle.load(f)
obs = state['obs']
configuration = state['configuration']
moves = agent(obs, configuration)
print(json.dumps({'moves': [list(m) for m in (moves or [])]}))
"""


def extract_state(replay_path: Path, step: int, seat: int) -> dict:
    with open(replay_path) as f:
        ep = json.load(f)
    obs = dict(ep['steps'][step][seat]['observation'])
    obs['player'] = seat
    return {
        'obs': obs,
        'configuration': ep.get('configuration', {}) or {},
        'teams': ep['info'].get('TeamNames', []),
    }


def planet_owner_map(obs: dict, our_seat: int) -> dict[int, str]:
    """Return {planet_id: 'own'/'enemy'/'neutral'}."""
    out = {}
    for p in obs.get('planets', []):
        pid = int(p[0])
        owner = int(p[1])
        if owner == our_seat:
            out[pid] = 'own'
        elif owner < 0:
            out[pid] = 'neutral'
        else:
            out[pid] = 'enemy'
    return out


def threatened_mine(obs: dict, our_seat: int) -> set[int]:
    """Heuristic: own planets that have an enemy fleet aimed somewhere
    'near' (rough — we don't import World.from_obs here to keep this
    free of side effects)."""
    # In production, the proposer uses model.time_to_enemy_threat. Here
    # for diagnostic purposes we just mark all own planets as
    # potentially threatened — the categorisation 'own' vs
    # 'threatened-own' is approximated by checking if there's any enemy
    # fleet at all on the board.
    fleets = obs.get('fleets', [])
    enemy_inflight = any(int(f[1]) >= 0 and int(f[1]) != our_seat
                         for f in fleets)
    if not enemy_inflight:
        return set()
    return {int(p[0]) for p in obs.get('planets', [])
            if int(p[1]) == our_seat}


def run_config(env_overrides: dict, state_path: Path, trace_path: Path) -> dict:
    env = dict(os.environ)
    env.update(env_overrides)
    src = WORKER % {
        'repo': str(REPO),
        'state_path': str(state_path),
        'trace_path': str(trace_path),
    }
    proc = subprocess.run(
        [sys.executable, '-c', src],
        env=env, cwd=str(REPO),
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        return {'error': proc.stderr[-2000:]}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {'error': f'parse failed: {e}; stdout: {proc.stdout[-500:]}'}


def summarize_trace(trace_path: Path, obs: dict, our_seat: int):
    if not trace_path.exists():
        return None
    owner_map = planet_owner_map(obs, our_seat)
    threatened = threatened_mine(obs, our_seat)
    by_class: dict[str, list] = {
        'own_defense': [],   # target is own + threatened
        'own_other': [],     # target is own + not threatened (rare; proposer shouldn't make these)
        'neutral': [],
        'enemy': [],
        'unknown': [],
    }
    n_total = 0
    with open(trace_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n_total += 1
            tgt_id = rec.get('tgt_id')
            owner = owner_map.get(tgt_id, 'unknown')
            if owner == 'own':
                cls = 'own_defense' if tgt_id in threatened else 'own_other'
            else:
                cls = owner
            by_class[cls].append({
                'src': rec.get('src_id'),
                'tgt': rec.get('tgt_id'),
                'ships': rec.get('ships'),
                'wait_N': rec.get('wait_N'),
                'eta': rec.get('eta'),
                'cheap': rec.get('cheap_delta'),
                'leaf': rec.get('leaf_delta'),
            })
    return {'n_total': n_total, 'by_class': by_class}


def print_class_summary(label: str, items: list):
    if not items:
        print(f'  {label}: 0 candidates')
        return
    leafs = sorted([float(it['leaf']) for it in items], reverse=True)
    pos = [v for v in leafs if v > 0]
    print(f'  {label}: n={len(items)}  '
          f'positive={len(pos)}  '
          f'max_leaf={leafs[0]:+.3f}  median_leaf={leafs[len(leafs)//2]:+.3f}  '
          f'min_leaf={leafs[-1]:+.3f}')
    # Show the top 3 by leaf_delta
    items_sorted = sorted(items, key=lambda x: float(x['leaf']), reverse=True)
    for it in items_sorted[:3]:
        print(f'    src={it["src"]:>2} tgt={it["tgt"]:>2} ships={it["ships"]:>3} '
              f'wait={it["wait_N"]} eta={it["eta"]:>2} '
              f'leaf={float(it["leaf"]):+.3f} cheap={float(it["cheap"]):+.3f}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('replay', type=Path)
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--seat', type=int, default=0)
    ap.add_argument('--vh', type=float, default=1.0,
                    help='BASELINE_VH_LAMBDA (default 1.0 = composite)')
    args = ap.parse_args()

    state = extract_state(args.replay, args.step, args.seat)
    teams = state.pop('teams')
    print(f'Replay: {args.replay}')
    print(f'Step: {args.step}  Seat: P{args.seat}')
    print(f'Teams: {teams}')
    own_planets = [p for p in state['obs']['planets'] if int(p[1]) == args.seat]
    opp_planets = [p for p in state['obs']['planets']
                   if int(p[1]) >= 0 and int(p[1]) != args.seat]
    print(f'State: own={len(own_planets)} garr={sum(int(p[6]) for p in own_planets)} | '
          f'opp={len(opp_planets)} garr={sum(int(p[6]) for p in opp_planets)} | '
          f'fleets={len(state["obs"].get("fleets", []))}')
    print()

    with tempfile.NamedTemporaryFile('wb', suffix='.pkl', delete=False) as f:
        pickle.dump(state, f)
        state_path = Path(f.name)
    trace_path = state_path.with_suffix('.jsonl')
    try:
        env_overrides = {
            'BASELINE_OPP_TIER': '2',
            'BASELINE_VH_LAMBDA': str(args.vh),
            'BASELINE_OPP_FILTER_THRESHOLD': '0.15',
            'BASELINE_PV_ETA': '1',
            'KINEMATIC_TABLE_ENABLED': '0',
        }
        result = run_config(env_overrides, state_path, trace_path)
        if 'error' in result:
            print('ERROR:', result['error'])
            return 1
        print(f'Agent returned {len(result["moves"])} moves')
        for m in result['moves']:
            src_id = int(m[0])
            print(f'  src={src_id} angle={m[1]:.3f} ships={int(m[2])}')
        print()

        summary = summarize_trace(trace_path, state['obs'], args.seat)
        if summary is None:
            print('No trace file produced.')
            return 1
        print(f'Total scored candidates: {summary["n_total"]}')
        print()
        print_class_summary('OWN/DEFENSE  (target = our threatened planet)',
                            summary['by_class']['own_defense'])
        print_class_summary('OWN/OTHER    (target = our untreatened planet)',
                            summary['by_class']['own_other'])
        print_class_summary('NEUTRAL      (target = neutral planet, expansion)',
                            summary['by_class']['neutral'])
        print_class_summary('ENEMY        (target = enemy planet, attack)',
                            summary['by_class']['enemy'])
    finally:
        state_path.unlink(missing_ok=True)
        # keep trace file for inspection
        print(f'\nTrace: {trace_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
