"""Decision-level diff: same seed, same opponent, variant A vs variant B —
show exactly which decisions changed and whether each change was good.

Outcome harnesses (margin_ab, clean_ab) say WHETHER a mechanism helps;
this shows each changed decision with its eventual fate, so a mechanism's
claim ("stops attacks into anticipated parries") is verified at the level
it operates on, from 2 games instead of 8+.

Per game and per variant:
  - every focal launch, classified (neutral/enemy/own) and annotated with
    its fate: flipped? stuck 20 steps? or annihilated (wasted)
  - decision-quality aggregates: wasted capture-sized attacks, ships
    thrown into failed attacks, sub-20-ship parcels, captures that stuck
Then a step-aligned diff of the launch streams around the first divergence.

Usage:
  python scripts/decision_diff.py submissions/_ns_multi_opp_def.py \
      submissions/producer_plus_veto2p_ffa_on.py \
      --opp agents/producer/main.py --seed 7 [--max-steps 200]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.mine_decision_rules import mine_replay  # noqa: E402

_GAME = r"""
import json, sys
sys.path.insert(0, {repo!r})
from kaggle_environments import make
cfg = {{'seed': {seed}}}
if {max_steps} > 0:
    cfg['episodeSteps'] = {max_steps}
env = make('orbit_wars', configuration=cfg, debug=False)
env.run([{focal!r}, {opp!r}])
out = {{'steps': [[{{'observation': a['observation'], 'action': a['action'],
                     'reward': a['reward'], 'status': a['status'], 'info': {{}}}}
                   for a in s] for s in env.steps],
        'rewards': [a['reward'] for a in env.steps[-1]],
        'info': {{'TeamNames': ['FOCAL', 'OPP']}}}}
json.dump(out, open({out!r}, 'w'))
print('ok')
"""


def play(focal: str, opp: str, seed: int, max_steps: int, out: str):
    code = _GAME.format(repo=str(REPO), seed=seed, focal=str(Path(focal).resolve()),
                        opp=str(Path(opp).resolve()), max_steps=max_steps, out=out)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=900, env={**os.environ})
    if "ok" not in (r.stdout or ""):
        raise RuntimeError(f"game failed: {(r.stderr or '')[-300:]}")


def launches_by_step(replay: dict, seat: int):
    """step -> list of (target_class, ships) for newly-born focal fleets,
    annotated with fate via mine_replay's attack records."""
    mined = mine_replay_from_dict(replay, seat)
    by_step = {}
    for a in mined["attacks"]:
        fate = "STUCK" if a["stuck"] else ("flip" if a["flipped"] else "WASTED")
        by_step.setdefault(a["step"], []).append(
            f"{a['tgt_class'][0]}{a['ships']:.0f}->g{a['garrison_launch']:.0f}:{fate}")
    return mined, by_step


def mine_replay_from_dict(replay: dict, seat: int):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(replay, f)
        path = f.name
    try:
        return mine_replay(path, seat)
    finally:
        os.unlink(path)


def quality(mined: dict):
    atk = mined["attacks"]
    sized = [a for a in atk if a["ships"] >= 1.2 * max(1.0, a["garrison_launch"])]
    failed = [a for a in sized if not a["flipped"]]
    stuck = [a for a in atk if a["stuck"]]
    return {
        "launches": len(atk),
        "capture_sized": len(sized),
        "wasted (sized, no flip)": len(failed),
        "ships into failed attacks": sum(a["ships"] for a in failed),
        "captures stuck 20+": len(stuck),
        "parcels < 20 ships": sum(1 for a in atk if a["ships"] < 20),
        "median fleet": (sorted(a["ships"] for a in atk)[len(atk) // 2] if atk else 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_a")
    ap.add_argument("variant_b")
    ap.add_argument("--opp", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--context", type=int, default=12,
                    help="steps of launch-diff shown from first divergence")
    args = ap.parse_args()

    replays = {}
    for tag, focal in (("A", args.variant_a), ("B", args.variant_b)):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out = f.name
        play(focal, args.opp, args.seed, args.max_steps, out)
        replays[tag] = json.load(open(out))
        os.unlink(out)

    print(f"seed={args.seed} opp={Path(args.opp).name}")
    mined, steps_tbl, names = {}, {}, {}
    for tag, path in (("A", args.variant_a), ("B", args.variant_b)):
        names[tag] = Path(path).name
        m, bs = launches_by_step(replays[tag], 0)
        mined[tag], steps_tbl[tag] = m, bs
        rew = replays[tag]["rewards"]
        print(f"\n[{tag}] {names[tag]}  -> {'WIN' if rew[0] > rew[1] else 'LOSS' if rew[0] < rew[1] else 'draw'}"
              f" in {m['n_steps']} steps (truncated run)")
        for k, v in quality(m).items():
            print(f"      {k}: {v:.0f}" if isinstance(v, float) else f"      {k}: {v}")

    # First divergence in the action streams.
    sa, sb = replays["A"]["steps"], replays["B"]["steps"]
    div = None
    for k in range(min(len(sa), len(sb))):
        if json.dumps(sa[k][0]["action"], sort_keys=True) != json.dumps(sb[k][0]["action"], sort_keys=True):
            div = k
            break
    if div is None:
        print("\nNo divergence — identical games.")
        return
    print(f"\nFirst divergence at step {div}. Launch diff (fate annotations:"
          f" n/e/o = neutral/enemy/own target, ->gN = garrison at launch):")
    print(f"{'step':>5}  {'A: ' + names['A'][:34]:38s} B: {names['B'][:34]}")
    shown = 0
    for k in range(div, min(len(sa), len(sb))):
        la = steps_tbl["A"].get(k, [])
        lb = steps_tbl["B"].get(k, [])
        if not la and not lb:
            continue
        if la == lb:
            continue
        print(f"{k:>5}  {', '.join(la) or '-':38s} {', '.join(lb) or '-'}")
        shown += 1
        if shown >= args.context:
            break


if __name__ == "__main__":
    main()
