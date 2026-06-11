"""In-process probe for the sync mechanism (PRODUCER_PLUS_SYNC).

Runs a real 2-player game with the focal agent imported in-process (so its
memory is inspectable) against a bundled referee, and reports per-step:
holds created / executed / released, and whether executed launches arrived
on the same tick as their far partners. Rule 38 verification that the
mechanism fires in the real environment.

Usage: python scripts/sync_probe.py [--seed 7] [--steps 200] [--opp submissions/v7_0_drop_one.py]
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "agents", "producer"))
sys.path.insert(0, os.path.join(REPO, "agents", "producer_plus"))

os.environ.setdefault("PRODUCER_PLUS_MULTI_SIZE", "1")
os.environ.setdefault("PRODUCER_PLUS_OPP_PROJECTION", "1")
os.environ.setdefault("PRODUCER_PLUS_RESPONSE_VETO", "1")
os.environ.setdefault("PRODUCER_PLUS_REACTIVE_FLOOR", "0.5")
os.environ.setdefault("PRODUCER_PLUS_SYNC", "1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--opp", default="submissions/v7_0_drop_one.py")
    args = ap.parse_args()

    from kaggle_environments import make
    import main as agent_mod

    mem = agent_mod._RUNTIME.memory
    execs = []
    orig_process = agent_mod._process_sync_holds

    def process_spy(memory, **kw):
        entries, debit = orig_process(memory, **kw)
        if entries is not None:
            for i in range(int(entries.valid.shape[0])):
                execs.append((kw["current_step"], float(entries.eta[i])))
                print(f"  step {kw['current_step']:3d}: EXEC launch "
                      f"src_slot={int(entries.source_slots[i])} "
                      f"tgt_slot={int(entries.target_slots[i])} "
                      f"ships={float(entries.ships[i]):.0f} eta={float(entries.eta[i]):.2f}")
        return entries, debit

    agent_mod._process_sync_holds = process_spy
    holds_created = 0
    holds_seen_steps = 0
    max_concurrent = 0
    prev_holds: list = []

    crashes = []

    def focal(obs):
        nonlocal holds_created, holds_seen_steps, max_concurrent, prev_holds
        # The env swallows agent exceptions and keeps playing (silent
        # forfeit-by-passivity) — surface them explicitly.
        try:
            action = agent_mod.agent(obs)
        except Exception:
            import traceback
            crashes.append(int(obs["step"]))
            sys.stderr.write(traceback.format_exc())
            raise
        holds = list(getattr(mem, "sync_holds", None) or [])
        if holds:
            holds_seen_steps += 1
            max_concurrent = max(max_concurrent, len(holds))
        prev_keys = {(h["src_id"], h["tgt_id"], h["arrival_step"]) for h in prev_holds}
        for h in holds:
            key = (h["src_id"], h["tgt_id"], h["arrival_step"])
            if key not in prev_keys:
                holds_created += 1
                print(f"  step {obs['step']:3d}: HOLD src={h['src_id']} "
                      f"tgt={h['tgt_id']} ships={h['ships']:.0f} "
                      f"arrive@{h['arrival_step']}")
        prev_holds = holds
        return action

    env = make("orbit_wars", configuration={
        "seed": args.seed, "episodeSteps": args.steps})
    env.run([focal, args.opp])
    r = [s.reward for s in env.state]
    print(f"\nseed={args.seed} steps={len(env.steps)} rewards={r}")
    print(f"holds created: {holds_created}  executed: {len(execs)}  "
          f"steps with pending holds: {holds_seen_steps}  "
          f"max concurrent: {max_concurrent}")
    if crashes:
        print(f"AGENT CRASHED at steps {crashes} — result is a forfeit, not a verdict")


if __name__ == "__main__":
    main()
