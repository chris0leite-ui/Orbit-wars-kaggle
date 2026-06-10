"""Expansion probe: does the planner see-and-reject neutral captures early?

Instruments the focal producer_plus planner (mass config by default) in a
single 2P game and logs, for every focal turn, what the greedy selector was
offered vs what it picked, split by target ownership class:

  - per step: my planet count, my banked ships, valid candidate count and
    best score per class (neutral / enemy / own), roi threshold, and the
    class + size of every fired wave.

Motivation (audit/2026-06-10-top-ladder-behavior.md): top-ladder agents hold
8 planets by step 40 vs our 6, funding expansion from the early stockpile.
Hypothesis: the flow scorer truncates a captured planet's production at the
horizon (H=18), so neutral captures whose payback period exceeds ~H steps
score below the roi threshold and are never fired — a modeling defect
(horizon truncation), not a tuning defect.

Usage:
  python scripts/expansion_probe.py --seed 7 [--steps 80]
      [--opp submissions/_ns_multi_opp_def.py] [--variant mass]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.bundle_producer_plus import ENV_VARIANTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=80, help="log focal turns up to this step")
    ap.add_argument("--opp", default="submissions/_ns_multi_opp_def.py")
    ap.add_argument("--variant", default="mass", help="ENV_VARIANTS key for the focal config")
    ap.add_argument("--out", default=None, help="JSONL output path (default: stdout summary only)")
    args = ap.parse_args()

    os.environ.update(ENV_VARIANTS[args.variant])

    for p in (str(ROOT / "agents" / "producer"), str(ROOT / "agents" / "producer_plus")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch  # noqa: F401
    import agents.producer_plus.main as ppm

    records: list[dict] = []
    ctx: dict = {}

    real_plan = ppm.plan_lite_waves
    real_greedy = ppm._greedy_select

    def spy_plan(**kw):
        # Stash per-call context for the greedy spy. Mirror (opponent
        # projection) calls also land here; the greedy spy filters by seat.
        ctx["obs"] = kw["obs"]
        ctx["obs_tensors"] = kw["obs_tensors"]
        ctx["garrison_status"] = kw["garrison_status"]
        return real_plan(**kw)

    def spy_greedy(**kw):
        out = real_greedy(**kw)
        obs = ctx.get("obs")
        if obs is None or int(obs.player_id) != 0:
            return out  # mirror call from the opponent seat, or no context
        step = int(ctx["obs_tensors"]["step"].max().item())
        if step > args.steps:
            return out
        gs_owner = ctx["garrison_status"].owner
        owner0 = gs_owner[..., 0] if gs_owner.dim() > 1 else gs_owner  # [P]
        pid = int(obs.player_id)
        score = kw["score"]
        tgt = kw["cand_tgt_slot"].clamp(0, int(owner0.shape[0]) - 1)
        tgt_owner = owner0[tgt]
        alive = obs.alive

        def _cls_mask(name):
            if name == "own":
                return tgt_owner == pid
            if name == "enemy":
                return (tgt_owner != pid) & (tgt_owner >= 0)
            return tgt_owner < 0  # neutral

        finite = torch.isfinite(score)
        row = {
            "step": step,
            "my_planets": int((obs.owned & alive).sum().item()),
            "my_banked": float(obs.ships[obs.owned & alive].sum().item()),
            "roi_threshold": float(kw["roi_threshold"]),
        }
        for name in ("neutral", "enemy", "own"):
            m = _cls_mask(name) & finite
            row[f"{name}_n"] = int(m.sum().item())
            row[f"{name}_best"] = float(score[m].max().item()) if bool(m.any()) else None
        entries, _leftover = out
        fired = []
        if int(entries.ships.shape[0]) > 0:
            e_tgt = entries.target_slots.clamp(0, int(owner0.shape[0]) - 1)
            e_owner = owner0[e_tgt]
            for i in range(int(entries.ships.shape[0])):
                if not bool(entries.valid[i]):
                    continue
                o = int(e_owner[i].item())
                cls = "own" if o == pid else ("enemy" if o >= 0 else "neutral")
                fired.append({"cls": cls, "ships": float(entries.ships[i].item())})
        row["fired"] = fired
        records.append(row)
        return out

    import torch
    ppm.plan_lite_waves = spy_plan
    # plan_lite_waves resolves _greedy_select via the module global:
    ppm._greedy_select = spy_greedy
    # run_turn resolves plan_lite_waves via the module global too (both the
    # top-level call and the plan_fn= handed to the mirror):
    focal = ppm.agent

    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.run([focal, str(ROOT / args.opp)])
    rewards = [s.reward for s in env.steps[-1]]
    n_steps = len(env.steps) - 1

    # ---- summary ----
    by_step = {r["step"]: r for r in records}
    print(f"seed={args.seed} opp={args.opp} variant={args.variant} "
          f"game_steps={n_steps} focal_reward={rewards[0]}")
    print(f"{'step':>4} {'plnt':>4} {'bank':>6} {'roiT':>7} "
          f"{'nN':>3} {'bestN':>8} {'nE':>3} {'bestE':>8} {'nO':>3} {'bestO':>8}  fired")
    for step in sorted(by_step):
        r = by_step[step]
        f = ",".join(f"{x['cls'][0]}{x['ships']:.0f}" for x in r["fired"]) or "-"
        def _fmt(v):
            return f"{v:8.2f}" if v is not None else "       -"
        print(f"{r['step']:>4} {r['my_planets']:>4} {r['my_banked']:>6.0f} "
              f"{r['roi_threshold']:>7.2f} "
              f"{r['neutral_n']:>3} {_fmt(r['neutral_best'])} "
              f"{r['enemy_n']:>3} {_fmt(r['enemy_best'])} "
              f"{r['own_n']:>3} {_fmt(r['own_best'])}  {f}")
    if args.out:
        with open(args.out, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {len(records)} rows -> {args.out}")


if __name__ == "__main__":
    main()
