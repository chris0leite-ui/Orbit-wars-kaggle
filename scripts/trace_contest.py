"""Rule-47 trace for the state-driven horizon-K lever (contest-aware design A).

Plays ONE game (focal config taken from the environment — set
BASELINE_STATE_DRIVEN_K=1 plus the champion config before running) and then,
for every focal turn, reconstructs the World + WorldModel and reports the
per-target horizon-K distribution:

  - how often a target is uncontested (opp_contest_tick is None -> K = ceil),
  - how often K is clamped to the floor (contested soon),
  - the median / quartiles of K and the share of targets with K > floor (the
    "freed long grabs" the lever exists to admit).

Confirms the K driver (`opp_contest_tick` = `time_to_enemy_threat`) is sane on
real games (not None-everywhere, not absurdly small) before trusting the lever.

Usage (champion config + lever ON):
  export BASELINE_LAUNCH_RULES=1 BASELINE_CAPTURE_HORIZON_K=10 \
         BASELINE_STATE_DRIVEN_K=1 ...(rest of champion config)...
  python scripts/trace_contest.py --vs submissions/baseline_champion_nokt.py --seed 42
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.baseline.launch_rules import (  # noqa: E402
    _capture_horizon_floor,
    capture_horizon_k,
    state_driven_k_enabled,
)
from lib.intent import World  # noqa: E402
from lib.world_model import WorldModel, opp_contest_tick  # noqa: E402


def _obs_of(step_entry, seat):
    """Extract a plain obs dict for `seat` from a kaggle_environments step."""
    o = step_entry[seat].observation
    return dict(o) if not isinstance(o, dict) else o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", default="agents/baseline")
    ap.add_argument("--vs", default="submissions/baseline_champion_nokt.py")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not state_driven_k_enabled():
        print("WARNING: BASELINE_STATE_DRIVEN_K is OFF — every K will be the "
              "floor. Set it to 1 to trace the lever.", file=sys.stderr)

    floor = _capture_horizon_floor()
    from fast import _load_callable, resolve_agent_spec  # reuse the agent loader

    from kaggle_environments import make
    p0 = _load_callable(resolve_agent_spec(args.focal)[1])
    p1 = _load_callable(resolve_agent_spec(args.vs)[1])
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.run([p0, p1])

    me = 0  # focal is P0
    all_k: list[int] = []
    n_uncontested = 0
    n_floor = 0
    n_targets = 0
    n_freed = 0          # K > floor
    contest_ticks: list[int] = []
    per_step_freed: list[tuple[int, int, int]] = []  # (step, n_targets, n_freed)

    for t, step_entry in enumerate(env.steps):
        obs = _obs_of(step_entry, me)
        if not obs.get("planets"):
            continue
        world = World.from_obs(obs)
        model = WorldModel.from_world(world)
        step_freed = 0
        step_tgts = 0
        for pid, p in world.planets_by_id.items():
            if int(p.owner) == me:
                continue
            n_targets += 1
            step_tgts += 1
            tick = opp_contest_tick(model, world, int(pid), me)
            k = capture_horizon_k(getattr(world, "step", None), tgt_id=int(pid),
                                  world=world, model=model, me=me)
            all_k.append(int(k))
            if tick is None:
                n_uncontested += 1
            else:
                contest_ticks.append(int(tick))
            if int(k) <= floor:
                n_floor += 1
            else:
                n_freed += 1
                step_freed += 1
        if step_tgts:
            per_step_freed.append((t, step_tgts, step_freed))

    print(f"=== state-driven-K trace  focal=P{me}  seed={args.seed}  "
          f"steps={len(env.steps)}  floor={floor} ===")
    if not all_k:
        print("no target evaluations — empty trace")
        return 1
    qs = statistics.quantiles(all_k, n=4) if len(all_k) > 1 else [all_k[0]] * 3
    print(f"target-evaluations: {n_targets}")
    print(f"  K distribution   min={min(all_k)} q1={qs[0]:.0f} "
          f"median={statistics.median(all_k):.0f} q3={qs[2]:.0f} max={max(all_k)}")
    print(f"  uncontested (K=ceil): {n_uncontested} "
          f"({100*n_uncontested/n_targets:.1f}%)")
    print(f"  clamped to floor    : {n_floor} ({100*n_floor/n_targets:.1f}%)")
    print(f"  freed (K>floor)     : {n_freed} ({100*n_freed/n_targets:.1f}%)")
    if contest_ticks:
        print(f"  contested targets' opp_contest_tick  "
              f"min={min(contest_ticks)} median="
              f"{statistics.median(contest_ticks):.0f} max={max(contest_ticks)}")
    # Sanity flags
    if n_uncontested == n_targets:
        print("  FLAG: every target uncontested — time_to_enemy_threat may be "
              "returning None everywhere (suspect).")
    if contest_ticks and statistics.median(contest_ticks) <= 1:
        print("  FLAG: contest ticks ~0 — opp threat implausibly immediate.")
    # Opening vs midgame freed share (the lever's intended early-expansion lift)
    opening = [f for (s, _tg, f) in per_step_freed if s <= 30]
    if opening:
        print(f"  opening (step<=30) mean freed targets/turn: "
              f"{statistics.mean(opening):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
