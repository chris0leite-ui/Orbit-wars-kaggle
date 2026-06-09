"""Capture-stickiness trace for sync coalitions (Lever 1 / size-to-hold).

Unlike scripts/joint_sync_probe.py (which only counts whether coalitions
EMIT), this measures whether the planets a sync coalition captures actually
STICK: for each emitted coalition we record (emit_step, target_id), then walk
the per-step ownership timeline from env.steps to classify the target as
  - captured-and-held  (we own it HOLD_WINDOW turns after we first take it)
  - captured-then-lost (we take it, lose it within HOLD_WINDOW)  <-- the leak
  - never-captured     (bounced, or budget-skipped under size-to-hold)

Runs the focal (champion config + sync ON) as player 0 vs a fixed opponent,
once with BASELINE_JOINT_SYNC_HOLD off and once on, and prints both censuses
so the capture-then-lost fraction can be compared. The hypothesis: HOLD on
drops captured-then-lost toward zero, moving some coalitions to never-captured
(budget-skipped — "don't take what we can't hold").

Usage: python scripts/sync_hold_trace.py [n_games] [opponent_path]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Champion production config + sync ON (matches the A/B focal). HOLD is set
# per-run below, NOT here, so the two arms differ only in that one flag.
_BASE = {
    "BASELINE_JOINT_AGGR": "1", "BASELINE_JOINT_TOP_K": "5",
    "BASELINE_JOINT_MAX_PAIRS": "60", "BASELINE_REINFORCE_EMIT": "1",
    "BASELINE_REINFORCE_ANTICIPATE": "1", "BASELINE_NEUTRAL_BONUS": "2.0",
    "BASELINE_NEUTRAL_EARLY_EXTRA": "1.5", "BASELINE_NEUTRAL_EARLY_HORIZON": "50",
    "BASELINE_ORBITAL_SAFETY": "1", "BASELINE_PV_ETA": "1",
    "BASELINE_LAUNCH_RULES": "1", "BASELINE_CAPTURE_HORIZON_K": "10",
    "BASELINE_JOINT": "1", "BASELINE_JOINT_SYNC": "1",
    "BASELINE_JOINT_SYNC_SRC_K": "3",
}
for k, v in _BASE.items():
    os.environ.setdefault(k, v)

from kaggle_environments import make  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agents.baseline.chooser_trajectory as ct  # noqa: E402
import agents.baseline.main as M  # noqa: E402

HOLD_WINDOW = 8
ME = 0  # focal is player 0
_orig_choose = ct.choose_trajectory
# coalitions: list of (emit_step, target_id) recorded this game
COALITIONS: list[tuple[int, int]] = []


def _find_world(args):
    for a in args:
        if hasattr(a, "planets_by_id") and hasattr(a, "step"):
            return a
    return None


def _counting_choose(*a, **k):
    moves, commits = _orig_choose(*a, **k)
    # arg index 3 is `me` (the seat). In self-play both seats call this patched
    # fn; only record OUR (player 0) coalitions so the census matches the
    # ownership timeline (classified for ME).
    seat = int(a[3]) if len(a) > 3 else ME
    if seat == ME:
        world = _find_world(a)
        step = int(getattr(world, "step", 0)) if world is not None else 0
        for c in commits:
            if c.get("sync_joint"):
                COALITIONS.append((step, int(c["tgt_id"])))
    return moves, commits


ct.choose_trajectory = _counting_choose


def _owner_timeline(env):
    """owner_at[t][pid] = owner of planet pid at step t (global view)."""
    timeline = []
    for stepstates in env.steps:
        obs = stepstates[0]["observation"]
        timeline.append({int(p[0]): int(p[1]) for p in obs["planets"]})
    return timeline


def _classify(timeline, emit_step, tgt):
    """Walk forward from emit_step; classify the target's fate."""
    n = len(timeline)
    # first step strictly after emit where WE own the target = the capture
    cap_t = None
    for t in range(emit_step + 1, n):
        if timeline[t].get(tgt) == ME:
            cap_t = t
            break
    if cap_t is None:
        return "never_captured"
    hold_t = min(cap_t + HOLD_WINDOW, n - 1)
    return "held" if timeline[hold_t].get(tgt) == ME else "lost"


def _run(hold: bool, n_games: int, opp: str):
    if hold:
        os.environ["BASELINE_JOINT_SYNC_HOLD"] = "1"
    else:
        os.environ.pop("BASELINE_JOINT_SYNC_HOLD", None)
    tally = {"emitted": 0, "held": 0, "lost": 0, "never_captured": 0, "wins": 0}
    label = "HOLD ON " if hold else "HOLD OFF"
    for g in range(n_games):
        COALITIONS.clear()
        env = make("orbit_wars", configuration={"seed": 100 + g}, debug=False)
        env.run([M.agent, opp])
        timeline = _owner_timeline(env)
        final = timeline[-1]
        my = sum(1 for o in final.values() if o == ME)
        opp_n = sum(1 for o in final.values() if o not in (ME, -1))
        tally["wins"] += 1 if my > opp_n else 0
        seen = set()
        for emit_step, tgt in COALITIONS:
            key = (emit_step, tgt)
            if key in seen:
                continue
            seen.add(key)
            tally["emitted"] += 1
            tally[_classify(timeline, emit_step, tgt)] += 1
        print(f"  [{label}] game {g} seed={100+g} done  "
              f"running emitted={tally['emitted']} held={tally['held']} "
              f"lost={tally['lost']} never={tally['never_captured']}",
              flush=True)
    return tally


def main() -> int:
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    opp = sys.argv[2] if len(sys.argv) > 2 else "submissions/v7_0_drop_one.py"
    # "self" runs M.agent on BOTH seats (the strong counter-attacker the leak
    # was originally observed against). Note: env is process-global, so HOLD
    # applies to both seats in self-play.
    opp_agent = M.agent if opp == "self" else opp
    print(f"opponent={opp}  games={n_games}  hold_window={HOLD_WINDOW}\n")
    for hold in (False, True):
        t = _run(hold, n_games, opp_agent)
        label = "HOLD ON " if hold else "HOLD OFF"
        cap = t["held"] + t["lost"]
        lost_frac = (t["lost"] / cap) if cap else 0.0
        print(f"[{label}] coalitions_emitted={t['emitted']}  "
              f"captured={cap} (held={t['held']} lost={t['lost']})  "
              f"never_captured={t['never_captured']}  "
              f"captured-then-lost={lost_frac:.0%}  focal_wins={t['wins']}/{n_games}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
