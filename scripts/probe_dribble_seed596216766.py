"""Probe: reproduce the dribble pattern at turns 27, 31, 54, 55, 58 of
episode 77389657 (seed 596216766).

Takes our slot's observation directly from the Kaggle replay JSON, then
runs the production chooser pipeline (propose + score_candidate_v4) to
intercept the candidate list AND the per-candidate delta score the
chooser would have computed at that turn.

Hypothesis: at the dribble turns, the 2-ship candidates we actually sent
scored delta > 0 (chooser approved them), AND no positive-delta
wait-and-bundle alternative was emitted by the proposer.

Run: python scripts/probe_dribble_seed596216766.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Match the production submission env vars (see
# submissions/baseline_joint_aggr_consolidated_orbitfix.py header).
# Set BEFORE importing agent code so module-level defaults take effect.
_PROD_ENV = {
    "BASELINE_JOINT_AGGR": "1",
    "BASELINE_JOINT_TOP_K": "5",
    "BASELINE_JOINT_MAX_PAIRS": "60",
    "BASELINE_REINFORCE_EMIT": "1",
    "BASELINE_REINFORCE_ANTICIPATE": "1",
    "BASELINE_NEUTRAL_BONUS": "2.0",
    "BASELINE_NEUTRAL_EARLY_EXTRA": "1.5",
    "BASELINE_NEUTRAL_EARLY_HORIZON": "50",
    "BASELINE_ORBITAL_SAFETY": "1",
}
for k, v in _PROD_ENV.items():
    os.environ.setdefault(k, v)
# Make sure the CRN env var (from our branch) is NOT set — prod doesn't.
os.environ.pop("BASELINE_OPP_TRAJ_TIER", None)

from agents.baseline.proposer import propose, MAX_HORIZON, MIN_HORIZON
from agents.baseline.chooser_trajectory import (
    score_candidate_v4, build_trajectory_baseline,
)
from agents.baseline.value import select_favor_fn
from lib.intent import World
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
from lib.world_model import WorldModel
from lib.fast_sim import from_obs as fs_from_obs


REPLAY_PATH = "/tmp/episodes/episode-77389657-replay.json"
ME = 3  # ChrisLeiteScha
INTERESTING_TURNS = [27, 31, 54, 55, 58]


def _num_seats_from_obs(obs_d) -> int:
    """Count distinct non-neutral owners across planets+fleets."""
    owners = set()
    for p in obs_d.get("planets", []) or []:
        if p[1] != -1:
            owners.add(int(p[1]))
    for f in obs_d.get("fleets", []) or []:
        if f[1] != -1:
            owners.add(int(f[1]))
    return max(2, len(owners))


def probe_turn(replay, turn: int) -> None:
    # Kaggle replay convention: steps[i].observation is the state AT END
    # of step i (i.e., the INPUT to step i+1). The action that was taken
    # AT step i was decided from steps[i-1].observation. So to reproduce
    # the agent's decision at turn N, feed it steps[N-1].observation.
    if turn <= 0:
        return
    input_step = replay["steps"][turn - 1]
    obs_d = dict(input_step[ME]["observation"])
    obs_d["player"] = ME
    # The action stored at steps[turn][ME].action was what we took at turn.
    actual_action = replay["steps"][turn][ME].get("action") or []

    me = ME
    planets = [Planet(*p) for p in obs_d["planets"]]
    fleets = [Fleet(*f) for f in obs_d.get("fleets", [])]
    my_planets = [p for p in planets if int(p.owner) == me]
    other_planets = [p for p in planets if int(p.owner) != me]

    print(f"\n{'='*78}")
    print(f"TURN {turn}  (our slot = {ME})")
    print(f"  my planets: {len(my_planets)}  "
          f"ships: {sum(int(p.ships) for p in my_planets)}  "
          f"prod: {sum(int(p.production) for p in my_planets)}")
    print(f"  per-planet: {[(int(p.id), int(p.ships), int(p.production)) for p in my_planets]}")
    print(f"  ACTUAL ACTION at this turn: {actual_action}")

    if not my_planets or not other_planets:
        print("  (no actionable state)")
        return

    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0))
    num_seats = _num_seats_from_obs(obs_d)

    threatened_mine = [
        p for p in my_planets
        if model.time_to_enemy_threat(int(p.id), me, world) is not None
    ]
    target_pool = other_planets + threatened_mine

    prerank = propose(
        my_planets, target_pool, world, model, me, omega,
        baseline_len=MAX_HORIZON + 1,
    )

    print(f"  proposer emitted {len(prerank)} candidates")
    if not prerank:
        return

    # Rebuild snap and baseline for score_candidate_v4. We need a kaggle
    # env obs structure (Struct), not a dict — fs_from_obs takes whatever
    # obs argument was passed to the agent.
    class _Struct(dict):
        def __getattr__(self, k):
            return self[k]

    obs_struct = _Struct(obs_d)
    snap_base = fs_from_obs(obs_struct, num_seats=num_seats)

    max_h = max(int(h) for _, _, _, _, _, _, h, _ in prerank)
    favor_fn = select_favor_fn()
    gamma = 0.99

    baseline_favors = build_trajectory_baseline(
        snap_base, me, num_seats, max_h, favor_fn, gamma, opp_traj=None,
    )

    # Score every candidate. Skip joint enumeration (separate code path).
    rows = []
    for cheap, src, tgt, ships, angle, eta_hint, h, wait_N in prerank:
        delta, status, eta = score_candidate_v4(
            snap_base, src, tgt, int(ships), float(angle),
            me, num_seats, world,
            baseline_favors=baseline_favors,
            favor_fn=favor_fn, gamma=gamma,
            horizon=int(h),
            wait_N=int(wait_N),
            opp_traj=None,
        )
        rows.append({
            "src": int(src.id), "tgt": int(tgt.id),
            "ships": int(ships), "wait_N": int(wait_N),
            "cheap": float(cheap), "delta": float(delta),
            "status": status, "eta": eta,
        })

    # Sort by delta descending; print top 15.
    rows.sort(key=lambda r: -r["delta"])
    print(f"  candidates (top 15 by chooser delta):")
    print(f"    {'src':>3} {'tgt':>3} {'ships':>5} {'wait':>4} "
          f"{'cheap':>7} {'delta':>9} {'status':>12} {'eta':>4}")
    for r in rows[:15]:
        delta_s = f"{r['delta']:+.4f}" if r['delta'] != float("-inf") else "-inf"
        eta_s = f"{r['eta']}" if r['eta'] is not None else "-"
        print(f"    {r['src']:>3} {r['tgt']:>3} {r['ships']:>5} {r['wait_N']:>4} "
              f"{r['cheap']:>+7.3f} {delta_s:>9} {r['status']:>12} {eta_s:>4}")

    # Highlight the candidate(s) matching what we actually sent.
    for act in actual_action:
        if not act:
            continue
        src_id, _angle, ships = int(act[0]), float(act[1]), int(act[2])
        match = [r for r in rows if r["src"] == src_id and r["ships"] == ships
                 and r["wait_N"] == 0]
        if match:
            r = match[0]
            print(f"  >>> ACTUAL launch (src={src_id} ships={ships}) was in prerank: "
                  f"delta={r['delta']:+.4f} status={r['status']}")
        else:
            print(f"  >>> ACTUAL launch (src={src_id} ships={ships}) "
                  f"NOT FOUND in scored prerank (maybe filtered or joint)")

    # Check: any wait_N>0 candidate with positive delta?
    bundles = [r for r in rows if r["wait_N"] > 0 and r["delta"] > 0]
    if bundles:
        print(f"  positive-delta wait-bundle candidates: {len(bundles)}")
        for r in bundles[:3]:
            print(f"    src={r['src']} tgt={r['tgt']} ships={r['ships']} "
                  f"wait_N={r['wait_N']} delta={r['delta']:+.4f}")
    else:
        print(f"  NO positive-delta wait-bundle candidates emitted "
              f"(hypothesis confirmed for this turn)")


def main():
    if not os.path.exists(REPLAY_PATH):
        print(f"Replay JSON not found at {REPLAY_PATH}")
        print("Download via: kaggle competitions replay 77389657 -p /tmp/episodes")
        sys.exit(1)
    replay = json.load(open(REPLAY_PATH))
    print(f"Replay: {replay['info']['EpisodeId']}  seed={replay['info']['seed']}  "
          f"slots={replay['info']['TeamNames']}")
    print(f"Probing OUR slot ({ME} = {replay['info']['TeamNames'][ME]})")

    for turn in INTERESTING_TURNS:
        try:
            probe_turn(replay, turn)
        except Exception as e:
            import traceback
            print(f"\nERROR at turn {turn}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
