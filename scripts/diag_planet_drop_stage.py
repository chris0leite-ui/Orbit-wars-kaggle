"""Diagnostic #2: per-turn, per-planet — at which stage is each owned
planet dropping out of the launch pipeline?

Stages:
  E. enumerated:   proposer produced >= 1 candidate with src=P
  S. scored:       chooser actually ran score_candidate_v4 on >= 1 of them
  P. positive:     >= 1 scored candidate had score > 0
  F. fired:        passed used_srcs/used_tgts emit logic and is in `moves`
  X. dropped_by_budget:  safe_deadline pre-bailed before this src was scored

This tells us whether idle planets:
  (a) get no candidates from the proposer (proposer issue)
  (b) get candidates but they're dropped by wallclock (compute limit)
  (c) get candidates, all score <= 0 (value-head saying 'hold')
  (d) get a positive candidate but lose the global rank race (emit logic)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("BASELINE_JOINT_AGGR", "1")
os.environ.setdefault("BASELINE_JOINT_TOP_K", "5")
os.environ.setdefault("BASELINE_JOINT_MAX_PAIRS", "60")
os.environ.setdefault("BASELINE_REINFORCE_EMIT", "1")
os.environ.setdefault("BASELINE_REINFORCE_ANTICIPATE", "1")
os.environ.setdefault("BASELINE_NEUTRAL_BONUS", "2.0")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_EXTRA", "1.5")
os.environ.setdefault("BASELINE_NEUTRAL_EARLY_HORIZON", "50")
os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")

from kaggle_environments import make
from fast import _load_callable
import agents.baseline.chooser_trajectory as ct_mod

# Per-turn capture buffer, indexed by (turn, src_id).
PER_TURN = []  # list of dicts per chooser call
CURRENT = {}


def _reset(turn):
    global CURRENT
    CURRENT = {
        "turn": turn,
        "enumerated": set(),    # src_ids in prerank
        "scored": set(),         # src_ids that score_candidate_v4 was called on
        "positive": set(),       # src_ids with score > 0
        "fired": set(),          # src_ids in final moves
        "dropped_by_budget": set(),  # src_ids in prerank but never scored
        "n_owned": 0,
        "owned_ids": set(),
    }


# Monkeypatch choose_trajectory to capture stage info.
_orig_choose = ct_mod.choose_trajectory


def instrumented_choose(snap_base, prerank, baseline_favors, me, num_seats,
                        wallclock_ms, min_horizon, max_horizon, gamma,
                        world, model,
                        reserved_srcs=None, reserved_for_new_commits=None,
                        wave_candidates=()):
    # Enumerated set.
    for cd, src, tgt, ships, angle, eta_hint, ph, wn in prerank:
        CURRENT["enumerated"].add(int(src.id))

    # Wrap score_candidate_v4 to log scored / positive.
    _orig_v4 = ct_mod.score_candidate_v4

    def wrapped_v4(snap_base, src, tgt, ships, angle, *args, **kwargs):
        sid = int(src.id)
        CURRENT["scored"].add(sid)
        result = _orig_v4(snap_base, src, tgt, ships, angle, *args, **kwargs)
        score, status, _ = result
        if status == "scored" and score > 0.0:
            CURRENT["positive"].add(sid)
        return result

    ct_mod.score_candidate_v4 = wrapped_v4
    try:
        moves, commits = _orig_choose(
            snap_base, prerank, baseline_favors, me, num_seats, wallclock_ms,
            min_horizon, max_horizon, gamma, world, model,
            reserved_srcs=reserved_srcs,
            reserved_for_new_commits=reserved_for_new_commits,
            wave_candidates=wave_candidates,
        )
    finally:
        ct_mod.score_candidate_v4 = _orig_v4

    # Fired set.
    for m in moves:
        # m = [src_id, angle, ships]
        CURRENT["fired"].add(int(m[0]))

    # Dropped-by-budget = enumerated AND NOT scored.
    CURRENT["dropped_by_budget"] = CURRENT["enumerated"] - CURRENT["scored"]
    return moves, commits


ct_mod.choose_trajectory = instrumented_choose


def play_and_trace(seed: int, num_seats: int, focal_idx: int, opp_path: str):
    env = make(
        "orbit_wars",
        configuration={"seed": seed, "episodeSteps": 500},
        debug=False,
    )
    env.reset(num_agents=num_seats)

    from agents.baseline.main import agent as orbitfix_agent
    opp = _load_callable(opp_path)
    agents_list = [opp] * num_seats
    agents_list[focal_idx] = orbitfix_agent
    # Phase F F13: detect arity ONCE via inspect.signature, not
    # try/except-TypeError around each per-turn call.
    import inspect
    def _wants_config(fn):
        try:
            return len(inspect.signature(fn).parameters) >= 2
        except (TypeError, ValueError):
            return True
    wants_config = [_wants_config(a) for a in agents_list]

    state = env.steps[0]
    PER_TURN.clear()
    n_steps = 0
    while True:
        # Reset capture for THIS turn.
        _reset(n_steps)

        # Pre-step: record focal's owned planets.
        focal_obs = state[focal_idx]["observation"] if isinstance(state[focal_idx], dict) else state[focal_idx].observation
        focal_obs_d = focal_obs if isinstance(focal_obs, dict) else dict(focal_obs)
        planets = focal_obs_d.get("planets", []) or []
        my_owned_ids = {int(p[0]) for p in planets if int(p[1]) == focal_idx}
        CURRENT["owned_ids"] = my_owned_ids
        CURRENT["n_owned"] = len(my_owned_ids)

        # Build actions.
        per_seat_actions = []
        for s_idx in range(num_seats):
            obs_s = state[s_idx]["observation"] if isinstance(state[s_idx], dict) else state[s_idx].observation
            a = (agents_list[s_idx](obs_s, env.configuration)
                 if wants_config[s_idx]
                 else agents_list[s_idx](obs_s))
            per_seat_actions.append(a)

        # Snapshot.
        PER_TURN.append({
            "turn": CURRENT["turn"],
            "n_owned": CURRENT["n_owned"],
            "owned_ids": set(CURRENT["owned_ids"]),
            "enumerated": set(CURRENT["enumerated"]),
            "scored": set(CURRENT["scored"]),
            "positive": set(CURRENT["positive"]),
            "fired": set(CURRENT["fired"]),
            "dropped_by_budget": set(CURRENT["dropped_by_budget"]),
        })

        state = env.step(per_seat_actions)
        n_steps = state[0]["observation"]["step"] if isinstance(state[0], dict) else state[0].observation.step
        s0 = state[0]
        status0 = s0.get("status") if isinstance(s0, dict) else getattr(s0, "status", "ACTIVE")
        if status0 != "ACTIVE" or n_steps >= 500:
            break
    return PER_TURN, n_steps


def summarize(rows, label):
    print(f"\n=== {label}  ({len(rows)} turns) ===")
    # Aggregate: per turn, how many planets at each stage.
    totals = {
        "owned":            0,
        "enumerated":       0,
        "scored":           0,
        "positive":         0,
        "fired":            0,
        "dropped_budget":   0,
    }
    # Idle-planet drop attribution.
    drop_by_stage = {
        "no_enum":     0,
        "no_score":    0,
        "no_positive": 0,
        "ranked_out":  0,  # had positive but didn't fire (emit-logic dedup)
    }
    for r in rows:
        owned = r["owned_ids"]
        enum = r["enumerated"] & owned
        scored = r["scored"] & owned
        positive = r["positive"] & owned
        fired = r["fired"] & owned

        totals["owned"] += len(owned)
        totals["enumerated"] += len(enum)
        totals["scored"] += len(scored)
        totals["positive"] += len(positive)
        totals["fired"] += len(fired)
        totals["dropped_budget"] += len(r["dropped_by_budget"] & owned)

        # For each owned planet that did NOT fire, attribute the drop:
        for pid in owned - fired:
            if pid not in enum:
                drop_by_stage["no_enum"] += 1
            elif pid not in scored:
                drop_by_stage["no_score"] += 1
            elif pid not in positive:
                drop_by_stage["no_positive"] += 1
            else:
                drop_by_stage["ranked_out"] += 1

    print(f"  per-planet totals across all turns:")
    print(f"    owned:        {totals['owned']:5d}")
    print(f"    enumerated:   {totals['enumerated']:5d}  ({100.0*totals['enumerated']/max(1,totals['owned']):.1f}% of owned)")
    print(f"    scored:       {totals['scored']:5d}  ({100.0*totals['scored']/max(1,totals['owned']):.1f}% of owned)")
    print(f"    positive:     {totals['positive']:5d}  ({100.0*totals['positive']/max(1,totals['owned']):.1f}% of owned)")
    print(f"    fired:        {totals['fired']:5d}  ({100.0*totals['fired']/max(1,totals['owned']):.1f}% of owned)")
    print(f"    dropped_by_budget: {totals['dropped_budget']:5d}  ({100.0*totals['dropped_budget']/max(1,totals['owned']):.1f}% of owned)")
    idle = totals["owned"] - totals["fired"]
    print(f"\n  idle-planet drop attribution (n={idle}):")
    for stage, n in drop_by_stage.items():
        pct = 100.0 * n / max(1, idle)
        bar = "#" * int(pct / 2)
        print(f"    {stage:>14s}  {n:5d} ({pct:5.1f}%)  {bar}")


def main():
    print("[diag2] per-planet stage attribution")
    rows_2p, steps_2p = play_and_trace(
        seed=42, num_seats=2, focal_idx=0,
        opp_path="submissions/v7_0_drop_one.py",
    )
    summarize(rows_2p, f"2P vs v7_0  seed=42  steps={steps_2p}")

    rows_4p, steps_4p = play_and_trace(
        seed=1511945213, num_seats=4, focal_idx=0,
        opp_path="submissions/v7_0_drop_one.py",
    )
    summarize(rows_4p, f"4P FFA  seed=1511945213  steps={steps_4p}")


if __name__ == "__main__":
    main()
