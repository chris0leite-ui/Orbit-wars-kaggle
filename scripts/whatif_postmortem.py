"""What-if rollout harness for baseline agent — Phase 1 of the
stateful-commit-ledger plan
(`/root/.claude/plans/so-now-research-and-zany-widget.md`).

Replays a Kaggle live-episode JSON forward from turn 0, driving the
simulation with `lib.fast_sim` and:
  - OUR seat's action: produced by `agents.baseline.main.agent` under a
    configurable chooser policy.
  - OPP seat(s) action: the opponent's RECORDED action at turn t+1.

This lets us ask "what if we'd used policy P on the same opponents and
the same seed?" The opp's recorded actions are applied verbatim even
after our simulated state diverges from the recorded state. Mismatched
actions (src no longer ours, etc.) are dropped by the engine — which
mirrors what would happen at live evaluation.

Supported policies (selected via env var `LEDGER_MODE`):
  - `baseline`   (default) — current chooser, no change.
  - `mpc`        — drop wait_N>0 candidates from the chooser's scored
                   list; pure receding-horizon emit. Implemented by
                   monkey-patching `chooser_trajectory.score_candidate_v4`
                   to return `(-inf, "skipped_wait", eta)` for wait_N>0.
  - `ledger`     — stateful commit ledger (placeholder; Phase 3 lands
                   the real implementation in `agents/baseline/main.py`).
                   For Phase 1 this falls back to `baseline`.

Output:
  audit/whatif/<sid>/<episode>/<policy>.json — per-turn detail + cumulative diff.
  audit/whatif/<sid>/<episode>/summary.json   — multi-policy comparison.

CLI:
  python -m scripts.whatif_postmortem 52827111 --episode 77140674 \
      --policy baseline mpc

Limitations:
- Opp's recorded actions are blindly replayed; once the sim diverges
  from reality, the opp's intent is preserved but the action may be
  ineffective (e.g., targeting a planet we now own).
- The sim drives fast_sim which uses the env's interpreter, so combat
  physics is bit-identical to the live env. Comet RNG is determined by
  `episode_seed` (read from replay `info.seed`).
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import agents.baseline.main as AGENT  # noqa: E402
import agents.baseline.chooser_trajectory as CHOOSER  # noqa: E402
import lib.fast_sim as FS  # noqa: E402
from scripts.live_episode_summary import detect_team_name  # noqa: E402


# Per-turn ledger telemetry — populated when LEDGER_MODE is `ledger`.
LEDGER_TRACE: list[dict] = []


# Policies we know how to run. Each maps to an (apply_hook, undo_hook)
# pair that prepares the agent module for the policy and cleans up
# after.
def _apply_baseline():
    return lambda: None  # no-op


def _apply_mpc():
    """Monkey-patch the trajectory chooser to drop wait_N>0 candidates.

    Returns a callable that undoes the patch.
    """
    orig = CHOOSER.score_candidate_v4

    def wrapped(snap_base, src, tgt, ships, angle, me, num_seats, world,
                baseline_favors, favor_fn, gamma, horizon,
                skip_admissibility=False, wait_N=0):
        if int(wait_N) != 0:
            return (float("-inf"), "skipped_wait", 0)
        return orig(snap_base, src, tgt, ships, angle, me, num_seats, world,
                    baseline_favors, favor_fn, gamma, horizon,
                    skip_admissibility=skip_admissibility, wait_N=wait_N)

    CHOOSER.score_candidate_v4 = wrapped

    def undo():
        CHOOSER.score_candidate_v4 = orig

    return undo


def _apply_ledger():
    """Activate the stateful commit ledger via `BASELINE_LEDGER=on`
    AND monkey-patch `_tick_ledger` to record per-turn commit/emit/drop
    telemetry into `LEDGER_TRACE`.
    """
    prev_env = os.environ.get("BASELINE_LEDGER")
    os.environ["BASELINE_LEDGER"] = "on"

    orig_tick = AGENT._tick_ledger

    def wrap_tick(me, world, model, omega):
        pending_before = list(AGENT._PENDING_LAUNCHES.get(int(me), []))
        # entries that REACHED zero this turn (pre-decrement wait==1)
        becoming_due = [e for e in pending_before
                        if int(e["wait_remaining"]) == 1]
        due_moves, survivors = orig_tick(me, world, model, omega)
        # Now becoming_due entries have either fired (got fired_at_step
        # set on them) or got drop_reason set.
        drops_by_reason: dict[str, int] = collections.Counter()
        for e in becoming_due:
            r = e.get("drop_reason")
            if r:
                drops_by_reason[r] += 1
        LEDGER_TRACE.append({
            "step": int(world.step),
            "pending_before": len(pending_before),
            "due_reached_zero": len(becoming_due),
            "emitted": len(due_moves),
            "dropped": sum(drops_by_reason.values()),
            "drops_by_reason": dict(drops_by_reason),
            "survivors": len(survivors),
            "due_moves": list(due_moves),
        })
        return due_moves, survivors

    AGENT._tick_ledger = wrap_tick

    def undo():
        AGENT._tick_ledger = orig_tick
        if prev_env is None:
            os.environ.pop("BASELINE_LEDGER", None)
        else:
            os.environ["BASELINE_LEDGER"] = prev_env

    return undo


def _apply_ledger_soft():
    """Same as `_apply_ledger`, but in soft mode: surviving pending
    entries do NOT reserve their src, so the chooser can opportunistically
    fire-now from a src that has an in-flight commit. The commit only
    fires at emit time if ≥`ships_planned` ships remain.
    """
    prev = os.environ.get("BASELINE_LEDGER_MODE")
    os.environ["BASELINE_LEDGER_MODE"] = "soft"
    undo_inner = _apply_ledger()

    def undo():
        undo_inner()
        if prev is None:
            os.environ.pop("BASELINE_LEDGER_MODE", None)
        else:
            os.environ["BASELINE_LEDGER_MODE"] = prev

    return undo


POLICIES = {
    "baseline": _apply_baseline,
    "mpc": _apply_mpc,
    "ledger": _apply_ledger,
    "ledger_soft": _apply_ledger_soft,
}


# ---------------------------------------------------------------------------
# State summarisation (per-turn snapshot + state diff)
# ---------------------------------------------------------------------------


def summarise_state(obs_or_snap) -> dict:
    """Counts of planets and ships per owner, for diff display."""
    if hasattr(obs_or_snap, "state"):
        obs = obs_or_snap.state[0].observation
    else:
        obs = obs_or_snap
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
    ships_by: dict[int, int] = collections.Counter()
    planets_by: dict[int, int] = collections.Counter()
    prod_by: dict[int, int] = collections.Counter()
    for p in planets:
        owner = int(p[1])
        planets_by[owner] += 1
        ships_by[owner] += int(p[5])
        prod_by[owner] += int(p[6])
    for f in fleets:
        owner = int(f[1])
        ships_by[owner] += int(f[6])
    return {
        "planets_by_owner": dict(planets_by),
        "ships_by_owner": dict(ships_by),
        "prod_by_owner": dict(prod_by),
        "n_fleets": len(fleets),
    }


def diff_states(sim: dict, recorded: dict) -> dict:
    """Compact diff between sim summary and recorded summary."""
    diff = {}
    owners = set(sim["planets_by_owner"]) | set(recorded["planets_by_owner"])
    for o in owners:
        sp = sim["planets_by_owner"].get(o, 0)
        rp = recorded["planets_by_owner"].get(o, 0)
        ss = sim["ships_by_owner"].get(o, 0)
        rs = recorded["ships_by_owner"].get(o, 0)
        if sp != rp or ss != rs:
            diff[f"owner_{o}"] = {
                "planets_sim_vs_rec": [sp, rp],
                "ships_sim_vs_rec": [ss, rs],
            }
    return diff


def normalise_action(a) -> frozenset:
    """Same shape as scripts.baseline_postmortem._normalise_action."""
    if not a:
        return frozenset()
    return frozenset(
        (int(x[0]), round(float(x[1]), 3), int(x[2]))
        for x in a
        if isinstance(x, list) and len(x) >= 3
    )


def run_policy(replay: dict, team_name: str, policy: str) -> dict:
    """Replay the game from turn 0 under `policy`, driving the agent's
    actions but feeding opp's recorded actions verbatim.
    """
    teams = replay["info"]["TeamNames"]
    our_seats = [i for i, t in enumerate(teams) if t == team_name]
    if not our_seats:
        return {"error": "team not in seats", "teams": teams}
    our_seat = our_seats[0]
    num_seats = len(teams)
    seed = replay["info"].get("seed") or 0
    cfg = replay["configuration"]

    # The replay's initial obs is at steps[0][seat]['observation']. fast_sim's
    # interpreter is happy with a single obs as input — seat-aliasing is
    # set up internally.
    obs0 = replay["steps"][0][our_seat]["observation"]
    snap = FS.from_obs(obs0, configuration=cfg, episode_seed=seed,
                       num_seats=num_seats)

    n_steps = len(replay["steps"])
    per_turn = []

    # Apply policy hooks
    LEDGER_TRACE.clear()
    undo_hook = POLICIES[policy]()
    diverged_at = None
    try:
        for t in range(n_steps - 1):
            # Pre-step: agent acts on the CURRENT sim state.
            our_obs = snap.state[our_seat].observation
            # Inject `player` so the agent knows which seat it is.
            # `from_obs` sets player=i on each seat already.
            t0 = time.perf_counter()
            try:
                predicted = AGENT.agent(our_obs)
            except Exception as e:
                predicted = []
                err = f"{type(e).__name__}: {e}"
            else:
                err = None
            dt_ms = (time.perf_counter() - t0) * 1000

            # Build per-seat action vector: ours = predicted, opps = recorded.
            actions = []
            for seat in range(num_seats):
                if seat == our_seat:
                    actions.append(predicted or [])
                else:
                    actions.append(replay["steps"][t + 1][seat].get("action") or [])

            # Snapshot of current sim state BEFORE step.
            sim_pre = summarise_state(snap)

            # Step the sim forward.
            snap = FS.step(snap, actions, in_place=False)

            # Compare to the recorded state at t+1.
            recorded_next = replay["steps"][t + 1][our_seat]["observation"]
            rec_sum = summarise_state(recorded_next)
            sim_sum = summarise_state(snap)
            d = diff_states(sim_sum, rec_sum)

            if diverged_at is None and d:
                diverged_at = t + 1

            recorded_action = replay["steps"][t + 1][our_seat].get("action") or []
            per_turn.append({
                "t": t,
                "dt_ms": round(dt_ms, 2),
                "err": err,
                "predicted_action": predicted or [],
                "recorded_action": recorded_action,
                "action_match": normalise_action(predicted) == normalise_action(recorded_action),
                "predicted_n": len(predicted) if predicted else 0,
                "recorded_n": len(recorded_action) if recorded_action else 0,
                "sim_after_step": sim_sum,
                "recorded_after_step": rec_sum,
                "state_diff": d,
            })
            if snap.fake_env.done:
                break
    finally:
        undo_hook()

    # Episode-end accounting.
    our_planets_final = snap.state[our_seat].observation.planets \
        if hasattr(snap.state[our_seat].observation, "planets") \
        else snap.state[our_seat].observation.get("planets", [])
    rewards = [snap.state[i].reward for i in range(num_seats)]
    n_idle_pred = sum(1 for pt in per_turn if pt["predicted_n"] == 0)
    n_idle_rec = sum(1 for pt in per_turn if pt["recorded_n"] == 0)
    n_match = sum(1 for pt in per_turn if pt["action_match"])

    # Ledger telemetry aggregate
    ledger_summary = None
    if LEDGER_TRACE:
        total_emitted = sum(t["emitted"] for t in LEDGER_TRACE)
        total_dropped = sum(t["dropped"] for t in LEDGER_TRACE)
        total_due = sum(t["due_reached_zero"] for t in LEDGER_TRACE)
        drops_by_reason_agg: dict[str, int] = collections.Counter()
        for t in LEDGER_TRACE:
            for k, v in (t.get("drops_by_reason") or {}).items():
                drops_by_reason_agg[k] += v
        ledger_summary = {
            "total_due_reached_zero": total_due,
            "total_emitted": total_emitted,
            "total_dropped_on_emit": total_dropped,
            "emit_success_rate": (total_emitted / total_due) if total_due else None,
            "drops_by_reason": dict(drops_by_reason_agg),
            "max_pending": max((t["pending_before"] for t in LEDGER_TRACE),
                               default=0),
            "trace": list(LEDGER_TRACE),
        }

    return {
        "policy": policy,
        "n_steps_simulated": len(per_turn),
        "diverged_at_turn": diverged_at,
        "rewards": rewards,
        "our_seat": our_seat,
        "n_idle_predicted": n_idle_pred,
        "n_idle_recorded": n_idle_rec,
        "action_match_count": n_match,
        "action_match_rate": (n_match / len(per_turn)) if per_turn else None,
        "final_planet_count_us": sum(1 for p in our_planets_final if int(p[1]) == our_seat),
        "per_turn": per_turn,
        "ledger_summary": ledger_summary,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_id")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--policy", nargs="+", default=["baseline"],
                        choices=list(POLICIES.keys()),
                        help="which chooser policies to run + compare")
    parser.add_argument("--team", default=None,
                        help="team name (defaults to the team from "
                             "summary.json, else auto-detect)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    sub_dir = REPO / "audit" / "live-episodes" / str(args.submission_id)
    candidates = list(sub_dir.glob(f"episode-{args.episode}*-replay.json"))
    if not candidates:
        print(f"ERROR: replay not found for episode {args.episode}")
        return 1
    replay_path = candidates[0]
    replay = json.load(open(replay_path))

    out_dir = Path(args.out) if args.out else \
        REPO / "audit" / "whatif" / str(args.submission_id) / args.episode
    out_dir.mkdir(parents=True, exist_ok=True)

    # Team name resolution order: --team flag, summary.json, auto-detect.
    if args.team:
        team_name = args.team
    else:
        summary_path = sub_dir / "summary.json"
        if summary_path.exists():
            team_name = json.load(open(summary_path))["team_name"]
        else:
            team_name = detect_team_name(
                [replay_path], os.environ.get("KAGGLE_USERNAME"))
    print(f"what-if submission={args.submission_id} ep={args.episode} as `{team_name}` "
          f"policies={args.policy}")

    results = {}
    for policy in args.policy:
        print(f"\n=== policy={policy} ===")
        t0 = time.perf_counter()
        res = run_policy(replay, team_name, policy)
        dt = time.perf_counter() - t0
        if "error" in res:
            print(f"  ERROR: {res['error']}")
            continue
        (out_dir / f"{policy}.json").write_text(
            json.dumps(res, indent=1, default=str) + "\n")
        results[policy] = res
        print(f"  simulated {res['n_steps_simulated']} turns in {dt:.1f}s")
        print(f"  rewards: {res['rewards']}  our_seat={res['our_seat']}")
        print(f"  action_match: {res['action_match_count']}/"
              f"{res['n_steps_simulated']} "
              f"({100*(res['action_match_rate'] or 0):.1f}%)")
        print(f"  predicted-idle turns: {res['n_idle_predicted']}/"
              f"{res['n_steps_simulated']} "
              f"({100*res['n_idle_predicted']/max(1,res['n_steps_simulated']):.1f}%)")
        print(f"  recorded-idle turns:  {res['n_idle_recorded']}/"
              f"{res['n_steps_simulated']}")
        print(f"  diverged at turn: {res['diverged_at_turn']}")
        print(f"  final planet count (us): {res['final_planet_count_us']}")

    # Multi-policy summary
    if len(results) >= 2:
        summary = {
            "submission_id": args.submission_id,
            "episode_id": args.episode,
            "team_name": team_name,
            "policies": {
                p: {
                    "n_idle_predicted": r["n_idle_predicted"],
                    "n_steps_simulated": r["n_steps_simulated"],
                    "diverged_at_turn": r["diverged_at_turn"],
                    "action_match_rate": r["action_match_rate"],
                    "rewards": r["rewards"],
                    "final_planet_count_us": r["final_planet_count_us"],
                }
                for p, r in results.items()
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("\n=== Multi-policy summary (out_dir written) ===")
        print(json.dumps(summary["policies"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
