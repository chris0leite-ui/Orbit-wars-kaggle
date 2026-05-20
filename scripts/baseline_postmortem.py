"""Replay-driven filter-rejection trace for `agents/baseline` (the current
production agent).

Replays a Kaggle live-episode JSON turn-by-turn through `agents.baseline.main.agent`,
with the three proposer filters monkey-patched to log every accept/reject decision
along with the key intermediate values the static math computed:

  - `_source_survives_launch`     (BUG#4 fix, default ON, opt-out via
                                   `PROPOSER_DRAIN_FILTER=off`)
  - `_target_holdable_after_capture`  (2026-05-18 PM, default ON, opt-out via
                                       `PROPOSER_HOLD_FEASIBILITY=off`)
  - `_target_cost_parity_ok`       (2026-05-19 PM, default ON, opt-out via
                                    `PROPOSER_COST_PARITY=off`)

Use case: confirm via data which filter is suppressing the candidates that
would have produced launches on the 49pct of turns we sat idle in the sary
sary loss (ep 77140674). See plan at
/root/.claude/plans/so-now-research-and-zany-widget.md for context.

CLI:
    python -m scripts.baseline_postmortem <submission_id> --episode <eid>
    python -m scripts.baseline_postmortem <submission_id> --limit 10

Output:
    audit/live-episodes/<sid>/postmortem/postmortem-<eid>.json (per-turn detail)
    audit/live-episodes/<sid>/postmortem/baseline-roll-up.json (aggregate)
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import agents.baseline.main as AGENT  # noqa: E402
import agents.baseline.proposer as PROP  # noqa: E402
import agents.baseline.chooser_trajectory as CHOOSER  # noqa: E402
from lib.fleet import speed as fleet_speed  # noqa: E402
from scripts.live_episode_summary import detect_team_name  # noqa: E402


# Per-turn filter rejection trace.
TRACE: list[dict[str, Any]] = []
# Per-turn chooser-score trace.
SCORE_TRACE: list[dict[str, Any]] = []


def install_filter_hooks() -> None:
    """Wrap the three proposer filters to log every call.

    Records: filter name, kept-or-dropped, source/target ids + counts, key
    intermediate values (the static math each filter computed). NO change
    to the filter's return value — the agent's actual decision is unchanged.
    """
    orig_drain = PROP._source_survives_launch
    orig_hold = PROP._target_holdable_after_capture
    orig_cost = PROP._target_cost_parity_ok

    def wrap_drain(src, ships, wait_N, world, model, me):
        result = orig_drain(src, ships, wait_N, world, model, me)
        threat_eta = model.time_to_enemy_threat(int(src.id), me, world)
        threat_force = 0
        if threat_eta is not None:
            threat_force = sum(
                sh for (eta_arr, owner, sh) in model.ledger.get(int(src.id), [])
                if owner != me and eta_arr <= int(threat_eta) + PROP.WAVE_LOOKAHEAD
            )
        TRACE.append({
            "filter": "drain",
            "kept": bool(result),
            "src": int(src.id),
            "src_ships": int(src.ships),
            "src_prod": int(src.production),
            "ships": int(ships),
            "wait_N": int(wait_N),
            "threat_eta": (None if threat_eta is None else int(threat_eta)),
            "threat_force": int(threat_force),
        })
        return result

    def wrap_hold(src, tgt, ships, wait_N, eta, world, model, me):
        result = orig_hold(src, tgt, ships, wait_N, eta, world, model, me)
        # Re-derive the key numbers (cheap; static math only) for the log.
        arrival_step = int(wait_N) + int(eta)
        if int(tgt.owner) == me:
            tgt_def = int(tgt.ships)
        elif int(tgt.owner) == -1:
            tgt_def = int(tgt.ships)
        else:
            tgt_def = int(tgt.ships) + int(tgt.production) * arrival_step
        delivered = int(ships) - tgt_def

        nearest_opp_id = None
        nearest_opp_dist = float("inf")
        nearest_opp_ships = 0
        nearest_opp_prod = 0
        for opp in world.planets_by_id.values():
            if int(opp.owner) == me or int(opp.owner) == -1:
                continue
            if int(opp.id) == int(tgt.id):
                continue
            if int(opp.ships) < 20:  # MIN_COUNTER_SHIPS
                continue
            d = math.hypot(float(opp.x) - float(tgt.x), float(opp.y) - float(tgt.y))
            if d < nearest_opp_dist:
                nearest_opp_dist = d
                nearest_opp_id = int(opp.id)
                nearest_opp_ships = int(opp.ships)
                nearest_opp_prod = int(opp.production)

        garrison = None
        counter_force = None
        if nearest_opp_id is not None and delivered >= 1:
            tgt_radius = float(getattr(tgt, "radius", 1.5))
            opp_radius = 1.5  # generic; only used for flight-time estimate
            flight = max(0.0, nearest_opp_dist - opp_radius - tgt_radius - 0.1)
            opp_speed = fleet_speed(nearest_opp_ships)
            t_op = int(math.ceil(flight / opp_speed)) if opp_speed > 0 else 0
            garrison = delivered + int(tgt.production) * t_op
            counter_force = nearest_opp_ships + nearest_opp_prod * (arrival_step + t_op)

        TRACE.append({
            "filter": "hold",
            "kept": bool(result),
            "src": int(src.id),
            "src_ships": int(src.ships),
            "tgt": int(tgt.id),
            "tgt_owner": int(tgt.owner),
            "tgt_ships": int(tgt.ships),
            "tgt_prod": int(tgt.production),
            "ships": int(ships),
            "wait_N": int(wait_N),
            "eta": int(eta),
            "tgt_def_at_arrival": int(tgt_def),
            "delivered": int(delivered),
            "nearest_opp": nearest_opp_id,
            "nearest_opp_dist": (None if nearest_opp_id is None
                                 else round(nearest_opp_dist, 2)),
            "nearest_opp_ships": (None if nearest_opp_id is None
                                  else int(nearest_opp_ships)),
            "garrison": (None if garrison is None else int(garrison)),
            "counter_force": (None if counter_force is None else int(counter_force)),
        })
        return result

    def wrap_cost(src, tgt, ships, wait_N, eta, world, model, me):
        result = orig_cost(src, tgt, ships, wait_N, eta, world, model, me)
        TRACE.append({
            "filter": "cost",
            "kept": bool(result),
            "src": int(src.id),
            "src_ships": int(src.ships),
            "tgt": int(tgt.id),
            "tgt_owner": int(tgt.owner),
            "tgt_ships": int(tgt.ships),
            "ships": int(ships),
            "wait_N": int(wait_N),
            "eta": int(eta),
        })
        return result

    PROP._source_survives_launch = wrap_drain
    PROP._target_holdable_after_capture = wrap_hold
    PROP._target_cost_parity_ok = wrap_cost


def install_chooser_hooks() -> None:
    """Wrap `chooser_trajectory.score_candidate_v4` to log every per-candidate
    score (delta, status, eta) along with the candidate identity.

    Solo-launch path only. Joint candidates (`score_candidate_v4_joint`) are
    invoked separately for 2P only and represent a small minority; the
    summary stats from solo cover the bulk of the chooser's decisions.
    """
    orig_score = CHOOSER.score_candidate_v4

    def wrap_score(snap_base, src, tgt, ships, angle, me, num_seats, world,
                   baseline_favors, favor_fn, gamma, horizon,
                   skip_admissibility=False, wait_N=0):
        delta, status, eta = orig_score(
            snap_base, src, tgt, ships, angle, me, num_seats, world,
            baseline_favors, favor_fn, gamma, horizon,
            skip_admissibility=skip_admissibility, wait_N=wait_N,
        )
        SCORE_TRACE.append({
            "src": int(src.id),
            "src_ships": int(src.ships),
            "src_prod": int(src.production),
            "tgt": int(tgt.id),
            "tgt_owner": int(tgt.owner),
            "tgt_ships": int(tgt.ships),
            "tgt_prod": int(tgt.production),
            "ships": int(ships),
            "wait_N": int(wait_N),
            "delta": (None if delta == float("-inf") else round(float(delta), 3)),
            "status": status,
            "eta": eta,
            "baseline_at_horizon": round(float(baseline_favors[min(horizon, len(baseline_favors)-1)]), 3),
        })
        return (delta, status, eta)

    CHOOSER.score_candidate_v4 = wrap_score


def analyse_episode(path: Path, team_name: str) -> dict:
    """Replay one episode through the baseline agent; return per-turn detail."""
    replay = json.load(open(path))
    teams = replay["info"]["TeamNames"]
    rewards = replay["rewards"]
    n_size = len(teams)

    our_seats = [i for i, t in enumerate(teams) if t == team_name]
    if not our_seats:
        return {"episode_id": path.stem, "error": "team not in seats", "teams": teams}
    our_seat = our_seats[0]
    our_player_id = replay["steps"][0][our_seat]["observation"].get("player", our_seat)

    if all(r is None for r in rewards):
        result = "crashed"
    else:
        max_r = max(r for r in rewards if r is not None)
        we_won = any(rewards[i] is not None and rewards[i] == max_r
                     for i in our_seats)
        result = "win" if we_won else "loss"

    steps = replay["steps"]
    n_steps = len(steps)
    per_turn: list[dict] = []

    for t in range(n_steps - 1):
        ours = steps[t][our_seat]
        if ours["status"] != "ACTIVE":
            continue
        obs = ours["observation"]
        if obs.get("step") is None:
            obs = dict(obs)
            obs["step"] = t
        recorded_action = steps[t + 1][our_seat].get("action") or []

        TRACE.clear()
        SCORE_TRACE.clear()
        t0 = time.perf_counter()
        try:
            predicted = AGENT.agent(obs)
        except Exception as e:
            predicted = None
            err = f"{type(e).__name__}: {e}"
        else:
            err = None
        dt_ms = (time.perf_counter() - t0) * 1000

        trace_snapshot = list(TRACE)
        score_snapshot = list(SCORE_TRACE)

        by_filter: dict = collections.defaultdict(
            lambda: {"kept": 0, "dropped": 0, "drops": []}
        )
        for entry in trace_snapshot:
            key = entry["filter"]
            if entry["kept"]:
                by_filter[key]["kept"] += 1
            else:
                by_filter[key]["dropped"] += 1
                by_filter[key]["drops"].append(entry)

        # Chooser-score summary: count by status; identify top-K deltas.
        status_counts: collections.Counter = collections.Counter()
        scored = []
        for s in score_snapshot:
            status_counts[s["status"]] += 1
            if s["status"] == "scored" and s["delta"] is not None:
                scored.append(s)
        scored.sort(key=lambda s: -s["delta"])
        top_pos = [s for s in scored if s["delta"] > 0][:5]
        top_neg_close = [s for s in scored if s["delta"] <= 0][:5]
        max_delta = scored[0]["delta"] if scored else None

        per_turn.append({
            "t": t,
            "dt_ms": round(dt_ms, 2),
            "err": err,
            "predicted": predicted if predicted is not None else [],
            "recorded": recorded_action,
            "predicted_n": len(predicted) if predicted else 0,
            "recorded_n": len(recorded_action) if recorded_action else 0,
            "filter": {k: dict(v) for k, v in by_filter.items()},
            "total_filter_calls": len(trace_snapshot),
            "n_scored": len(score_snapshot),
            "score_status": dict(status_counts),
            "max_delta": max_delta,
            "n_positive_delta": sum(1 for s in scored if s["delta"] > 0),
            "top_positive": top_pos,
            "top_below_zero": top_neg_close,
        })

    return {
        "episode_id": path.stem.replace("-replay", ""),
        "size": n_size,
        "result": result,
        "teams": teams,
        "our_seat": our_seat,
        "our_player_id": our_player_id,
        "n_steps": n_steps,
        "per_turn": per_turn,
    }


def aggregate(per_episode: list[dict]) -> dict:
    """Roll-up across episodes."""
    by_result: collections.Counter = collections.Counter()
    by_filter_kept: collections.Counter = collections.Counter()
    by_filter_dropped: collections.Counter = collections.Counter()
    n_turns = 0
    n_idle_turns = 0
    n_recorded_idle = 0
    n_predicted_idle = 0
    dt_all: list[float] = []

    for ep in per_episode:
        if "error" in ep:
            continue
        by_result[ep["result"]] += 1
        for pt in ep["per_turn"]:
            n_turns += 1
            dt_all.append(pt["dt_ms"])
            if pt["predicted_n"] == 0:
                n_predicted_idle += 1
            if pt["recorded_n"] == 0:
                n_recorded_idle += 1
            if pt["predicted_n"] == 0 and pt["recorded_n"] == 0:
                n_idle_turns += 1
            for f, counts in pt["filter"].items():
                by_filter_kept[f] += counts["kept"]
                by_filter_dropped[f] += counts["dropped"]

    # Chooser-side aggregates.
    idle_with_candidates = 0
    idle_with_pos_delta = 0
    idle_scored_total = 0
    idle_pos_delta_total = 0
    nonidle_scored_total = 0
    nonidle_pos_delta_total = 0
    for ep in per_episode:
        if "error" in ep: continue
        for pt in ep["per_turn"]:
            n_pos = pt.get("n_positive_delta", 0) or 0
            n_scored = pt.get("n_scored", 0) or 0
            if pt["predicted_n"] == 0:
                if n_scored > 0:
                    idle_with_candidates += 1
                if n_pos > 0:
                    idle_with_pos_delta += 1
                idle_scored_total += n_scored
                idle_pos_delta_total += n_pos
            else:
                nonidle_scored_total += n_scored
                nonidle_pos_delta_total += n_pos

    p95 = sorted(dt_all)[int(0.95 * (len(dt_all) - 1))] if dt_all else 0
    return {
        "n_episodes": len([e for e in per_episode if "error" not in e]),
        "by_result": dict(by_result),
        "n_turns": n_turns,
        "n_predicted_idle": n_predicted_idle,
        "n_recorded_idle": n_recorded_idle,
        "n_both_idle": n_idle_turns,
        "by_filter_kept": dict(by_filter_kept),
        "by_filter_dropped": dict(by_filter_dropped),
        "by_filter_drop_rate": {
            f: round(by_filter_dropped[f]
                     / (by_filter_kept[f] + by_filter_dropped[f]), 4)
            for f in set(list(by_filter_kept) + list(by_filter_dropped))
            if (by_filter_kept[f] + by_filter_dropped[f]) > 0
        },
        "idle_turns_with_candidates": idle_with_candidates,
        "idle_turns_with_positive_delta": idle_with_pos_delta,
        "idle_scored_total": idle_scored_total,
        "idle_positive_delta_total": idle_pos_delta_total,
        "nonidle_scored_total": nonidle_scored_total,
        "nonidle_positive_delta_total": nonidle_pos_delta_total,
        "dt_ms_p95": round(p95, 2),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_id")
    parser.add_argument("--episode", default=None,
                        help="run a single episode-<id> only")
    parser.add_argument("--limit", type=int, default=None,
                        help="limit number of episodes")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--only-result", default=None,
                        choices=["win", "loss"], help="restrict to outcome")
    args = parser.parse_args(argv)

    sub_dir = REPO / "audit" / "live-episodes" / str(args.submission_id)
    if not sub_dir.is_dir():
        print(f"ERROR: {sub_dir} does not exist.")
        return 1

    out_dir = Path(args.out) if args.out else sub_dir / "postmortem"
    out_dir.mkdir(parents=True, exist_ok=True)

    replays = sorted(sub_dir.glob("episode-*-replay.json"))
    if args.episode:
        replays = [r for r in replays if args.episode in r.stem]
    if not replays:
        print("ERROR: no replays match.")
        return 1

    team_name = detect_team_name(replays, os.environ.get("KAGGLE_USERNAME"))
    print(f"baseline_postmortem submission={args.submission_id} as `{team_name}` "
          f"({len(replays)} replays before --only-result/--limit)")

    install_filter_hooks()
    install_chooser_hooks()

    per_episode = []
    t0 = time.perf_counter()
    skipped_wrong_result = 0
    for path in replays:
        if args.limit and len(per_episode) >= args.limit:
            break
        eid = path.stem.replace("-replay", "")
        try:
            ep = analyse_episode(path, team_name)
        except Exception as e:
            print(f"  {eid}: ERROR {type(e).__name__}: {e}")
            continue
        if args.only_result and ep.get("result") != args.only_result:
            skipped_wrong_result += 1
            continue
        out_path = out_dir / f"postmortem-{eid}.json"
        out_path.write_text(json.dumps(ep, indent=1) + "\n")
        per_episode.append(ep)
        n_idle_pred = sum(1 for p in ep["per_turn"] if p["predicted_n"] == 0)
        n_drops_hold = sum(p["filter"].get("hold", {}).get("dropped", 0)
                           for p in ep["per_turn"])
        n_drops_cost = sum(p["filter"].get("cost", {}).get("dropped", 0)
                           for p in ep["per_turn"])
        n_drops_drain = sum(p["filter"].get("drain", {}).get("dropped", 0)
                            for p in ep["per_turn"])
        print(f"  {eid} size={ep['size']} result={ep['result']} "
              f"n_steps={ep['n_steps']} pred_idle={n_idle_pred} "
              f"drops hold={n_drops_hold} cost={n_drops_cost} drain={n_drops_drain}")

    roll = aggregate(per_episode)
    roll["submission_id"] = args.submission_id
    roll["team_name"] = team_name
    roll["elapsed_s"] = round(time.perf_counter() - t0, 1)
    roll["skipped_wrong_result"] = skipped_wrong_result
    (out_dir / "baseline-roll-up.json").write_text(
        json.dumps(roll, indent=2) + "\n"
    )

    print()
    print(f"=== ROLL-UP submission {args.submission_id} ===")
    print(f"  episodes={roll['n_episodes']}  by_result={roll['by_result']}")
    print(f"  n_turns={roll['n_turns']}")
    print(f"  predicted-idle turns: {roll['n_predicted_idle']} "
          f"({100*roll['n_predicted_idle']/roll['n_turns']:.1f}pct)")
    print(f"  recorded-idle turns:  {roll['n_recorded_idle']} "
          f"({100*roll['n_recorded_idle']/roll['n_turns']:.1f}pct)")
    print(f"  filter kept:    {roll['by_filter_kept']}")
    print(f"  filter dropped: {roll['by_filter_dropped']}")
    print(f"  filter drop-rate: {roll['by_filter_drop_rate']}")
    print(f"  --- chooser side ---")
    print(f"  idle turns with candidates scored: "
          f"{roll['idle_turns_with_candidates']}/{roll['n_predicted_idle']}")
    print(f"  idle turns with at least one Δ>0:  "
          f"{roll['idle_turns_with_positive_delta']}/{roll['n_predicted_idle']}")
    print(f"  idle:    candidates scored={roll['idle_scored_total']}  "
          f"Δ>0={roll['idle_positive_delta_total']}")
    print(f"  nonidle: candidates scored={roll['nonidle_scored_total']}  "
          f"Δ>0={roll['nonidle_positive_delta_total']}")
    print(f"  agent dt p95={roll['dt_ms_p95']}ms")
    print(f"  elapsed: {roll['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
