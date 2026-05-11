"""Replay-driven postmortem for a Kaggle submission's live episodes.

Re-runs the v3_snipe agent through every observation recorded in each replay,
with monkey-patched telemetry on the mission proposers, planner, and mechanism
pipeline. Cross-references our launched fleets against the replay's actual
forward state to classify outcomes (captured / bounced / too-late / sun / oob /
hit-other-planet / vanished-unaccounted).

The diagnostic is the strategic-utility view that
audit/2026-05-11-capture-success-probe.json is missing: it counted "reached"
(97.2%) but said nothing about whether the fleet actually helped us win.

Hook attachment assumes `from lib.mechanism import DEFAULT_MECHANISMS` resolves
to a mutable list reference (true today, agents/v3_snipe/main.py:33). The
diagnostic mutates that list in-place; do not import this script alongside a
live game.

CLI:
    python -m scripts.episode_postmortem <submission_id> [--episode <eid>]
    [--out <dir>] [--limit N]

Output:
    audit/live-episodes/<sid>/postmortem/{postmortem-<eid>.json, roll-up.json}
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Imports below depend on sys.path containing the repo root.
import lib.mechanism as M  # noqa: E402
import lib.missions.snipe as MS  # noqa: E402
import lib.missions.reinforce as MR  # noqa: E402
import lib.planner as MP  # noqa: E402
from lib.intent import World, realize  # noqa: E402
from lib.world_model import WorldModel  # noqa: E402
import agents.v3_snipe.main as AGENT  # noqa: E402
from scripts.live_episode_summary import detect_team_name  # noqa: E402


TELEMETRY: dict = {}


def reset_telemetry() -> None:
    TELEMETRY.clear()
    TELEMETRY["n_snipe_candidates"] = 0
    TELEMETRY["n_reinforce_candidates"] = 0
    TELEMETRY["n_settled"] = 0
    TELEMETRY["n_sources"] = 0
    TELEMETRY["n_sources_idle"] = 0
    TELEMETRY["drops"] = collections.Counter()
    TELEMETRY["picked_scores"] = []
    TELEMETRY["runnerup_margin"] = []


def install_hooks() -> None:
    """Monkey-patch the proposers, planner, mechanisms, and realize gate.

    Idempotent: re-installing wraps the already-wrapped functions, which is
    fine for per-episode counting because reset_telemetry() zeros everything
    each turn. But callers should only invoke this once per process.
    """
    orig_propose_snipe = MS.propose_snipe_missions
    orig_propose_reinforce = MR.propose_reinforce_missions
    orig_settle = MP.settle_plan

    def wrapped_snipe(world, model):
        out = orig_propose_snipe(world, model)
        TELEMETRY["n_snipe_candidates"] += len(out)
        return out

    def wrapped_reinforce(world, model):
        out = orig_propose_reinforce(world, model)
        TELEMETRY["n_reinforce_candidates"] += len(out)
        return out

    def wrapped_settle(missions, world, model):
        # Count sources and idle sources after the planner runs.
        intents = orig_settle(missions, world, model)
        TELEMETRY["n_settled"] += len(intents)
        my_planets = [p for p in world.planets_by_id.values() if p.owner == world.my_id]
        n_sources = len(my_planets)
        used_srcs = {i.src_id for i in intents}
        TELEMETRY["n_sources"] += n_sources
        TELEMETRY["n_sources_idle"] += max(0, n_sources - len(used_srcs))
        # Per-source picked score + runner-up margin.
        by_src: dict = collections.defaultdict(list)
        for m in missions:
            by_src[m.src_id].append(m.score)
        for src in used_srcs:
            scores = sorted(by_src.get(src, []), reverse=True)
            if scores:
                TELEMETRY["picked_scores"].append(scores[0])
            if len(scores) >= 2:
                TELEMETRY["runnerup_margin"].append(scores[0] - scores[1])
        return intents

    # In-place patch (the agent imports symbols by name, not module).
    MS.propose_snipe_missions = wrapped_snipe
    MR.propose_reinforce_missions = wrapped_reinforce
    MP.settle_plan = wrapped_settle
    AGENT.propose_snipe_missions = wrapped_snipe
    AGENT.propose_reinforce_missions = wrapped_reinforce
    AGENT.settle_plan = wrapped_settle

    # Wrap each mechanism in DEFAULT_MECHANISMS by mutating the list in-place.
    orig_mechs = list(M.DEFAULT_MECHANISMS)

    def _wrap_mech(name, fn):
        def shim(intents, world):
            n_in = len(intents)
            out = fn(intents, world)
            TELEMETRY["drops"][name] += max(0, n_in - len(out))
            return out
        shim.__name__ = name
        return shim

    M.DEFAULT_MECHANISMS[:] = [_wrap_mech(f.__name__, f) for f in orig_mechs]


def _planet_owner_at(step_obs: dict, planet_id: int):
    """Return (owner, ships) for planet_id in the given step's observation."""
    for p in step_obs.get("planets", []):
        if p[0] == planet_id:
            return p[1], p[5]
    return None, None


def _fleet_in_step(step_obs: dict, fleet_id: int):
    """Find fleet by ID in the step's observation; return entry or None."""
    for f in step_obs.get("fleets", []):
        if f[0] == fleet_id:
            return f
    return None


def attribute_fleets(replay: dict, our_seat: int, our_player_id: int) -> list:
    """Classify every fleet WE launched into outcome buckets.

    Scans steps[t][our_seat]["observation"]["fleets"]:
      - Each new fleet (id not seen before, owner==our_player_id) is a launch.
      - Track its appearance until it vanishes; inspect the same step + next
        step in EVERY seat's observation to see what happened to the target.

    Returns a list of dicts (one per fleet we launched).
    """
    steps = replay["steps"]
    n_steps = len(steps)
    seen_fleets: set = set()
    fleets_out = []

    # Build a fast per-step view: any seat's observation gives the global state,
    # so we use seat 0's observation when available (it's still ACTIVE), else
    # fall back to our seat.
    def _global_obs(t):
        # Prefer a seat whose status is ACTIVE so we get up-to-date state.
        for seat in range(len(steps[t])):
            if steps[t][seat]["status"] == "ACTIVE":
                return steps[t][seat]["observation"]
        return steps[t][0]["observation"]

    # Fleet ID -> turn-of-first-sighting + initial entry
    first_sight: dict = {}
    target_guess: dict = {}  # fleet_id -> target_id (closest planet at launch direction)

    for t in range(n_steps):
        obs = _global_obs(t)
        for f in obs.get("fleets", []):
            fid, fowner = f[0], f[1]
            if fowner != our_player_id:
                continue
            if fid not in seen_fleets:
                seen_fleets.add(fid)
                first_sight[fid] = (t, f, dict(obs))

    # For each of our fleets, walk forward until it disappears.
    for fid, (t_launch, init_entry, init_obs) in first_sight.items():
        owner = init_entry[1]
        x0, y0 = init_entry[2], init_entry[3]
        angle = init_entry[4]
        ships_launch = init_entry[5]

        last_seen_t = t_launch
        last_entry = init_entry
        vanish_t = None
        for t in range(t_launch + 1, n_steps):
            obs_t = _global_obs(t)
            cur = _fleet_in_step(obs_t, fid)
            if cur is None:
                vanish_t = t
                break
            last_seen_t = t
            last_entry = cur

        # Determine outcome from where the fleet was last seen + what changed
        # on the planet nearest the last-seen position at vanish_t.
        outcome = "unknown"
        target_id = None
        flipped_to_us = False
        if vanish_t is None:
            outcome = "alive_at_end"
        else:
            obs_vanish = _global_obs(vanish_t)
            # Find planet closest to fleet's last-seen position.
            best_d = float("inf")
            best_pid = None
            for p in obs_vanish.get("planets", []):
                pid, _po, px, py = p[0], p[1], p[2], p[3]
                d = math.hypot(px - last_entry[2], py - last_entry[3])
                if d < best_d:
                    best_d = d
                    best_pid = pid
            # Was the vanish co-located with a planet? Distance threshold: planet
            # radius is small (~1.5-3 units); use 5 units as a generous bound.
            # Check for sun proximity too: sun is at center (50, 50) radius 10.
            sun_d = math.hypot(last_entry[2] - 50.0, last_entry[3] - 50.0)
            in_bounds = 0.0 <= last_entry[2] <= 100.0 and 0.0 <= last_entry[3] <= 100.0

            if sun_d < 10.5:
                outcome = "sun"
            elif not in_bounds:
                outcome = "oob"
            elif best_d < 5.0 and best_pid is not None:
                # Hit a planet. Was it our target (highest-angle-correlation guess)?
                target_id = best_pid
                # Did ownership change in our favour?
                owner_before, ships_before = _planet_owner_at(_global_obs(vanish_t - 1), best_pid)
                owner_after, ships_after = _planet_owner_at(obs_vanish, best_pid)
                if owner_after == our_player_id and owner_before != our_player_id:
                    outcome = "captured"
                    flipped_to_us = True
                elif owner_after == our_player_id and owner_before == our_player_id:
                    outcome = "reinforced_self"  # arrived at our own planet
                elif owner_after != our_player_id and owner_before != our_player_id and owner_before is not None:
                    # Planet did not flip to us. Was it enemy-held? Bounce.
                    if owner_before == -1:
                        outcome = "bounced_neutral"
                    else:
                        outcome = "bounced_enemy"
                elif owner_after != our_player_id and owner_before == our_player_id:
                    # We owned it; lost it on the same step we arrived. Bad luck.
                    outcome = "arrived_but_lost"
                else:
                    outcome = "hit_planet_unknown_flip"
            else:
                outcome = "vanished_in_space"

        fleets_out.append({
            "fleet_id": fid,
            "launched_t": t_launch,
            "ships": ships_launch,
            "angle": angle,
            "launch_xy": [x0, y0],
            "vanish_t": vanish_t,
            "lifetime": (vanish_t - t_launch) if vanish_t else (n_steps - t_launch),
            "target_id": target_id,
            "outcome": outcome,
            "flipped_to_us": flipped_to_us,
        })

    return fleets_out


def _normalise_action(a):
    """Coerce a recorded action list into a comparable tuple-set.

    Action format: list of [src_id, angle, ships]. Round angle for comparison.
    """
    if not a:
        return frozenset()
    return frozenset(
        (int(x[0]), round(float(x[1]), 3), int(x[2]))
        for x in a
        if isinstance(x, list) and len(x) >= 3
    )


def analyse_episode(path: Path, team_name: str) -> dict:
    """Process one replay JSON; return per-episode postmortem dict."""
    replay = json.load(open(path))
    teams = replay["info"]["TeamNames"]
    rewards = replay["rewards"]
    n_size = len(teams)

    # Find our seat(s). For a self-match (all seats are us), prefer seat 0.
    our_seats = [i for i, t in enumerate(teams) if t == team_name]
    if not our_seats:
        return {"episode_id": path.stem, "error": "team not in seats", "teams": teams}
    our_seat = our_seats[0]

    # The agent's "player" field in obs is the canonical id (== seat in 2P/4P).
    our_player_id = replay["steps"][0][our_seat]["observation"].get("player", our_seat)

    steps = replay["steps"]
    n_steps = len(steps)

    # Determine win/loss/draw outcome.
    if all(r is None for r in rewards):
        result = "crashed"
    else:
        max_r = max(r for r in rewards if r is not None)
        we_won = any(rewards[i] is not None and rewards[i] == max_r for i in our_seats)
        we_lost = not we_won
        result = "win" if we_won else ("loss" if we_lost else "draw")

    # Per-turn re-execution.
    per_turn = []
    n_match = 0
    n_emit_match = 0  # both empty counts as "match"
    n_compared = 0

    for t in range(n_steps - 1):
        ours = steps[t][our_seat]
        if ours["status"] != "ACTIVE":
            continue
        obs = ours["observation"]
        recorded_action = ours.get("action") or []

        reset_telemetry()
        t0 = time.perf_counter()
        try:
            predicted = AGENT.agent(obs)
        except Exception as e:
            predicted = None
            err = f"{type(e).__name__}: {e}"
        else:
            err = None
        dt_ms = (time.perf_counter() - t0) * 1000

        rec_set = _normalise_action(recorded_action)
        pred_set = _normalise_action(predicted) if predicted is not None else None
        match = (pred_set == rec_set) if pred_set is not None else False
        if pred_set is not None:
            n_compared += 1
            if match:
                n_match += 1
            if not recorded_action and predicted == []:
                n_emit_match += 1

        per_turn.append({
            "t": t,
            "n_snipe": TELEMETRY["n_snipe_candidates"],
            "n_reinforce": TELEMETRY["n_reinforce_candidates"],
            "n_settled": TELEMETRY["n_settled"],
            "n_sources": TELEMETRY["n_sources"],
            "n_sources_idle": TELEMETRY["n_sources_idle"],
            "drops": dict(TELEMETRY["drops"]),
            "picked_scores": TELEMETRY["picked_scores"],
            "runnerup_margin": TELEMETRY["runnerup_margin"],
            "agent_action_matches_recorded": match,
            "dt_ms": round(dt_ms, 2),
            "err": err,
            "recorded_n_launches": len(recorded_action) if recorded_action else 0,
            "predicted_n_launches": (len(predicted) if predicted else 0) if predicted is not None else None,
        })

    fleets = attribute_fleets(replay, our_seat, our_player_id)
    fleet_outcomes = collections.Counter(f["outcome"] for f in fleets)

    return {
        "episode_id": path.stem.replace("-replay", ""),
        "size": n_size,
        "result": result,
        "our_seat": our_seat,
        "our_player_id": our_player_id,
        "teams": teams,
        "n_steps": n_steps,
        "action_match_rate": (n_match / n_compared) if n_compared else None,
        "n_turns_compared": n_compared,
        "n_fleets_launched": len(fleets),
        "fleet_outcomes": dict(fleet_outcomes),
        "fleets": fleets,
        "per_turn": per_turn,
    }


def aggregate(per_episode: list) -> dict:
    """Roll-up across episodes."""
    drops_total: collections.Counter = collections.Counter()
    fleet_outcomes_total: collections.Counter = collections.Counter()
    n_sources_total = 0
    n_sources_idle_total = 0
    n_snipe_total = 0
    n_reinforce_total = 0
    n_settled_total = 0
    n_turns = 0
    dt_ms_all: list = []
    match_ok = 0
    match_total = 0
    by_result: collections.Counter = collections.Counter()
    by_size_result: dict = collections.defaultdict(collections.Counter)

    for ep in per_episode:
        if "error" in ep:
            continue
        by_result[ep["result"]] += 1
        by_size_result[ep["size"]][ep["result"]] += 1
        for k, v in (ep.get("fleet_outcomes") or {}).items():
            fleet_outcomes_total[k] += v
        for pt in ep.get("per_turn", []):
            n_turns += 1
            for k, v in pt.get("drops", {}).items():
                drops_total[k] += v
            n_sources_total += pt.get("n_sources", 0)
            n_sources_idle_total += pt.get("n_sources_idle", 0)
            n_snipe_total += pt.get("n_snipe", 0)
            n_reinforce_total += pt.get("n_reinforce", 0)
            n_settled_total += pt.get("n_settled", 0)
            dt_ms_all.append(pt.get("dt_ms", 0))
            if pt.get("agent_action_matches_recorded") is True:
                match_ok += 1
            if pt.get("predicted_n_launches") is not None:
                match_total += 1

    p95 = sorted(dt_ms_all)[int(0.95 * (len(dt_ms_all) - 1))] if dt_ms_all else 0
    p99 = sorted(dt_ms_all)[int(0.99 * (len(dt_ms_all) - 1))] if dt_ms_all else 0
    return {
        "n_episodes": len([e for e in per_episode if "error" not in e]),
        "by_result": dict(by_result),
        "by_size_result": {str(k): dict(v) for k, v in by_size_result.items()},
        "n_turns": n_turns,
        "action_match_rate": (match_ok / match_total) if match_total else None,
        "n_turns_compared": match_total,
        "candidates_per_turn": {
            "snipe": (n_snipe_total / n_turns) if n_turns else 0,
            "reinforce": (n_reinforce_total / n_turns) if n_turns else 0,
        },
        "settled_per_turn": (n_settled_total / n_turns) if n_turns else 0,
        "idle_source_rate": (n_sources_idle_total / n_sources_total) if n_sources_total else 0,
        "drops_per_turn": {k: v / n_turns for k, v in drops_total.items()} if n_turns else {},
        "drops_total": dict(drops_total),
        "fleet_outcomes_total": dict(fleet_outcomes_total),
        "dt_ms_p95": round(p95, 2),
        "dt_ms_p99": round(p99, 2),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_id")
    parser.add_argument("--episode", default=None, help="run a single episode-<id> only")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--limit", type=int, default=None, help="limit number of episodes")
    args = parser.parse_args(argv)

    sub_dir = REPO / "audit" / "live-episodes" / str(args.submission_id)
    if not sub_dir.is_dir():
        print(f"ERROR: {sub_dir} does not exist. Run live_episode_summary first.")
        return 1

    out_dir = Path(args.out) if args.out else sub_dir / "postmortem"
    out_dir.mkdir(parents=True, exist_ok=True)

    replays = sorted(sub_dir.glob("episode-*-replay.json"))
    if args.episode:
        replays = [r for r in replays if args.episode in r.stem]
    if args.limit:
        replays = replays[: args.limit]
    if not replays:
        print("ERROR: no replays match.")
        return 1

    team_name = detect_team_name(replays, os.environ.get("KAGGLE_USERNAME"))
    print(f"Postmortem submission {args.submission_id} as `{team_name}`, {len(replays)} episodes")

    install_hooks()

    per_episode = []
    t0 = time.perf_counter()
    for i, path in enumerate(replays):
        eid = path.stem.replace("-replay", "")
        # Skip self-matches: all seats == team_name and all rewards equal.
        try:
            ep = analyse_episode(path, team_name)
        except Exception as e:
            print(f"  [{i+1}/{len(replays)}] {eid}: ERROR {type(e).__name__}: {e}")
            continue
        out_path = out_dir / f"postmortem-{eid}.json"
        out_path.write_text(json.dumps(ep, indent=1) + "\n")
        per_episode.append(ep)
        am = ep.get("action_match_rate")
        am_str = f"{am*100:.0f}%" if am is not None else "n/a"
        print(f"  [{i+1}/{len(replays)}] {eid} size={ep.get('size')} result={ep.get('result')} "
              f"fleets={ep.get('n_fleets_launched')} act-match={am_str}")

    roll = aggregate(per_episode)
    roll["submission_id"] = args.submission_id
    roll["team_name"] = team_name
    roll["elapsed_s"] = round(time.perf_counter() - t0, 1)
    (out_dir / "roll-up.json").write_text(json.dumps(roll, indent=2) + "\n")
    print()
    print(f"=== ROLL-UP submission {args.submission_id} ===")
    print(f"  episodes={roll['n_episodes']}  by_result={roll['by_result']}")
    print(f"  by_size_result={roll['by_size_result']}")
    print(f"  action_match_rate={roll['action_match_rate']}")
    print(f"  candidates/turn  snipe={roll['candidates_per_turn']['snipe']:.1f}  "
          f"reinforce={roll['candidates_per_turn']['reinforce']:.1f}")
    print(f"  settled/turn={roll['settled_per_turn']:.2f}  "
          f"idle_source_rate={roll['idle_source_rate']*100:.1f}%")
    print(f"  drops/turn = {roll['drops_per_turn']}")
    print(f"  fleet_outcomes = {roll['fleet_outcomes_total']}")
    print(f"  agent dt p95={roll['dt_ms_p95']}ms  p99={roll['dt_ms_p99']}ms")
    print(f"  elapsed: {roll['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
