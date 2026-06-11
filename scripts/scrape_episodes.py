"""Scrape Orbit Wars ladder episodes: metadata crawl + replay download.

The Kaggle episode service lets anyone list the episodes of any submission
(`ListEpisodes` with {"submissionId": N}) and download any episode replay
(the same endpoint the `kaggle competitions replay` CLI uses). Each episode
record carries every seat's submissionId / teamId / rating at game time, and
each response carries every referenced team's `publicLeaderboardSubmissionId`
(their current best bot). So starting from OUR submission ids we can walk the
match graph straight up the rating ladder to the top teams without touching
the 19 GB Meta Kaggle dump.

Usage:
  python scripts/scrape_episodes.py crawl --max-queries 200
  python scripts/scrape_episodes.py download --min-score 1350 --max-replays 800
  python scripts/scrape_episodes.py stats

State (data/external/):
  crawl_state.json    visited submission ids + team/submission metadata
  episodes.jsonl      one line per unique episode (agents, scores, time)
  replays/            episode-<id>-replay.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXT = os.path.join(BASE, "data", "external")
REPLAY_DIR = os.path.join(EXT, "replays")
STATE_PATH = os.path.join(EXT, "crawl_state.json")
EPISODES_PATH = os.path.join(EXT, "episodes.jsonl")

LIST_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
REPLAY_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/GetEpisodeReplay"

# Our own submissions (seed the crawl; harvested from `kaggle competitions
# submissions orbit-wars`).
SEED_SUBMISSIONS = [53564198, 53558897, 53556728, 53547475, 53542171, 53529884]

SLEEP_S = 0.6  # politeness: single-threaded, ~1.6 req/s max


def _post(url: str, payload: dict, timeout: int = 40):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"visited": {}, "teams": {}, "submissions": {}, "episode_ids": []}


def _save_state(state: dict) -> None:
    os.makedirs(EXT, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)


def _append_episodes(rows: list) -> None:
    os.makedirs(EXT, exist_ok=True)
    with open(EPISODES_PATH, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _known_episode_ids() -> set:
    ids = set()
    if os.path.exists(EPISODES_PATH):
        with open(EPISODES_PATH) as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    return ids


def crawl(max_queries: int, min_score_frontier: float) -> None:
    state = _load_state()
    visited = state["visited"]            # subId(str) -> query time
    teams = state["teams"]                # teamId(str) -> {name, bestSub}
    subs = state["submissions"]           # subId(str) -> {teamId, maxScore}
    known_eps = _known_episode_ids()

    def note_sub(sid, team_id=None, score=None):
        rec = subs.setdefault(str(sid), {"teamId": team_id, "maxScore": 0.0})
        if team_id is not None:
            rec["teamId"] = team_id
        if score is not None and score > rec.get("maxScore", 0.0):
            rec["maxScore"] = score

    for sid in SEED_SUBMISSIONS:
        # seeds query first regardless of the frontier floor
        if str(sid) not in visited:
            note_sub(sid, score=99999.0)

    queries = 0
    while queries < max_queries:
        # frontier: unvisited submission with the highest known score
        frontier = [
            (rec.get("maxScore", 0.0), sid) for sid, rec in subs.items()
            if sid not in visited and rec.get("maxScore", 0.0) >= min_score_frontier
        ]
        if not frontier:
            print("frontier empty, stopping")
            break
        frontier.sort(reverse=True)
        score, sid = frontier[0]
        try:
            d = _post(LIST_URL, {"submissionId": int(sid)})
        except Exception as e:
            if "429" in str(e):
                # rate limited: back off hard, do NOT mark visited
                print(f"  429 on {sid}, backing off 60s")
                time.sleep(60)
                continue
            print(f"  query {sid} FAILED: {e}")
            visited[sid] = {"t": time.time(), "error": str(e)}
            _save_state(state)
            time.sleep(SLEEP_S * 4)
            continue
        visited[sid] = {"t": time.time()}
        queries += 1

        new_rows = []
        for ep in d.get("episodes", []):
            for a in ep.get("agents", []):
                asid = a.get("submissionId")
                if asid:
                    note_sub(asid, a.get("teamId"),
                             max(a.get("initialScore") or 0.0,
                                 a.get("updatedScore") or 0.0))
            if ep["id"] not in known_eps:
                known_eps.add(ep["id"])
                new_rows.append({
                    "id": ep["id"],
                    "createTime": ep.get("createTime"),
                    "agents": [
                        {"submissionId": a.get("submissionId"),
                         "teamId": a.get("teamId"),
                         "index": a.get("index", 0),
                         "reward": a.get("reward"),
                         "score": a.get("updatedScore") or a.get("initialScore")}
                        for a in ep.get("agents", [])
                    ],
                })
        for t in d.get("teams", []) or []:
            tid = str(t.get("id"))
            teams[tid] = {"name": t.get("teamName"),
                          "bestSub": t.get("publicLeaderboardSubmissionId")}
            if t.get("publicLeaderboardSubmissionId"):
                # a team's current best bot is a prime crawl target
                note_sub(t["publicLeaderboardSubmissionId"], t.get("id"))
        for s in d.get("submissions", []) or []:
            if s.get("status") and s["status"] != "COMPLETE":
                # dead submissions list no useful episodes
                visited.setdefault(str(s["id"]), {"t": time.time(),
                                                  "skip": s["status"]})

        _append_episodes(new_rows)
        _save_state(state)
        top = max((r.get("maxScore", 0) for r in subs.values()), default=0)
        print(f"[{queries}/{max_queries}] sub {sid} (score {score:.0f}): "
              f"+{len(new_rows)} eps, {len(subs)} subs known, best seen {top:.0f}")
        time.sleep(SLEEP_S)

    _save_state(state)
    print(f"done: {len(_known_episode_ids())} episodes, {len(subs)} submissions, "
          f"{len(teams)} teams")


def download(min_score: float, max_replays: int, min_steps_hint: int = 0) -> None:
    os.makedirs(REPLAY_DIR, exist_ok=True)
    have = {f for f in os.listdir(REPLAY_DIR)}
    rows = []
    with open(EPISODES_PATH) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    # strongest-agent-in-episode score, newest first as tiebreak
    def key(r):
        return (max((a.get("score") or 0.0) for a in r["agents"]),
                r.get("createTime") or "")
    rows.sort(key=key, reverse=True)
    got = sum(1 for r in rows if f"episode-{r['id']}-replay.json" in have)
    print(f"{len(rows)} episodes known, {got} already downloaded")
    n = 0
    for r in rows:
        if n >= max_replays:
            break
        best = max((a.get("score") or 0.0) for a in r["agents"])
        if best < min_score:
            continue
        fname = f"episode-{r['id']}-replay.json"
        if fname in have:
            continue
        try:
            # the CLI's replay endpoint is authenticated; reuse it
            import subprocess
            res = subprocess.run(
                ["kaggle", "competitions", "replay", str(r["id"]),
                 "-p", REPLAY_DIR],
                capture_output=True, text=True, timeout=120)
            path = os.path.join(REPLAY_DIR, fname)
            out = (res.stderr or "") + (res.stdout or "")
            if "429" in out or "Too Many Requests" in out:
                print("  429 on replay download, backing off 60s")
                time.sleep(60)
                continue
            if res.returncode != 0 or not os.path.exists(path):
                raise ValueError(out.strip()[:200])
            n += 1
            if n % 25 == 0:
                print(f"  {n} replays downloaded (latest ep {r['id']}, best {best:.0f})")
        except Exception as e:
            print(f"  ep {r['id']} FAILED: {e}")
            time.sleep(SLEEP_S * 4)
        time.sleep(SLEEP_S)
    print(f"downloaded {n} new replays into {REPLAY_DIR}")


def stats() -> None:
    state = _load_state()
    subs = state["submissions"]
    rows = []
    if os.path.exists(EPISODES_PATH):
        with open(EPISODES_PATH) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    print(f"submissions known: {len(subs)}, episodes known: {len(rows)}")
    by_team = {}
    for sid, rec in subs.items():
        t = str(rec.get("teamId"))
        if rec.get("maxScore", 0) > by_team.get(t, (0, None))[0]:
            by_team[t] = (rec["maxScore"], sid)
    teams = state.get("teams", {})
    top = sorted(by_team.items(), key=lambda kv: -kv[1][0])[:25]
    for tid, (score, sid) in top:
        name = (teams.get(tid) or {}).get("name", "?")
        print(f"  {score:7.1f}  team {name} (sub {sid})")
    if rows:
        import collections
        band = collections.Counter()
        for r in rows:
            best = max((a.get("score") or 0.0) for a in r["agents"])
            band[int(best // 100) * 100] += 1
        print("episodes by strongest-agent score band:")
        for b in sorted(band, reverse=True):
            print(f"  {b}-{b+99}: {band[b]}")
    n_replays = len(os.listdir(REPLAY_DIR)) if os.path.isdir(REPLAY_DIR) else 0
    print(f"replays on disk: {n_replays}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crawl")
    c.add_argument("--max-queries", type=int, default=200)
    c.add_argument("--min-score-frontier", type=float, default=900.0,
                   help="don't bother querying submissions below this rating")
    d = sub.add_parser("download")
    d.add_argument("--min-score", type=float, default=1350.0)
    d.add_argument("--max-replays", type=int, default=800)
    sub.add_parser("stats")
    args = ap.parse_args()
    if args.cmd == "crawl":
        crawl(args.max_queries, args.min_score_frontier)
    elif args.cmd == "download":
        download(args.min_score, args.max_replays)
    else:
        stats()


if __name__ == "__main__":
    main()
