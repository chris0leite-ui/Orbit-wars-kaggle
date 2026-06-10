"""crawl_top_replays.py — climb the public episode graph to the ladder's top.

Kaggle's EpisodeService accepts {"submissionId": X} unauthenticated and
returns episodes whose agents carry (submissionId, teamId, updatedScore).
Starting from our own submissions, repeatedly hop to the highest-rated
opponent submissions seen so far ("ladder climbing"): after a few hops the
frontier reaches the global top teams. Then download replays for the best
teams' episodes via `kaggle competitions replay`.

Usage:
    python scripts/crawl_top_replays.py --seed-subs 53527125 53523036 \
        --hops 6 --out audit/top-replays
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LIST_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"


def list_episodes(submission_id: int) -> list[dict]:
    req = urllib.request.Request(
        LIST_URL,
        data=json.dumps({"submissionId": int(submission_id)}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r).get("episodes", [])
        except Exception as e:
            if attempt == 2:
                print(f"  WARN list({submission_id}): {type(e).__name__} {str(e)[:120]}",
                      file=sys.stderr)
                return []
            time.sleep(2 * (attempt + 1))
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-subs", type=int, nargs="+", required=True)
    ap.add_argument("--hops", type=int, default=6)
    ap.add_argument("--frontier-width", type=int, default=3,
                    help="how many top unexplored submissions to expand per hop")
    ap.add_argument("--out", default="audit/top-replays")
    ap.add_argument("--top-teams", type=int, default=3,
                    help="download replays for this many top teams at the end")
    ap.add_argument("--replays-per-team", type=int, default=40)
    args = ap.parse_args()

    # sub_id -> {"score": float, "teamId": int, "explored": bool}
    seen: dict[int, dict] = {}
    episodes_by_sub: dict[int, list[dict]] = {}

    def ingest(eps: list[dict]):
        for ep in eps:
            for ag in ep.get("agents", []):
                sid = ag.get("submissionId")
                if sid is None:
                    continue
                score = ag.get("updatedScore") or ag.get("initialScore") or 0.0
                cur = seen.setdefault(
                    sid, {"score": float(score), "teamId": ag.get("teamId"),
                          "explored": False},
                )
                cur["score"] = max(cur["score"], float(score))

    for s in args.seed_subs:
        eps = list_episodes(s)
        episodes_by_sub[s] = eps
        seen.setdefault(s, {"score": 0.0, "teamId": None, "explored": True})
        seen[s]["explored"] = True
        ingest(eps)
        print(f"seed {s}: {len(eps)} episodes, frontier {len(seen)} subs")

    for hop in range(args.hops):
        frontier = sorted(
            (sid for sid, m in seen.items() if not m["explored"]),
            key=lambda sid: -seen[sid]["score"],
        )[: args.frontier_width]
        if not frontier:
            break
        for sid in frontier:
            seen[sid]["explored"] = True
            eps = list_episodes(sid)
            episodes_by_sub[sid] = eps
            ingest(eps)
            time.sleep(0.5)
        best = max(seen.values(), key=lambda m: m["score"])
        print(f"hop {hop+1}: explored {frontier} -> known subs {len(seen)}, "
              f"best score seen {best['score']:.0f} (team {best['teamId']})")

    # rank teams by best submission score
    team_best: dict[int, tuple[float, int]] = {}
    for sid, m in seen.items():
        t = m["teamId"]
        if t is None:
            continue
        if t not in team_best or m["score"] > team_best[t][0]:
            team_best[t] = (m["score"], sid)
    ranked = sorted(team_best.items(), key=lambda kv: -kv[1][0])
    print("\ntop teams discovered:")
    for t, (sc, sid) in ranked[:10]:
        print(f"  team {t}: best sub {sid} score {sc:.0f}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"teams": []}
    for t, (sc, sid) in ranked[: args.top_teams]:
        eps = episodes_by_sub.get(sid) or list_episodes(sid)
        eps = [e for e in eps if e.get("state") == "COMPLETED"]
        tdir = out_root / f"team-{t}-sub-{sid}"
        tdir.mkdir(exist_ok=True)
        n = 0
        for ep in eps[: args.replays_per_team]:
            eid = ep["id"]
            outf = tdir / f"episode-{eid}-replay.json"
            if outf.is_file():
                n += 1
                continue
            proc = subprocess.run(
                ["kaggle", "competitions", "replay", str(eid), "-p", str(tdir), "-q"],
                capture_output=True, text=True,
            )
            if proc.returncode == 0:
                n += 1
            else:
                print(f"  WARN replay {eid}: {proc.stderr.strip()[:100]}",
                      file=sys.stderr)
            time.sleep(0.3)
        manifest["teams"].append(
            {"teamId": t, "submissionId": sid, "score": sc, "replays": n,
             "dir": str(tdir)})
        print(f"team {t} (score {sc:.0f}): {n} replays in {tdir}")
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
