"""Convert kaggle_environments replays → flat fingerprint format → 15-feature matrix.

KE replay format:
  d["steps"][step][player_idx] = {"observation": {...}, "action": [...], "reward", "status"}
  obs has "planets", "fleets", "comets", "step", "player", ...
  action is a list of [from_pid, angle, ships, ...] (we only use first 3 fields)

Maps to lib/fingerprint.py expected schema:
  {"steps": [{"planets": [[id, owner, x, y, radius, ships, prod], ...],
              "fleets":  [[id, owner, x, y, angle, from_pid, ships], ...],
              "action_p<i>": [[from_pid, angle, ships], ...]}, ...]}

Output:
  audit/2026-05-11-top-performer-fingerprints.json
  {"rows": [{"label": team, "rank": rank_or_None, "seat": int,
             "size": 2or4, "won": bool, "K": prefix_turns,
             "features": [f1..f15], "episode": str, "src": filename}, ...],
   "feature_names": [...]}

CLI:
  python3 scripts/fingerprint_external.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Make lib importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.fingerprint import FEATURE_NAMES, fingerprint  # noqa: E402

REPLAY_DIR = Path("audit/external/replays")
OUT_PATH = Path("audit/2026-05-11-top-performer-fingerprints.json")
PREFIX_TURNS = 100  # K used by the existing manifold_check + meta-router


def ke_to_flat(replay: dict, focal_idx: int) -> dict:
    """Convert one KE replay → flat fingerprint-format replay for `focal_idx`.

    The fingerprint module expects 2-player schema (player_id in {0, 1}),
    so we relabel: focal_idx → 0, "everyone else's actions" lumped to 1.
    """
    steps_out = []
    for step_idx, step in enumerate(replay.get("steps", [])):
        # Each step is list of per-player {observation, action, reward, status}.
        # We need obs.planets and obs.fleets — take from any player (they're
        # global). Use focal_idx if present, else 0.
        if focal_idx < len(step):
            obs = step[focal_idx].get("observation", {})
        else:
            obs = step[0].get("observation", {})
        planets = obs.get("planets", []) or []
        fleets = obs.get("fleets", []) or []

        # Action: focal → action_p0, every non-focal action lumped into action_p1
        focal_action = step[focal_idx].get("action") if focal_idx < len(step) else []
        focal_action = focal_action or []
        # Normalise to first-3 fields
        focal_action_norm = []
        for a in focal_action:
            if a and len(a) >= 3:
                focal_action_norm.append([a[0], a[1], a[2]])
        others_action_norm = []
        for j, pstate in enumerate(step):
            if j == focal_idx:
                continue
            a_list = pstate.get("action") or []
            for a in a_list:
                if a and len(a) >= 3:
                    others_action_norm.append([a[0], a[1], a[2]])

        steps_out.append(
            {
                "planets": planets,
                "fleets": fleets,
                "action_p0": focal_action_norm,
                "action_p1": others_action_norm,
            }
        )
    return {
        "seed": replay.get("info", {}).get("seed", 0),
        "agent_p0": "focal",
        "agent_p1": "rest",
        "n_steps": len(steps_out),
        "rewards": replay.get("rewards", []),
        "statuses": replay.get("statuses", []),
        "steps": steps_out,
    }


# Filename convention: r<rank>-<team>-<2P|4P>-<W|L>-<episode>.json
# midpack-<...>-<size>P-<episode>.json
FNAME_RE = re.compile(r"^r(?P<rank>\d{2})-(?P<team>.+?)-(?P<size>[24]P)-(?P<wl>[WL])-(?P<eid>\d+)\.json$")
MIDPACK_RE = re.compile(r"^midpack-.*?-(?P<size>[24]P)-(?P<eid>\d+)\.json$")


def main():
    rows = []
    files = sorted(REPLAY_DIR.glob("*.json"))
    print(f"Found {len(files)} replay files.")
    for fpath in files:
        m = FNAME_RE.match(fpath.name) or MIDPACK_RE.match(fpath.name)
        if not m:
            if not fpath.name.startswith("test-"):
                print(f"  SKIP unparsable: {fpath.name}")
            continue
        try:
            replay = json.loads(fpath.read_text())
        except Exception as e:
            print(f"  ERR loading {fpath.name}: {e}")
            continue

        info = replay.get("info", {})
        team_names = info.get("TeamNames", [])
        rewards = replay.get("rewards", [])

        is_midpack = fpath.name.startswith("midpack-")
        if is_midpack:
            rank = None
            label = "midpack"
            # focal: any seat held by ChrisLeiteScha (our v2). Take the first.
            focal_idx = next((i for i, t in enumerate(team_names)
                              if t == "ChrisLeiteScha"), 0)
            focal_team = "ChrisLeiteScha"
        else:
            d = m.groupdict()
            rank = int(d["rank"])
            label = d["team"].replace("_", " ").replace("at ", "@ ").replace(" lookaside ", "lookaside")
            # Find focal seat: team name match (try with underscores → spaces)
            label_variants = {label, label.replace("@ ", "@ ")}
            focal_idx = None
            for i, t in enumerate(team_names):
                if t in label_variants or t.replace(" ", "_") in {fpath.name.split("-")[1]}:
                    focal_idx = i
                    break
            if focal_idx is None:
                # Try fuzzy: contains the team token
                team_token = fpath.name.split("-")[1]
                for i, t in enumerate(team_names):
                    if team_token.lower() in t.lower().replace(" ", "_"):
                        focal_idx = i
                        break
            if focal_idx is None:
                print(f"  WARN no focal match for {fpath.name}, team_names={team_names}")
                focal_idx = 0
            focal_team = team_names[focal_idx]

        size = int(m.group("size")[0])
        won = focal_idx < len(rewards) and rewards[focal_idx] == max(rewards)
        eid = m.group("eid")

        flat = ke_to_flat(replay, focal_idx)
        try:
            feats = fingerprint(flat, player_id=0, prefix_turns=PREFIX_TURNS)
        except Exception as e:
            print(f"  ERR fingerprint {fpath.name}: {e}")
            continue

        rows.append(
            {
                "label": focal_team,
                "filename_team": label,
                "rank": rank,
                "seat": focal_idx,
                "size": size,
                "won": won,
                "K": PREFIX_TURNS,
                "n_steps_total": len(flat["steps"]),
                "episode_id": eid,
                "src": fpath.name,
                "rewards": rewards,
                "team_names": team_names,
                "features": feats.tolist(),
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "feature_names": FEATURE_NAMES,
        "prefix_turns": PREFIX_TURNS,
        "n_rows": len(rows),
        "rows": rows,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH}: {len(rows)} rows × {len(FEATURE_NAMES)} features.")
    # Quick per-rank summary
    by_rank: dict = {}
    for r in rows:
        key = r["rank"] if r["rank"] else "midpack"
        by_rank.setdefault(key, []).append(r)
    for k in sorted(by_rank, key=lambda x: (x == "midpack", x)):
        print(f"  rank={k}: {len(by_rank[k])} rows, "
              f"sizes={sorted(set(r['size'] for r in by_rank[k]))}")


if __name__ == "__main__":
    main()
