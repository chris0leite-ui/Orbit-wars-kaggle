"""Per-team per-archetype winrate gap analysis.

Walks two replay corpora:
  - top-10:  audit/external/replays/r<rank>-<team>-<size>-<W|L>-<eid>.json
  - ours:    audit/replays/live/52710995/episode-<eid>-replay.json

For each replay we extract:
  - focal seat (filename for top-10 corpus; team-name match for ours)
  - won/lost (filename W/L for top-10; reward sign for ours)
  - archetype (via lib.archetype_binning.archetype_of_replay on turn-0 obs)

Then we tally per-archetype winrate for our submission and for the
top-10 mass aggregate, and compute the GAP = top10_winrate - our_winrate.

Output:
  audit/2026-05-18-team-archetype-gap.json
  audit/2026-05-18-team-archetype-gap.md
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.archetype_binning import archetype_of_replay
from lib.archetype_strategy import ARCHETYPES, KNOWN_REGRESSIONS

TOP10_DIR = REPO / "audit" / "external" / "replays"
OUR_DIR = REPO / "audit" / "replays" / "live" / "52710995"
OWN_TEAM_NAMES = {"ChrisLeiteScha", "Chris Leite Scha"}

# Top-10 filenames look like:  r<rank>-<team>-<size>-<W|L>-<eid>.json
TOP10_RE = re.compile(
    r"^r(?P<rank>\d{2})-(?P<team>.+?)-(?P<size>[24]P)-(?P<wl>[WL])-(?P<eid>\d+)\.json$"
)


def _wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - spread) / denom, (centre + spread) / denom


def _load_top10() -> list[dict]:
    """Returns list of {team, rank, size, won, archetype, episode_id, seat}."""
    out = []
    for fp in sorted(TOP10_DIR.glob("*.json")):
        m = TOP10_RE.match(fp.name)
        if not m:
            continue
        size = int(m.group("size")[0])  # 2 or 4
        rep = json.loads(fp.read_text())
        # Focal seat: read rewards + team names to find which seat = top-10 team
        team_name = m.group("team")
        info = rep.get("info") or {}
        team_names = info.get("TeamNames") or info.get("teamNames") or []
        focal_idx = 0
        for i, t in enumerate(team_names):
            if t.replace(" ", "").lower() == team_name.replace("_", "").lower():
                focal_idx = i
                break
        try:
            arch = archetype_of_replay(rep, focal_idx=focal_idx)
        except Exception as e:
            print(f"  skip {fp.name}: {e}")
            continue
        out.append({
            "team": team_name,
            "rank": int(m.group("rank")),
            "size": size,
            "won": m.group("wl") == "W",
            "archetype": arch,
            "episode_id": m.group("eid"),
            "seat": focal_idx,
        })
    return out


def _load_ours() -> list[dict]:
    """Walk our submission 52710995 replays. won = focal reward > opp reward."""
    out = []
    for fp in sorted(OUR_DIR.glob("episode-*-replay.json")):
        rep = json.loads(fp.read_text())
        info = rep.get("info") or {}
        team_names = info.get("TeamNames") or info.get("teamNames") or []
        focal_idx = None
        for i, t in enumerate(team_names):
            if t in OWN_TEAM_NAMES:
                focal_idx = i
                break
        if focal_idx is None:
            continue
        size = len(team_names)
        rewards = rep.get("rewards") or []
        if focal_idx >= len(rewards) or rewards[focal_idx] is None:
            continue
        # Won iff focal reward strictly greater than every other seat's reward
        focal_r = rewards[focal_idx]
        won = all(
            (r is None) or (r < focal_r)
            for i, r in enumerate(rewards) if i != focal_idx
        )
        try:
            arch = archetype_of_replay(rep, focal_idx=focal_idx)
        except Exception as e:
            print(f"  skip {fp.name}: {e}")
            continue
        out.append({
            "team": "us(52710995)",
            "rank": None,
            "size": size,
            "won": won,
            "archetype": arch,
            "episode_id": fp.stem.removeprefix("episode-").removesuffix("-replay"),
            "seat": focal_idx,
        })
    return out


def _agg(rows: list[dict]) -> dict[str, tuple[int, int]]:
    """archetype -> (wins, n) aggregated across all rows."""
    bucket: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        bucket[r["archetype"]].append(r["won"])
    return {a: (sum(b), len(b)) for a, b in bucket.items()}


def main() -> int:
    print(f"Loading top-10 corpus from {TOP10_DIR} ...")
    top10 = _load_top10()
    print(f"  {len(top10)} replays")
    print(f"Loading our corpus from {OUR_DIR} ...")
    ours = _load_ours()
    print(f"  {len(ours)} replays")

    # Split by size — 2P and 4P play very differently, panel is 2P-only.
    top10_2p = [r for r in top10 if r["size"] == 2]
    ours_2p = [r for r in ours if r["size"] == 2]
    top10_4p = [r for r in top10 if r["size"] == 4]
    ours_4p = [r for r in ours if r["size"] == 4]
    print(f"2P split: top10={len(top10_2p)}, ours={len(ours_2p)}")
    print(f"4P split: top10={len(top10_4p)}, ours={len(ours_4p)}\n")

    # Focus on 2P (panel is 2P)
    t10 = _agg(top10_2p)
    us = _agg(ours_2p)

    print(f"=== Per-archetype gap (2P) ===")
    print(f"{'archetype':<55s} {'top10':>10s} {'ours':>10s} {'gap':>8s}")
    print("-" * 90)
    rows = []
    for arch in ARCHETYPES:
        tw, tn = t10.get(arch, (0, 0))
        uw, un = us.get(arch, (0, 0))
        t_rate = tw / tn if tn else None
        u_rate = uw / un if un else None
        gap = (t_rate - u_rate) if (t_rate is not None and u_rate is not None) else None
        rows.append({
            "archetype": arch,
            "top10_wins": tw, "top10_n": tn, "top10_rate": t_rate,
            "ours_wins": uw, "ours_n": un, "ours_rate": u_rate,
            "gap": gap,
            "known_regression": arch in KNOWN_REGRESSIONS,
        })

    # Sort: largest gap first (None gap pushed to bottom)
    rows.sort(key=lambda r: (-1e9 if r["gap"] is None else -r["gap"]))

    for r in rows:
        t_str = f"{r['top10_wins']}/{r['top10_n']}" if r["top10_n"] else "  -"
        u_str = f"{r['ours_wins']}/{r['ours_n']}" if r["ours_n"] else "  -"
        gap_str = f"{r['gap']:+.0%}" if r["gap"] is not None else "  -"
        flag = " <reg>" if r["known_regression"] else ""
        print(f"{r['archetype']:<55s} {t_str:>10s} {u_str:>10s} {gap_str:>8s}{flag}")

    # Cells with biggest INFORMATIVE gap (both corpora have samples)
    informative = [r for r in rows if r["gap"] is not None and r["ours_n"] >= 2]
    print(f"\n=== Top-10 gap-cells (informative only, n_ours >= 2) ===")
    for r in informative[:10]:
        print(f"  {r['archetype']}: top10 {r['top10_wins']}/{r['top10_n']} = {r['top10_rate']:.0%}, "
              f"ours {r['ours_wins']}/{r['ours_n']} = {r['ours_rate']:.0%}, "
              f"gap={r['gap']:+.0%}")

    # Persist
    out_json = REPO / "audit" / "2026-05-18-team-archetype-gap.json"
    out_json.write_text(json.dumps({
        "top10_2p_n": len(top10_2p),
        "ours_2p_n": len(ours_2p),
        "top10_4p_n": len(top10_4p),
        "ours_4p_n": len(ours_4p),
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {out_json}")

    # Markdown report
    out_md = REPO / "audit" / "2026-05-18-team-archetype-gap.md"
    md = ["# Per-archetype winrate gap: top-10 vs our submission 52710995",
          "",
          f"Corpora: top-10 = {len(top10_2p)} 2P + {len(top10_4p)} 4P replays "
          f"(from 2026-05-11 curation); ours = {len(ours_2p)} 2P + {len(ours_4p)} 4P "
          "from submission 52710995 ladder games.",
          "",
          "All top-10 corpus entries are WINS by construction (5 wins per team curated). "
          "Their per-archetype winrate is therefore 100% wherever they have samples; "
          "the analytical signal is **which archetypes top-10 plays + wins on** vs "
          "**which our submission wins**. A high gap means top-10 wins in cells we "
          "lose — IL target.",
          "",
          "## Ranked gap table (2P, sorted by gap)",
          "",
          "| archetype | top-10 W/n | ours W/n | gap |",
          "|---|---|---|---|"]
    for r in rows:
        t_str = f"{r['top10_wins']}/{r['top10_n']}" if r["top10_n"] else "—"
        u_str = f"{r['ours_wins']}/{r['ours_n']}" if r["ours_n"] else "—"
        g_str = f"{r['gap']:+.0%}" if r["gap"] is not None else "—"
        suffix = " ⚠" if r["known_regression"] else ""
        md.append(f"| `{r['archetype']}`{suffix} | {t_str} | {u_str} | {g_str} |")
    md += ["", "⚠ = baseline known-regression archetype from the 2026-05-18 A/B vs v7_0."]
    out_md.write_text("\n".join(md) + "\n")
    print(f"wrote {out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
