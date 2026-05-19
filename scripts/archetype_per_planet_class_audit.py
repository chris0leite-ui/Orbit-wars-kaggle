"""Per-archetype × per-planet-class behavioural rollup: top-10 vs ours.

For each (archetype, planet-class) cell, computes how often top-10
launches a fleet AT a planet of that class versus how often we do, plus
when/whether each side ends up owning a planet of that class. The
headline metric is ``target_intensity_delta`` (top-10 minus ours): a
class with positive delta is a planet-type top-10 prizes and we ignore.

Outputs:
  audit/2026-05-19-archetype-per-planet-class.json
  audit/2026-05-19-archetype-per-planet-class.md

Usage:
  python scripts/archetype_per_planet_class_audit.py \
      --archetype med_high_prod__mixed_static__big_static
  python scripts/archetype_per_planet_class_audit.py --all-cells
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.archetype_binning import archetype_of_replay
from lib.per_planet_class import (
    ALL_CLASS_LABELS,
    aggregate_breakdowns,
    per_planet_breakdown,
)
from scripts.fingerprint_external import ke_to_flat


TOP10_DIR = REPO / "audit" / "external" / "replays"
OUR_DIR = REPO / "audit" / "replays" / "live" / "52710995"
OWN_TEAM_NAMES = {"ChrisLeiteScha", "Chris Leite Scha"}

TOP10_RE = re.compile(
    r"^r(?P<rank>\d{2})-(?P<team>.+?)-(?P<size>[24]P)-(?P<wl>[WL])-(?P<eid>\d+)\.json$"
)

GAP_PATH = REPO / "audit" / "2026-05-18-team-archetype-gap.json"
INTENSITY_DELTA_THRESHOLD = 0.15  # |delta| above this is flagged in the cross-cell summary


def _focal_of_top10(rep: dict, team_name: str) -> int:
    info = rep.get("info") or {}
    team_names = info.get("TeamNames") or info.get("teamNames") or []
    for i, t in enumerate(team_names):
        if t.replace(" ", "").lower() == team_name.replace("_", "").lower():
            return i
    return 0


def _focal_of_ours(rep: dict) -> int | None:
    info = rep.get("info") or {}
    team_names = info.get("TeamNames") or info.get("teamNames") or []
    for i, t in enumerate(team_names):
        if t in OWN_TEAM_NAMES:
            return i
    return None


_PRELOAD_CACHE: dict | None = None


def _preload_all(prefix_turns: int) -> dict[str, dict[str, list[dict]]]:
    """Walk both corpora once; build per-replay class breakdowns keyed by archetype.

    Returns ``{archetype: {"top10": [breakdown, ...], "ours": [breakdown, ...]}}``.
    Memoised at module level keyed on ``prefix_turns``.
    """
    global _PRELOAD_CACHE
    if _PRELOAD_CACHE is not None and _PRELOAD_CACHE.get("prefix_turns") == prefix_turns:
        return _PRELOAD_CACHE["by_arch"]

    by_arch: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"top10": [], "ours": []}
    )

    for fp in sorted(TOP10_DIR.glob("*.json")):
        m = TOP10_RE.match(fp.name)
        if not m or m.group("size") != "2P" or m.group("wl") != "W":
            continue
        try:
            rep = json.loads(fp.read_text())
        except Exception:
            continue
        focal_idx = _focal_of_top10(rep, m.group("team"))
        try:
            arch = archetype_of_replay(rep, focal_idx=focal_idx)
        except Exception:
            continue
        flat = ke_to_flat(rep, focal_idx)
        by_arch[arch]["top10"].append(
            per_planet_breakdown(flat, player_id=0, prefix_turns=prefix_turns)
        )

    for fp in sorted(OUR_DIR.glob("episode-*-replay.json")):
        try:
            rep = json.loads(fp.read_text())
        except Exception:
            continue
        focal_idx = _focal_of_ours(rep)
        if focal_idx is None:
            continue
        info = rep.get("info", {})
        team_names = info.get("TeamNames") or info.get("teamNames") or []
        if len(team_names) != 2:
            continue
        try:
            arch = archetype_of_replay(rep, focal_idx=focal_idx)
        except Exception:
            continue
        flat = ke_to_flat(rep, focal_idx)
        by_arch[arch]["ours"].append(
            per_planet_breakdown(flat, player_id=0, prefix_turns=prefix_turns)
        )

    _PRELOAD_CACHE = {"prefix_turns": prefix_turns, "by_arch": dict(by_arch)}
    return _PRELOAD_CACHE["by_arch"]


def _run_one(target_arch: str, prefix_turns: int, quiet: bool = False) -> dict | None:
    cache = _preload_all(prefix_turns)
    bucket = cache.get(target_arch, {"top10": [], "ours": []})
    n_top10 = len(bucket["top10"])
    n_ours = len(bucket["ours"])
    if n_top10 == 0 or n_ours == 0:
        if not quiet:
            print(f"  SKIP {target_arch}: top10_n={n_top10} ours_n={n_ours}")
        return None

    top10 = aggregate_breakdowns(bucket["top10"], prefix_turns)
    ours = aggregate_breakdowns(bucket["ours"], prefix_turns)

    rows = []
    for c in ALL_CLASS_LABELS:
        t = top10[c]
        o = ours[c]
        rows.append({
            "class": c,
            "n_planets_per_game": t["n_planets_per_game"],
            "home_per_game_top10": t["home_per_game"],
            "home_per_game_ours": o["home_per_game"],
            "target_count_per_game_top10": t["target_count_per_game"],
            "target_count_per_game_ours": o["target_count_per_game"],
            "target_intensity_top10": t["target_intensity"],
            "target_intensity_ours": o["target_intensity"],
            "target_intensity_delta": t["target_intensity"] - o["target_intensity"],
            "target_share_top10": t["target_share"],
            "target_share_ours": o["target_share"],
            "target_share_delta": t["target_share"] - o["target_share"],
            "first_capture_turn_top10": t["first_capture_turn"],
            "first_capture_turn_ours": o["first_capture_turn"],
            "end_owned_rate_top10": t["end_owned_rate"],
            "end_owned_rate_ours": o["end_owned_rate"],
        })

    return {
        "archetype": target_arch,
        "n": {"top10": n_top10, "ours": n_ours},
        "rows": rows,
    }


def _print_one(result: dict) -> None:
    arch = result["archetype"]
    n_top10 = result["n"]["top10"]
    n_ours = result["n"]["ours"]
    print(f"\n=== {arch} === (top10_n={n_top10}, ours_n={n_ours})")
    print(f"{'class':<26s}  {'n_pl':>5s}  "
          f"{'int_top10':>10s}  {'int_ours':>10s}  {'Δintensity':>10s}  "
          f"{'share_t10':>10s}  {'share_ours':>10s}  {'Δshare':>9s}  "
          f"{'end_t10':>8s}  {'end_ours':>8s}")
    print("-" * 120)
    ranked = sorted(result["rows"], key=lambda r: -r["target_share_delta"])
    for r in ranked:
        print(
            f"  {r['class']:<24s}  "
            f"{r['n_planets_per_game']:>5.2f}  "
            f"{r['target_intensity_top10']:>10.3f}  "
            f"{r['target_intensity_ours']:>10.3f}  "
            f"{r['target_intensity_delta']:>+10.3f}  "
            f"{r['target_share_top10']:>10.3f}  "
            f"{r['target_share_ours']:>10.3f}  "
            f"{r['target_share_delta']:>+9.3f}  "
            f"{r['end_owned_rate_top10']:>8.2f}  "
            f"{r['end_owned_rate_ours']:>8.2f}"
        )


def _render_markdown(results: list[dict], cross_summary: dict, prefix_turns: int) -> str:
    md: list[str] = []
    md.append("# Archetype × planet-class rollup audit")
    md.append("")
    md.append(
        "Per-archetype, per-planet-class behaviour comparison: top-10 winning "
        "play vs our submission 52710995 across the first "
        f"{prefix_turns} turns of every 2P game."
    )
    md.append("")
    md.append(
        "Class label = `{prod}_{kin}_{prox}`: production above/below per-board "
        "median, rotating vs static (`is_orbiting`), inner vs outer (orbital "
        "radius above/below per-board median). 8 classes total."
    )
    md.append("")
    md.append("Headline metric is **`target_intensity_delta`** (top-10 minus ours):")
    md.append("- Strongly positive → top-10 prizes this class, we ignore it.")
    md.append("- Strongly negative → we waste shots on this class, top-10 ignores it.")
    md.append("- Near zero with low end-owned-rates → true filler planets.")
    md.append("")
    md.append(
        "Caveat: the prior aggregate audit established top-10 launches "
        "~2x as often in absolute terms, so most classes show a uniformly "
        "positive intensity delta. **`target_share_delta`** (per-class % of "
        "total launches; top-10 share minus ours share) normalises out the "
        "universal aggression deficit and surfaces class-conditional "
        "allocation differences."
    )
    md.append("")

    # Headline findings: classes ranked by absolute mean share-delta across cells.
    share_ranking = sorted(
        ALL_CLASS_LABELS,
        key=lambda c: -abs(cross_summary[c]["share"]["mean_delta"])
    )
    nontrivial = [c for c in share_ranking if abs(cross_summary[c]["share"]["mean_delta"]) >= 0.01]
    md.append("## Headline findings")
    md.append("")
    md.append(
        "Ranked by |mean share-delta| across all 16 informative cells. Positive = "
        "top-10 over-allocates relative to ours; negative = we over-allocate."
    )
    md.append("")
    for c in nontrivial[:5]:
        s = cross_summary[c]["share"]
        n_pos = len(s["positive_cells"])
        n_neg = len(s["negative_cells"])
        direction = "top-10 prizes" if s["mean_delta"] > 0 else "we over-allocate to"
        md.append(
            f"- **`{c}`**: mean Δshare {s['mean_delta']:+.3f} — {direction} this "
            f"class ({n_pos} cells positive / {n_neg} negative at |Δ| ≥ "
            f"{SHARE_DELTA_THRESHOLD:.2f})."
        )
    md.append("")

    md.append("## Cross-cell summary")
    md.append("")
    md.append("For each class, cells where the |intensity delta| crosses "
              f"{INTENSITY_DELTA_THRESHOLD:.2f}. Direction = sign of (top-10 minus ours).")
    md.append("")
    md.append("**By `target_intensity_delta`** (top-10 minus ours, raw):")
    md.append("")
    md.append("| class | top-10-prizes (positive Δ) | we-overshoot (negative Δ) | mean Δ | n cells |")
    md.append("|---|---|---|---|---|")
    for c in ALL_CLASS_LABELS:
        s = cross_summary[c]["intensity"]
        pos = ", ".join(s["positive_cells"]) or "—"
        neg = ", ".join(s["negative_cells"]) or "—"
        md.append(
            f"| `{c}` | {pos} | {neg} | {s['mean_delta']:+.3f} | {s['n_cells']} |"
        )
    md.append("")
    md.append("**By `target_share_delta`** (class-share of total launches; "
              "removes the universal aggression deficit):")
    md.append("")
    md.append("| class | top-10-prizes (positive Δ) | we-overshoot (negative Δ) | mean Δ | n cells |")
    md.append("|---|---|---|---|---|")
    for c in ALL_CLASS_LABELS:
        s = cross_summary[c]["share"]
        pos = ", ".join(s["positive_cells"]) or "—"
        neg = ", ".join(s["negative_cells"]) or "—"
        md.append(
            f"| `{c}` | {pos} | {neg} | {s['mean_delta']:+.3f} | {s['n_cells']} |"
        )
    md.append("")

    md.append("## Per-cell tables")
    for r in results:
        md.append("")
        md.append(f"### `{r['archetype']}` — top10 n={r['n']['top10']}, ours n={r['n']['ours']}")
        md.append("")
        md.append(
            "| class | n/game | int t10 | int ours | Δ int | share t10 | share ours | Δ share | end t10 | end ours |"
        )
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in sorted(r["rows"], key=lambda x: -x["target_share_delta"]):
            md.append(
                f"| `{row['class']}` "
                f"| {row['n_planets_per_game']:.2f} "
                f"| {row['target_intensity_top10']:.3f} "
                f"| {row['target_intensity_ours']:.3f} "
                f"| {row['target_intensity_delta']:+.3f} "
                f"| {row['target_share_top10']:.3f} "
                f"| {row['target_share_ours']:.3f} "
                f"| {row['target_share_delta']:+.3f} "
                f"| {row['end_owned_rate_top10']:.2f} "
                f"| {row['end_owned_rate_ours']:.2f} |"
            )
    md.append("")
    return "\n".join(md) + "\n"


SHARE_DELTA_THRESHOLD = 0.05  # 5% share gap on a 100% pie


def _build_cross_summary(results: list[dict]) -> dict[str, dict]:
    """Per class: which archetypes show a strong positive / negative delta.

    Tracks both ``target_intensity_delta`` (raw, biased by the universal
    aggression deficit) and ``target_share_delta`` (allocation-normalised).
    """
    def _empty():
        return {"positive_cells": [], "negative_cells": [], "deltas": [], "n_cells": 0}

    summary: dict[str, dict] = {
        c: {"intensity": _empty(), "share": _empty()} for c in ALL_CLASS_LABELS
    }
    for r in results:
        arch = r["archetype"]
        for row in r["rows"]:
            c = row["class"]
            for metric, key, thresh in (
                ("intensity", "target_intensity_delta", INTENSITY_DELTA_THRESHOLD),
                ("share", "target_share_delta", SHARE_DELTA_THRESHOLD),
            ):
                d = row[key]
                s = summary[c][metric]
                s["deltas"].append(d)
                s["n_cells"] += 1
                if d >= thresh:
                    s["positive_cells"].append(arch)
                elif d <= -thresh:
                    s["negative_cells"].append(arch)
    for c, branches in summary.items():
        for s in branches.values():
            s["mean_delta"] = (sum(s["deltas"]) / len(s["deltas"])) if s["deltas"] else 0.0
    return summary


def _run_all_cells(prefix_turns: int) -> int:
    gap_data = json.loads(GAP_PATH.read_text())
    candidates = [r for r in gap_data["rows"] if r["top10_n"] >= 1 and r["ours_n"] >= 1]
    print(f"Loaded {len(candidates)} informative cells from {GAP_PATH.name}\n")

    results: list[dict] = []
    for c in candidates:
        arch = c["archetype"]
        print(f"--> {arch} (gap={c['gap']:+.0%} top10={c['top10_n']} ours={c['ours_n']})")
        res = _run_one(arch, prefix_turns, quiet=True)
        if res is None:
            continue
        res["panel_gap"] = c["gap"]
        results.append(res)

    if not results:
        print("No cells produced a comparison.")
        return 2

    print(f"\nRan rollup on {len(results)} cells; building cross-cell summary...\n")
    cross = _build_cross_summary(results)

    print("\n=== Cross-cell summary (target_share_delta) ===")
    print(f"{'class':<26s}  {'mean Δshare':>12s}  positive / negative cells")
    print("-" * 80)
    for c in ALL_CLASS_LABELS:
        s = cross[c]["share"]
        pos = len(s["positive_cells"])
        neg = len(s["negative_cells"])
        print(f"  {c:<24s}  {s['mean_delta']:>+12.3f}  +{pos} / -{neg}")

    print("\n=== Cross-cell summary (target_intensity_delta) ===")
    print(f"{'class':<26s}  {'mean Δint':>10s}  positive / negative cells")
    print("-" * 80)
    for c in ALL_CLASS_LABELS:
        s = cross[c]["intensity"]
        pos = len(s["positive_cells"])
        neg = len(s["negative_cells"])
        print(f"  {c:<24s}  {s['mean_delta']:>+10.3f}  +{pos} / -{neg}")

    for r in results:
        _print_one(r)

    out_json = REPO / "audit" / "2026-05-19-archetype-per-planet-class.json"
    out_json.write_text(json.dumps({
        "prefix_turns": prefix_turns,
        "intensity_delta_threshold": INTENSITY_DELTA_THRESHOLD,
        "cells": results,
        "cross_summary": cross,
    }, indent=2))
    print(f"\nwrote {out_json}")

    out_md = REPO / "audit" / "2026-05-19-archetype-per-planet-class.md"
    out_md.write_text(_render_markdown(results, cross, prefix_turns))
    print(f"wrote {out_md}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--archetype",
        default="med_high_prod__mixed_static__big_static",
        help="Target archetype name (panel taxonomy).",
    )
    ap.add_argument("--prefix-turns", type=int, default=100)
    ap.add_argument(
        "--all-cells",
        action="store_true",
        help="Iterate over every cell with both top-10 and ours samples "
             "in audit/2026-05-18-team-archetype-gap.json, write the full "
             "JSON / MD outputs.",
    )
    args = ap.parse_args()

    if args.all_cells:
        return _run_all_cells(args.prefix_turns)

    print(f"Target archetype: {args.archetype}")
    print(f"Prefix turns:     {args.prefix_turns}\n")
    res = _run_one(args.archetype, args.prefix_turns)
    if res is None:
        return 2
    _print_one(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
