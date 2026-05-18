"""Per-archetype behavior audit: top-10 vs our submission.

For a chosen target archetype (default: the highest-gap cell from the
Phase-A meta-analysis), compute the 15-d behavioral fingerprint for
every replay in both corpora that lands in that archetype, then report
per-feature deltas with confidence intervals.

Outputs:
  audit/2026-05-18-archetype-action-audit.md
  audit/2026-05-18-archetype-action-audit.json

Usage:
  python scripts/archetype_action_audit.py [--archetype NAME] [--prefix-turns N]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.archetype_binning import archetype_of_replay
from lib.fingerprint import FEATURE_NAMES, fingerprint
from scripts.fingerprint_external import ke_to_flat

TOP10_DIR = REPO / "audit" / "external" / "replays"
OUR_DIR = REPO / "audit" / "replays" / "live" / "52710995"
OWN_TEAM_NAMES = {"ChrisLeiteScha", "Chris Leite Scha"}

TOP10_RE = re.compile(
    r"^r(?P<rank>\d{2})-(?P<team>.+?)-(?P<size>[24]P)-(?P<wl>[WL])-(?P<eid>\d+)\.json$"
)


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


def _won(rep: dict, focal_idx: int) -> bool | None:
    rewards = rep.get("rewards") or []
    if focal_idx >= len(rewards) or rewards[focal_idx] is None:
        return None
    focal_r = rewards[focal_idx]
    return all(
        (r is None) or (r < focal_r)
        for i, r in enumerate(rewards) if i != focal_idx
    )


_PRELOAD_CACHE: dict | None = None


def _preload_all(prefix_turns: int) -> dict:
    """Walk both corpora ONCE, computing each replay's archetype + fingerprint.

    Returns ``{archetype: {"top10_win": [...], "ours_win": [...], "ours_loss": [...]}}``.
    Memoised at module level keyed on prefix_turns.
    """
    global _PRELOAD_CACHE
    if _PRELOAD_CACHE is not None and _PRELOAD_CACHE.get("prefix_turns") == prefix_turns:
        return _PRELOAD_CACHE["by_arch"]

    by_arch: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"top10_win": [], "ours_win": [], "ours_loss": []}
    )

    for fp in sorted(TOP10_DIR.glob("*.json")):
        m = TOP10_RE.match(fp.name)
        if not m or m.group("size") != "2P":
            continue
        rep = json.loads(fp.read_text())
        focal_idx = _focal_of_top10(rep, m.group("team"))
        try:
            arch = archetype_of_replay(rep, focal_idx=focal_idx)
        except Exception:
            continue
        flat = ke_to_flat(rep, focal_idx)
        feats = fingerprint(flat, player_id=0, prefix_turns=prefix_turns)
        by_arch[arch]["top10_win"].append(feats)

    for fp in sorted(OUR_DIR.glob("episode-*-replay.json")):
        rep = json.loads(fp.read_text())
        focal_idx = _focal_of_ours(rep)
        if focal_idx is None:
            continue
        info = rep.get("info", {})
        if len(info.get("TeamNames", []) or info.get("teamNames", [])) != 2:
            continue
        try:
            arch = archetype_of_replay(rep, focal_idx=focal_idx)
        except Exception:
            continue
        won = _won(rep, focal_idx)
        if won is None:
            continue
        flat = ke_to_flat(rep, focal_idx)
        feats = fingerprint(flat, player_id=0, prefix_turns=prefix_turns)
        by_arch[arch]["ours_win" if won else "ours_loss"].append(feats)

    _PRELOAD_CACHE = {"prefix_turns": prefix_turns, "by_arch": dict(by_arch)}
    return _PRELOAD_CACHE["by_arch"]


def _load_corpus(target_arch: str, prefix_turns: int) -> dict:
    """Returns {'top10_win': [np.ndarray], 'ours_win': [...], 'ours_loss': [...]}
    for a single archetype, using the module-level cache."""
    cache = _preload_all(prefix_turns)
    return cache.get(target_arch, {"top10_win": [], "ours_win": [], "ours_loss": []})


def _summarise(samples: list[np.ndarray]) -> dict:
    """Per-feature mean + stdev + n."""
    if not samples:
        return {"n": 0, "mean": [None] * len(FEATURE_NAMES), "std": [None] * len(FEATURE_NAMES)}
    arr = np.stack(samples, axis=0)  # (n, 15)
    return {
        "n": int(arr.shape[0]),
        "mean": [float(x) for x in arr.mean(axis=0)],
        "std": [float(x) for x in arr.std(axis=0, ddof=0)],
    }


def _delta(a_mean: float | None, b_mean: float | None) -> float | None:
    if a_mean is None or b_mean is None:
        return None
    return a_mean - b_mean


def _effect_size(top10_mean, top10_std, ours_mean, ours_std, n_top10, n_ours) -> float | None:
    """Cohen's d using pooled stdev. None if either sample is empty / 0-std."""
    if None in (top10_mean, ours_mean) or n_top10 < 2 or n_ours < 2:
        return None
    s = math.sqrt(((n_top10 - 1) * (top10_std ** 2) + (n_ours - 1) * (ours_std ** 2))
                  / max(n_top10 + n_ours - 2, 1))
    if s == 0.0:
        return None
    return (top10_mean - ours_mean) / s


def _run_one(target_arch: str, prefix_turns: int, quiet: bool = False) -> dict | None:
    """Run the behavior diff on one archetype; return a result dict
    (or None if either corpus is empty)."""
    buckets = _load_corpus(target_arch, prefix_turns)
    t10 = _summarise(buckets.get("top10_win", []))
    o_w = _summarise(buckets.get("ours_win", []))
    o_l = _summarise(buckets.get("ours_loss", []))
    o_all = _summarise(buckets.get("ours_win", []) + buckets.get("ours_loss", []))

    if t10["n"] == 0 or o_all["n"] == 0:
        if not quiet:
            print(f"  SKIP {target_arch}: top10_n={t10['n']}, ours_n={o_all['n']}")
        return None

    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        t_m, t_s = t10["mean"][i], t10["std"][i]
        oa_m, oa_s = o_all["mean"][i], o_all["std"][i]
        eff = _effect_size(t_m, t_s, oa_m, oa_s, t10["n"], o_all["n"])
        rows.append({
            "feature": name,
            "top10_mean": t_m, "top10_std": t_s,
            "ours_all_mean": oa_m, "ours_all_std": oa_s,
            "ours_win_mean": o_w["mean"][i], "ours_loss_mean": o_l["mean"][i],
            "delta_top10_minus_ours_all": _delta(t_m, oa_m),
            "effect_size_d": eff,
        })
    return {
        "archetype": target_arch,
        "n": {"top10_win": t10["n"], "ours_win": o_w["n"], "ours_loss": o_l["n"]},
        "rows": rows,
    }


def _print_one(result: dict) -> None:
    """Print the per-cell behavior diff table."""
    t10_n = result["n"]["top10_win"]
    oa_n = result["n"]["ours_win"] + result["n"]["ours_loss"]
    ol_n = result["n"]["ours_loss"]
    print(f"\n=== {result['archetype']} === (top10_n={t10_n}, ours_all={oa_n}, ours_loss={ol_n})")
    print(f"{'feature':<28s}  {'top10':<14s}  {'ours-all':<14s}  {'delta':>8s}  {'d':>6s}")
    print("-" * 88)
    for r in result["rows"]:
        def _f(v, w=8, p=3):
            return f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}s}"
        print(f"  {r['feature']:<26s}  {_f(r['top10_mean'],8,3)}±{_f(r['top10_std'],5,2)}  "
              f"{_f(r['ours_all_mean'],8,3)}±{_f(r['ours_all_std'],5,2)}  "
              f"{_f(r['delta_top10_minus_ours_all'],8,3)}  {_f(r['effect_size_d'],6,2)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetype", default="med_high_prod__mixed_static__big_static",
                    help="Target archetype name (panel taxonomy).")
    ap.add_argument("--prefix-turns", type=int, default=100,
                    help="Prefix length for fingerprint computation.")
    ap.add_argument("--all-gap-cells", action="store_true",
                    help="Iterate over every cell with non-zero top10 + ours "
                         "samples in audit/2026-05-18-team-archetype-gap.json "
                         "and produce a cross-cell summary.")
    args = ap.parse_args()

    if args.all_gap_cells:
        return _run_all_cells(args.prefix_turns)

    print(f"Target archetype: {args.archetype}")
    print(f"Prefix turns:     {args.prefix_turns}\n")

    result = _run_one(args.archetype, args.prefix_turns)
    if result is None:
        return 2
    _print_one(result)

    rows = result["rows"]
    ranked = sorted(
        [r for r in rows if r["effect_size_d"] is not None],
        key=lambda r: -abs(r["effect_size_d"]),
    )
    print("\nTop deltas (|d| desc):")
    for r in ranked[:8]:
        direction = "+" if r["effect_size_d"] > 0 else "-"
        print(f"  {direction} {r['feature']:<28s} top-10={r['top10_mean']:7.3f} "
              f"ours={r['ours_all_mean']:7.3f} d={r['effect_size_d']:+.2f}")

    out_json = REPO / "audit" / "2026-05-18-archetype-action-audit.json"
    out_json.write_text(json.dumps({**result, "prefix_turns": args.prefix_turns}, indent=2))
    print(f"\nwrote {out_json}")

    out_md = REPO / "audit" / "2026-05-18-archetype-action-audit.md"
    out_md.write_text(_render_markdown_single(result, args.prefix_turns))
    print(f"wrote {out_md}")
    return 0


def _render_markdown_single(result: dict, prefix_turns: int) -> str:
    rows = result["rows"]
    ranked = sorted(
        [r for r in rows if r["effect_size_d"] is not None],
        key=lambda r: -abs(r["effect_size_d"]),
    )
    md = [f"# Archetype action audit: `{result['archetype']}`",
          "",
          f"Top-10 winning play vs our submission 52710995 play in the highest-gap "
          f"archetype from `audit/2026-05-18-team-archetype-gap.md`.",
          "",
          f"Sample sizes (2P, fingerprint prefix = {prefix_turns} turns):",
          f"- top-10 wins: **{result['n']['top10_win']}**",
          f"- our wins: **{result['n']['ours_win']}**",
          f"- our losses: **{result['n']['ours_loss']}**",
          "",
          "## Per-feature comparison",
          "",
          "| feature | top-10 | ours-all | delta | Cohen's d |",
          "|---|---|---|---|---|"]
    for r in rows:
        def fmt(v, p=3):
            return f"{v:.{p}f}" if v is not None else "—"
        md.append(f"| `{r['feature']}` | {fmt(r['top10_mean'])} | {fmt(r['ours_all_mean'])} | "
                  f"{fmt(r['delta_top10_minus_ours_all'])} | {fmt(r['effect_size_d'], 2)} |")
    md += ["", "## Ranked by |Cohen's d|", ""]
    for r in ranked[:8]:
        direction = "**higher**" if r["effect_size_d"] > 0 else "**lower**"
        md.append(f"- `{r['feature']}`: top-10 is {direction} ({r['top10_mean']:.3f} vs "
                  f"{r['ours_all_mean']:.3f}, d = {r['effect_size_d']:+.2f})")
    return "\n".join(md) + "\n"


def _run_all_cells(prefix_turns: int) -> int:
    """Iterate over informative gap cells; produce a cross-cell summary.

    Reads ``audit/2026-05-18-team-archetype-gap.json`` and selects every
    cell where both corpora have at least 1 sample. Runs the behavior
    diff on each, then aggregates per-feature signed-d statistics:
    how many cells go top-10-positive, how many top-10-negative, and
    what the mean |d| is. The goal is to verify whether the
    aggressive-top-10 / hoarding-ours pattern observed in the
    target cell holds across all gap cells.
    """
    gap_path = REPO / "audit" / "2026-05-18-team-archetype-gap.json"
    gap_data = json.loads(gap_path.read_text())
    candidates = [r for r in gap_data["rows"]
                  if r["top10_n"] >= 1 and r["ours_n"] >= 1]
    print(f"Loaded {len(candidates)} informative cells from {gap_path}\n")

    per_cell: list[dict] = []
    for c in candidates:
        arch = c["archetype"]
        print(f"--> {arch} (gap={c['gap']:+.0%} top10={c['top10_n']} ours={c['ours_n']})")
        res = _run_one(arch, prefix_turns, quiet=True)
        if res is None:
            continue
        # Annotate with gap context
        res["panel_gap"] = c["gap"]
        per_cell.append(res)

    if not per_cell:
        print("No cells produced a comparison.")
        return 2

    print(f"\nRan diff on {len(per_cell)} cells; aggregating per feature...\n")

    # Per-feature aggregate across cells (weighted by min(top10_n, ours_n)).
    per_feature: dict[str, dict] = {}
    for i, name in enumerate(FEATURE_NAMES):
        ds = []
        weights = []
        signs = []
        cells_with_d = []
        for res in per_cell:
            r = res["rows"][i]
            if r["effect_size_d"] is None:
                continue
            ds.append(r["effect_size_d"])
            w = min(res["n"]["top10_win"], res["n"]["ours_win"] + res["n"]["ours_loss"])
            weights.append(w)
            signs.append(1 if r["effect_size_d"] > 0 else -1)
            cells_with_d.append(res["archetype"])
        if not ds:
            per_feature[name] = {
                "n_cells": 0, "weighted_mean_d": None, "mean_abs_d": None,
                "n_positive": 0, "n_negative": 0,
            }
            continue
        wm = sum(d * w for d, w in zip(ds, weights)) / max(sum(weights), 1)
        per_feature[name] = {
            "n_cells": len(ds),
            "weighted_mean_d": wm,
            "mean_abs_d": statistics.fmean(abs(d) for d in ds),
            "n_positive": sum(1 for s in signs if s > 0),
            "n_negative": sum(1 for s in signs if s < 0),
            "per_cell_d": {a: d for a, d in zip(cells_with_d, ds)},
        }

    # Print the cross-cell summary
    print(f"{'feature':<28s}  {'cells':>5s}  {'wmean d':>9s}  {'|d|mean':>8s}  {'+':>3s}/{'-':>3s}")
    print("-" * 70)
    ranked = sorted(per_feature.items(),
                    key=lambda kv: -(abs(kv[1]["weighted_mean_d"]) if kv[1]["weighted_mean_d"] is not None else 0))
    for name, agg in ranked:
        wm = agg["weighted_mean_d"]
        md = agg["mean_abs_d"]
        if wm is None:
            print(f"{name:<28s}  {agg['n_cells']:>5d}  {'-':>9s}  {'-':>8s}  {'-':>3s}/{'-':>3s}")
        else:
            print(f"{name:<28s}  {agg['n_cells']:>5d}  {wm:>+9.3f}  {md:>8.3f}  "
                  f"{agg['n_positive']:>3d}/{agg['n_negative']:>3d}")

    # Persist
    out_json = REPO / "audit" / "2026-05-18-archetype-action-audit-allcells.json"
    out_json.write_text(json.dumps({
        "prefix_turns": prefix_turns,
        "n_cells": len(per_cell),
        "per_cell": per_cell,
        "per_feature_summary": per_feature,
    }, indent=2))
    print(f"\nwrote {out_json}")

    # Markdown report
    out_md = REPO / "audit" / "2026-05-18-archetype-action-audit-allcells.md"
    md = [f"# Cross-cell behavior audit: top-10 vs us across all gap cells",
          "",
          f"Behavior-diff (15-d fingerprint) on every cell from "
          f"`audit/2026-05-18-team-archetype-gap.json` that has at "
          f"least one top-10 and one ours sample. **{len(per_cell)} cells "
          f"aggregated.** Each cell contributes Cohen's d per feature; "
          f"the cross-cell weighted-mean d is the headline.",
          "",
          "## Cross-cell summary (sorted by |weighted-mean d|)",
          "",
          "| feature | cells | weighted-mean d | mean \\|d\\| | + / − cells |",
          "|---|---|---|---|---|"]
    for name, agg in ranked:
        wm = "—" if agg["weighted_mean_d"] is None else f"{agg['weighted_mean_d']:+.3f}"
        md_d = "—" if agg["mean_abs_d"] is None else f"{agg['mean_abs_d']:.3f}"
        md.append(f"| `{name}` | {agg['n_cells']} | {wm} | {md_d} | "
                  f"{agg['n_positive']}/{agg['n_negative']} |")
    md += ["",
           "**Reading:**",
           "- `weighted-mean d` = mean Cohen's d across cells, weighted by min(top-10-n, ours-n).",
           "- **Positive** d → top-10 has the feature HIGHER than us.",
           "- **Negative** d → top-10 has the feature LOWER than us.",
           "- **+/− cells** = how many cells show each sign. A feature with 10 cells "
           "all-positive or all-negative is a **universal** behavioral gap; mixed signs "
           "suggest the behavior is archetype-specific.",
           "",
           "## Per-cell deltas",
           ""]
    for res in per_cell:
        n_t = res["n"]["top10_win"]
        n_o = res["n"]["ours_win"] + res["n"]["ours_loss"]
        md.append(f"### `{res['archetype']}` (top10_n={n_t}, ours_n={n_o}, "
                  f"panel-gap={res['panel_gap']:+.0%})")
        md.append("")
        md.append("| feature | top-10 | ours | delta | d |")
        md.append("|---|---|---|---|---|")
        for r in res["rows"]:
            def fmt(v, p=3):
                return f"{v:.{p}f}" if v is not None else "—"
            md.append(f"| `{r['feature']}` | {fmt(r['top10_mean'])} | "
                      f"{fmt(r['ours_all_mean'])} | "
                      f"{fmt(r['delta_top10_minus_ours_all'])} | "
                      f"{fmt(r['effect_size_d'], 2)} |")
        md.append("")
    out_md.write_text("\n".join(md) + "\n")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
