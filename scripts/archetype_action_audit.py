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


def _load_corpus(target_arch: str, prefix_turns: int) -> dict:
    """Returns {'top10_win': [np.ndarray], 'ours_win': [...], 'ours_loss': [...]}.

    Top-10 corpus is all-wins by curation so we only have a 'win' bucket
    for them; for our corpus we split wins vs losses so we can compare
    'top-10 winning play' against 'our LOSING play in the same cell'
    (the high-value contrast).
    """
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)

    # Top-10
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
        if arch != target_arch:
            continue
        flat = ke_to_flat(rep, focal_idx)
        feats = fingerprint(flat, player_id=0, prefix_turns=prefix_turns)
        buckets["top10_win"].append(feats)

    # Ours: split by outcome
    for fp in sorted(OUR_DIR.glob("episode-*-replay.json")):
        rep = json.loads(fp.read_text())
        focal_idx = _focal_of_ours(rep)
        if focal_idx is None:
            continue
        # 2P only
        if len(rep.get("info", {}).get("TeamNames", []) or rep.get("info", {}).get("teamNames", [])) != 2:
            continue
        try:
            arch = archetype_of_replay(rep, focal_idx=focal_idx)
        except Exception:
            continue
        if arch != target_arch:
            continue
        won = _won(rep, focal_idx)
        if won is None:
            continue
        flat = ke_to_flat(rep, focal_idx)
        feats = fingerprint(flat, player_id=0, prefix_turns=prefix_turns)
        buckets["ours_win" if won else "ours_loss"].append(feats)

    return dict(buckets)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetype", default="med_high_prod__mixed_static__big_static",
                    help="Target archetype name (panel taxonomy).")
    ap.add_argument("--prefix-turns", type=int, default=100,
                    help="Prefix length for fingerprint computation.")
    args = ap.parse_args()

    print(f"Target archetype: {args.archetype}")
    print(f"Prefix turns:     {args.prefix_turns}\n")

    buckets = _load_corpus(args.archetype, args.prefix_turns)
    t10 = _summarise(buckets.get("top10_win", []))
    o_w = _summarise(buckets.get("ours_win", []))
    o_l = _summarise(buckets.get("ours_loss", []))
    o_all = _summarise(buckets.get("ours_win", []) + buckets.get("ours_loss", []))

    print(f"n samples: top10_win={t10['n']}  ours_win={o_w['n']}  ours_loss={o_l['n']}  ours_total={o_all['n']}\n")

    if t10["n"] == 0 or o_all["n"] == 0:
        print("ERROR: at least one corpus is empty for this archetype.")
        return 2

    # Per-feature comparison: top10_win vs ours_all, also vs ours_loss
    print(f"{'feature':<28s}  {'top10 (n=' + str(t10['n']) + ')':<18s}  {'ours-all (n=' + str(o_all['n']) + ')':<18s}  "
          f"{'delta':>10s}  {'d':>6s}  {'ours-loss':<18s}  {'delta-vs-loss':>14s}")
    print("-" * 130)
    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        t_m, t_s = t10["mean"][i], t10["std"][i]
        oa_m, oa_s = o_all["mean"][i], o_all["std"][i]
        ol_m, ol_s = o_l["mean"][i], o_l["std"][i]
        d_all = _delta(t_m, oa_m)
        d_loss = _delta(t_m, ol_m)
        eff = _effect_size(t_m, t_s, oa_m, oa_s, t10["n"], o_all["n"])
        rows.append({
            "feature": name,
            "top10_mean": t_m, "top10_std": t_s,
            "ours_all_mean": oa_m, "ours_all_std": oa_s,
            "ours_win_mean": o_w["mean"][i], "ours_loss_mean": ol_m,
            "delta_top10_minus_ours_all": d_all,
            "delta_top10_minus_ours_loss": d_loss,
            "effect_size_d": eff,
        })

        def _f(v, w=8, p=3):
            return f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}s}"

        print(f"{name:<28s}  {_f(t_m,8,3)}±{_f(t_s,6,2)}  "
              f"{_f(oa_m,8,3)}±{_f(oa_s,6,2)}  "
              f"{_f(d_all,10,3)}  {_f(eff,6,2)}  "
              f"{_f(ol_m,8,3)}±{_f(ol_s,6,2)}  {_f(d_loss,14,3)}")

    # Sort by absolute effect size to surface the biggest behavioral gaps
    print("\n=== Top behavioral deltas (|Cohen's d| descending) ===")
    ranked = sorted(
        [r for r in rows if r["effect_size_d"] is not None],
        key=lambda r: -abs(r["effect_size_d"]),
    )
    for r in ranked[:8]:
        direction = "↑" if r["effect_size_d"] > 0 else "↓"
        print(f"  {direction} {r['feature']:<28s} top-10 = {r['top10_mean']:7.3f}, "
              f"ours = {r['ours_all_mean']:7.3f} (d={r['effect_size_d']:+.2f})")

    # Persist
    out_json = REPO / "audit" / "2026-05-18-archetype-action-audit.json"
    out_json.write_text(json.dumps({
        "archetype": args.archetype,
        "prefix_turns": args.prefix_turns,
        "n": {"top10_win": t10["n"], "ours_win": o_w["n"], "ours_loss": o_l["n"]},
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {out_json}")

    # Markdown
    out_md = REPO / "audit" / "2026-05-18-archetype-action-audit.md"
    md = [f"# Archetype action audit: `{args.archetype}`",
          "",
          f"Comparing top-10 winning play against our submission 52710995 play",
          f"in the highest-gap archetype from `audit/2026-05-18-team-archetype-gap.md`.",
          "",
          f"Sample sizes (2P, fingerprint prefix = {args.prefix_turns} turns):",
          f"- top-10 wins: **{t10['n']}**",
          f"- our wins: **{o_w['n']}**",
          f"- our losses: **{o_l['n']}** (the high-value contrast)",
          "",
          "## Per-feature comparison",
          "",
          "| feature | top-10 mean | ours-all mean | delta | Cohen's d | ours-loss mean |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        def fmt(v, p=3):
            return f"{v:.{p}f}" if v is not None else "—"
        md.append(f"| `{r['feature']}` | {fmt(r['top10_mean'])} | {fmt(r['ours_all_mean'])} | "
                  f"{fmt(r['delta_top10_minus_ours_all'])} | {fmt(r['effect_size_d'], 2)} | "
                  f"{fmt(r['ours_loss_mean'])} |")
    md += ["", "## Ranked by |Cohen's d|", ""]
    for r in ranked[:8]:
        direction = "**higher**" if r["effect_size_d"] > 0 else "**lower**"
        md.append(f"- `{r['feature']}`: top-10 is {direction} ({r['top10_mean']:.3f} vs "
                  f"{r['ours_all_mean']:.3f}, d = {r['effect_size_d']:+.2f})")
    out_md.write_text("\n".join(md) + "\n")
    print(f"wrote {out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
