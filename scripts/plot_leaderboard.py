"""Plot the Orbit Wars public-leaderboard score distribution."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("audit/leaderboard/orbit-wars-publicleaderboard-2026-05-22T14:02:57.csv")
OUT_PATH = Path("audit/leaderboard/score_distribution.png")

# Our team + key submissions to annotate.
OUR_TEAM = "ChrisLeiteScha"
ROLLING_PAIR = {
    "_phase4_step1_FND.py": 1116.0,   # latest live sub
    "baseline_full.py": 1083.0,        # older half of rolling pair
}
TEAM_PEAK_EVICTED = 1149.2  # composite_a2_hybrid, sub 52744856
ORBITFIX_PRED = (1120.0, 1135.0)  # predicted band for the suggested submission

scores: list[float] = []
ranks: list[int] = []
our_score: float | None = None
our_rank: int | None = None
with CSV_PATH.open(encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        s = float(row["Score"])
        scores.append(s)
        ranks.append(int(row["Rank"]))
        if row["TeamName"] == OUR_TEAM:
            our_score = s
            our_rank = int(row["Rank"])

arr = np.array(scores)
print(f"n={len(arr)} teams")
print(f"top1={arr.max():.1f}  median={np.median(arr):.1f}  p25={np.percentile(arr,25):.1f}  "
      f"p10={np.percentile(arr,10):.1f}  min={arr.min():.1f}")
print(f"our team {OUR_TEAM!r}: rank={our_rank} score={our_score}")
percentile = (len(arr) - our_rank + 1) / len(arr) * 100 if our_rank else None
print(f"our percentile: top {100 - percentile:.1f}% (i.e. {percentile:.1f}th percentile from bottom)")

# Plot: histogram + ECDF overlay.
fig, (ax_hist, ax_ecdf) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

bins = np.linspace(arr.min(), arr.max(), 80)
ax_hist.hist(arr, bins=bins, color="#3b82f6", edgecolor="white", alpha=0.85)
ax_hist.set_ylabel("teams (per bin)")
ax_hist.set_title(
    f"Orbit Wars public leaderboard — score distribution ({len(arr)} teams, snapshot 2026-05-22 14:02 UTC)"
)

# Reference markers.
for q, label, color in [
    (50, "median", "#6b7280"),
    (25, "25th %ile", "#9ca3af"),
    (10, "10th %ile", "#9ca3af"),
]:
    v = np.percentile(arr, q)
    ax_hist.axvline(v, color=color, linestyle=":", linewidth=1)
    ax_hist.text(v, ax_hist.get_ylim()[1] * 0.92, f"{label}\n{v:.0f}", color=color,
                 ha="center", va="top", fontsize=8)

# Our rolling pair.
ax_hist.axvline(ROLLING_PAIR["_phase4_step1_FND.py"], color="#dc2626", linewidth=1.5,
                label=f"FND (latest, μ={ROLLING_PAIR['_phase4_step1_FND.py']})")
ax_hist.axvline(ROLLING_PAIR["baseline_full.py"], color="#f59e0b", linewidth=1.5, linestyle="--",
                label=f"baseline_full (older, μ={ROLLING_PAIR['baseline_full.py']})")
ax_hist.axvline(TEAM_PEAK_EVICTED, color="#16a34a", linewidth=1.5, linestyle=":",
                label=f"team peak EVICTED (μ={TEAM_PEAK_EVICTED})")

# Predicted band for orbitfix candidate.
ax_hist.axvspan(*ORBITFIX_PRED, color="#16a34a", alpha=0.12,
                label=f"orbitfix candidate band [{ORBITFIX_PRED[0]:.0f}, {ORBITFIX_PRED[1]:.0f}]")

# Our current displayed-ladder score from the CSV.
if our_score is not None:
    ax_hist.axvline(our_score, color="#7c3aed", linewidth=2,
                    label=f"OUR TEAM ({OUR_TEAM}) μ={our_score} rank={our_rank}/{len(arr)}")

ax_hist.legend(loc="upper left", fontsize=8, framealpha=0.92)
ax_hist.grid(axis="y", alpha=0.25)

# ECDF.
sorted_arr = np.sort(arr)
ecdf_y = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr) * 100
ax_ecdf.plot(sorted_arr, ecdf_y, color="#1d4ed8", linewidth=1.5)
ax_ecdf.set_xlabel("public-leaderboard score (μ)")
ax_ecdf.set_ylabel("percentile (cumulative %)")
ax_ecdf.grid(alpha=0.3)

# Mark the same key scores on the ECDF.
def _pct_at(s: float) -> float:
    return float((sorted_arr <= s).mean() * 100)

for s, color, label in [
    (ROLLING_PAIR["_phase4_step1_FND.py"], "#dc2626", "FND latest"),
    (ROLLING_PAIR["baseline_full.py"], "#f59e0b", "baseline_full"),
    (TEAM_PEAK_EVICTED, "#16a34a", "team peak (evicted)"),
]:
    p = _pct_at(s)
    ax_ecdf.axvline(s, color=color, linestyle=":", linewidth=1)
    ax_ecdf.scatter([s], [p], color=color, s=40, zorder=5)
    ax_ecdf.text(s, p + 2, f"{label}\n{p:.0f}th %ile", color=color, fontsize=8,
                 ha="center")

if our_score is not None:
    p = _pct_at(our_score)
    ax_ecdf.scatter([our_score], [p], color="#7c3aed", s=60, zorder=6, marker="*")
    ax_ecdf.text(our_score, p - 6, f"us\n{p:.0f}th %ile", color="#7c3aed", fontsize=8,
                 ha="center", fontweight="bold")

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=140)
print(f"wrote {OUT_PATH}")
