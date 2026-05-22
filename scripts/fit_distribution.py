"""Characterise the Orbit Wars leaderboard score distribution.

Fits candidate distributions, prints shape stats + KS goodness-of-fit,
and produces diagnostic plots (Q-Q, log-log survival, density overlay).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

CSV_PATH = Path("audit/leaderboard/orbit-wars-publicleaderboard-2026-05-22T14:02:57.csv")
OUT_PATH = Path("audit/leaderboard/distribution_fit.png")

scores: list[float] = []
with CSV_PATH.open(encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        scores.append(float(row["Score"]))
x = np.sort(np.array(scores, dtype=float))
n = len(x)

# --- Shape stats ---
mu = float(np.mean(x))
med = float(np.median(x))
sd = float(np.std(x, ddof=1))
sk = float(stats.skew(x))
ku = float(stats.kurtosis(x, fisher=True))  # excess kurtosis
mn, mx = float(x.min()), float(x.max())

print(f"n={n}")
print(f"mean={mu:.1f}  median={med:.1f}  sd={sd:.1f}  min={mn:.1f}  max={mx:.1f}")
print(f"skewness={sk:.3f}  excess kurtosis={ku:.3f}")
print(f"(normal: skew=0, kurt=0 ; log-normal: skew>0 ; Pareto: skew>>0, fat tail)")
print()

# --- Candidate fits ---
def fit_and_score(name: str, dist, **kw):
    """Fit dist to x, return (params, KS-D, KS-p, AIC, log-likelihood)."""
    try:
        params = dist.fit(x, **kw)
    except Exception as e:
        return name, None, np.nan, np.nan, np.nan, np.nan, str(e)
    # KS test against the fitted distribution.
    D, p = stats.kstest(x, dist.cdf, args=params)
    # AIC = 2k - 2 log L.
    logL = float(np.sum(dist.logpdf(x, *params)))
    k = len(params)
    aic = 2 * k - 2 * logL
    return name, params, D, p, aic, logL, None


print("=" * 70)
print("Candidate distribution fits (max-likelihood, then KS + AIC)")
print("=" * 70)
results = []
# Pareto type I: pdf x^{-(α+1)} on [x_m, ∞). scipy.stats.pareto is shape=α,
# loc=x_m, scale=... by default. We pin loc=0 and let scale absorb x_m? No,
# pareto.cdf(x; b, loc, scale) = 1 - ((x-loc)/scale)^(-b). To fit a Pareto,
# we want loc=x_m, scale=1 (or scale=x_m, loc=0). MLE will find a fit.
candidates = [
    ("Normal", stats.norm, {}),
    ("Log-normal", stats.lognorm, {"floc": 0}),
    ("Exponential", stats.expon, {}),
    ("Gamma", stats.gamma, {"floc": 0}),
    ("Weibull (min)", stats.weibull_min, {"floc": 0}),
    ("Pareto (Type I, x_min=min(x))", stats.pareto, {"floc": 0, "fscale": mn}),
    ("Generalised Pareto (GPD)", stats.genpareto, {"floc": mn}),
]
for name, dist, kw in candidates:
    out = fit_and_score(name, dist, **kw)
    results.append(out)
    name, params, D, p, aic, logL, err = out
    if err:
        print(f"{name:38s}  FIT FAILED: {err}")
        continue
    params_str = ", ".join(f"{v:.3g}" for v in params)
    print(f"{name:38s}  params=({params_str})")
    print(f"{'':38s}  KS D={D:.4f} p={p:.2e}  logL={logL:.1f}  AIC={aic:.1f}")
print()

# --- Pareto-specific test: log-log linearity of survival ---
print("=" * 70)
print("Power-law (Pareto) check: log-log linearity of empirical survival")
print("=" * 70)
# Use threshold = top-quartile (typical Pareto tail starts in upper tail).
for q in (0.50, 0.75, 0.90, 0.95):
    cut = float(np.quantile(x, q))
    tail = x[x >= cut]
    if len(tail) < 30:
        continue
    # Hill estimator for alpha.
    log_tail = np.log(tail / cut)
    alpha_hill = 1.0 / np.mean(log_tail)
    # Log-log linear fit of survival.
    sorted_tail = np.sort(tail)
    sv = 1.0 - np.arange(len(sorted_tail)) / len(sorted_tail)
    # Drop last bin (survival hits 0). Fit log(sv) vs log(x).
    mask = sv > 0
    slope, intercept, r, pv, _ = stats.linregress(
        np.log(sorted_tail[mask]), np.log(sv[mask])
    )
    print(f"q={q:.2f} cutoff={cut:>7.1f} n_tail={len(tail):>4}  "
          f"Hill α≈{alpha_hill:.2f}  log-log slope={slope:.2f}  R²={r**2:.3f}")
print("(Pareto would give R² ≈ 1 and a stable Hill α across cutoffs)")
print()

# --- Log-normal check: shapiro on log(x) ---
print("=" * 70)
print("Log-normal check: normality of log(x)")
print("=" * 70)
logx = np.log(x)
print(f"log(x) mean={np.mean(logx):.3f}  sd={np.std(logx):.3f}  "
      f"skew={stats.skew(logx):.3f}  ex-kurt={stats.kurtosis(logx):.3f}")
# Shapiro doesn't handle n > 5000 well; use Anderson-Darling.
ad = stats.anderson(logx, dist="norm")
print(f"Anderson-Darling on log(x): A²={ad.statistic:.3f}  "
      f"crit(5%)={ad.critical_values[2]:.3f}  ⇒ "
      f"{'reject log-normal' if ad.statistic > ad.critical_values[2] else 'cannot reject log-normal'}")
print()

# --- Plot: 4-panel diagnostic ---
fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# (1) Histogram with overlaid fit densities.
ax1 = ax[0, 0]
bins = np.linspace(mn, mx, 80)
ax1.hist(x, bins=bins, density=True, color="#3b82f6", alpha=0.55,
         edgecolor="white", label="empirical")
xs = np.linspace(mn, mx, 600)
# Pick best by AIC among the converged fits.
ok = [r for r in results if r[1] is not None]
ok_sorted = sorted(ok, key=lambda r: r[4])  # by AIC
for name, params, D, p, aic, logL, _ in ok_sorted[:3]:
    dist = dict((c[0], c[1]) for c in candidates)[name]
    try:
        ax1.plot(xs, dist.pdf(xs, *params), linewidth=1.5,
                 label=f"{name} (KS={D:.3f}, AIC={aic:.0f})")
    except Exception:
        pass
ax1.set_xlabel("score (μ)")
ax1.set_ylabel("density")
ax1.set_title("Empirical density + top-3 fits (by AIC)")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.25)

# (2) Q-Q plot vs log-normal (the best fit, usually).
ax2 = ax[0, 1]
lognorm_params = dict((c[0], r) for c, r in zip(candidates, results) if c[0] == "Log-normal")
lognorm_res = [r for r in results if r[0] == "Log-normal"][0]
shape, loc, scale = lognorm_res[1]
theo = stats.lognorm.ppf((np.arange(1, n + 1) - 0.5) / n, shape, loc=loc, scale=scale)
ax2.scatter(theo, x, s=6, alpha=0.5, color="#1d4ed8")
lim = [min(theo.min(), mn), max(theo.max(), mx)]
ax2.plot(lim, lim, color="#dc2626", linestyle="--", linewidth=1)
ax2.set_xlabel("log-normal theoretical quantile")
ax2.set_ylabel("empirical quantile")
ax2.set_title(f"Q-Q vs log-normal  (KS={lognorm_res[2]:.3f})")
ax2.grid(alpha=0.25)

# (3) Log-log survival plot (Pareto check).
ax3 = ax[1, 0]
sv = 1.0 - np.arange(n) / n
sv_y = np.where(sv > 0, sv, np.nan)
ax3.loglog(x, sv_y, color="#1d4ed8", linewidth=1.4, label="empirical survival")
# Overlay Pareto fit (above the top-quartile cutoff).
cut = float(np.quantile(x, 0.75))
tail = x[x >= cut]
log_tail = np.log(tail / cut)
alpha_hill = 1.0 / np.mean(log_tail)
xs_tail = np.linspace(cut, mx, 200)
sv_pareto = (cut / xs_tail) ** alpha_hill * (len(tail) / n)
ax3.loglog(xs_tail, sv_pareto, color="#dc2626", linestyle="--", linewidth=1.5,
           label=f"Hill Pareto α={alpha_hill:.2f} above 75th %ile")
# Overlay log-normal survival.
sv_lognorm = stats.lognorm.sf(xs, shape, loc=loc, scale=scale)
ax3.loglog(xs, sv_lognorm, color="#16a34a", linestyle=":", linewidth=1.5,
           label="log-normal survival")
ax3.set_xlabel("score (log)")
ax3.set_ylabel("P(X > x) (log)")
ax3.set_title("Log-log survival: Pareto would be straight, log-normal curves down")
ax3.legend(fontsize=8)
ax3.grid(which="both", alpha=0.25)

# (4) Q-Q vs Pareto (Type I, x_m = min).
ax4 = ax[1, 1]
pareto_res = [r for r in results if r[0].startswith("Pareto")][0]
if pareto_res[1] is not None:
    b, loc_p, scale_p = pareto_res[1]
    theo_p = stats.pareto.ppf((np.arange(1, n + 1) - 0.5) / n, b, loc=loc_p, scale=scale_p)
    ax4.scatter(theo_p, x, s=6, alpha=0.5, color="#1d4ed8")
    lim = [min(theo_p.min(), mn), max(theo_p.max(), mx)]
    ax4.plot(lim, lim, color="#dc2626", linestyle="--", linewidth=1)
    ax4.set_title(f"Q-Q vs Pareto (Type I, α={b:.2f})  (KS={pareto_res[2]:.3f})")
    ax4.set_xlabel("Pareto theoretical quantile")
    ax4.set_ylabel("empirical quantile")
    ax4.grid(alpha=0.25)

fig.suptitle(f"Orbit Wars leaderboard — distribution diagnostics (n={n})",
             fontsize=12, y=0.995)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=140)
print(f"wrote {OUT_PATH}")
