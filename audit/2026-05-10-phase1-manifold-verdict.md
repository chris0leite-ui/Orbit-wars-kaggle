# 2026-05-10 — Phase 1 verdict: manifold hypothesis (partial refute, useful failure)

> Branch: `claude/simple-trading-strategies-QS0xV`
> Plan: `/root/.claude/plans/read-the-handover-next-imperative-whisper.md`
> Companion prior art: `audit/2026-05-10-meta-strategy-prior-art.md`
> Reports:
> - `audit/manifold/20260510T141114Z/report.md` (7-class — full panel)
> - `audit/manifold/20260510T141409Z/report.md` (5-class — the gate target)
> Capture: `audit/replays/20260510T132957Z/` (1568 games, 404 MB, gitignored).

## What the gate said

> Phase 1 gate (per plan §"Gate to Phase 2"): random-forest accuracy ≥ 90%
> at K ≤ 100 turns on 5-fold CV-by-seed for the 5-strategy zoo.

## What we got

**Gate: ❌ NOT CLEARED.** Best K ≤ 100 result on the 5-strategy zoo:
random forest 80.5%, logistic regression 80.6%, both at K=100. We
are ~10 percentage points short.

| K   | RF (5-class) | LR (5-class) | RF (7-class incl. v1+baseline) | LR (7-class) |
| --- | ------------ | ------------ | ------------------------------ | ------------ |
| 25  | 68.1%        | 65.5%        | 54.2%                          | 53.7%        |
| 50  | 66.3%        | 73.5%        | 53.0%                          | 61.8%        |
| 100 | **80.5%**    | **80.6%**    | 68.5%                          | 68.5%        |
| 200 | 77.7%        | 76.8%        | 65.9%                          | 66.7%        |

Confusion matrix on the 5-class problem at K=200 (rows=true, cols=pred):

```
                    enemy_first  nearest  production  roi  weakest
enemy_first             267        14         17       9      13
nearest                  12       230          5      48      25
production               16        17        225      56       6
roi                       0        40         41     233       6
weakest                  10        12         10       1     287
```

## What this actually tells us (the partial-confirm)

The hypothesis "competitor strategies live on a small-dim manifold"
is **partly** supported and **partly** refuted by this data:

1. **Coarse classes ARE distinguishable.** `weakest` (89.7% per-class
   accuracy), `enemy_first` (83.4%), and `baseline` (95% in the
   7-class run) sit in their own basins. Within-row diagonal mass is
   high; within-row off-diagonal mass is low and concentrated in
   adjacent classes. This matches the AlphaStar "discrete basins"
   prior more than a smooth low-dim manifold.
2. **The "ROI-family" is one basin, not three.** `nearest`,
   `production`, and `roi` are mutually confusable at K ≤ 200:
   - `nearest` → `roi`: 48/320 = 15.0%
   - `roi` → `production`: 41/320 = 12.8%
   - `production` → `roi`: 56/320 = 17.5%
   They share the `DEFAULT_MECHANISMS` stack
   (`[validate, arrival_size, lead_aim]`) and all three use a
   distance- or production-aware score function, so their behavioural
   footprints overlap. Our 15-feature fingerprint can't separate them
   cleanly — the discriminating signal sits in *which planet they
   targeted first* and our `_infer_target` ray-cast proxy is too
   noisy to recover that consistently.
3. **`nearest` ≡ `v1_orbitfix` is a designed-duplicate, not a
   classifier failure.** They share `propose_intents()` line-by-line
   plus the same A.6 RNG seed; the 7-class confusion matrix shows
   them mutually predicting each other 29% of the time, which is
   correct in-data behaviour. The 5-class report (which excludes
   v1_orbitfix) is the one to read for the gate.

## Implications for the meta-strategy framework

The manifold framework still works for **broad-class** routing — a
meta-router that switches based on `{snipe-cheap, pressure-opp,
production-aware-greedy}` is buildable today with our existing
features. What's *not* buildable is a fine-grained 5-way
classification that distinguishes nearest-vs-production-vs-roi from
behaviour alone.

**Practical consequence for Phase 2/3:** there is no submission
incentive to distinguish the ROI-family members anyway. ROI dominates
the panel; if the live opponent is in that basin, our best response
is ROI. The meta-router gains EV when it can identify weakest-style
or enemy_first-style opponents and switch *away* from ROI to
something that exploits their gap. That's a 3- or 4-class problem,
not a 5-class one — and our fingerprint already handles it (89.7%
accuracy on weakest, 83.4% on enemy_first).

## Two paths forward (tagged for PI decision)

**A) Coarsen the labels and proceed.** Group the ROI-family
(`nearest`, `production`, `roi`) into a single class
`production_aware_greedy` and re-run the gate. Predicted result: RF
≥ 92% at K ≤ 100 (the misclassifications inside the family vanish;
the cross-family errors are already small). Lets us proceed to Phase
2/3 with a **3-class** meta-router (`production_aware_greedy /
weakest / enemy_first`) where the class identifies the broad basin
and the best-response rule picks the correct exploiter.

**B) Extend the fingerprint and re-run.** Add features specifically
designed to separate the ROI-family:
- target-distance distribution shape (mean, p5, p95, skew — distinguishes
  nearest's tight close-distance bias from roi's mid-range bias)
- target-production distribution shape (production picks p=5 most often
  → low variance; roi spreads across mid-production)
- early-vs-late behaviour split (first 50 turns vs steps 51-100)
- target-id-set diversity (Shannon entropy of which target IDs we
  attack — high for ROI, low for weakest/enemy_first which spam the
  cheapest/closest each turn)
Bumps `FEATURE_VERSION` to 2 (invalidates any classifier trained
against v1, but Phase 1 hasn't trained one for production yet so
this is free). Predicted result: gate clears for the full 5-class
problem.

**C) (Last resort) Learned embedding** — Grover et al. ICML 2018
protocol: small MLP trained on a triplet/discriminative loss over
fingerprint pairs, projecting to 4-dim. Heavier; only justified if A
and B both fail.

**Recommended:** A first, then B. A is a one-line change in the
manifold script (`--label-merge nearest=production_aware_greedy
production=production_aware_greedy roi=production_aware_greedy`) and
unblocks Phase 2 immediately. B is a half-day of feature engineering
that pays back if/when we want fine-grained adaptation between
ROI-family members. C only if both fail.

## What's NOT changed by this verdict

- **`roi` is still our strongest standalone agent** at the 32-seed
  panel: **97.1% mean panel winrate, 100% (64/64) vs v1_orbitfix.**
  The Phase 1 result has *no* implication for ROI's submission case;
  the meta-router is additive, not load-bearing.
- The submission-economy gate (rolling-last-2; do not push speculative
  variants on the same UTC day as a known-good submit) is unchanged.
  When v1.1's μ settles, ROI can submit on its own merit; the
  meta-router lands later.
- The 32-seed capture (`audit/replays/20260510T132957Z/`) is a
  **load-bearing artifact for everything downstream** — Phase 2's
  zoo expansion will reuse the same fingerprint corpus shape, and
  Phase 3's BR-table builder reads the per-game outcomes already in
  `audit/tournaments/20260510T140907Z.json`. Don't delete the
  replay dir even though it's gitignored.

## Numbers to remember

- 32-seed × 7 agents = 1568 games, 39 min wallclock, 404 MB on disk.
- 5-class gate: 80.5% RF at K=100 (target 90%).
- ROI standalone: 97.1% mean WR vs panel, 100% vs v1_orbitfix at 32 seeds.
- Confusable cluster: nearest / production / roi (all "production-aware
  greedy" in the AlphaStar-basin sense).
- Cleanly separated classes: weakest (89.7%), enemy_first (83.4%),
  baseline (95% in 7-class).
