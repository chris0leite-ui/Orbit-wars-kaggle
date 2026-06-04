# state/STRATEGY.md — current strategy (branch JzIAr)

> **READ FIRST.** This is the canonical "what are we running" doc. Everything
> in `CLAUDE.md` and `HANDOVER.md` points back here.

## The strategy: `baseline_adaptive_k` + the teamwork refiner

**Current live agent:** `champ_refine_adaptivek.py`, sub **53336920** (2026-06-03
17:53 UTC). Settled **μ ≈ 1179** on the current field (was warm-up noise at 860 in
the first ~20 min — see the warm-up note below).
**Reproducible build:** `scripts/_build_refine_adaptivek_bundle.sh`.
**Base it builds on:** `baseline_adaptive_k` (champion `launch_rules_universal` +
adaptive horizon K), whose identical agent settled **μ = 1170.4** (sub 53265480) /
**1188.3** frozen (sub 53324164, now evicted).

### What it is in one sentence

The all-time-champion config (`launch_rules_universal`, 12 env vars: joint-aggressive
multi-source coalitions / reinforcement / neutral-bonus / orbital-safety /
present-value-discount / universal K=10 launch-discipline ceiling) **plus two**
levers: (1) an **adaptive horizon K**, and (2) an **augment-not-replace teamwork
refiner**.

### Lever 1 — adaptive horizon K

`K(step) = max(10, round(20 - (20-10) · step / 30))` — opens at K=20 in the
predictable opening, decays linearly to the disciplined champion floor K=10 by step
30. Static K=10 hid ~75% of the opening expansion map (median neutral ETA ≈ 22); the
opening is genuinely predictable (few in-flight fleets, known positions) so far
launches are safe there, while midgame the K=10 floor is the right discipline. See
`audit/2026-06-01-adaptive-horizon-k-investigation.md`. Read by
`agents/baseline/launch_rules.capture_horizon_k(step)` →
(1) launch-discipline gate, (2) proposer far-candidate prune, (3) sync-coalition cap.

### Lever 2 — teamwork refiner (`BASELINE_CHOOSER=refine`)

Run the champion chooser **verbatim** (never removes a champion launch), then **add
only** oracle-positive two-source "sync coalitions" — pairs where neither planet can
solo-capture a defended target but both combined can — that don't conflict with the
champion's committed launches. Augment, not replace (greedy-replace was falsified
9/16). The seam is real: vs a strong opponent the generator yields ~100+ coalitions
per game (driven by the resource ratio — they appear when our planets are contested /
out-resourced). See `audit/2026-06-03-postmortem-teamwork-reversal.md`.

### Evidence — honest status (do NOT overstate)

- **Local A/B (strong, but single-opponent):** refine vs the adaptive-K champion
  scored **70.2% h2h (n=57, Wilson-lo 0.573), paired +9 net** on 16 matched seeds.
  This is clear evidence of an edge **against that one opponent** — by our own
  discipline (archived Rule 43) a single-opponent A/B is **not** calibrated to the
  live field.
- **Live μ comparison is confounded — NOT a clean ranking.** The adaptive-K base
  (1188) is **evicted/frozen** at the 2026-06-03 field; refine (1179) is **active**
  against today's larger, stronger field. A later submission earns its μ against
  tougher opponents, so the same true skill shows a lower number. **You cannot
  directly compare a frozen-old-field μ to an active-current-field μ.**
- **One clean same-field datapoint:** refine **1179** ≈ computeByShips **1180** (both
  active today) — parity between our strong-A/B agent and a parity-A/B agent. Not a
  separation, not a contradiction.
- **Bottom line:** strong local edge, live-uncontradicted, **not yet live-confirmed**.
  To confirm: a same-field comparison (resubmit the base so both play today's field)
  or a multi-opponent local panel (the archived Rule 43 gate).

## Open thread — the opening "we wait too long" question

PI replay (seed 722289020): we sit idle early, an aggressive opponent (Merchant API)
takes the ring. **Diagnosed (`audit/2026-06-04-opening-wait-diagnostic.md`): NOT a
horizon problem** — 0/31 opening turns were horizon-starved; the proposer offered
2–12 candidates every turn and the agent launched on only 4/31. The lever is the
**value function's early-expansion appetite**, not the horizon constant. Untested:
whether the waiting is *exploited* by aggressive early-expanders (it's symmetric in
self-play). Next experiment: appetite vs an aggressive expander (not self-play), cut
by opponent class. Tool: `scripts/opening_starvation.py`.

### TrueSkill warm-up — DO NOT panic at early μ

Kaggle's TrueSkill starts every new submission at **μ ≈ 600** and climbs over ~24 h.
Do not draw conclusions from the first few hours. An **evicted** submission's μ is
**frozen** at the field it last played and is not comparable to an active one's.

## Iteration protocol — observation-driven

1. **PI observes** something concrete (replay, leaderboard move, opponent trace).
2. **PI reports** it in plain English (Rule 0).
3. **AI diagnoses** the modeling cause — minimal investigation (Rule 40: model the
   right thing; do NOT bump a constant).
4. **AI proposes** the smallest change addressing the cause, default-OFF gated so the
   bundle stays byte-identical until proven.
5. **PI signs off.**
6. **AI implements** in `agents/baseline/`; smokes per Rule 46.
7. **AI submits** per Rules 1 / 12 / 42; appends a row to the `state/MULTI_BRANCH.md`
   push-claim board.
8. Wait for the next observation. Go to 1.

One observation → one mechanism → one push. No multi-axis exploration.

## How to bundle, smoke, and submit

```
# 1. Build (env-vars baked; outputs submissions/champ_refine_adaptivek.py).
bash scripts/_build_refine_adaptivek_bundle.sh

# 2. Rule 46 smoke (required before every submit).
python -m pytest tests/test_bundle.py -q
python fast.py play submissions/champ_refine_adaptivek.py \
       --vs submissions/baseline.py --seed 7              # max turn < 1000 ms

# 3. Rule 42 gate — read the rolling pair, then append a claim row.
kaggle competitions submissions orbit-wars | head -5

# 4. Submit with explicit PI sign-off.
kaggle competitions submit -c orbit-wars \
    -f submissions/champ_refine_adaptivek.py \
    -m "<plain-English description + evicted sub-id/μ + new bundle sha256>"
```

## How to modify

A new mechanism lives in `agents/baseline/`, gated behind a `BASELINE_<MECHANISM>`
env var, default OFF → byte-identical champion when unset. The build script bakes the
var ON in the bundle header (see the env-header block atop the bundle). Tests go in
`tests/`; the Rule 46 bundle test protects byte-parity of the default-OFF path.

## Pointers

- `CLAUDE.md` — process rules (kept lean).
- `HANDOVER.md` — next-session brief (points here).
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42).
- `state/TOOLS.md` — tools registry (A/B harnesses, diagnostics, bundler).
- `comp-context.md` — settled-once competition facts.
- `audit/` — append-only postmortems, investigations, replays.
- `knowledge-base/` — PI second-brain (`thoughts/`, `concepts/`, `flags/`, `questions/`).
- `state/_archive/CLAUDE-JzIAr-full-49rules.md` — the full pre-slim rule set.
