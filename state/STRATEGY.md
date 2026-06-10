# state/STRATEGY.md — current main strategy

> **READ FIRST.** This is the canonical "what are we running" doc. Everything
> in `CLAUDE.md` and `HANDOVER.md` points back here.
> Updated 2026-06-09 — the baseline_adaptive_k era doc this replaces is in
> git history (and its mechanism survives, default-OFF, inside producer_plus).

## The strategy: `producer_plus_multi_opp_def`

**Engine:** the vendored third-party "Producer" agent (Slawek Biel, MIT —
`agents/producer/`, provenance in `agents/producer/PROVENANCE.md`), a
torch-based greedy wave planner that out-performed our home-grown baseline
on the live ladder.
**Our additions** (`agents/producer_plus/` + gated code in
`agents/producer/orbit_lite/`), each behind a `PRODUCER_PLUS_*` env var,
default OFF (OFF path is action-stream-identical to vanilla Producer —
guarded by `tests/test_producer_plus_opp_proj.py::test_off_path_bit_identical_to_producer`):

1. **Multi-size candidate enumeration** (`PRODUCER_PLUS_MULTI_SIZE`) —
   three ship-count variants per (source, target) pair (capture floor,
   2× floor, safe drain) folded into the candidate axis.
2. **Producer-mirror opponent projection** (`PRODUCER_PLUS_OPP_PROJECTION`)
   — runs Producer's own planner from the opponent's seat and feeds its
   predicted launches into the scorer as background, plus an opp-aware
   defensive shortlist (defense reacts to predicted attacks, not only
   fleets already in flight).

**Reproducible build:**
`python scripts/bundle_producer_plus.py --variant multi_opp_def`
→ `submissions/producer_plus_multi_opp_def_on.py`.

**Validation:** local n=32 seat-balanced clean A/B vs vanilla producer =
24/32 = 75%, Wilson [0.579, 0.867] — measured 2026-06-05 and reproduced
exactly on the rebuilt bundle 2026-06-09. Live settle 1263–1287
(sub 53384340, evicted 06-07 by a manual multi_size resubmit).

### Mechanisms measured and rejected (do not re-ship without new diagnosis)

- **Coalitions** (L=2 multi-source): parity at best, -1 win composed.
- **Multi-tick opp projection + recapture penalty**: live μ = 1099.3
  (sub 53390700) vs the 1285 backstop — large live regression.
- **Strategic-value bonuses** (denial/opening) at default weight 0.1:
  0/4 clean A/B — weights dominated the score; re-tune to ≈ 0.005–0.02
  against the dumped competitive_score distribution before retry.
- **Force-concentration** (relaxed one-wave-per-target mutex), 2026-06-09:
  standalone 6/32, lean 7/32, multi-tick 5/32 vs producer — hard null.
- **Strategic-value bonuses at CALIBRATED weights** (2026-06-10; denial
  0.01, opening 0.04 — sized so the median bonus is 5-7% of the median
  acted-on score, per the calibration probe): denial 16/32, opening
  15/32, composed 18/32 — all far below the 75% base. The terms are
  directionally wrong, not mis-weighted. Do not retry by re-tuning.
- **Scorer horizon 18 → 24** (`PRODUCER_PLUS_HORIZON_2P/4P=24`,
  2026-06-10): 17/32 — regression; the engine's H=18 is calibrated
  against its other constants.
- **Adaptive K on the producer engine**: 8/16 parity (it mattered on our
  old baseline, not on Producer's calibration).

The pattern across nine measured mechanisms: every addition to the
proven multi_size + opp_projection stack regresses against vanilla
producer. The engine is at a local optimum w.r.t. the 2P vs-producer
yardstick.

### The 4P front (opened 2026-06-10 — live-replay diagnosis)

The ladder plays 2P AND 4P; from 195 live episodes of our best sub
(53384340): **4P is 60% of game volume, our 4P winrate 29% vs 63% in
2P.** 82 of 83 4P losses end with us ELIMINATED (median step 120),
carved by 2+ opponents. Winner profile across all 468 seats: same
first-attack timing as losers but 3× the enemy captures and 3× the
ships by mid-game — an extermination meta won by whoever wins the
brawls, not by farming. **Any future 4P mechanism must be measured in
4P** — the 2P A/B cannot see this axis at all.

Tools for the 4P axis:
- `scripts/clean_ffa.py` — subprocess-isolated 4P harness (focal + 3
  background agents, rotating seat, first-place rate).
- Namespaced bundles (`submissions/_ns_multi_opp_def.py`): replacing
  the `PRODUCER_PLUS_` env prefix in a bundle copy gives disjoint gate
  keys, so producer_plus bundles can share a process — enables
  4P self-play pools and 2P head-to-head vs our own best.
- First local 4P calibration: multi_opp_def first-place vs 3× vanilla
  producer = 13/32 (random = 25%).

First mechanism on this axis — `PRODUCER_PLUS_FFA_SCORE` (strength-
weighted opponent term, kills the mutual-damage-trade bias): no lift vs
the 3×producer pool (11/32 vs baseline 13/32, 12 paired outcomes
flipped). Self-play-pool verdict pending; 2P byte-identical by
construction.

**Loss-anatomy mining (2026-06-10,
`audit/2026-06-10-4p-loss-anatomy-mining.md`):** 4P losses are decided
in the step-20..80 brawl window — we are ship-rank 1 at step 20 even in
losses, production peaks ~step 40 then declines while the eventual
winner's doubles. NOT the separators: self-drained-then-carved rate
(60% in wins AND losses), neutral-expansion count (stalls at 3 in
both), defensive-shortlist width (≥3 planets falling within horizon on
only 9% of loss steps). Multi-front carving (2+ rivals, 54/85 losses
vs 6/31 wins) is the end state of an economy already lost.

Measured on the 4P axis (3×producer pool, seeds 0–31, baseline 13/32):
- `tick4p` (4P-only multi-tick opp projection K=3, 2P byte-identical):
  **10/32 — null.** The mirror re-spends rival ships across rounds
  (no budget debit) → phantom aggression. Do not re-ship without the
  debit fix.
- `reinforce_deficit` (defense candidate sizing: pre-flip floor =
  post-flip survivor + 1 instead of 1): **9/32, paired 5–1 AGAINST
  baseline — mild genuine regression on this pool.** The full-drain
  rescue it displaces buys speed (fleet speed rises with size) and
  post-hold surplus. Do not ship standalone.

Baseline 13/32 reproduced exactly by deterministic re-run; per-seed
pool logs now archived under `audit/pools/`.

### TrueSkill warm-up — DO NOT panic at early μ

Kaggle's TrueSkill starts every new submission at μ ≈ 600 and climbs over
~24 h. Also: the field is strengthening fast — identical multi_size code
settled 1282 on 06-04 but 1181 on 06-07. Compare a resubmission against
the field of its own day, not against historical settles.

### Iteration protocol — observation-driven

1. **PI observes** something concrete — a single-game replay, a specific loss
   pattern, a leaderboard move, a turn-by-turn trace, an opponent behaviour.
2. **PI reports** the observation in plain English. (Per CLAUDE.md Rule 0.)
3. **AI diagnoses** — minimal investigation, surface the modeling cause (per
   CLAUDE.md Rule 40: model the right thing; do NOT bump a constant).
4. **AI proposes** the smallest change that addresses the cause, gated behind a
   default-OFF env var so the proven bundle stays byte-identical until proven.
5. **PI signs off** on the proposal.
6. **AI implements**; smoke gates per CLAUDE.md Rule 46.
7. **AI submits** per CLAUDE.md Rules 1 / 12 / 42; appends a row to
   `state/MULTI_BRANCH.md` push-claim board.
8. Wait for the next observation. Go to step 1.

No multi-axis exploration. No speculative ports. One observation → one mechanism
→ one push. The PI is the observation source; the AI is the mechanism builder.

## How to bundle, smoke, and submit

```
# 1. Build (env-vars baked into the bundle header).
python scripts/bundle_producer_plus.py --variant multi_opp_def

# 2. Rule 46 smoke (required before every submit).
python -m pytest tests/test_bundle.py -q                              # 15/15 expected
python fast.py play submissions/producer_plus_multi_opp_def_on.py \
       --vs submissions/v7_0_drop_one.py --seed 7                     # max turn < 1000 ms

# 3. Local lift gate (Rule 45): clean n=32 A/B vs vanilla producer.
python scripts/clean_ab.py submissions/<bundle>.py agents/producer/main.py \
       --seeds 16 --workers 2

# 4. Rule 42 gate (check evicted-μ).
kaggle competitions submissions orbit-wars | head -3                  # see rolling pair
#   append a claim row to state/MULTI_BRANCH.md before pushing

# 5. Submit with explicit PI sign-off.
kaggle competitions submit -c orbit-wars \
    -f submissions/producer_plus_multi_opp_def_on.py \
    -m "<plain-English description + sub-id of evicted submission + sha256 of new bundle>"
```

**A/B caveat:** never A/B two producer_plus bundles in the same process —
both read the same `PRODUCER_PLUS_*` env keys, so they cross-contaminate.
Always measure against vanilla producer (`agents/producer/main.py`), one
game per subprocess (`scripts/clean_ab.py` does this).

## How to modify

A new mechanism goes in `agents/producer/orbit_lite/` (engine-level) or
`agents/producer_plus/main.py` (planner-level), gated behind a
`PRODUCER_PLUS_<MECHANISM>` env var, default OFF → vanilla-Producer
behaviour when unset. Add the variant's env-var set to `ENV_VARIANTS` in
`scripts/bundle_producer_plus.py`; the bundler bakes the vars into the
bundle header. Tests go in `tests/` — note they must clear `PRODUCER_PLUS_*`
env keys between in-process games and compare action streams (rewards are
±1 in current kaggle_environments).

## Pointers

- `CLAUDE.md` — process rules (kept lean).
- `HANDOVER.md` — next-session brief (kept lean, points here).
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42); historical track registry trimmed.
- `state/TOOLS.md` — tools registry (A/B harnesses, diagnostics, validation, bundler).
- `comp-context.md` — settled-once competition facts (env spec, deadline, gate clearance).
- `audit/` — append-only audit trail (postmortems, investigations, replays).
- `knowledge-base/` — PI second-brain (`thoughts/`, `concepts/`, `flags/`, `questions/`).
