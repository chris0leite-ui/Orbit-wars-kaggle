# Per-geometry-class priority prior — design for next session

> Permanent reference. Written 2026-05-19 after the
> `audit/2026-05-19-archetype-per-planet-class.{json,md}` rollup
> showed top-10 over-allocates +10 percentage-points of fleet share
> to low-production rotating inner planets (8 of 16 cells positive,
> 2 negative) and we over-allocate to big-static planets
> (high-production static inner −4.7 pp, outer −4.9 pp). Plain
> English, no abbreviations. The data is real; the architecture
> below is what we plan to build, NOT what is in the agent today.

## The one-line version

Encode "which planets matter most" as a small per-geometry-class
prior weight that nudges the proposer's candidate ranking, plus a
running estimate of what the opponent has been targeting that
shifts the prior in-game. The prior is interpretable, computed
from data, and slots into one existing function in the proposer.

## What problem we're solving

The audit established two things:

1. We launch ~half as often as top-10 in absolute terms (the
   universal "aggression deficit" — already known from earlier
   audits, not what this design fixes).
2. **Within our launches, we allocate them to the wrong planet
   classes.** Top-10 spends roughly a fifth of its launches on
   low-production rotating inner planets; we spend an eighth on
   that class and instead put more weight on the heavily-garrisoned
   big-static planets that look high-value-on-paper but cost more
   ships to take.

The agent's current proposer (`agents/baseline/proposer.py:205`)
ranks candidate launches by a single cheap formula:

    cheap_marginal_value = 0.05 * target.production * pv_horizon
                          (for the capture branch)

That formula is proportional to production, which is exactly the
bias that pushes us toward high-production planets. The prior we're
about to design adjusts that ranking so top-10's actual allocation
pattern emerges from the agent's choices, not from a hard-coded
class boost. This follows Rule 40 (prefer modeling-correctness
over restriction-tuning): we're correcting the value function, not
adding a band-aid.

## The three layers

### Layer 1 — static class prior `alpha(c)`

Eight numbers, one per geometry class
(`lib/per_planet_class.py::ALL_CLASS_LABELS`). Each number is the
difference between top-10's share of launches on that class and
our share, averaged over all 16 informative archetype cells:

| class | alpha |
|---|---|
| `low_prod_rotating_inner` | **+0.100** |
| `low_prod_static_inner` | +0.017 |
| `low_prod_static_outer` | +0.003 |
| `high_prod_rotating_outer` | 0.000 |
| `low_prod_rotating_outer` | 0.000 |
| `high_prod_rotating_inner` | −0.024 |
| `high_prod_static_inner` | −0.047 |
| `high_prod_static_outer` | −0.049 |

(Source: `audit/2026-05-19-archetype-per-planet-class.json`'s
`cross_summary.share.mean_delta`. The two outer-rotating zeroes
come from the rotation-radius limit excluding outer planets from
the rotating bin — those classes are empty in practice and stay
neutral.)

Positive alpha means "top-10 prizes this class above us"; we boost
candidates that target it. Negative alpha means "we waste shots on
this class relative to top-10"; we down-weight it.

### Layer 2 — per-planet feature function (unchanged)

The proposer already computes a per-candidate score from
production, capture cost (via `capture_size`), arrival time, and
threat ETA. That stays as-is. The prior multiplies the result; it
does not replace the per-planet math. This keeps the existing
threat / reinforce branches working, and means a broken prior
(`lambda = 0`) reduces the agent to its current behaviour exactly.

### Layer 3 — opponent posterior `gap(c, t)`

Each turn, look at every in-flight enemy fleet (we already see
them in the observation), infer each fleet's intended target via
`lib.fingerprint._infer_target`, classify the target with
`lib.per_planet_class.classify_planet`, and tally the share per
class. That snapshot is the opponent's recent targeting
distribution — no per-turn state needed because fleets in flight
are themselves a sliding window of recent launches.

Compare each class's observed opp share against the top-10
empirical share for the same class:

    gap(c, t) = top10_share(c) - opp_share_in_flight(c, t)

Positive gap means the opponent is under-targeting this class
relative to a strong-play baseline — we can grab those planets
cheaply. Negative gap means the opponent is concentrating on this
class — contested, so down-weight. (For v1 we treat both
directions symmetrically; if the symmetric treatment underperforms
we can split into separate "grab cheap" and "avoid contested"
coefficients in v2.)

### Combining the three

For each candidate target with class `c`:

    priority(c, t) = exp( lambda_alpha * alpha(c)  +  lambda_gap * gap(c, t) )

    weighted_cheap_delta = cheap_marginal_value * priority(c, t)

Exponential combination has three properties we want:

- Always positive, so multiplying a positive `cheap_delta` keeps
  the sign and a negative one stays a rejection candidate.
- Composes additively in the exponent, so the two effects layer
  cleanly.
- A `lambda = 0` knob disables either effect for ablation.

Default magnitudes (back-of-envelope from the alpha numbers and
the production ratio between high-prod and low-prod planets,
which is roughly 3×): `lambda_alpha = 3`, `lambda_gap = 2`. A
class with alpha = +0.10 and zero gap then gets a `exp(0.3) ≈ 1.35×`
boost. Sweep over `lambda_alpha ∈ {0, 1, 3, 6}` on the seed panel
to find the empirical optimum.

## Where it slots in

Single injection point: `cheap_marginal_value` in
`agents/baseline/proposer.py:169`. The function currently returns
a float per `(src, tgt, ships, eta)`. Wrap its return value with
the priority multiplier. Pre-compute `board_medians` and
`opp_share_in_flight` once per agent call (turn-0 setup in
`agent()` at `agents/baseline/main.py:80`) and pass them through
`propose(...)` as a small dict.

This is a deliberately narrow injection. The chooser
(`chooser.py:60`) keeps doing fast-sim validation with the
existing leaf value function (`value.py:33`). The prior steers
which 60 candidates reach Stage-2 validation; the chooser
remains the final arbiter, so a bad prior cannot directly cause
a bad action — it can only fail to surface a good one.

## What state we need

The agent today is a pure function of the observation. The opponent
posterior is computed from in-flight fleets only, so it stays pure
too — no module-level state, no episode reset, no risk of
cross-game contamination.

If a later iteration wants a longer memory (an exponential
moving average over launches across turns, not just in-flight
fleets), that requires per-episode state. Defer until v1 ships.

## How we'll validate

1. **Self-play vs current submission** on the 128-seed geometry
   panel (`python fast.py eval <new> --vs <current> --geometry-panel
   --by-archetype`). Target: >55 % winrate against the current
   live submission across the panel, no regression > 5 pp in any
   single archetype.
2. **Per-class share recomputation.** Run the new agent through
   `scripts/archetype_per_planet_class_audit.py --all-cells` against
   a strong opponent on fresh replays. The post-fix
   `target_share_delta` for `low_prod_rotating_inner` should drop
   from +0.10 toward 0; over-allocation on `high_prod_static_*`
   should shrink toward 0 too. Reporting metric: how much of the
   share gap on the top-3 classes closes.
3. **Lambda sweep.** `lambda_alpha ∈ {0, 1, 3, 6}` ×
   `lambda_gap ∈ {0, 2, 5}` on a smaller 32-seed panel. Read off
   the (winrate, share-gap-closed) Pareto frontier; pick a single
   point for the seed-panel A/B.
4. **Submission gate** (per Rule 12 / Rule 27): if the chosen
   point beats the current submission by ≥1 standard deviation on
   the seed panel AND closes >50 % of the share gap, push as one
   of the day's submissions.

## Six-question preflight (Rule 16)

1. **Already explored?** No. The v15–v20 line was chooser fixes
   (Stage-2 validation). v8 had a class concept (in the
   `top-performer-strategies` doc) but never wired class weights
   into the proposer. This is genuinely new ground.
2. **Rank-lock-vulnerable?** Moderate. The prior coefficients
   come from a 16-cell, ≤6-games-per-cell sample of top-10
   replays; the +10 pp signal on `low_prod_rotating_inner` could
   shrink with more data. Mitigation: keep `lambda_alpha`
   small enough that the underlying production-driven ranking
   remains dominant; the prior nudges, not overrides.
3. **Predicted standalone result?** +1 to +3 percentage-point
   winrate on the seed panel against the current submission,
   driven by share-gap closure on the three flagged classes.
   Plausible larger effect if the share gap is causally tied to
   the absolute aggression deficit, less if it's purely
   correlational.
4. **Correlation with previous fixes?** The Tier-1-opp chooser
   experiment (asymmetric, reverted in `f28c9fc`) was working on
   the same problem from the chooser side and broke on
   common-random-number panels. This works from the proposer side,
   which is orthogonal; if it ships, the chooser-side experiment
   can be retried on top.
5. **Precedent?** Best-response with opponent-conditioned priors
   is standard in poker (hand-strength prior updated by betting
   patterns) and in partially-observable RL (attention-weighted
   value functions). The closed-form Bayesian version here is the
   minimal viable instance.
6. **Does the training objective match the comp metric?** Comp
   metric is TrueSkill rating from head-to-head games. The prior
   targets per-game winrate against strong opponents (top-10
   share-matching). Winrate compounds into TrueSkill, so yes.

## What could go wrong

- **The share gap is not causal.** Maybe top-10 hits
  `low_prod_rotating_inner` because their main-line strategy
  generates spillover fleets that happen to land on those planets,
  not because the class is valuable. If true, the prior boosts a
  symptom, not a cause; fix won't help. Detector: ablation
  (run with `lambda_alpha = 0`, see if winrate matches the
  current submission within noise).
- **Geometry classes are too coarse.** Eight classes might lump
  together planets with very different roles. Detector: per-cell
  winrate gap on the panel — if some cells improve and others
  regress sharply, the binning isn't capturing the real
  distinction.
- **Opponent posterior is too noisy.** With only a handful of
  in-flight fleets at any moment (~3–8 enemy fleets), the share
  estimate has high variance. Detector: run with
  `lambda_gap = 0` (static-only) — if static beats static-plus-gap,
  the posterior is hurting.
- **Default coefficients are wrong.** `lambda_alpha = 3` is a
  guess. Detector: the sweep itself; we pick the empirical max.

## Out of scope for v1

- IL on top-10 trajectories. Defer until we know the closed-form
  prior is at least neutral on the panel; jumping to IL skips the
  diagnostic step.
- Per-archetype prior tuning. v1 uses one global per-class alpha;
  per-archetype refinement (`alpha(c | archetype)`) is v2 if the
  global prior helps but plateaus.
- Trained opponent classifier (`Tier-2`). The placeholder in
  `lib/opp_model.py:128` exists but is unimplemented; we don't
  need it for this design.
- 4-player games. The audit is 2-player; extending to 4P needs a
  separate per-class share recomputation against the 4P top-10
  corpus.

## Next-session checklist

1. Read this doc, `HANDOVER.md`, and
   `audit/2026-05-19-archetype-per-planet-class.md`.
2. Write `lib/priority_prior.py`: contains the alpha table, the
   in-flight opp-share calculator, and the
   `priority(class, board_medians, world, my_id, lambda_alpha,
   lambda_gap) -> dict[planet_id, float]` function.
3. Modify `agents/baseline/main.py::agent` to compute
   `board_medians` and `opp_share_in_flight` once per call, and
   pass them into the proposer.
4. Modify `agents/baseline/proposer.py::cheap_marginal_value` to
   accept an optional `priority_by_planet` argument and multiply
   the return value when provided.
5. Add a unit test: with `lambda_alpha = lambda_gap = 0`, the
   agent's per-turn actions are byte-identical to the current
   submission on a fixed seed (Rule 38: fix-verification
   reproduces the failure state — here, the baseline state).
6. Lambda sweep, panel A/B, submission decision (per the
   validation section above).
