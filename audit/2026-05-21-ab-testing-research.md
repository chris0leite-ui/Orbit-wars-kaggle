# 2026-05-21 — Fast and reliable A/B testing for Orbit Wars

> Research synthesis. No new code; inventory of what exists + the
> failure history that calibrated it + a decision tree the next
> session can follow without re-deriving from first principles.

## TL;DR — what to run, in order

For any candidate agent, walk this ladder. Stop at the first FAIL.

1. **Smoke (60 s).** `python fast.py smoke <agent>` — vs random + nearest, n=32 each. Catches dead-on-arrival and silent crashes (turn-ms p95 < 30 ms = agent short-circuiting).
2. **Single-opponent triage (5-10 min).** `python fast.py eval <agent>` — adaptive Wilson-gated vs v7_0. Default `--max-seeds 64`, early-stop when Wilson 95% CI clears or fails the 0.55 gate. **Triage only.**
3. **Champion h2h, n ≥ 32 (10-15 min).** `python fast.py eval <agent> --vs <current_rolling_champion>` with explicit `--max-seeds 32 --gate 0.50`. **Required by Rule 43.**
4. **Multi-opponent panel + champion h2h, n ≥ 32 each (30-45 min).** `python fast.py eval <agent> --vs-panel default --require-h2h <champion> --max-seeds 32`. EVERY opp must clear Wilson-lo ≥ 0.50; pooled winrate does NOT count. **Pre-submit gate, Rule 43.**
5. **Geometry panel for tail regressions (60 min, optional).** `python fast.py eval <agent> --geometry-panel --by-archetype --full-panel` — 32 archetypes × 2 seats. Use when (a) the aggregate winrate looks fine but you suspect a flavour-conditional hole, or (b) you're considering a default change rather than an opt-in addition.

Only after all of (1)-(4) clear: bundle + parity smoke (Rule 46), then submit (Rule 42 push claim board first).

---

## Existing infrastructure (one row per tool, scannable)

| Tool | Question it answers | Default n | CRN | Verdict cost |
|---|---|---|---|---|
| `fast.py smoke` | Crashes? Beats random + nearest? | 32 per opp | yes (balanced seats) | 60 s |
| `fast.py eval <a>` | Beats v7_0 baseline? | adaptive 16→32→64 | yes | 5-10 min |
| `fast.py eval <a> --vs X` | Beats specific opp X? | adaptive | yes | 5-10 min |
| `fast.py eval <a> --vs-panel default --require-h2h C` | Beats the 3-opp panel AND champion C? | adaptive per opp | yes | 30-45 min |
| `fast.py eval <a> --geometry-panel --by-archetype` | Consistent across 32 archetypes? | 128 seeds × 2 seats | yes | 17-60 min |
| `fast.py play <a> --seed S` | WHY did seed S go this way? | 1 game, verbose | n/a | 30 s |
| `fast.py bench <a>` | p95 turn-ms vs 1000 ms budget? | 3 games | n/a | ~3 min |
| `scripts/tournament.py` | Round-robin matrix across N agents | configurable | yes | scales |
| `scripts/ffa_panel.py` | 4P FFA focal-vs-focal with fixed bg | configurable | yes | scales |
| `scripts/ab_variants.py` | Multi-anchor gate (per-anchor not pooled) | configurable | yes | scales |

Substrate that makes A/B fast:

- `lib/fast_sim.py` — ~20× speedup vs `env.clone()+step()` for any A/B whose agents simulate forward.
- `ProcessPoolExecutor` with 8 workers (fast.py default). One game ≈ 8-20 s; 32 games × 2 seats ≈ 80-160 s wall-clock.
- Adaptive tier-doubling with Wilson early-stop: decisive wins/losses settle at n=16-32; only inconclusive cases pay for n=64.

---

## Reliability principles (each paid for in friction.md)

### 1. n ≥ 32 minimum for any submission gate (Rule 45)

Wilson 95% CI width at n=16 is ≈ 0.45. You cannot distinguish parity from a 20-pp regression. Two confirmed false-positive submissions paid for this rule:

- `n16-falsely-shows-parity`: v21 at n=16 = 8/16 = 50.0% (Wlo=0.28) read as parity; same agent at n=32 = 10/32 = 31.2% Wlo=0.18, clear FAIL. Burned 4 ablations before the n=32 reveal.
- `small-n-ab-noise-misled-panel`: 5/8 = 62.5% smoke escalated to a 70-min panel that landed at 12/32 = 37.5%.

n=16 is for **triage only** ("agent doesn't crash, isn't obviously broken"). n=32 minimum for any verdict that gates a submission.

### 2. Multi-opponent panel + champion h2h (Rule 43)

Single-opponent A/B is BANNED as sole evidence for a submit. Reason:

- **Non-transitive loops exist.** A>B>C>A pairwise: 14/20 vs B, 6/20 vs C, pooled 60% — looks borderline; per-anchor view rejects. `_per_anchor_summarise` + `_anchor_gate` in `scripts/ab_variants.py` is the catch-all; the fast.py `--vs-panel` flag inherits the same semantics (every opp must clear, not the average).
- **Ladder mixture ≠ any one panel opponent.** `local-AB-not-calibrated-to-live-ladder`: 0/16 vs one opponent submitted as plausible; settled live at μ=711.5 against a 1122 baseline.
- **Panel-pass without champion h2h fires same-family blindspot.** 4× recurrence (v13/v14/v17/v18 all panel-PASSED, h2h vs v12 was 47%). `fast.py eval --vs-panel` is hard-coded to REFUSE without `--require-h2h <champion>` (escape valve `FAST_PY_SKIP_H2H_GATE=1`, do not use).

The fast.py source carries the gate; the workflow rule is: never strip `--require-h2h`.

### 3. CRN (common-random-numbers) is the variance lever — but features can cancel in Δ

Same seed × both seats (P0 + P1) for variance reduction. `_balanced_pairs` in `fast.py` does this automatically.

**Caveats:**

- `crn-cancellation-blunts-leaf-scorer-features`: leaf-scorer additions where the same opp_traj appears in both baseline and candidate leaves cancel in Δ. F4 vulnerability-penalty regressed across 3 variants for this reason. Modifications to leaf-scoring need a stronger evaluator than panel lift — head-to-head vs same-family is the real signal.
- `dogpile-overestimates-without-reactive-opp`: K-step fixed-opp-rollout invariance means joint candidates over-estimate without reactive opp. Action-space expansion needs reactive opp FIRST.
- `crn-symmetry-broken-without-reading-prior-audits`: asymmetric chooser (different opp_traj in baseline vs candidate) gave 0/32. Both legs of `leaf(action) - baseline` MUST use the SAME opp trajectory.

### 4. Don't trust in-process A/B when configs differ

`same-process-pv-shared-state`: `lib.scoring.PV_GAMMA` is a module-level constant set once at import. Whichever agent triggers the import first wins for both. Always use `fast.py eval` (separate worker processes); never trust same-process numbers when agents need different env vars.

### 5. Mechanism sanity-print before trusting the winrate

`broken-mechanism-yields-fake-positive-signal`: v2 cluster overlay panel showed 67% lift vs v7_0 — actually a classifier bug that force-routed every board into cluster 3 (a high-cadence template). After fixing the classifier, v3 collapsed to 53%. The "encouraging" number was the bug.

**Cheap fix:** before treating an A/B result as signal, run a 30-second `fast.py play --seed <N>` and confirm the agent is actually doing the thing it's supposed to do (cluster distribution, launch count distribution, mission-mix). Three lines of print > a 70-min panel rerun.

### 6. Live μ in the first ~10 games is noise (Rule 36 caveat)

`early-trueskill-mu-unreliable`: v12's ladder settled from 1217.7 → 1099.3. The +97μ "huge gain" was an early-window low-sample artifact. TrueSkill σ ≈ 300 at submit, shrinks ∝ 1/√N. Wait ≥ 6 h post-submit before basing strategic decisions on a new submission's μ; ≥ 24 h before treating it as settled.

### 7. Bundle + parity smoke before submit (Rule 46)

Bundler has 5 known silent-fail modes (multi-line imports, aliased imports, cross-agent imports, float tie-breaking, missing symbols). `composite_a2_hybrid` (sub #52744234) ERROR'd on an absolute import the local tests didn't catch — ~1 LB slot lost. Required:

```
python scripts/bundle_agent.py <agent>
pytest tests/test_bundle.py
python fast.py play <bundled_submission>
```

---

## Speed levers (paid-for, ranked by lift)

1. **Adaptive Wilson early-stop.** Decisive PASS/FAIL settles at n=16-32; only borderline cases pay the n=64 cost. Built into `_eval_vs_one`. (Caveat: sequential testing inflates type-I; for gate-calibration use case the effect is small at our tier thresholds 16/32/64, but treat any "PASS at n=16, marginal" with suspicion and re-run at n=32 anyway.)
2. **`lib/fast_sim.py`.** ~20× speedup over `env.clone()+step()`. Any A/B whose agents simulate forward should use it. Parity-pinned by `tests/test_fast_sim_parity.py`.
3. **8 parallel workers** (fast.py default `--workers 8`). One game ≈ 8-20 s; 32 games × 2 seats / 8 workers ≈ 80-160 s wall.
4. **Smoke before eval.** `fast.py smoke` is 60 s; catches the agent-crash-returns-empty failure mode (`agent-exception-swallowed-by-kaggle-env`) before you spend 10 min on an A/B that's measuring a broken agent.
5. **Single-game trace before re-running panels.** When an A/B disagrees with intuition, `fast.py play --seed <losing_seed>` answers "why" cheaper than rerunning the A/B at higher n. The btjeK H44 finding (65% fleet-destroyed-in-flight) came from exactly this loop.

---

## Open infrastructure gaps (actionable)

| Gap | Impact | Effort |
|---|---|---|
| `--vs-panel` is opt-in; no hard pre-submit gate at the workflow layer | Rule 43 is enforced by social convention + `--require-h2h` refusal, not by a submission wrapper. A `scripts/pre_submit_check.py` that runs the panel and refuses if any opp Wilson-lo < 0.50 would close the loop | small (~50 LOC) |
| No sample-size calculator front-and-center | PI has to know Wilson width ≈ 0.45 at n=16. A `fast.py n_for_gate <gate>` or doc line would help sizing | trivial (~10 LOC) |
| Geometry panel uniform; live ladder runs up to 9% on some cells, 0% on others | Tail-regression detection good; sample-efficiency for ladder-frequent cells underweight. A `SEEDS_DISTRIBUTION_MATCHED_64` would complement | medium |
| Wilson sequential early-stop is technically peeking-inflated | Practically small at our tier sizes (16/32/64); no doc says so explicitly. A one-line caveat + a sanity simulation would close this | small |
| 4P A/B coverage is thin | `scripts/ffa_panel.py` exists; not in `fast.py`. The ladder has 4P games; same-family non-transitivity in 4P is undertested | medium |
| Mechanism sanity-print isn't an enforced step | `broken-mechanism-yields-fake-positive-signal` could refire. A `fast.py introspect <agent> --seed 0` printing cluster/launch/mission-mix would make it cheap to run | small (~30 LOC) |

---

## Decision tree — what do I run for THIS change?

- **New chooser internals, expect lift in same-family head-to-head.** Run (3) champion h2h n=32 FIRST. If FAIL, single-game trace 4-6 losing seeds. Do NOT run the panel — same-family lift is the real signal; panel adds noise (`panel-misleads-head-to-head`, 4× fired).
- **New mechanism / value-head / proposer addition.** Run (1) smoke → (2) eval vs v7_0 → if PASS, (4) panel + champion. The panel catches non-transitive regressions a single-opp A/B misses (`local-AB-not-calibrated-to-live-ladder`).
- **Default-change (replacing a baseline behaviour, not adding opt-in).** Add (5) geometry-panel `--by-archetype`. Defaults regressing on flavour-tails hits the ladder hard because the panel uniform sampling underweights the tail (`med_low_prod__mixed_*` cluster, see `audit/2026-05-18-seed-panel.md`).
- **Hyperparameter sweep (one knob, several values).** Use `scripts/ab_variants.py` with `_anchor_gate` so the verdict is per-anchor not pooled. Don't pivot the knob and re-sweep more than 3× (Rule 37).
- **4P-specific change.** `scripts/ffa_panel.py` with focal-vs-focal + fixed background. The 2P panel is silent on 4P seat dynamics.

---

## What "fast and reliable" means in numbers

| Question | Cost (8 workers) | Reliability statement |
|---|---|---|
| "Doesn't crash, beats trivial floors" | 60 s | Smoke: Wilson-lo ≥ 0.55 vs random AND nearest, n=64 |
| "Beats v7_0 cleanly" | 5-10 min | Adaptive eval: Wilson-lo ≥ 0.55 at decisive tier |
| "Beats current champion" | 10-15 min | n=32 vs champion, Wilson-lo ≥ 0.50 |
| "Beats the panel + champion (submit-ready)" | 30-45 min | All 4 opp Wilson-lo ≥ 0.50 at n=32 each |
| "Won't regress on any geometry archetype" | 60 min | 32 archetypes × 2 seats, none < 25% winrate |
| "Why is seed 7 losing?" | 30 s | Single-game trace |

A complete submit-ready evaluation is 60-90 min of wall-clock on this container if you run all five tiers. Most candidates die at tier 2 or 3 in <15 min total; only candidates that look real pay for the full stack.

---

## Pointers

- Tools registry: `state/TOOLS.md`.
- Cross-branch state: `state/MULTI_BRANCH.md`.
- Rules: `CLAUDE.md` Rules 42-47 are the A/B-discipline rules.
- Failure history: `audit/friction.md` (tags above link to specific entries).
- Geometry panel: `audit/2026-05-18-seed-panel.md` + `lib/seed_panel.py`.
- Anchor-gate tests: `tests/test_ab_variants_gate.py` (5 unit tests; the pooled-vs-per-anchor case at line 148 is the demo of why pooled hides regressions).
- Bundle gate: Rule 46 + `tests/test_bundle.py`.
