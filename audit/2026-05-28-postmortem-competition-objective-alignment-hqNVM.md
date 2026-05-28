# Postmortem — 2026-05-28 competition-objective-alignment-hqNVM

## Session summary

Phase A of the learned-value-head program: distill the proven
`favor_hybrid` scalar value function (the head behind the EVICTED
μ=1149 team peak) into the 21889-param MLP from 40 hand-crafted
features, then A/B `baseline_learned` vs `baseline_hybrid` to test
whether the chooser-with-learned-head wiring is sound.

Result: **14/32 = 43.8 % wins, Wilson 95 % CI [0.282, 0.607],
INCONCLUSIVE near-parity.** v1 (margin-on-lite_greedy self-play) was
2/32 = 6.2 % on the same harness. The 38 pp jump is what we wanted
to see — wiring works, features are mostly sufficient, the v1 failure
was target + data not architecture.

Distillation R²: ~99.8 % (val_rmse 48.4 vs y_std ~1029).
Inference latency under chooser load: p50=164ms, p95=240ms, max=459ms
(env actTimeout 1000ms — well within budget).

## What went wrong

Nothing flagged. The Phase A cycle ran cleanly:
- Distillation training: clean convergence, val_rmse 370 → 48 over
  18 epochs.
- A/B harness: auto-bumped n=16 → n=32 (Rule 45 compliance).
- Inference latency: within budget at chooser-call density.
- Commits: clean, descriptive, no session-URL leak (Rule 39 held).

Two observations worth tagging (in `audit/friction.md` 2026-05-28) but
neither a real failure:
1. `phase-a-bench-vs-evicted-not-rolling-pair` — A/B was vs the
   EVICTED μ=1149 baseline, not the live rolling-last-2. This was
   correct for a substrate diagnostic but means we have ZERO live
   calibration. Mandatory before Phase B submit: Rule 43 panel +
   Rule 45 n≥32 vs current rolling champion.
2. `n32-inconclusive-still-a-pass-for-diagnostic` — Wilson CI
   [0.282, 0.607] crosses 50 %, formally INCONCLUSIVE. Calling Phase
   A a "pass" is justified by the magnitude jump from v1 (6 % → 44 %)
   on a SUBSTRATE diagnostic. A SUBMISSION decision at the same
   evidence level would be a hard FAIL under Rule 45 — the two
   thresholds are NOT the same and conflating them later would be a
   real error.

## Frictions logged this session

See `audit/friction.md` `## 2026-05-28` heading:
- `phase-a-bench-vs-evicted-not-rolling-pair`
- `n32-inconclusive-still-a-pass-for-diagnostic`

## Promotion candidates (PI ratified: PENDING)

**Candidate 1 — diagnostic-vs-submission gate distinction.**

```markdown
### [ ] CLAUDE.md — distinguish diagnostic gates from submission gates

**Tag:** `n32-inconclusive-still-a-pass-for-diagnostic`
(2026-05-28 Phase A wrap)

**Where to insert:** sub-clause under Rule 45 (n ≥ 32 minimum for
A/B lift claims).

**What to add:**
Rule 45 (n ≥ 32, Wilson-lo ≥ 0.50) is a SUBMISSION gate. A SUBSTRATE
diagnostic ("does this wiring work?", "do these features carry signal?",
"does this loss converge?") is graded on magnitude-of-change vs a
known-broken or known-strong reference, not on Wilson-lo crossing a
threshold. A diagnostic-pass artifact must NEVER be promoted to a
submission candidate without re-clearing the submission gate.

**Why:** Phase A wrap on 2026-05-28 — `baseline_learned` 14/32 vs
`baseline_hybrid`, Wilson [0.282, 0.607], called PASS for diagnostic
(v1 was 2/32) but would FAIL as a submission gate. The wrap correctly
held the artifact back from a submission slot; conflating the two
thresholds in a future session would burn a slot.
```

**Candidate 2 — substrate diagnostics need a sibling "live-ladder
calibration" pass before promotion.**

```markdown
### [ ] CLAUDE.md — substrate-diagnostic must pair with live-ladder calibration

**Tag:** `phase-a-bench-vs-evicted-not-rolling-pair`
(2026-05-28 Phase A wrap)

**Where to insert:** addition to Rule 43 (multi-opponent panel
mandatory pre-submit).

**What to add:**
When a substrate-only diagnostic clears (wiring sound, features
sufficient, etc.), the artifact remains a substrate result — not a
submission candidate — until a SEPARATE A/B against the CURRENT
rolling-last-2 has been run. The substrate-diagnostic opponent (often
historical, often EVICTED) is the wrong calibration target for the
ladder. A "substrate green" badge is not a "live green" badge.

**Why:** Phase A `baseline_learned` A/B target was `favor_hybrid`
(μ=1149, EVICTED). Substrate test was clean but did NOT measure
performance against the live rolling pair (μ=806 / μ=829), which is
what the ladder evaluates against. Skipping the live-calibration step
on a Phase B candidate would risk a rolling-pair eviction event
similar to the 2026-05-20 five-step chain (~320 μ lost in 24 h).
```

Both candidates are PI-pending. Do not commit to `improvements.md`
without explicit ratification.

## PI additions (from step 4)

PENDING — to be appended after PI replies to the verbatim postmortem
question.

## Framework version at session-end

- Branch: `claude/competition-objective-alignment-hqNVM`
- HEAD commit: `fb74d22` (Phase A distillation cycle — wiring verified)
- Ahead of `origin/main`: 5 commits
- Active rules: 1..47 (CLAUDE.md `## Operating rules — concise`;
  rules 41-47 added 2026-05-20)
- Loaded skills this session: `postmortem` (this skill);
  `kaggle-comp` (implicit via project loop)
- Comp deadline: 2026-06-23 23:59 UTC (26 days remain at session-end)
