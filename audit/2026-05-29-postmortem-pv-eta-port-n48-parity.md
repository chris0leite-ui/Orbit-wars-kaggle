# Postmortem — 2026-05-29 PM (claude/game-theory-winning-strategy-SEU7P)

Session focus: resolve the conflicted merge of
`origin/claude/kaggle-submission-review-gZsCu` into SEU7P (perf
chain branch), then decide whether the resulting bundle is submit-
worthy via subprocess-isolated A/B vs the frozen µ=1163.5 PV_ETA
anchor.

## What went wrong

1. **Wrapper-bundle SyntaxError ate 41 min of compute.** The first
   subprocess-isolated A/B against the anchor came back **0/32**.
   Read as a strategic catastrophe. Was a duplicate
   `from __future__ import annotations` line — one from the anchor
   preamble I prepended (line 4), one from the SEU7P bundle body
   (now at line 23 in the concatenated file). Python requires the
   directive at file start; the second occurrence at line 23 raises
   `SyntaxError`. The kaggle env caught it at agent load and
   forfeited every game. A 10-s single-game smoke or
   `ast.parse(open(p).read())` would have caught this in pre-flight.
   Rule 46 was applicable in spirit (bundle smoke before
   compute-intensive use) but not in letter (we weren't
   submitting). See friction
   `wrapper-bundle-duplicate-from-future`.
2. **First-batch positive estimate didn't corroborate.** First
   subprocess n=32 batch came in at 19/32 = 59.4%, Wilson
   [0.42, 0.74]. Mentally framed as "real positive signal." The
   +16-game extension (seeds 16-23) landed at 6/16 = 37.5%, pulling
   combined n=48 to 25/48 = **52.1%**, Wilson **[0.38, 0.66]** —
   parity. Rule 45's Wilson-lo ≥ 0.50 gate protected the submit
   decision (no submission went out); the interpretive lesson is
   that a single n=32 point-estimate of 59.4% should not have been
   characterized as "real signal" without n=48+ corroboration. See
   friction `n32-first-batch-positive-signal-doesnt-corroborate`.
3. **Nearly shipped the wait-grid strip on commit-body trust.**
   Commit `418ab08` ("strip the dead wait-grid mechanism")
   contained a rationale paragraph: *"PI's framing: committing to
   a future launch is the wrong semantics — information updates
   every turn, so re-deciding 'fire now or wait' each turn is
   correct."* I read this as PI-ratified and was on track to
   incorporate the strip via merge. PI intervened: *"Be careful.
   They removed the waiting."* Investigation confirmed the strip
   removed wait_N>0 candidate *scoring* entirely (not just the
   silently-disabled ledger): `wait_then_fire_variants` and
   `min_wait_affordable` in `agents/baseline/proposer.py`
   enumerated the "src can't fire-now-afford tgt, but could after
   N turns" option, which the strip eliminated. Aborted merge,
   cherry-picked `c45cf00` (PV_ETA alone), preserved the wait-grid.
   Without PI intervention I would have shipped a regression. See
   friction `pi-intervention-prevented-shipping-the-strip`.

## Decisions taken — quality assessment

| Decision | Quality | Note |
|---|---|---|
| Abort gZsCu merge, cherry-pick PV_ETA alone | GOOD | Correct response to PI's flag; preserved the wait-grid. |
| Build A/B wrapper by concat (anchor preamble + SEU7P body) | BAD | Did not strip duplicate `from __future__`; no pre-flight smoke. Cost: 41 min CPU on a wipeout that 30 s of pre-flight would have caught. |
| Accept cherry-pick's ancillary knobs wholesale (FOLLOWON / MIN_DELTA / SHIP_TURN_KAPPA / NEUTRAL_BONUS multiplications) | ACCEPTABLE | All default-OFF, byte-for-byte parity preserved; but I didn't surface the breadth of the cherry-pick to PI at decision-time. |
| Hold at n=32 instead of submitting at Wilson-lo 0.42 | GOOD | Rule 45 followed. |
| Run +16 extension at PI's request | GOOD | Confirmed first batch was variance. |
| Hold at n=48 = 52.1% parity, do not submit | GOOD | Rule 45 + Rule 43 both fail; n=48 cannot distinguish lift from regression. |

## PI overrides this session

- **1 corrective intervention**: "Be careful. They removed the
  waiting." Calibration: my reading of the commit body had me on
  track to ship the strip. PI's domain knowledge of what the
  wait-grid does was load-bearing. Score: I had not internalised
  the strip's behavioural impact from the diff before trusting the
  commit message.
- **2 directive interventions** (no override): "do A/B", "run
  another 16."

## Frictions logged this session

Cross-linked to `audit/friction.md` 2026-05-29 PM block:

- `wrapper-bundle-duplicate-from-future`
- `rule-46-bundle-smoke-skipped-before-ab`
- `n32-first-batch-positive-signal-doesnt-corroborate`
- `pi-intervention-prevented-shipping-the-strip`
- `cherry-pick-pulled-ancillary-knobs-silently`

## Promotion candidates

Three drafted (Rule 46 extension to ≥5-min compute steps; Rule 45
interpretation sub-clause on n=32 first-batch framing; Rule 44
sub-clause on verifying the diff over the commit body).

**PI verdict: nothing to add or promote.** Candidates not
promoted to `.claude/skills/kaggle-comp/improvements.md`. Frictions
remain in `audit/friction.md` for future re-evaluation if the same
pattern recurs and the cost evidence accumulates.

## PI additions (from step 4)

None.

## Outcome summary

- **Branch state at session end:** ahead 30 / behind 0 vs
  `origin/main`. Tip: `07b2067` (anchor file imported from gZsCu).
- **A/B result:** focal `baseline_pv_eta_seu7p.py` (SEU7P perf
  chain + PV_ETA cherry-pick + preserved wait-grid) vs frozen anchor
  `baseline_pv_eta_anchor_1163.py`. **n=48: 25/48 = 52.1%, Wilson
  [0.383, 0.655].** Per-seat: P0 = 9/24 = 37.5%, P1 = 16/24 = 66.7%
  (30-pp seat asymmetry, unexplained, deferred).
- **Submission:** none. Rolling pair unchanged
  (`baseline_validated` µ=1109.9, `baseline_leaf_pv_2p` µ=1086.4).
- **No LB slot consumed.**

## Framework version at session-end

- Commit SHA: `07b2067` (pre-wrap)
- Active rules: 1..49 per `CLAUDE.md ## Operating rules`
- Loaded skills this session: `postmortem`, `kaggle-comp` (implicit)
