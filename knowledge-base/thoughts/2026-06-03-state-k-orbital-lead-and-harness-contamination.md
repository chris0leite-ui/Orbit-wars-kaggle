# 2026-06-03 — state-K orbital-lead fix + a harness-contamination finding

Session: reviewed the three "adaptive K" levers, picked the state-driven
horizon (Lever 1) to improve, shipped a fix, found a harness bug.

## What shipped

`baseline_state_k_orbital_lead` (sub 53316984) — state-driven-K with the
**orbital-lead** fix to its contest tick. The shipped state-K computed
"earliest enemy interference" by aiming the enemy at the target's CURRENT
position; for orbiting targets that mis-estimates the intercept. The fix
keeps the conservative launch-now timing but leads the moving target.

The path mattered more than the destination:

1. **Probe-first killed the obvious fix.** The plan's approved one-liner
   (`arrival_eta=our_eta`) would have *raised* K ~+12 everywhere — it
   assumes the enemy stays asleep until we arrive. A 4-game probe showed
   this before any A/B spend. The *correct* fix (launch-now + lead) is
   unbiased (~50/50 up/down, mean ≈ 0). Probe-first earned its keep twice:
   once to confirm the defect is real (94%), once to reject the naive fix.

2. **The real test is the leaderboard.** Local evidence was ~parity (n=4),
   so PI sent it to the ladder rather than burning slots chasing a
   contaminated local signal. Correct call given #3.

## The finding that may matter more than the fix

`clean_ab` runs both A/B agents in ONE process; config via
`os.environ.setdefault` is process-global, so single-variable A/Bs
(variant-ON vs variant-OFF of the same baseline) **contaminate** — both
agents run the first-loaded value → X-vs-X → noise. Demonstrated directly
(audit/2026-06-03-clean-ab-env-contamination.md).

If this has been silently corrupting single-variable A/Bs, then some
`mechanism-ledger` "dead" verdicts — and the chronic local-vs-live gap
(Rule 43) — may be partly **measurement error, not real nulls**. The
unsettling implication: we may have discarded working ideas. Worth a
deliberate re-audit of a couple of high-value "falsified" levers with a
contamination-proof harness.

## Open questions

- Does orbital-lead beat state-K's 1155.4 on the ladder, or land ~parity?
- How many ledger nulls were single-variable `clean_ab` runs? (triage list)
- Should the bundler fix (multi-line imports) merge to `main`? It's a
  pre-existing blocker independent of strategy.
