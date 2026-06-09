# Commit body is rationale, not contract — verify behaviour from the diff

**Session:** 2026-05-29 PM, claude/game-theory-winning-strategy-SEU7P
**Context:** merging the gZsCu wait-grid strip + PV_ETA cherry-pick
into the SEU7P perf chain.

## What happened

Commit `418ab08` ("strip the dead wait-grid mechanism") was framed
in its commit body as a bug fix PI had pre-ratified:

> PI's framing: committing to a future launch is the wrong semantics
> — information updates every turn, so re-deciding "fire now or
> wait" each turn is correct. The accidental silent-prune ALREADY
> produced that semantics; the strip reclaims the wait-N scoring
> compute for additional fire-now candidates.

I read this as "PI agreed; the strip is semantically neutral; merge
through." I was on track to ship the strip.

PI intervened: *"Be careful. They removed the waiting."*

When I actually read the diff, the deleted code in
`agents/baseline/proposer.py` included `wait_then_fire_variants`
and `min_wait_affordable` — functions that enumerated "source X
cannot fire-now-afford target Y, but COULD after N turns of
production." Post-strip, those (src, tgt) pairs never enter prerank.
The agent loses the option to plan ahead for wait-then-fire
trajectories. This is a semantic change, not a bug fix.

The strip's commit body was *rationale* — what the author thought
the change did. It was not *contract* — what the diff actually does.
The two diverged. The author may genuinely have believed the strip
was neutral; PI's domain knowledge said otherwise; the diff
adjudicated.

## The general pattern

Commit messages capture **author intent at write time**, often under
incomplete information. The diff captures **what shipped**. When the
two disagree, the diff wins. This is especially true when:

1. The commit body cites an authority figure ("PI agreed",
   "consensus on the call", "reviewer signed off") to justify a
   semantic shift. The cited authority may not have seen the diff
   in detail; their endorsement may have been on a stated
   abstraction, not the concrete behaviour.
2. The change is framed as a "cleanup" or "bug fix" of code the
   author believes to be dead. Code is dead more rarely than authors
   think. The wait-grid was "dead" because the ledger that
   committed wait_N>0 launches was disabled; the wait-grid itself
   was very much alive in candidate *scoring*.
3. The diff is large but the body is short. Body length and diff
   length should track. A 5-line body explaining a 200-line strip
   is a smell.

## The cost

A near-miss on shipping a real regression. Caught by PI, not by
process. Should have been caught by me reading the diff before
trusting the body.

## What I'd want different

The minimal version: when a cherry-pick or merge commit's body
references "PI agreed / endorsed / ratified" on a semantic shift,
read the diff for the affected subsystem first, then verify the
behaviour change matches my expectation. Don't take rationale as
contract. This was drafted as a promotion candidate (Rule 44
sub-clause) but PI declined to promote — the friction stays in the
log as a one-off until/unless the pattern recurs.

## Related friction tags

- `pi-intervention-prevented-shipping-the-strip` (this session)
- `crn-symmetry-broken-without-reading-prior-audits` (origin of
  Rule 44 — same general failure mode: edit without reading
  state-of-truth)
- `wrong-file-recon-skipped-state-md` (origin of Rule 44)
