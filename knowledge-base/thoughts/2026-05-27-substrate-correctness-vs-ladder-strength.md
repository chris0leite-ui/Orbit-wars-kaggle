# 2026-05-27 — substrate correctness vs ladder strength

This session's random-elim-gate work surfaced a useful distinction
worth naming: **substrate correctness** and **ladder strength** are
separate axes, and a bundle can fail one while excelling at the
other.

## The empirical evidence

- `submissions/baseline_joint_aggr_consolidated_orbitfix.py`
  settled at μ=1165.4 on the live ladder (sub 52912707) — the team
  peak at the time. It also **fails the candidate Rule 48 gate
  vs nearest** (14/16 ELIM, 2 step-500 score-wins with focal
  owning 27/40 and 35/40 planets vs opp's single defended
  pocket).

- The bug that causes the gate failure is the documented
  `midgame-filter-overrejects-in-dominant-endgame` pattern: the
  B1 hold filter on `agents/baseline/proposer.py:
  _target_holdable_after_capture` refuses to attack opp's last
  planet because the (own!) neighbouring planets are flagged as
  potential counters. Documented in session-EqJuT's postmortem,
  fix committed (`68c24be`) only on `agents/lagrange_simple`.

- The bug doesn't surface on the ladder because real
  competitive opponents virtually never reach the "1 stubborn
  enemy planet" endgame state. Competitive games end via concede
  / score-tiebreak / midgame elimination, not via single-planet
  endgame stalemate.

## What this means

A substrate-correctness gate (vs random, vs nearest) probes
**a different region of game-state space** than the ladder.
Specifically:

- Gates probe **dominant-endgame mechanics**: can the agent
  actually finish a game it has clearly won?
- Ladder probes **competitive mid-game tempo**: can the agent
  beat peer-level agents in the 0-200 step window before
  resolution?

An agent can excel at one without the other:
- **Pass gate, weak on ladder**: an agent that always plays for
  full elimination but is bad at competitive opening / midgame
  tactics. Wins clean games it dominates; loses to peers in the
  midgame.
- **Strong on ladder, fail gate** (the orbitfix case): an agent
  with strong midgame tactics + tempo, but a substrate bug that
  only fires in the rare full-elimination state.

These call for different fixes. Ladder regressions ask for
*more strategy / better mechanism*. Gate failures ask for
*substrate correctness* — usually a small filter / predicate
fix, low risk, low ladder impact.

## Implication for promotion of Rule 48

PI promoted "100% win-by-elim vs random at n=16" in session-EqJuT
as a pre-submit gate. This session's evidence suggests the gate
catches a class of bugs (dominant-endgame correctness) that the
ladder distribution doesn't probe. The gate's value is **not**
"correlates with ladder μ" — it's **"catches substrate bugs the
ladder won't"**. Treating it as a strength proxy would be a
miscalibration; treating it as a correctness filter is correct.

## Side observation

The Vjaz9 kt_p23 v8 result this session is interesting in this
frame: v8 added an endgame-elim bonus to the value head, but my
measurement shows it didn't improve the random gate (still 6/16
ELIM) AND v8 actively LOSES to nearest 5/16. That means v8 is a
substrate-correctness REGRESSION on the kt_p23 lineage even
though it was framed as a substrate-correctness FIX. The
explanation is probably that the v7→v8 commit shipped a value-head
swap (`BASELINE_VALUE_HEAD=attack_pull`) in the same change as the
endgame bonus, and the head swap is the regressor. Cause-isolation
discipline matters — bundling two changes makes the A/B unable
to attribute.

## Adjacent

- `audit/random_elim_gate_results.md` — empirical numbers.
- `state/MULTI_BRANCH.md` — cross-branch finding section.
- session-EqJuT's `2026-05-23-postmortem-session-EqJuT.md` —
  the PI promotion of the gate.