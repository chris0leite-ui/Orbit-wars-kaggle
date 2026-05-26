# Postmortem — 2026-05-23 session-EqJuT

## Session summary

PI directive: build the **simplest possible Lagrangian agent** that uses
our high-precision physics substrate + shadow-price coordination.
Maintainable, don't reinvent the wheel. Required gate: **100% win-by-
elimination vs `random`** at n=16 with random seat assignment.

Delivered: `agents/lagrange_simple/` (~400 LOC across 4 files) +
`scripts/random_elim_gate.py` + `tests/test_lagrange_simple.py` (12
tests). Final gate: **16/16 ELIM in 180s wallclock**. Cherry-picked to
`claude/session-EqJuT` and pushed.

## What went wrong

Decision-quality flags (per `knowledge-base/concepts/decision-quality-
vs-outcome-quality.md` — assessed against priors-at-decision-time,
not hindsight):

- **Bug 1 — Python truthiness gotcha on planet id 0.** Wrote
  `int(fate.hit_planet_id or -1) != int(tgt.id)` — when `hit_id == 0`,
  `0 or -1` evaluates to -1, silently dropping every shot at planet 0.
  Caused gate failure at seed 32966. Decision quality at write-time:
  BAD. The idiom is a known Python gotcha; should have written
  `x is not None` for nullable IDs from the start. Rule-gap: no rule
  flagged `or default` for nullable integer ids. Cost: 1 gate cycle +
  ~30 min debug.

- **Bug 2 — B1 hold filter applied uniformly across game phases.**
  Copied the orbitfix B1 hold filter from `agents/baseline/proposer.py`
  without considering whether it's correctly calibrated for ALL game
  states. The filter is load-bearing for +63 μ on the ladder (midgame
  recapture-risk modeling), but in dominant endgame (29 my-planets vs
  3 opp) it rejects every attack on opp's pocket because the few
  remaining opp planets "could counter" — which is technically true
  but operationally meaningless given our overwhelming base. Decision
  quality at write-time: arguably OK given the +63 μ prior; rule-gap
  surfaced. Cost: 1 gate cycle.

- **Calibration miss — poll-spam vs Monitor.** Issued ~30
  `cat /tmp/lagrange_*.out` tool calls polling A/B progress when a
  single `Monitor` or `until-loop` background Bash would have produced
  one notification on completion. No compute waste, but noisy session
  log. Decision quality: calibration miss; the tools are already
  documented.

- **Mild Rule 44 ambiguity — started on wrong branch.** Session
  resumed on `claude/strategy-axis-decision-3437` from prior session
  state, but the harness designated `claude/session-EqJuT`. Recovered
  cleanly via cherry-pick at end. Decision quality: did read state
  docs (`MULTI_BRANCH.md`) but didn't cross-check branch identity with
  harness directive until after development. Not promoting (this is
  harness-config specific, not a generalizable code-comp pattern).

## PI overrides (calibration data)

- **Redirected gate target.** I framed the natural comparison as "vs
  `agents/baseline`" and ran an n=8 A/B (0/8 LOSS). PI corrected:
  "first achieve 100% against nearest in random seats. only accept
  wins by elimination." This reframed the gate from
  ladder-competitiveness to correctness-against-weakest-opponent.
  Calibration takeaway: random is the substrate-correctness probe;
  baseline is the ladder-competitiveness probe; the two are distinct
  gates with different thresholds. The random-elim gate caught both
  bugs above; no amount of n=8-vs-baseline iteration would have.

- **Push-target clarification.** PI confirmed
  `claude/session-EqJuT` per harness rule. Cherry-pick onto the
  orbitfix base applied cleanly.

## Frictions logged this session

See `audit/friction.md` § "2026-05-23 (claude/session-EqJuT — simplest
Lagrangian agent)" for one-line entries:

- `tag: python-truthiness-gotcha-planet-id-zero`
- `tag: midgame-filter-overrejects-in-dominant-endgame`
- `tag: started-on-wrong-branch`
- `tag: tail-poll-instead-of-monitor`

## Promotion candidates (PI ratified)

PI ratified ONE candidate this session:

- **PROMOTED** — `tag: random-elim-gate-mandatory`. Added as the new
  TOP-PRIORITY pending entry in `.claude/skills/kaggle-comp/
  improvements.md`. Rule: any agent being considered for a Kaggle
  submission must first pass `scripts/random_elim_gate.py --n 16` at
  100% wins, ALL by elimination. Wins by score / step-500 timeout do
  NOT count. n=16 hard floor; n=32 recommended. Sub-clause of Rule 12
  (submission discipline) + Rule 43 (multi-opponent panel). Cost
  evidence: 2 latent failure modes caught in this session's gate run
  alone.

Candidates NOT ratified this session:

- The truthiness-gotcha rule and the game-phase-aware filter rule
  were both surfaced but PI did not promote. The regression test +
  in-friction fix are deemed sufficient; revisit if either pattern
  recurs.

## PI additions

> "promote 100% win against random as requirement"

Promoted as above.

## Framework version at session-end

- Branch: `claude/session-EqJuT` (HEAD pushed to origin)
- Commit SHA: `68c24be` (HEAD before postmortem stage)
- Cherry-picked from `claude/strategy-axis-decision-3437`:
  - `d0ada32` feat: lagrange_simple — minimal precision-physics
    Lagrangian agent
  - `b89daac` infra: random-elim gate
  - `142de3e` fix: planet id 0 silent-drop
  - `68c24be` fix: relax hold filter in dominant-endgame
- Active rules: CLAUDE.md Rules 1-47 (no new rule added this session;
  pending Rule-48 candidate filed in improvements.md)
- Skills invoked this session: postmortem (final wrap step)

## Carry-forward for next session

- `agents/lagrange_simple` passes the random-elim gate but loses
  0/8 vs `agents/baseline`. Single-source-per-target capture is the
  structural ceiling. Multi-source dogpiling (~50 LOC) is the obvious
  next iteration but was deferred per PI's "simplest" directive.
- The new pending Rule-48 in `improvements.md` should be promoted to
  `CLAUDE.md` in the next audit pass, and applied retroactively to
  every existing submission candidate.

---

## Verification follow-up (later same day, no source change in tree)

PI asked: "check if we beat the simple strategies by a hundred percent
by elimination as outlined in the docs of this branch" — then "try
rung 3 and explain why you think it is not possible".

### What was re-run

All three rungs from `knowledge-base/questions/2026-05-23-multi-
source-dogpile-next.md` PI direction note (`01f7544`), against the
EqJuT HEAD `ce70160`. Artifacts saved under `audit/probe-results/`.

| Rung | Opp        | n  | Wins | ELIMs | Losses | Verdict |
|------|------------|----|------|-------|--------|---------|
| 1    | `random`   | 16 | 16   | 9     | 0      | FAIL    |
| 2    | `starter`  | 16 | 15   | 9     | 1      | FAIL    |
| 3    | `baseline` | 16 | 0    | 0     | 16 (15 by opp-ELIM) | FAR FAIL |

### What went wrong (verification-side)

- **Calibration miss in `ce70160` commit message.** Body claims "14/16
  ELIM at default rng-seed = 2026". Re-run with the same default seed
  produced 9/16. No source change between commit and verification.
  Best read: the commit's self-reported gate score was either not
  reproducible at write-time or relied on warmed kinematic-table
  cache state not present in a cold run. The original session's
  Rule-38 check (fix-verification reproduces failure) covered the
  bug-side, not the gate-score-side.
- **PI direction note overtaken by within-session evidence.** The
  note in `multi-source-dogpile-next.md` recommends rung 3 = dogpile.
  But 6 dogpile variants regressed in the same session, axis closed
  under Rule 37. The note was never amended; it still recommends a
  same-day-falsified mechanism.
- **Rung 3 is structurally out of reach for this agent.** Detailed
  reasoning in `knowledge-base/thoughts/2026-05-23-lagrange-simple-
  rung-verification.md`. Summary: direction reversed (opp ELIMs us,
  not vice-versa); single-source ceiling load-bearing in
  `score.py:233`; the only multi-source path (dogpile) is closed.
  Reaching rung 3 needs cross-source defensive coordination + opening
  theory + game-phase-aware filter tuning — a different agent, not
  an extension.

### PI additions

> "Nothing to add — proceed"

### Promotion candidates (this verification pass)

PI did not ratify either candidate (no-preference at wrap-up). Both
remain drafted-only below; not added to `improvements.md`. Revisit
if either pattern recurs.

- **Candidate A — `gate-score-claim-needs-saved-artifact`.** Any
  commit message claiming a gate / panel / A/B score must reference
  a checked-in artifact path (e.g. `audit/probe-results/YYYY-MM-DD-
  <slug>.out`) so the claim is independently verifiable. Sub-clause
  of Rule 38. Cost evidence: this session lost ~25 min re-running
  the rung-1 gate to discover the 14/16 → 9/16 discrepancy that an
  attached artifact would have surfaced instantly.
- **Candidate B — `same-session-pi-note-amendment-on-axis-closure`.**
  When a session closes a design axis under Rule 37 (or otherwise
  falsifies a mechanism), and that axis was the subject of a
  same-session PI direction note, the wrap-up must amend the note
  inline rather than silently leaving the recommendation. Cost
  evidence: the rung-3 dogpile recommendation in
  `multi-source-dogpile-next.md` is still live despite same-session
  falsification.

### Framework version at this verification

- Branch: `claude/session-EqJuT` HEAD `ce70160` (unchanged from
  prior postmortem; verification added no source files).
- New tracked artifacts: `audit/probe-results/2026-05-23-rung{1,2,3}-
  *-n16.out`, `knowledge-base/thoughts/2026-05-23-lagrange-simple-
  rung-verification.md`, friction.md appends.
