# Postmortem — 2026-05-25 competitive-programming-strategy-ESwSv

Session: baseline_wave v5 / v5.1 implementation + A/B + ablation.

## What went wrong

- **A/B-without-diagnostic.** Implemented v5 (3 fixes, 1-2 h
  compute including tests + bundle + n=16 A/B) directly from the
  Aidan replay screenshot. The cheaper move was: trace a single
  game on Aidan's seed (799069305) with our v3.1 agent FIRST, to
  learn *why* v3.1 fires 0 waves there. Possible explanations
  (geometry-blocking, value-head-rejecting, orbitfix RA
  intercepts) all had different downstream patches; I didn't
  distinguish before coding. n=1 replay → 3-edit patch with no
  intermediate diagnostic step is the bad-decision template.

- **"Mechanical fire" mistaken for "good fire".** The wave-
  emission probe (89 turns of wave activity on Aidan's seed vs
  v3.1's 0) felt like positive validation. It only proved the
  generator works — not that the generated waves WIN. The right
  post-fix probe was a per-leg outcome trace on a game vs
  orbitfix (do waves connect? get reinforced against?). I went
  straight to A/B; verdict was 2/16 wins (worse than v3.1's
  3/16). Quality probe would have caught it before the 595 s A/B
  spend.

- **Rule 37 iteration counting was implicit.** Plan documented
  "this is the 3rd structural iteration on the wave-proposer
  axis (v3, v3.1, v5)" but I proceeded based on the Aidan replay
  priors. The plan's "STOP" gate was a post-A/B-failure
  verdict ("if fails, then Rule 37"); it should have been a
  pre-implementation gate ("Rule 37 already says STOP; require
  PI sign-off before code"). The 1.5 h compute was incurred on a
  mechanism with prior-iteration null-failures, against
  written rule.

## Frictions logged this session

See `audit/friction.md` 2026-05-25 block:

- `wave-mechanical-vs-quality-test-gap` — emission counter is
  not a quality signal; need per-leg outcome traces.
- `replay-as-strong-prior-trap` — n=1 replay → 3-edit patch w/o
  intermediate diagnostic.
- `rule-37-iteration-counting-ambiguity` — what counts as an
  axis-iteration; when does the STOP gate fire.

## Promotion candidates (PI ratified: NONE)

PI elected to keep all three tags in `audit/friction.md` only;
no promotion to `.claude/skills/kaggle-comp/improvements.md`
this cycle. Tags age in friction.md for one more session before
re-eligibility per friction.md convention.

## PI additions (from step 4)

PI replied "Looks complete, write it." — no verbatim additions.

## Framework version at session-end

- Commit SHA: `7fc52cf` (HEAD before postmortem stage)
- Active rules: 0..47 (CLAUDE.md `## Operating rules — concise`)
- Loaded skills this session: `postmortem`

## A/B evidence (this session's measurements)

n=16 vs orbitfix (Wilson lo on all rows: 0.00):

| variant | elim/16 | wins/16 | avg steps |
|---|---:|---:|---:|
| v3.1 (reference, prior session) | 0 | 3 | ~186 |
| v5 full (stockpile re-enable) | 0 | 2 | 184.5 |
| v5.1 (stockpile dropped — shipped) | 0 | 4 | 195.3 |

Wave-emission probe (in-process, seed 799069305, vs random):

| variant | wave-emit turns | elim steps |
|---|---:|---:|
| v3.1 | 0 | 200 (cap) |
| v5.1 | 89 | 102 |

The generator-works-but-doesn't-win pattern is the load-bearing
finding. Mechanical correctness (Rule 38) ≠ A/B correctness;
Rule 38 needs a sibling rule for candidate-generator quality.
