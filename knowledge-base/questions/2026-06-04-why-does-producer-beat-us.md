# Open question — why does the Producer beat our whole agent line?

**Opened:** 2026-06-04. **Status:** open.

## The observation that prompted it

In n=16 triage, a vendored third-party public agent ("Producer", torch
planner — now `agents/producer/`) beat **both** our live champion
(`champ_adaptiveK_on`, μ≈1170) and our latest push
(`champ_refine_adaptivek`, μ=951.5) at 3/16 each — same count despite a
≈370 μ gap between the two focal agents. Full measured detail:
`audit/2026-06-04-producer-eval-observations.md`.

## Why it is a question and not a conclusion

The result is direction-only (n=16, Rule 45). It is **consistent with**
several different stories that have not been distinguished yet:
- a shared chooser-independent loss mode in our family;
- a geometry-specific weakness (`low_prod/big_rotating` was 0/2);
- our local A/B panel being a monoculture that never exercised the
  failure (it A/Bs within our own lineage);
- something mundane (seed luck at small n).

## To answer it (cheapest first)

1. Re-run at n ≥ 32 vs the Producer; get a tight CI.
2. Check whether the 13 losses are the *same games* across both focal
   agents (shared loss mode vs coincidence).
3. Read one single-game trace of a loss on a `low_prod/big_rotating` seed.
4. Only then ask "what modeling change?" — do not jump to a fix before the
   mechanism is seen (Rule 40: modeling-correctness over restriction-tuning).

## Cross-branch note

This was observed on `claude/competition-approach-strategy-a7baQ`, which is
behind `origin/main`. The Producer opponent + this question are likely
relevant to the active tracks in `state/MULTI_BRANCH.md`; sync when those
branches next pick up panel work.
