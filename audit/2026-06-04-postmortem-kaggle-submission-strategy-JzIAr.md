# Postmortem — 2026-06-04 kaggle-submission-strategy-JzIAr

Session: opponent-agnostic / Producer-lens investigation (passive-opponent
spike + Option B net-ship-swing leaf-swap).

## What went wrong

- **Nothing flagged at the decision-quality level.** The headline outcome was
  negative, but the *decisions* were sound given priors: we formed a falsifiable
  hypothesis (Producer's edge = its value lens, per the 3%-vs-78% producer_lite
  result), validated the leaf-swap as mathematically equivalent to Producer's
  `competitive_score` (Plan agent + a load-bearing conservation unit test),
  built the minimal faithful test, measured it cleanly at n=32, and let it
  speak. It refuted the hypothesis. I would retake every step given the same
  priors. A bad outcome from a good decision is a good decision.

- **The result (for the record).** The opponent-agnostic *assumption* — not the
  leaf — is what hurts our chooser:
  - net-swing lens vs Producer: **7/32 = 21.9%** (Wilson [0.11, 0.39]) — adopting
    Producer's lens bought us nothing vs Producer (≈ the champion's own loss rate).
  - net-swing lens vs our champion: **12/32 = 37.5%** [0.23, 0.55] — worse than `favor`.
  - passive-opponent ALONE (favor leaf) vs champion: **10/32 = 31.2%** [0.18, 0.49]
    — the disambiguator: passive-alone already loses, the net-swing leaf on top is
    marginally better but still loses. **Removing the reactive opponent from our
    rollout is the culprit** — its recapture penalty was load-bearing; dropping it
    causes overextension. Producer wins *despite* being opponent-passive because of
    its other machinery, not because opponent-agnostic scoring is itself superior.

- **Process frictions (not decision faults):**
  1. `pkill -f orbit_wars` self-matched the killer shell's own command line and
     killed the relaunch before it wrote its script — cost a wasted background
     relaunch + visible confusion. Recovered by killing leftovers by PID and
     writing/launching in separate steps.
  2. `clean_ab` at n=32 with the heavy bundles runs ≈ 30 min/A/B (~1.5 h for a
     3-A/B sweep) — I estimated ~10 min. Forced a mid-flight reprioritization
     (decisive Producer gate had been queued last); only happened because PI
     nudged. Lesson: budget the cost and order A/B chains decisive-gate-first.
  3. Missable faint signal: the spike's single both-passive game (seed 7) lost
     early; correctly flagged "not a verdict (n=1)" at the time. Hindsight only.

## Frictions logged this session

- Not separately appended to `audit/friction.md` (postmortem invoked directly,
  not via the WRAPUP friction step). The three above are the session's frictions;
  PI elected not to log or promote them (see below).

## Promotion candidates (PI ratified: NO)

- Drafted two (`pkill-self-match`, `clean-ab-cost-and-order`). **PI: "Nothing to
  add or to promote."** Not promoted to `improvements.md`.

## PI additions (from step 4)

- None. PI: "Nothing to add or to promote." Session called: "this is leading
  nowhere" → wrap up after the negative result was settled.

## Framework version at session-end

- Commit SHA: 10b035f8137a3136b29555ab46006e9daa9cae8b
- Active rules (CLAUDE.md): 0, 1, 12, 32, 35, 36, 38, 39, 40, 42, 45, 46.
- Loaded skills this session: postmortem.
