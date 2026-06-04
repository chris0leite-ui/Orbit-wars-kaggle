# CLAUDE.md — Orbit Wars (observation-driven iteration on a single strategy)

This is the rules + pointers index. The strategy itself lives in
`state/STRATEGY.md` — read that first.

## Rule 0 — How to communicate with the PI

**Plain English, every time. No abbreviations the PI hasn't already used.**
No letter-number experiment codes (E1, F2, h1d, K=27, τ=100k) in chat —
describe what each thing does. If you have to introduce a technical term,
define it inline on first use. If you find yourself reaching for an
acronym, that's a smell.

## Working mode — one strategy, observation-driven

The competition strategy is `baseline_adaptive_k` + the teamwork refiner
(`state/STRATEGY.md`). We are no longer running parallel exploration tracks.
The loop is:

1. **PI observes** something concrete (replay, leaderboard, opponent trace).
2. **PI reports** the observation in plain English.
3. **AI diagnoses** the modeling cause — minimal investigation.
4. **AI proposes** the smallest fix, default-OFF gated.
5. **PI signs off** on the proposal.
6. **AI implements + smokes** (Rule 46).
7. **AI submits** (Rules 1 / 12 / 42).
8. Wait for the next observation.

One observation → one mechanism → one push. No multi-axis exploration.

## Operating rules

1. **Submission discipline.** Every `kaggle competitions submit` is single-shot,
   explicitly approved by the PI. No retry/until/while loops.
12. **Daily submission budget = 5/day.** Kaggle auto-keeps your **rolling last 2
    submissions** for final evaluation — not 2 PI-selected. A new submit
    auto-evicts the older of the previous two. Plan submission order accordingly.
    The strategy starts at μ ≈ 600 (TrueSkill warm-up) and climbs over ~24 h;
    do not draw conclusions from the first few hours of leaderboard data. **An
    evicted submission's μ is frozen at the field it last played — it is NOT
    directly comparable to an active submission's μ on today's (stronger,
    larger) field.**
32. **Session-start git fetch.** `git fetch origin && git log HEAD..origin/main`
    BEFORE any new compute.
35. **PI thoughts are append-only.** Transcribe PI voice-dumps to
    `knowledge-base/thoughts/YYYY-MM-DD-slug.md`. Never overwrite, delete, or
    archive on cleanup. Folder is permanent.
36. **Session-end second-brain update.** Before wrap-up, add at least one entry
    to `knowledge-base/thoughts/`; log open questions in `questions/`; surface
    persistent flags in `flags/`.
38. **Fix-verification reproduces the failure state.** When you fix a friction,
    verify by: (a) reproducing the original failing state, (b) applying the
    fix, (c) confirming the failure mode is gone. Unit-testing the new code
    path on synthetic input is NECESSARY but NOT SUFFICIENT — it cannot detect
    fix-doesn't-apply-to-real-environment.
39. **No Claude session URLs in commits / PR bodies.** Do NOT append
    `https://claude.ai/code/session_…` (or any equivalent session identifier)
    to commit messages, PR titles, PR bodies, issue comments, or any artifact
    pushed to a repository. Session IDs stay in chat replies only.
40. **Prefer modeling-correctness over restriction-tuning.** When a failure
    mode can be addressed by either (a) bumping a constant / threshold /
    hard-cap, or (b) fixing the underlying model (better leaf scoring, better
    target prediction, better physics, better opponent model), prefer (b).
    Restrictions are band-aids on a model that misvalues actions; the right
    behaviour should emerge from a correct model, not from an artificial cap.
42. **Pre-submit coordination gate.** Before any `kaggle competitions submit`:
    (a) run `kaggle competitions submissions orbit-wars | head -5` to read the
    current rolling pair, (b) append a claim row to the
    `state/MULTI_BRANCH.md` push-claim board with branch / agent / predicted
    μ / which sub_id + μ will be evicted, (c) if evicted-μ EXCEEDS predicted-μ,
    the submit is **BLOCKED** until explicit PI sign-off.
45. **n ≥ 32 minimum for any A/B lift claim.** n = 8 / 16 Wilson CI is too
    wide to distinguish parity from a 20-pp regression. Lift claims gated to
    submission require n ≥ 32 with Wilson-lo ≥ 0.50. A single-opponent A/B is
    evidence of an edge vs *that opponent only* — not calibration to the live
    field; confirm a submit-relevant lift on a multi-opponent panel. Exception:
    a triage at n = 16 may proceed to n = 32 confirmation before submit, but
    n = 16 alone is never a submit gate. **Note:** for a calibration probe (no
    lift claim, just re-measuring a known result), Rule 45 does not apply.
46. **Bundle + parity smoke before any submission.** Every submission MUST
    clear: (a) `bash scripts/_build_refine_adaptivek_bundle.sh` succeeds;
    (b) `pytest tests/test_bundle.py` GREEN; (c) `python fast.py play
    submissions/champ_refine_adaptivek.py --vs submissions/baseline.py
    --seed 7` runs one full game with max turn < 1000 ms.

## Pointers

- `state/STRATEGY.md` — **the strategy.** Read first.
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42).
- `state/TOOLS.md` — tools registry (A/B harnesses, bundler, smoke, diagnostics).
- `comp-context.md` — settled-once competition facts.
- `HANDOVER.md` — next-session brief.
- `SETUP.md` — onboarding checklist for a fresh container.
- `audit/` — append-only postmortems, investigations, replays.
- `knowledge-base/` — PI second-brain (Rules 35-36).
- `state/_archive/` — superseded state docs — read only if specifically relevant.

## Archived rules

The pre-strategy-lock CLAUDE.md (the full 49-rule set covering parallel-track
exploration, chooser-axis falsification (Rule 37), the confound-sweep rule
(Rule 41), the multi-opponent panel mandate (Rule 43), production-share
evaluation (Rule 48), and the joint-coordination planner doctrine (Rule 49))
is preserved at `state/_archive/CLAUDE-JzIAr-full-49rules.md`. Re-promote rules
from there as the observation-driven iteration surfaces a need for them.
