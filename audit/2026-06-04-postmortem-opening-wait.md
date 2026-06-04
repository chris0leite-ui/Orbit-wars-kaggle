# Postmortem — 2026-06-04 (opening-wait diagnostic)

Branch: `claude/kaggle-submission-strategy-JzIAr`. Measurement-only session; no
submission.

## What happened (one paragraph)

PI flagged a live loss on a perimeter-ring/central-sun map (seed 722289020) with a
specific mechanism story: sparse map → targets beyond horizon K → no candidates →
idle opening. Built `scripts/opening_starvation.py` and tested it directly. The
horizon hypothesis was **refuted**: 0% of opening boards have their nearest neutral
past K_OPEN=20, and on the repro seed the agent had 2–12 candidates every opening turn
and launched on only 4 of 31 — 0 turns horizon-starved. The real finding: we wait by
**value-function choice** (27/31 opening turns sat on candidates), not by reachability.
Relocated the lever from a horizon constant to the chooser's early-expansion appetite,
and scoped the genuinely-untested question (is the waiting *exploited* by aggressive
expanders, vs merely symmetric in self-play). See
`audit/2026-06-04-opening-wait-diagnostic.md`.

## What went well

- **Tested the mechanism instead of acting on the story.** A plausible, PI-offered
  hypothesis with a supporting docstring stat was refuted by direct measurement before
  any code change — exactly the Rule-40 discipline (the symptom was not where the knob
  was). Avoided a wasted K_OPEN bump / far-launch-fallback build.
- **Cheap/expensive split made it fast.** The map-level question (step-0 reach) needs
  no game play; the behavioral question (launch overlay) needs one focal game. Keeping
  them separate gave a clean answer inside the compute budget after the first
  both-seats aggregate blew the timeout.
- **Did not over-claim.** Stopped at "we wait by choice" (proven) and explicitly did
  NOT assert "this is why we lost" (needs an aggressive opponent) — and did not submit.

## What went wrong (→ friction.md 2026-06-04)

1. **Leaned toward the hypothesis before testing it** — partly endorsed the horizon
   read and drafted a "shipped fix is under-powered" line from a docstring stat that
   actually answered an adjacent question. (`leaned-toward-hypothesis-before-testing`)
2. **Both-seats × per-board `propose()` aggregate blew the 300s cap.**
   (`both-seats-propose-aggregate-too-slow`)
3. **Arg-parse crashed on the `--scan` placeholder** (int-coerce before mode check).
   (`argv-parse-before-mode-check`)

## Promotion candidates (for PI ratification → kaggle-comp/improvements.md)

- **"When a plausible mechanism story is offered (especially by the PI), build the
  discriminating measurement FIRST; don't pre-commit to the framing or quote a stat
  that answers an adjacent question."** Today's near-miss; complements the standing
  Rule-40 instinct. Strong candidate.
- **(Re-raised from 2026-06-03, still unratified)** "A/B on the live champion config,
  not the repo default; confirm a ~50% parity anchor before trusting any lift," and
  "re-test closed/null findings against the CURRENT opponent field." Both reinforced
  by today's opponent-class confound framing.

(Not auto-promoting — flagged for PI ratification next session per the postmortem
skill, which blocks on PI replies.)

## Calibration / overrides

No submission and no override this session (measurement-only). Standing
`pi-stamp-risk` note: the 2026-06-03 session had a PI submit override; this session
had none — override cadence is healthy, not rubber-stamping.

## Result

Killed a hypothesis cleanly, relocated the lever, and left a sharp, cheap,
no-build next experiment (opening appetite vs an aggressive expander, cut by opponent
class). Net-positive measurement session; live floor still pending the refine-settle
check (flag 2026-06-04).
