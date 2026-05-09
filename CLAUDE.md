# CLAUDE.md — Orbit Wars (code/agent competition)

This file is the rules + pointers index. Detailed state lives in
`state/`, `HANDOVER.md`, and the references at the bottom of this file.

Truth lives on `origin/main`. Session-start git fetch is Rule 32.

## Rule 0 — How to communicate with the PI

**Plain English, every time. No abbreviations the PI hasn't already used.**
No letter-number experiment codes (E1, F2, h1d, K=27, τ=100k) in chat —
describe what each thing does. The glossary is for agent-to-agent
reference and for reading older audit notes; it is **not** something the
PI should ever have to look up mid-conversation.

If you have to introduce a technical term, define it inline on first use.
If you find yourself reaching for an acronym, that's a smell.

## Operating rules — concise

Rules tagged `[TABULAR-ONLY]` below were inherited from a tabular
reference comp (s6e5) and **do not apply to Orbit Wars**; they are kept
for cross-comp rule-number stability. Rule bodies are preserved verbatim.

1. **Submission discipline.** Every `kaggle competitions submit` (or
   `kaggle kernels push`) is single-shot, explicitly approved by the PI.
   No retry/until/while loops.
2. **Smoke + 1-fold time-probe + 1h GPU cap.** Smoke at 1 fold / 50k rows;
   if 5-fold projection ≥ 1 hour, shrink. Kill any kernel that
   pre-processes ≥30 min without fold output. (Code-comps: substitute
   "1-fold" with "1 self-play episode batch ≈100 games"; the 1h cap stays.)
3. [TABULAR-ONLY — N/A for Orbit Wars. Code-comp analogue: baseline-opponent-panel beat-rate gate; see guardrails.md G13.] **4-gate leakage filter pre-LB-probe** (G1 OOF clears anchor; G2 blend
   lift; G3 net rare-class flip ratio ≥0.5; G4 direction asymmetry).
4. **Never give up; saturation is bounded.** After every null, brainstorm
   3 untried mechanisms. Locking is for the final 3-day window only.
5. **Keep CLAUDE.md lean.** Hits the 150-line cap (Rule 9); anything
   that grows beyond rules + pointers goes in a dedicated file.
6. **Heuristics before heavy compute.** Closed-form rule before Optuna /
   GPU / RL training run.
7. **Research before saturation.** At 3 nulls / 5 same-LB saturations /
   2 days no lift: web search, 2 prior-comp writeups, 5 untried
   mechanisms with citations.
8. **Settled-once facts** in `comp-context.md`. Never re-ask.
9. **~160-line guideline on session-read docs** (`CLAUDE.md`,
   `HANDOVER.md`, `state/current.md`, `audit/friction.md`). Reference
   docs (mechanism-ledger, audit archive, postmortems) are exempt —
   pulled on demand, not read by default.
10. **Pull-style updates.** No proactive minute-level chatter; on PI pull,
    1-2 sentences with the latest fact.
11. **Model routing.** Haiku read-only; Sonnet default; Opus hard. 10/day.
12. **Spend the full daily submission budget (Orbit Wars: 5/day).**
    Submissions are calibration probes; the predicted-vs-actual-μ-rating
    gap per agent family is data. **Caveat for Orbit Wars:** Kaggle
    keeps your **rolling last 2 submissions** for final evaluation —
    not 2 PI-selected. A late submit auto-evicts the previous one.
    Plan submission order accordingly: never push speculative variants
    after a known-good submit unless you're willing to lose the good
    one's ladder spot.
13. **Kaggle GPU is part of compute budget.** Local CPU-only; Kaggle
    notebooks are the GPU path. Port any 5-fold > 1h CPU before declaring
    "not cost-justified."
14. **Strategy-critic-loop fires automatically** at end-of-day, on
    predicted-vs-actual-rank drift ≥1 bracket on consecutive submits,
    before any new mechanism family, at 50% checkpoint, and at any plateau.
15. **Handover protocol.** PI says "handover" → read `HANDOVER.md`.
    PI says "prepare handover" → follow `WRAPUP.md` section B.
16. **6-question pre-flight check** before any candidate ≥10 min CPU/GPU.
    Q1 already-explored? Q2 rank-lock-vulnerable family? Q3-Q5 predict
    standalone result + correlation + cite precedent. **Q6 does training
    objective match the comp metric (Orbit Wars: tournament TrueSkill / Elo)?**
    Q6 unanswered = forced SKIP.
17. **Wrap-up triggers.** "Wrap up" → `WRAPUP.md` section A; "prepare
    handover" → both sections.
18. **Issue-tree claim before compute.** Before any probe ≥10 min, claim
    an unclaimed `open` leaf in `ISSUES.md`.
19. **Experimentation harness.** Document nulls too. Code-comp tooling
    (probe / gate / calibration scripts) gets built per-comp; do NOT
    port s6e5 tabular probes (`probe.py`, `pre_submit_diff.py`,
    `research_seed.py`) — they assume tabular OOF arrays.
20. **Single-model-first / kitchen-sink FE before stacking.** Build the
    feature factory and a single strong model first; stacking adds, it
    doesn't replace. (Code-comps: build a single competent agent before
    ensembling agents or training agent populations.)
21. **Family falsification requires ≥3 variants** of the key
    hyperparameter.
22. **Public-notebook scan at every plateau.** Pull top 5 Kaggle notebooks
    (≥10 votes); list approach + result + agent class; build the gap.
23. **Framework is scaffolding, not authorship.** Discipline optimises
    HOW to evaluate, not WHAT. ≥1 slot per 3-day cycle for free-form
    agent design.
24. [TABULAR-ONLY — N/A for Orbit Wars] **Fold-safe label-conditional aggregates.** ANY feature derived from
    labels via groupby aggregation MUST be re-fit per CV fold using
    training rows only. Diagnostic: 80/20 holdout test in <10 min CPU.
25. [TABULAR-ONLY — N/A for Orbit Wars] **Transductive features need adversarial-validation check.** Before
    any combined train+test transform, run AV; if AV-AUC > 0.55, fit on
    train only.
26. **PI interaction protocol.** PI is read+strategy, not keyboard.
    Every BOTE asks PI: (i) Q6 metric alignment, (ii) which precedent
    are we pricing this against? Once-per-session devil's-advocate
    ritual on the strongest current recommendation.
27. [TABULAR-ONLY — N/A for Orbit Wars. Code-comp analogue: `kaggle_environments.evaluate()` head-to-head ≥10 games against the previously-submitted agent — if winrate is 50%±5%, the new agent likely doesn't outclass it; reconsider the slot spend.] **Pre-submit prediction diff is mandatory.** `scripts/pre_submit_diff.py`
    against the previous submit. If Spearman > 0.999, abort — LB will tie.
28. **Subagent dispatch limits.** Don't dispatch general-purpose subagents
    for Python jobs > 5 min. Long-running compute launches from the main
    thread.
29. **Same-session friction must apply same-session.** When a friction is
    logged, immediately apply the fix to all in-flight invocations.
30. **GPU kernel template gate.** Any new torch / RL kernel uses
    `"machine_shape": "GpuT4x2"`. P100 is for tree-GPU only.
31. **Concurrent compute cap ≤2 CPU-heavy jobs.** Schedule cheap probes
    (<30 s) ahead of slow ones.
32. **Session-start git fetch.** `git fetch origin && git log
    HEAD..origin/main && git diff HEAD..origin/main HANDOVER.md` BEFORE
    any new compute.
33. [TABULAR-ONLY — N/A for Orbit Wars] **Inner-CV-validate post-hoc OOF transformations.** Isotonic / Platt /
    per-group rescaling; never trust the in-sample lift.
34. **Experiments use descriptive names in docs.** Letter-number codes
    may stay in artifact filenames and old audits, but new prose says
    what each thing IS.
35. **PI thoughts are append-only.** Transcribe PI voice-dumps to
    `knowledge-base/thoughts/YYYY-MM-DD-slug.md`. Never overwrite,
    delete, or archive on cleanup. Folder is permanent.
36. **Session-end second-brain update.** Before wrap-up, add at least
    one entry to `knowledge-base/thoughts/`; log open questions in
    `questions/`; surface persistent flags in `flags/`.

## Defaults from prior-comp postmortem

[The five R-defaults below are tabular reference-comp (s6e5) postmortem
outputs. They do **not** apply directly to Orbit Wars; they are kept
here for cross-comp comparison and so future tabular comps inherit them.
Orbit Wars accumulates its own postmortem-derived defaults via
WRAPUP step 4b → improvements.md.]

- **R1** [TABULAR-ONLY] Two-anchor OOF.
- **R2** [TABULAR-ONLY] Final selection along public-LB axis. PRIMARY =
  best public. HEDGE = best OOF that regressed ≤30 bp on public.
  **Code-comp default for Orbit Wars:** the platform auto-keeps your
  rolling last 2 submits — there is NO PI selection at the deadline.
  The strategic question is "when to submit," not "what to submit at
  the end." Heuristic: only push a new agent when its expected μ-gain
  exceeds the current σ; the previous-2 submit is locked the moment a
  3rd is pushed, so weak late submits are unrecoverable for ~24 h.
- **R5** [TABULAR-ONLY] Final-window OOF-best regression probe.
- **R7** [TABULAR-ONLY] Override-mechanism rules. Flip count <200 →
  HEDGE only; >200 needs explicit PI sign-off.
- **R8** End-of-comp: log final percentile to
  `~/.claude/skills/kaggle-comp/improvements.md`.

## Pointers

State (mutable, refresh each session):
- `state/current.md` — current submitted agent, tournament rank, submission count.
- `state/calibration-ladder.md` — predicted-rank vs actual-rank table for new-candidate sizing.
- `state/hypothesis-board.md` — open ideas, killed list, hedge ladder.
- `state/mechanism-ledger.md` — every agent family tried (heuristic / search / IL / RL / hybrid), with results.

Process docs (read once / on trigger):
- `SETUP.md` — onboarding checklist for a new comp (read on day 1 of a fresh repo).
- `HANDOVER.md` — next-session brief (Rule 15).
- `WRAPUP.md` — wrap-up + prepare-handover procedure (Rule 17).
- `ISSUES.md` — live problem decomposition / claim board (Rule 18).
- `comp-context.md` — settled-once facts (kickoff, env spec, gate clearance).
- `audit/INDEX.md` — map of audit/ subdirs.
- `audit/friction.md` — current friction summary (concise).

- `knowledge-base/` — PI second-brain (permanent; Rules 35-36).
  Subdirs: `thoughts/`, `concepts/`, `friction/`, `flags/`, `questions/`.

Skills:
- `.claude/skills/postmortem/SKILL.md`, `.claude/skills/kaggle-comp/`.
