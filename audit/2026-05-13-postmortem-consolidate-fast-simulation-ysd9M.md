# Postmortem — 2026-05-13 consolidate-fast-simulation-ysd9M

Session: JAX sprint sub-phases 8e (parity gaps) + 8f (code-review
findings). Branch reaches merge to main this session.

## What went wrong

- **Sub-phase 8e shipped without a scalar-vs-JAX end-to-end parity
  test (T2).** I knew at decision time that "JAX matches its numpy
  mirror" is not source-of-truth parity. The code review surfaced
  exactly this gap. Cost: one extra push cycle to add the test in
  8f. Bad decision given priors.
- **First Kaggle kernel push didn't probe the dataset mount path.**
  Four push cycles before a diagnostic `print` revealed
  `/kaggle/input/datasets/<owner>/<slug>/...`. Cost: ~30 min cycle
  time. Pattern: when first deploying to a new platform, add a
  layout-probe `print` to v1, don't assume documented paths.

## PI overrides this session

- "Resolve the issues thoroughly" → expanded sub-phase 8f scope from
  "real bugs only" to "all C/P/Q/T findings". My initial scoping was
  too narrow. Calibration: when PI hasn't scoped, assume the larger
  scope is the intent.
- "How easy will it be to adapt new strategies?" → pushed me to
  audit infrastructure gaps I hadn't volunteered. Calibration: I
  tend to report what shipped, not what's missing. Should surface
  friction-on-future-work as part of normal status, not on prompt.

## Frictions logged this session

Cross-link to `audit/friction.md` under the
`## 2026-05-13 (consolidate-fast-simulation-ysd9M — JAX sprint wrap)`
heading. Four entries:

- `silent-engine-capacity-loss` — `fleet_launch` slot off-by-one;
  bug C1 in code review.
- `parity-tests-vs-mirror-not-source-of-truth` — JAX parity tests
  inherited the numpy-mirror's bugs; bug C2 in code review.
- `harness-knob-without-plumbing` — `A_AGGRESSIVE` env var silently
  ignored for 4 kernel cycles; bug G in code review.
- `kaggle-cli-kgat-auth-2hr-detour` — new-style KGAT token needed
  `KAGGLE_API_TOKEN` env var, not `kaggle.json`.

## Promotion candidates (PI ratification: pending at session end)

Per protocol, drafted but NOT committed to
`.claude/skills/kaggle-comp/improvements.md`. Three candidates:

### Candidate 1 — parity-tests-vs-source-of-truth

When porting algorithm X (scalar reference) to platform Y (JAX, GPU,
numpy batch), parity tests MUST compare Y-output against X-output —
NOT Y-output against a-second-port-of-X. The intermediate port can
inherit the same constant / sentinel / off-by-one as Y, masking the
bug. Bug C2 today (lead-aim tolerance 0.5 vs scalar 0.3) survived
117 parity tests because every JAX parity test compared against the
numpy mirror, which had the same 0.5.

### Candidate 2 — knob-smoke-test

Every exposed config knob (CLI flag, env var, kwarg with semantic
effect) must be exercised by a smoke test that flips it and asserts
the output differs. Knobs that don't change anything when flipped
are dead — either remove them or fix the plumbing. Bug G today
(`A_AGGRESSIVE` silently unwired) ran 4 kernel cycles before
surfacing.

### Candidate 3 — in-loop-state-mutation-needs-structural-test

Any algorithm that mutates state inside a Python-unrolled or
`lax.scan` loop needs a structural / contiguity test, not just a
per-element parity test. Slot allocators, cumsum-based pickers,
"next free index" patterns are common offenders. Bug C1 today
(`fleet_launch` slot off-by-one) silently halved fleet capacity
for the entire 8c-8e period; per-element parity tests passed
because they compared by fleet id, not by slot density.

## PI additions (from step 4)

(PI did not respond before merge — promotion candidates left as
drafts in this postmortem; ratification deferred to next session.)

## Framework version at session-end

- Commit SHA at postmortem time: `b67c868`
- Active rules: 1-36 (per `CLAUDE.md ## Operating rules`)
- Loaded skills this session: `postmortem`
- Test status: 117 / 118 JAX tests green
- Branch: `claude/consolidate-fast-simulation-ysd9M`, merging to main.
