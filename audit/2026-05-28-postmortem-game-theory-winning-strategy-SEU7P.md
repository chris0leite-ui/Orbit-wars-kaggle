# Postmortem — 2026-05-28 game-theory-winning-strategy-SEU7P

The session that wrapped the reach-frontier doctrine investigation.
Three nulls. No submission. Several durable artefacts shipped despite
the strategic null. Scored on decision quality, not outcome quality
(`knowledge-base/concepts/decision-quality-vs-outcome-quality.md`).

## What went wrong

**Dominant pattern: build-before-test on doctrine prescriptions.**

Three operationalisations of the same underlying hypothesis ("doctrine
prescriptions beat baseline") got built and falsified in sequence
without an upstream counter-experiment that would have caught the
correlation-not-causation issue at lower cost:

- **v1 chooser** — 0/20 vs baseline. ~1 session of build + diagnosis.
  At decision-time priors were sound (doctrine math is correct, n=92
  share-separation is real); the build was reasonable. The miss was
  not running a counter-experiment first ("force baseline to make
  doctrine-aligned moves, measure"). Cost: ~1 session.
- **v2 chooser** (hold-floor + gang-up) — 0/32 vs baseline. ~0.5
  session. Same pattern, smaller scale: after v1's diagnosis I
  committed to a v2 build before testing whether ANY closed-form-per-
  turn chooser could clear baseline. A 5-minute "force baseline to
  match v1's launch frequency" probe would have shown the issue isn't
  reward calibration. Cost: ~0.5 session.
- **4P cushion wrapper** — 4/32 vs baseline's 26/32 against same
  background. ~0.5 session. Same pattern, smallest scale. By this
  point the trail was getting cold; this third try barely cleared
  the Rule 37 "same axis" line (it's a different operationalisation,
  but the underlying logic was identical).

Total cost: ~2 sessions of work. With a counter-experiment-first
protocol it would have been ~1 session of work to reach the same
"doctrine line closed" verdict.

**Other notable decisions:**

- **λ_loss recalibration from 1.0 → 0.1 in v1** was reactive, not
  principled. Tuned a symptom (chooser too conservative) rather than
  diagnosing the underlying issue (hold=0 collapse, Bug 1). ~30 min
  wasted on wrong calibration before the root-cause trace exposed
  the real bug. Defensible at decision-time but a faster trace would
  have saved the detour.

- **Bundler CLI parity gate workaround.** When the CLI gate hit a
  kaggle_environments sys.path conflict, I worked around via
  `--skip-parity-gate` and relied on `tests/test_bundle.py` for
  parity. Correct call (the pytest gate did catch parity-correctness
  via the bundled_reach_frontier fixture), but the bundler bug is
  unpatched. Future modular agents will re-hit it.

## PI overrides

None this session. PI's questions ("is there any hope?",
"look at our best submission directly on Kaggle") were redirects
that improved the trajectory rather than overrides. The "look at
best submission" pivot was the most productive prompt of the day —
surfaced the 4P 0/2 gap that motivated the cushion experiment
(which then falsified the doctrine prescription cleanly).

## Rule-bypass failures

None observed. Rule 47 (physics-trace) followed by construction
(every emit physics-validated). Rule 45 (n≥32 for lift claims)
followed for the cushion A/B. Rule 1 (no unauthorised submit)
followed (no submits attempted). Rule 42 (cross-branch submit
coordination) not triggered (no submits).

## Rule-gap failures

**One identified.** There is no rule that mandates a counter-
experiment before operationalising a doctrine-derived prescription.
This gap directly produced three consecutive nulls on the same
underlying hypothesis. Three same-class failures + ~1 session of
avoidable cost → Rule 4 ("never give up") would suggest "iterate
on the axis"; Rule 37 ("3 consecutive same-axis nulls = stop") was
near its cap but not crossed (each variant claimed a distinct sub-
axis). Need a more specific rule:

> Before building a doctrine-derived "improve baseline" variant,
> run a counter-experiment that forces baseline to match the
> doctrine fingerprint. n ≥ 16 vs same-strength opp. If lift is
> not positive, the fingerprint is correlation-not-causation;
> close the axis.

**Cost evidence:** ~1 session avoidable today. Same pattern
plausibly fires in other research contexts (descriptive empirical
study → prescriptive operational variant). Promotion candidate.

## Frictions logged this session

Cross-link to `audit/friction.md` 2026-05-28 block (just-appended):

- `tag: doctrine-empirical-correlation-not-causation` — the main
  rule-gap; promotion candidate above.
- `tag: bundler-cli-parity-gate-vs-pytest-parity-divergence` —
  kaggle_environments sys.path conflict broke CLI gate; pytest
  fixture worked around. Not promoted (one-off bundler-CLI bug).
- `tag: lib-joint-solver-broken-strategic-lp-import` — latent
  cross-agent import in `lib/joint_solver/lp.py`. Promotion
  candidate: "lib modules shouldn't import from agents/".
- `tag: bundler-default-lib-order-stale-kinematic-table` — third
  same-class incident (`bundler-missing-block-e-modules` 2026-05-11,
  `new-lib-module-silently-broken-bundle` 2026-05-13). Promotion
  candidate: rule the lib-add + bundle-order-add be one commit.

## PI additions (from step 4)

PI: "Nothing to add or to promote." (2026-05-28). No frictions missed,
no decisions to flag, no candidates ratified.

## Promotion candidates (PI ratified: **no** — none promoted)

The three candidates below were drafted but explicitly declined by PI
in step 4. Left in the postmortem as historical record (so a future
session sees the same friction logged + already-considered + already-
declined, rather than re-drafting the same promotions).

### [ ] CLAUDE.md — Counter-experiment before doctrine operationalisation

**Tag:** `doctrine-empirical-correlation-not-causation` (2026-05-28,
3× same-axis nulls this session).

**Where to insert:** CLAUDE.md after Rule 37 (consecutive-falsification
cap), as Rule 49.

**What to add:**

```
49. **Counter-experiment before doctrine-derived operationalisation.**
    When an empirical study (Rule 48 / doctrine docs / n≥30 pattern
    audit) identifies a fingerprint that correlates with winning,
    do NOT build a prescriptive variant before running a counter-
    experiment that forces baseline to match the fingerprint and
    measures lift. n ≥ 16 vs same-strength opp. Three consecutive
    falsifications on the reach-frontier doctrine (v1 chooser 0/20,
    v2 chooser 0/32, 4P cushion wrapper 4/32 vs baseline's 26/32)
    were avoidable if a "force baseline to make doctrine-aligned
    moves" probe had landed first. Cost evidence: ~1 session of
    work avoidable today. Same pattern plausibly fires anywhere a
    descriptive empirical fingerprint gets operationalised as a
    prescriptive policy. Origin: 2026-05-28 postmortem
    `audit/2026-05-28-postmortem-game-theory-winning-strategy-SEU7P.md`.
```

**Why:** Three nulls today on the same underlying hypothesis. The
doctrine math is sound (n=92 separation 0.488 is real); the
prescriptive operationalisations all fail. Without an explicit
counter-experiment rule, future agents will re-burn the same cycles
on whichever empirical-fingerprint-to-policy transfer they encounter.

### [ ] kaggle-comp skill — lib-add-AND-bundle-order-add same-commit

**Tag:** `bundler-default-lib-order-stale-kinematic-table` (2026-05-28,
third same-class friction in this comp).

**Where to insert:** `.claude/skills/kaggle-comp/improvements.md`
under "Pending — promotion needed".

**What to add:**

```markdown
### [ ] [ADAPT-FOR-CODE-COMP] bundler: new lib module MUST land with DEFAULT_LIB_ORDER update

Three same-class frictions this comp:
- `bundler-missing-block-e-modules` (2026-05-11)
- `new-lib-module-silently-broken-bundle` (2026-05-13)
- `bundler-default-lib-order-stale-kinematic-table` (2026-05-28)

Pattern: someone adds `lib/X.py` that's lazily imported by another
lib module (typically `lib/trajectory.py`). The bundler's
`_assert_lib_imports_resolved` guard fires only when an agent
bundle reaches the new lib. Latent for weeks/months between landing
and first use.

**Fix:** add a process rule (CLAUDE.md or scripts/bundle_agent.py
header docstring): "Any commit that adds `lib/X.py` MUST also add
`X` to `DEFAULT_LIB_ORDER` in scripts/bundle_agent.py in the same
commit. PR / commit hook can enforce."

Or stronger: a pre-commit hook that diffs `lib/*.py` adds against
`DEFAULT_LIB_ORDER` adds and fails the commit if mismatched.
```

**Why:** Three same-class incidents in this comp → meets the
"promote when 2+ incidents" threshold from improvements.md header.

### [ ] kaggle-comp skill — lib/ MUST NOT import from agents/

**Tag:** `lib-joint-solver-broken-strategic-lp-import` (2026-05-28).

**Where to insert:** `.claude/skills/kaggle-comp/improvements.md`
under "Pending — promotion needed".

**What to add:**

```markdown
### [ ] [ADAPT-FOR-CODE-COMP] layering: lib/* must not `from agents.* import`

`lib/joint_solver/lp.py:37` imported `from agents.baseline.strategic_lp
import _greedy_assignment`. The referenced module doesn't exist on this
branch (checked-out artefact of another branch's refactor). Latent
until a non-baseline agent tried to use lp.py — then surfaced as
ModuleNotFoundError mid-import-chain. Same module also contains
similar imports in `mpc.py`, `opening_planner.py`, `value.py`.

**Fix:** add a layering rule. `lib/*` is the substrate; `agents/*` is
the consumer. Reverse imports indicate copy-paste needed: either
inline the function (preferred for simple helpers) or extract the
shared symbol into a lib module both agents can import.

**Enforcement option:** a unit test that walks `lib/` and asserts no
`from agents.* import` patterns. ~10 LOC, blocks the next instance.
```

**Why:** Class of bug that's hard to find statically (lazy imports +
strip-by-bundler hide it from `python -m compileall`). Better to
prevent than detect.

## Framework version at session-end

- Commit SHA: c20036b (HEAD; clean tree after wrap-up)
- Branch: claude/game-theory-winning-strategy-SEU7P (ahead 14 of main)
- Active rules: 1-48 per CLAUDE.md (47 = physics trace before agent design, 48 = production-share primary)
- Loaded skills this session: postmortem (this), kaggle-comp (passive context)
- Substrate added today: `fast.py --save-replays`, `scripts/measure_hold_times.py --replay-dir`, bundler DEFAULT_LIB_ORDER fix
