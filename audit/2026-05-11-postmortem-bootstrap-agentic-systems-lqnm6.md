# Postmortem — 2026-05-11 bootstrap-agentic-systems-lqnm6

## What went wrong

This session shipped 14 commits and a PI-approved submission (#52544634).
Quality issues to flag for calibration:

- **Bad decision: published a lift headline at n=8.** The Block E
  lookahead MVP's first panel showed v3_lookahead beating v2 11/16 =
  68.8% on 8 seeds (Wilson CI [44%, 86%]). I reported this as a
  positive result before running the 32-seed retest, which collapsed
  to 50/50 parity. The plan file's gate had always required 32 seeds
  × both seats; I jumped ahead. **Rule under-applied:** Rule 21
  ("Family falsification requires ≥3 variants of the key hyperparameter")
  in spirit — n=8 is not enough to claim a lift over noise.

- **Bad decision: spun up a redundant 32-seed panel based on a false
  process-finished signal.** The `until [ -z "$(pgrep ...)" ]; do
  sleep 30; done` polling pattern returned a false-completion during
  a worker transition. I concluded "panel was killed" and started a
  second concurrent panel — which then competed for CPU with the
  original (still-running) panel. Wasted ~10 minutes of compute and
  drew the wrong conclusion ("the 8-seed result was inflated by P1
  seat") from a partially-complete 23/64-game dataset. **Rule
  under-applied:** Rule 31 ("Concurrent compute cap ≤2 CPU-heavy
  jobs") was nominally satisfied — but the underlying intent (don't
  duplicate work) was violated.

- **Bad decision: first bundled submission with broken bundle.**
  `python -m scripts.bundle_agent agents/v3_snipe` produced a 53KB
  file that crashed 10/10 self-play (`NameError: 'propose_snipe_
  missions' is not defined`). Root cause was that
  `DEFAULT_LIB_ORDER` hadn't been updated as Block E modules
  landed. **Caught at the E.2 gate** (which is exactly what the
  gate is for, so the gate did its job) — but the failure cost
  ~5 minutes and the diagnostic was load-bearing. This is the
  second time this session a bundler-import gap silently broke
  bundles (earlier we caught the `lib/trajectory` gap via the
  bundle parity test). The bundler is fragile.

- **PI override (mild): submission discipline.** PI approved the
  v3_snipe submit; the submission was rational. But the trade-off
  (evicting v1.2/roi at μ=1006.9 vs v2 at μ=974.3) wasn't recorded
  as an explicit decision document before the push. Post-hoc it
  is correct — v3 ≫ v1.2 in 2P (97% h2h) — but the lack of an
  expected-Δμ pre-registration is a calibration gap.

- **Rule-gap: planner regression at v3.0 was caught by smoke
  testing, not by tests.** When `settle_plan`'s first attempt with
  no-double-commit regressed 0/8 vs v2, the failure surfaced via
  smoke A/B, not via a unit test. The unit tests had been written
  to the new semantics, so they all passed even though the agent
  behaviour was wrong. **Promotion candidate:** add a Mission-pipeline
  parity test that asserts `propose_*` + `settle_plan` produces the
  same Intent stream as the previous-version v2 strategy on a fixed
  obs panel, to catch silent regressions earlier.

## Frictions logged this session

`audit/friction.md` 2026-05-11 entries (cross-linked):

- `guard-mechanisms-check-only-to-predicted-endpoint` — load-bearing.
- `bundler-missing-block-e-modules` — second occurrence of the
  bundler-stale-DEFAULT_LIB_ORDER pattern in one day.
- `8-seed-mvp-result-is-noise` — calibration data point.
- `until-loop-spurious-process-detection` — workflow.
- `rolling-last-2-tradeoff-needs-explicit-decision-record` —
  process gap.

## Promotion candidates (PI ratification pending)

### [ ] `.claude/skills/kaggle-comp/improvements.md` — bundler auto-discovery

**Tag:** `bundler-missing-block-e-modules` (and earlier
`bundler-missing-trajectory`)

**Where to insert:** new entry under "Pending" / `[CODE-COMP-DISCOVERED]`.

**What to add:**

> ### [ ] [CODE-COMP-DISCOVERED] bundle_agent.py: auto-discover required lib modules from agent imports
>
> Origin: Orbit Wars 2026-05-11. Twice in one day a new `lib/*.py`
> module was added (`lib/trajectory`, `lib/mission` + `lib/missions/*`
> + `lib/planner`) and the bundler's hand-maintained `DEFAULT_LIB_ORDER`
> wasn't updated. First time the bundle parity test caught it; second
> time the E.2 self-play gate caught it. Pattern: any module added to
> `lib/` since the bundler was last touched silently breaks bundles
> when imported transitively.
>
> Fix: replace the `DEFAULT_LIB_ORDER` constant with AST-based
> discovery — parse the agent's `main.py`, traverse `from lib...
> import ...` statements, build a topologically-sorted module list,
> bundle in order. Eliminates the manual maintenance burden.

**Why:** two distinct cost events this session (one production bundle
crash, one parity-test catch). Pattern recurs every time the lib
surface area grows.

### [ ] `.claude/skills/kaggle-comp/improvements.md` — gate rule on lift claims

**Tag:** `8-seed-mvp-result-is-noise`

**Where to insert:** new entry under "Pending" / `[CROSS-CUTTING]`.

**What to add:**

> ### [ ] [CROSS-CUTTING] CLAUDE.md / Rule 19 addendum: ≥32 seeds for any lift claim
>
> Origin: Orbit Wars 2026-05-11. The Block E v3.1 lookahead MVP's
> 8-seed result (11/16 = 68.8% h2h vs v2) was reported as a positive
> lift; 32-seed retest collapsed to 50/50 parity. Wilson 95% CI on
> 8-seed = [44%, 86%] is already wide enough to contain parity. The
> plan file's recommended gate was always 32 seeds; we jumped ahead.
>
> Fix: encode in Rule 19 (Experimentation harness) — "A lift claim
> (vs control or PRIMARY) requires ≥32 seeds × both-seats unless the
> point estimate is ≥75% with n ≥ 16 AND the Wilson lower bound is ≥
> 55%." Smaller-n claims must be labeled "smoke" not "lift."

**Why:** mis-reporting an 8-seed result as a lift is a recurring
pattern across comps. The cost here was attention diverted to a
"why did v3.1 lift?" investigation that turned out to have nothing
to investigate.

### [ ] `.claude/skills/kaggle-comp/improvements.md` — rolling-last-2 decision record

**Tag:** `rolling-last-2-tradeoff-needs-explicit-decision-record`

**Where to insert:** new entry under "Pending" / `[CODE-COMP-DISCOVERED]`.

**What to add:**

> ### [ ] [CODE-COMP-DISCOVERED] CLAUDE.md / Rule 12 addendum: pre-submit eviction record
>
> Origin: Orbit Wars 2026-05-11. v3_snipe push at 12:16 UTC evicted
> v1.2/roi (μ=1006.9, the older/higher-rated of the two prior slots),
> leaving [v2 (974.3, buggy), v3_snipe (PENDING)]. The trade-off was
> rational (v3 ≫ v1.2 locally) but wasn't pre-registered.
>
> Fix: Rule 12 sub-clause — "Before every Orbit Wars submission, write
> a one-line decision record: 'evicts <submission_id (μ=...)>; expected
> Δμ over current best = +X based on <local panel result>.' Append to
> state/current.md::last_submission_message."

**Why:** rolling-last-2 makes every submission an explicit trade
between known and unknown. Pre-registration forces honest
calibration: if expected-Δμ doesn't justify the eviction, don't push.

## PI additions (from step 4)

> PI: pending PI input. Block until PI replies via WRAPUP-step-4 ask.

(If PI does not surface additions, the three candidates above stand
as drafted, awaiting yes/no/edit ratification.)

## Framework version at session-end

- Commit SHA at wrap-up start: 657cddd (`Submit v3_snipe as #52544634;
  fix bundler to inline Block E modules`).
- Active rules (CLAUDE.md `## Operating rules — concise`): 1, 2, 4, 5,
  6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
  26, 27, 28, 29, 30, 31, 32, 34, 35, 36 (TABULAR-ONLY rules 3, 24,
  25, 33 skipped per CLAUDE.md tagging).
- Loaded skills this session: kaggle-comp (implicit via project),
  postmortem (invoked at wrap-up).
- Branch: `claude/bootstrap-agentic-systems-lqnm6`, 14 commits ahead
  of `origin/main`.
