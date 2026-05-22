# kaggle-comp skill — cross-comp improvements log

> Cross-comp distilled patterns. Items here qualify when a friction
> pattern appears in 2+ comps, costs > 1 LB slot or > 1h of agent time,
> or required a human nag. Each entry shows status, scope, and
> required edit. **Keep this file ≤150 lines.** When it grows past,
> rotate per self-improvement.md: archive applied/superseded items to
> `improvements-archive-YYYY-MM-DD.md`.
>
> Last rotation: 2026-05-14 (claude/audit-workflow-friction-XD56a).
> Full prior history at `improvements-archive-2026-05-14.md`.

## Tags

- `[CROSS-CUTTING]` — applies to any Kaggle comp (tabular or code/agent).
- `[TABULAR-ONLY]` — applies only to tabular comps; ignore on Orbit Wars.
- `[ADAPT-FOR-CODE-COMP]` — concept transfers but the implementation differs in code/agent comps; adaptation note inline.
- `[CODE-COMP-DISCOVERED]` — new lessons that originate from a code/agent comp.

## Pending — promotion needed

### [ ] [CROSS-CUTTING] Stop-hook should not force commit-before-verify

`tag: stop-hook-pressure-commits-speculative-WIP` (2026-05-16,
v13 session).

Stop-hook `~/.claude/stop-hook-git-check.sh` warns on every turn
with uncommitted changes. Pattern: agent commits speculative work
to silence the hook, then has to revert when verification reveals
regression. Cost: 1 wasted commit/revert pair in the v13 session
(lite_greedy-neutral-fix committed before panel ran; panel showed
Wlo 0.700 → 0.483; reverted).

**Fix:** when a change is being VERIFIED (panel running, tests
running), use `git stash` to silence the stop-hook without
committing speculative work. Stash, run verification, pop+commit
only on PASS. Document this in CLAUDE.md or kaggle-comp skill so
the pattern doesn't recur. Alternative: extend stop-hook to skip
warning when a background verification job is in flight (less
robust; relies on detecting in-flight jobs).

### [ ] [CROSS-CUTTING] **TOP PRIORITY** SessionStart hook: bootstrap + git fetch

`tag: fix-not-validated-against-real-failing-state` (2026-05-14),
`tag: agent-introspection-skipped-bootstrap` (2026-05-13),
`tag: handover-stale-at-session-start-no-git-log-check` (2026-05-13).
Three same-class incidents in two days. Pattern: a rule written in
friction.md doesn't bind because friction notes don't gate behaviour.
Promoted to CLAUDE.md Rule 38 (fix-verification reproduces failure
state) but that still relies on the agent remembering. The structural
fix is a SessionStart hook.

**Fix:** create `.claude/skills/session-start-hook` content that runs:
1. `git fetch origin && git log -5 --oneline HEAD`
2. `bash bootstrap.sh` (idempotent — guards skip if already present)
3. `python -m pytest tests/ -q --no-header -x --tb=line 2>&1 | tail -5`

Each step's output posted before the first agent response. Stops:
- Handover-staleness (git step)
- Bootstrap-skipped (bootstrap step; runs unconditionally because
  the patched bootstrap.sh internally checks for `data/main.py`)
- Test-baseline-unknown (pytest step; gives the agent a fresh
  "16 fail" or "0 fail" reading)

Cost evidence (3 incidents): ~30 min designing duplicated work + 16
spurious pytest failures the agent labelled "pre-existing" + 10 min
manual recovery on day-1 audit branch.

### [ ] [CODE-COMP-DISCOVERED] AST-walk import discovery in bundle_agent.py

`tag: bundler-missing-block-e-modules` and 5 related. Origin: Orbit
Wars 2026-05-11 through 2026-05-14. **Partially mitigated** by the
loud-error guard added 2026-05-14 (`_assert_lib_imports_resolved`).
That stops the silent-NameError failure but still requires manual
maintenance of `DEFAULT_LIB_ORDER`.

**Fix:** replace the hand-maintained `DEFAULT_LIB_ORDER` with AST
discovery. Parse the agent's `main.py`, traverse `from lib... import`
statements transitively, build a topologically-sorted module list.
Eliminates the manual maintenance burden. Defer until the next time
a new `lib/*.py` is added.

### [ ] [CODE-COMP-DISCOVERED] Make `--vs-panel` mandatory before submission

`tag: local-vs-v7_0-only-misses-ladder-distribution`. Origin: Orbit
Wars 2026-05-12 (v3.5.1 -150μ) and 2026-05-14 (geo v3.1 -80μ). The
flag landed in source 2026-05-14; the workflow rule has not.

**Fix:** add a CLAUDE.md sub-clause to Rule 12 (submission
discipline) requiring a `--vs-panel` PASS verdict in the pre-submit
decision record. Belongs in CLAUDE.md, not in a hook, because the
PI is in the loop on every submit anyway.

### [ ] [CODE-COMP-DISCOVERED] fast.py `--require-h2h` skip-by-name misses env-var dispatch

`tag: require-h2h-skip-by-name-misses-env-var-dispatch`. Origin:
Orbit Wars 2026-05-17 PM (claude/audit-workflow-performance-btjeK).
The gate that REFUSES `fast.py eval --vs-panel` without
`--require-h2h <champion>` correctly forces an in-family h2h
declaration — but the panel-loop's "same-agent skip" at
`fast.py:_eval_vs_one` compares opponent NAME to focal NAME and
silently skips when they match. Modular agents with env-var
dispatch (e.g. `agents/baseline` with `BASELINE_VALUE_HEAD=composite`
vs unset) share a path and name; the gate skips the h2h that
matters most. Caught only because PI asked for a cross-branch
survey; otherwise we'd have submitted without validating
composite-vs-favor.

**Fix:** either (a) include the env-var snapshot in the focal-vs-
opponent identity check, or (b) accept a `--force-h2h` flag that
overrides the same-agent skip. Recurrence risk applies to every
future modular-agent + env-var-dispatch combination — the new
production default is exactly this pattern.

Cost evidence: 1 session of incomplete in-family h2h validation
on a candidate submission.

### [ ] [CODE-COMP-DISCOVERED] Bundler should inject submission env manifest

`tag: bundler-ships-with-wrong-default-env-var`. Origin: Orbit
Wars 2026-05-17 PM (claude/audit-workflow-performance-btjeK).
`scripts/bundle_agent.py` concatenates lib/ + agent files into a
single submission .py but knows nothing about runtime env-var
requirements. The first bundle of `agents/baseline` after the A2
merge shipped with `BASELINE_VALUE_HEAD` unset → default `favor`
→ MISSED the composite-in-2P lift the A/B validated. Worked around
by adding `os.environ.setdefault("BASELINE_VALUE_HEAD", "hybrid")`
manually at the top of `agents/baseline/main.py`. Fragile.

**Fix:** extend `scripts/bundle_agent.py` to read a per-agent file
`agents/<name>/SUBMISSION_ENV` (simple KEY=value lines), and emit
matching `os.environ.setdefault(KEY, value)` lines at the top of
the bundle. Eliminates the per-agent "did you remember setdefault?"
gotcha. Manifest absent → bundle behaves as today.

Cost evidence: 1 extra bundle cycle this session (~5 min) +
hidden-regression risk if not caught at re-test.

### [ ] [CROSS-CUTTING] do-and-dont.md — ISO date convention; never invent Day-N

`tag: day-counter-drift`. Origin: s6e5 2026-05-08 PM. Prose uses ISO
dates (`2026-05-14`) or comp-day-N anchored to comp start. **Never
invent a "Day N" counter that is not calendar-anchored.** The `dN`
short-codes in script names are FROZEN sequencing identifiers and
MUST NOT be reused as date references.

**Fix:** session-start sanity check: grep `state/*.md HANDOVER.md`
for `Day N` patterns where N > days-since-comp-start, surface as
friction. Add to `do-and-dont.md` once.

### [ ] [CROSS-CUTTING] CLAUDE.md / Rule 12 addendum: pre-submit eviction record

`tag: rolling-last-2-tradeoff-needs-explicit-decision-record`.
Origin: Orbit Wars 2026-05-11. Rolling-last-2 makes every submission
an explicit trade between known and unknown; without
pre-registration the agent defaults to "ship the new thing" without
sizing the eviction cost.

**Fix:** Rule 12 sub-clause — before every Orbit Wars submission,
write a one-line decision record citing (i) the submission_id and
μ being evicted, (ii) the expected Δμ over the surviving slot,
(iii) the rationale if the evicted submission has higher μ than
what stays. Append to `state/current.md::last_submission_message`
before the submit call.

### [ ] [CROSS-CUTTING] PI-protocol — no-unexplained-abbreviations rule

`tag: pi-comm-no-unexplained-abbreviations`. PI verbatim 2026-05-07:
"I often struggle to understand what we are doing with so many
abbreviations and specific methods and slang." Promoted to Rule 0
of Orbit Wars's CLAUDE.md but state-doc compliance is poor —
`state/current.md`, `hypothesis-board.md`, `mechanism-ledger.md`
remain heavy on coded references (v7_X, K=10, μ, σ, drop-one,
σ-equiv, Wilson lo).

**Fix:** Rule 0 already exists. The remaining work is enforcement
in HANDOVER.md prose specifically (the doc the PI actually reads
between sessions). State files MAY stay coded for agent-to-agent
reference.

### [ ] [CROSS-CUTTING] Unify Rule 7 + 14 + 22 into single plateau-runbook

`tag: plateau-response-fragmented`. Three separate rules all fire on
plateau (Rule 7 research, Rule 14 strategy-critic, Rule 22
public-notebook scan). Agent picks one and ignores the others.

**Fix:** new file `.claude/skills/kaggle-comp/plateau-runbook.md`
that sequences all three. Reference from a single new rule that
supersedes 7/14/22.

### [ ] [CROSS-CUTTING] Closed-form "no action" is not a bug — investigate the candidate space

`tag: analytical-zero-not-bug` (2026-05-20, slice 8c session). PI
override mid-session: agent proposed "relax the Δ > 0 emit gate"
when differential chooser produced long idle stretches. PI
corrected: "the fix is certainly not to relax the delta — Δ=+0.0
is a feature, not a bug. The candidate space is incomplete."

Generalises Rule 40 (modeling-correctness > restriction-tuning)
to the CHOOSER OUTPUT axis. When closed-form math returns "no
action," the analytical correctness is reporting honestly.
"Loosen the bound" is an anti-pattern — it injects noise back
into a clean substrate. The right response is to investigate
the candidate space:

- Are there move classes the proposer doesn't emit (migration,
  multi-source coalitions, time-shifted joints, etc.)?
- Is the analytical primitive missing scenarios (defensive
  reinforce without inbound threat, speculative pre-positioning)?
- Is the value formula missing terms (positional value,
  denied-production value)?

**Fix:** new CLAUDE.md rule:

> Rule 42 (proposed). **"Closed-form zero" is honest, not broken.**
> When an analytical chooser returns no action / Δ=0 / empty emit,
> do NOT relax the gate. The math is reporting correctly; the
> failure is in the input space. Investigate candidate generators
> and missing primitives instead. Origin: 2026-05-20 differential-
> chooser Slice 8c session; PI corrected mid-iteration.

### [ ] [CODE-COMP-DISCOVERED] Stacking analytical commits on top of a rollout chooser is noise

`tag: stack-not-replace-analytical-on-rollout` (2026-05-19 to
2026-05-20, slices 4-10 session). Pattern observed across 7
slices: every time we added closed-form commits (W1/W2/L1/L2 in
Slices 4-5, LP in Slice 6, migration in Slice 9) ON TOP OF an
existing rollout chooser, they produced noise — same or worse
win rate vs trajectory baseline. Replacing the substrate
(differential in Slice 8, joint LP in Slice 10) directionally
right but value-calibration / candidate-space gaps still bit.

Architectural lesson:

- The rollout chooser was doing implicit PLANNING via its leaf
  state (favor encoded whole-turn move-set consequences).
- Analytical commits added on top override the rollout's
  per-source allocation, not augment it.
- "Two decision-makers" architecture → conflicts → noise.

**Fix:** new CLAUDE.md rule:

> Rule 43 (proposed). **Analytical work either REPLACES the
> chooser substrate or stays in the heuristic input layer
> (cheap_delta, prerank ordering). Don't stack it on top of an
> existing chooser's decisions.** If you want to use closed-form
> commits, they must be the ONLY decision-maker, not a parallel
> commit pass over a rollout. Origin: 7 negative-result slices
> over 2 sessions; documented across audit/2026-05-19-slice*.md
> and audit/2026-05-20-slice*.md.

### [ ] [CODE-COMP-DISCOVERED] Submission-gating A/Bs MUST include the live rolling-pair leader

`tag: synthetic-baseline-doesnt-predict-live` (2026-05-23,
claude/strategy-axis-decision-3437 — items 1+3+4+5 session).

7 A/Bs of `alpha_beta_on` vs derived `*_off` bundles ranged
50.0–62.5% (looked like +12pp directional lift). Two A/Bs of
the SAME bundle vs the actual live agents:
  - vs `_phase4_step1_FND` (μ=1101): 3/8 = 37.5%
  - vs LIVE `orbitfix` (μ=1165): 3/8 = 37.5%

Cost: would have shipped α+β based on synthetic +12pp and
evicted a μ=1165 rolling-pair half (Rule 12 — rolling-last-2
eviction is unrecoverable for ~24h). Caught only when PI
prompted "have you A/B tested against our latest submission?"

Root cause: synthetic baselines (focal vs. a derived `*_off`
variant of the same source) measure "did my code do anything,"
not "does my code beat the deployed agent." They share 99%+ of
their behaviour; only the gate functions differ. Live agents are
different code lineages with different proposers, choosers, and
env-var defaults.

**Fix:** new CLAUDE.md rule (PI ratified 2026-05-23):

> Rule 48 (proposed). **Submission-gating A/Bs MUST include the
> current live rolling-pair leader as one opponent.** Synthetic-
> baseline A/Bs (focal vs. a derived `*_off` variant of the same
> source) are diagnostic, not gating. Any A/B intending to
> justify a `kaggle competitions submit` MUST include the bundle
> built from the live rolling-pair leader's commit as one
> opponent, with Wilson-lo ≥ 0.50 required. The leader is
> identified by re-pulling `kaggle competitions submissions
> orbit-wars` at A/B-design time (Rule 43) and selecting the
> highest-μ rolling-pair half. Synthetic A/Bs remain useful for
> "is the feature firing" / isolation but cannot satisfy Rule
> 12 / 43 alone. Origin: 2026-05-23
> `synthetic-baseline-doesnt-predict-live`; full table in
> `audit/2026-05-23-postmortem-strategy-axis-decision-3437.md`.

### [ ] [CODE-COMP-DISCOVERED] Hardcoded-constant variants need lazy gate functions consistently used at every dispatch site

`tag: maximin-router-read-env-not-lazy-fn` (2026-05-23,
sister-session fix at commit b436e05).

`scripts/build_topology_variants.py` produces variant bundles by
regex-rewriting the BODY of lazy `_*_enabled()` functions to
`return True` / `return False`. Variant bundles are immune to
env-var pollution by construction. BUT: a dispatcher elsewhere
in the code that reads `os.environ.get(...)` directly bypasses
the lazy function and reads the (possibly contaminated) env-var.
Result: hardcoded `True` and `False` variants both end up
running the same branch.

Concrete instance: `agents/analytical_phase_c/main.py:_decision_router`
called `os.environ.get("LP_MAXIMIN_SEARCH")` directly. Variant
`maximin_on` (with `_maximin_enabled()` hardcoded True) and
`maximin_off` (hardcoded False) both routed to `depth2_search`
because env-var was unset. Maximin's 4W/4L A/B result actually
measured LP-vs-LP. Sister session caught at b436e05.

Cost: ~3 min wallclock A/B invalidated; full Phase ε.1 isolation
result lost confidence. Cheap to fix structurally.

**Fix:** new CLAUDE.md rule (PI ratified 2026-05-23):

> Rule 49 (proposed). **When a feature gate is hardcoded in a
> variant bundle (e.g. via `build_topology_variants.py`-style
> regex rewrite of a lazy `_*_enabled()` function body), EVERY
> dispatch site for that feature MUST call the lazy gate
> function — never `os.environ.get(...)` directly.** A
> dispatcher reading env directly bypasses the hardcoded variant
> and routes both bundles to the same branch. Pin tests must
> assert the dispatch path was taken with each gate state, not
> just that the gate function returns the expected value in
> isolation. Origin: 2026-05-23
> `maximin-router-read-env-not-lazy-fn`; sister-fixed at b436e05;
> full incident in
> `audit/2026-05-23-postmortem-strategy-axis-decision-3437.md`.

## Applied in 2026-05-14 audit pass

Moved out of this file to keep it lean. Full details in the
`improvements-archive-2026-05-14.md`.

- `[APPLIED]` Rule 37 — consecutive-falsification cap → CLAUDE.md
- `[APPLIED]` Rule 2 extension — two-tier kernel smoke → CLAUDE.md
- `[APPLIED]` bootstrap.sh data-presence canonical-file check
- `[APPLIED]` bootstrap.sh KGAT-token + harness-env-var detection
- `[APPLIED]` bootstrap.sh blinker preflight + cred smoke
- `[APPLIED]` bundle_agent.py loud-error on missing lib modules
- `[APPLIED]` bundle_agent.py refuse-tracked-overwrite (`--force`)
- `[APPLIED]` fast.py `--vs-panel` 3-opponent calibration
