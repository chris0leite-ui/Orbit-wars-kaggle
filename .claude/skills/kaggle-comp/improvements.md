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

### [ ] [CROSS-CUTTING] CLAUDE.md Rule 41 — confound-sweep before correlational conclusion

`tag: territory-share-confound-on-distance-metric` (2026-05-19,
parked-ship-research session, PI override).

**Where to insert:** `## Operating rules — concise` in CLAUDE.md,
after Rule 40. Rule 41.

**What to add:**

```
41. **Confound-sweep before correlational conclusion.** Before
    reporting a correlation between a metric and an outcome (win/loss,
    high-LB/low-LB, treatment/control) as evidence FOR or AGAINST a
    hypothesis, enumerate ≥2 mechanical confounds the metric is
    sensitive to. If any confound is plausible AND not controlled for,
    label the result "correlational, not causal" and propose either
    a controlled subset (e.g. restrict the window) or a different
    metric. Distance-to-class-X, ratio-of-Y, and time-to-event metrics
    are the highest-risk family — they shift with the very quantity
    you're testing against. Origin: 2026-05-19 parked-ship analysis;
    rear = `min_dist_to_nonour ≥ 35` grows automatically with
    territory share, so the win/loss split was tautological with
    "we are winning," not informative about chooser behavior.
```

**Why:** PI override 2026-05-19; same root cause was visible in the
2026-05-17 `audit/replays/idle-trajectory-2026-05-17.md` framing
(43.8 % rear-ship rate reported from a pool with 87.5 % winrate) and
ran for ~36 h before challenge. Two recurrences on the same axis
within 48 h. Full postmortem: `audit/2026-05-19-postmortem-parked-ship-confound.md`.

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
