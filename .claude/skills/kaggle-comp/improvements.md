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

### [ ] [CODE-COMP-DISCOVERED] Rule 41 — verify primitives before iterating chooser

`tag: 5-version-iteration-without-verification` (2026-05-19 PM),
`tag: chooser-architecture-strategically-neutral` (2026-05-19 PM2),
`tag: building-from-scratch-dominates-0-of-32` (2026-05-19 PM2).
Three same-class incidents in two consecutive sessions on the same
branch. Prior session shipped 5 trajectory_roi iterations (v1→v3.1)
on physically-broken primitives — all 0-1/32 vs baseline. Current
session shipped 5 MORE architectures (Phase B veto/hybrid/baseline_veto,
goal_planner with/without validation, greedy_expand MVP) — same
outcome. MVP test confirmed: 60 LOC greedy ≈ 500 LOC architected
planner (14/32, Wilson [0.28, 0.61]). The chooser layer doesn't
matter when the primitives haven't been validated against the live
env behaviour.

**Where to insert:** `CLAUDE.md` ## Operating rules — concise — add
Rule 41 after Rule 40, with a `[CODE-COMP-DISCOVERED]` lineage note.

**What to add:**
```
41. **Verify primitives before iterating choosers.** Before shipping
    v_(n+1) on top of primitives p_1..p_k, prove EACH p_i is correct
    against env behaviour (not just self-consistent on constructed
    scenarios). For code-comps: every closed-form output that becomes
    a launch must round-trip through the env's ground-truth physics
    (e.g. `lib.trajectory.predict_fleet_fate`) before being depended
    on by a higher layer. If primitive verification fails, fix the
    primitive — do not patch the chooser around the gap. Origin: 5/19
    PM + PM2 sessions on `claude/ml-competition-strategy-PFhzM`,
    10 total iterations across 2 sessions, all 0-1/32 vs baseline
    because primitives were geometrically blind to sun / OOB / planet-
    blocking. Builds on Rule 40 (modeling-correctness over
    restriction-tuning) by adding a precondition: WHEN modeling-
    correctness is in question, verify the primitive first.
```

**Why:** Two sessions of 5-iteration sprees, 0-1/32 outcomes, $100s of
agent-time wasted on chooser polish that couldn't compensate for
broken physics. Same pattern is structurally likely on any future
code-comp where the env has physics constraints (collisions,
boundaries, conservation laws).

### [ ] [CODE-COMP-DISCOVERED] Physics-validation gate is mandatory

`tag: physics-primitives-not-used-by-our-line` (2026-05-19 PM2).
Our entire experimental agent line ignored `lib.trajectory.predict_fleet_fate`
for the whole session despite baseline.py using it as a drop-filter
in `lib/mechanism.py:593` (`sun_avoid`), `:686`
(`path_clears_other_planets`), `:775` (`oob_guard`). Cost: 4 A/Bs
burned on agents with ~6.8% physics-wasted launches (1.0% sun, 5.8%
OOB per replay-probe across 2 episodes). Discovered only via PI's
"do we use it?" question.

**Where to insert:** `CLAUDE.md` ## Operating rules — concise — add
Rule 42 (or fold into Rule 41 if compactness preferred).

**What to add:**
```
42. **Physics-validation gate.** Every emit produced by an
    experimental agent must round-trip through the env's ground-
    truth physics primitive (`lib.trajectory.predict_fleet_fate` for
    Orbit Wars) before reaching the env. Pattern: late-with-fallback
    (collect ranked candidates, validate cheapest-first, accept first
    that passes; cap fallback iterations). Mirrors
    `lib/mechanism.py:593,686,775`. Skip the gate → you'll ship sun-
    dying launches in real games while your unit tests stay green.
    Origin: 2026-05-19 PM2 audit (`audit/2026-05-19-postmortem-
    PFhzM-physics-gate-and-mvp.md`).
```

**Why:** Cost evidence is direct (4 A/Bs × ~10 min = ~40 min of
compute) plus undetectable strategic distortion (we A/B'd choosers
against an opponent that was making physically valid moves while we
weren't). Generalises: any code-comp with environmental physics
needs the equivalent.

### [ ] [CODE-COMP-DISCOVERED] Scenarios + replay-position oracle

`tag: synthetic-scenarios-miss-constructor-blind-spots` (2026-05-19 PM2).
All 17 goal_planner unit tests passed on constructed geometries;
agent shipped physically broken (launches through the sun). The
prior session's pivot already declared "scenarios are the gate"
(`knowledge-base/thoughts/2026-05-19-roi-pivot-scenario-gated-clean-
architecture.md`), but constructed scenarios only catch bugs the
constructor anticipated. Same blindness pattern in
`tests/test_cluster_solver.py:35-77` — all clear-line geometry, no
sun-blocked cases.

**Where to insert:** Amend the principle in
`knowledge-base/thoughts/2026-05-19-roi-pivot-scenario-gated-clean-
architecture.md` AND add to `CLAUDE.md` Rule 19 (experimentation
harness).

**What to add (CLAUDE.md Rule 19 addendum):**
```
   **Scenario gate must be paired with a replay-position oracle.**
   Synthetic scenarios alone catch only the bugs the constructor
   anticipated. For every new primitive, also run on N >= 10 real
   replay positions (`audit/live-episodes/*.json`) and validate the
   output against the env's ground-truth behaviour (e.g.
   `predict_fleet_fate` for launches). Bugs that only manifest in
   real-board geometry slip past synthetic gates routinely.
   See `scripts/probe_emits_via_fate.py` for the diagnostic pattern.
```

**Why:** Cost evidence — 17/17 tests green while agent was completely
broken vs Kaggle baseline (0/32). The principle "scenarios are the
gate" was already promoted last session; the amendment specifies the
NECESSARY companion check.

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
