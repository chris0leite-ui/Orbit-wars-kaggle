# kaggle-comp skill — cross-comp improvements log

> Cross-comp distilled patterns. Items here qualify when a friction
> pattern appears in 2+ comps, costs > 1 LB slot or > 1h of agent time,
> or required a human nag. Each entry shows status, scope, and
> required edit. **Keep this file ≤150 lines.** When it grows past,
> rotate per self-improvement.md: archive applied/superseded items to
> `improvements-archive-YYYY-MM-DD.md`.
>
> Last rotation: 2026-05-20 (claude/review-skills-improvements-moKOR).
> Full prior history at `improvements-archive-2026-05-20.md`
> (which itself supersedes `improvements-archive-2026-05-14.md`).

## Tags

- `[CROSS-CUTTING]` — applies to any Kaggle comp (tabular or code/agent).
- `[TABULAR-ONLY]` — applies only to tabular comps; ignore on Orbit Wars.
- `[ADAPT-FOR-CODE-COMP]` — concept transfers but implementation differs in code/agent comps.
- `[CODE-COMP-DISCOVERED]` — new lessons that originate from a code/agent comp.

## Pending — promotion needed

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
4. **NEW (2026-05-20):** `kaggle competitions submissions orbit-wars | head -5`
   surfaced inline (drives the MULTI_BRANCH live-Kaggle table that
   Rule 42 now references).

Cost evidence (3 incidents): ~30 min designing duplicated work + 16
spurious pytest failures the agent labelled "pre-existing" + 10 min
manual recovery on day-1 audit branch.

### [ ] [CROSS-CUTTING] Stop-hook should not force commit-before-verify

`tag: stop-hook-pressure-commits-speculative-WIP` (2026-05-16,
v13 session).

Stop-hook `~/.claude/stop-hook-git-check.sh` warns on every turn
with uncommitted changes. Pattern: agent commits speculative work
to silence the hook, then has to revert when verification reveals
regression. Cost: 1 wasted commit/revert pair in the v13 session.

**Fix:** when a change is being VERIFIED (panel running, tests
running), use `git stash` to silence the stop-hook without
committing speculative work. Stash, run verification, pop+commit
only on PASS.

### [ ] [CODE-COMP-DISCOVERED] AST-walk import discovery in bundle_agent.py

`tag: bundler-missing-block-e-modules` and 5 related. Origin: Orbit
Wars 2026-05-11 through 2026-05-14. **Partially mitigated** by the
loud-error guard (`_assert_lib_imports_resolved`) and the EpMVP
2026-05-20 "inline agent submodules + explicit-name imports"
upgrade (addresses 3 of 5 silent-fail modes).

**Fix:** (i) merge up the EpMVP bundler upgrade as part of the
code-consolidation pass; (ii) replace hand-maintained
`DEFAULT_LIB_ORDER` with AST discovery. Parse the agent's `main.py`,
traverse `from lib... import` statements transitively, build a
topologically-sorted module list. Eliminates manual maintenance.

### [ ] [CROSS-CUTTING] PI-protocol — no-unexplained-abbreviations rule

`tag: pi-comm-no-unexplained-abbreviations`. PI verbatim 2026-05-07:
"I often struggle to understand what we are doing with so many
abbreviations and specific methods and slang." Promoted to Rule 0
of Orbit Wars's CLAUDE.md but state-doc compliance is mixed.

**Fix:** Rule 0 already exists. The remaining work is enforcement
in `HANDOVER.md` prose specifically (the doc the PI actually reads
between sessions). State files MAY stay coded for agent-to-agent
reference.

### [ ] [CROSS-CUTTING] Unify Rule 7 + 14 + 22 into single plateau-runbook

`tag: plateau-response-fragmented`. Three separate rules all fire on
plateau (Rule 7 research, Rule 14 strategy-critic, Rule 22
public-notebook scan). Agent picks one and ignores the others.

**Fix:** new file `.claude/skills/kaggle-comp/plateau-runbook.md`
that sequences all three. Reference from a single new rule that
supersedes 7/14/22.

### [ ] [CODE-COMP-DISCOVERED] **NEW** Code-consolidation merge gate (6-step) into pre-submit runbook

`tag: cross-branch-code-divergence` (2026-05-20).

Eight active branches building parallel agents with overlapping but
not-merged code: `lib/trajectory_layer.py` (PFhzM), `agents/precision/`
(precision branch), `chooser_roi.py` (btjeK), EpMVP bundler upgrade,
EpMVP oracle tests. The 6-step consolidation gate now lives in
`state/TOOLS.md`. Needs promotion into the kaggle-comp skill's
day-loop / experiment-loop so that branch-only artifacts get a
formal merge-up review rather than dying with the branch.

**Fix:** add `consolidation-runbook.md` to the skill (separate from
plateau-runbook); reference it from day-loop step 1 alongside
`state/MULTI_BRANCH.md`. Defer the file write until next session;
the state docs already encode the gate, so this is the discoverability
layer.

### [ ] [CODE-COMP-DISCOVERED] **NEW** Three-track parallel work as canonical pattern for code-comps

`tag: parallel-track-fragmentation` (2026-05-20 cross-branch survey).

Eight active branches collapsed into THREE methodological tracks
(Analytical / Hybrid-Sim / Verify-first) plus shared substrate
(Tier-1 closed-form vs Tier-2 simulation). This split should be
the default mental model for any Simulation-class Kaggle comp —
not invented per-comp.

**Fix:** add "Track registry" section to kickoff-runbook so new
code-comps start with the three-track placeholder filled. Defer
until next code-comp kickoff (no urgency for Orbit Wars).

## Applied in 2026-05-20 audit pass

Moved out of this file to keep it lean. Full details in
`improvements-archive-2026-05-20.md`.

- `[APPLIED]` Rule 41 — confound-sweep before correlational conclusion → CLAUDE.md
- `[APPLIED]` Rule 42 — pre-submit cross-branch coordination gate → CLAUDE.md + `state/MULTI_BRANCH.md`
- `[APPLIED]` Rule 43 — multi-opponent panel mandatory pre-submit → CLAUDE.md (supersedes "--vs-panel mandatory" pending)
- `[APPLIED]` Rule 44 — state-of-truth read before subsystem edits → CLAUDE.md (supersedes "read state docs" pending)
- `[APPLIED]` Rule 45 — n ≥ 32 minimum for A/B lift claims → CLAUDE.md
- `[APPLIED]` Rule 46 — bundle + parity smoke before submission → CLAUDE.md
- `[APPLIED]` Rule 47 — physics-primitive verification before agent design → CLAUDE.md
- `[APPLIED]` `state/MULTI_BRANCH.md` + `state/TOOLS.md` as single source of truth
- `[APPLIED]` SKILL.md step 0 "load MULTI_BRANCH + TOOLS first"
- `[APPLIED]` day-loop.md step 1 amendment for code-comp branch coordination

## Applied in 2026-05-14 audit pass

Full details in `improvements-archive-2026-05-14.md`.

- `[APPLIED]` Rule 37 — consecutive-falsification cap → CLAUDE.md
- `[APPLIED]` Rule 2 extension — two-tier kernel smoke → CLAUDE.md
- `[APPLIED]` bootstrap.sh data-presence canonical-file check
- `[APPLIED]` bootstrap.sh KGAT-token + harness-env-var detection
- `[APPLIED]` bootstrap.sh blinker preflight + cred smoke
- `[APPLIED]` bundle_agent.py loud-error on missing lib modules
- `[APPLIED]` bundle_agent.py refuse-tracked-overwrite (`--force`)
- `[APPLIED]` fast.py `--vs-panel` 3-opponent calibration
