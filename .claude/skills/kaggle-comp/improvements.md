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

### [ ] [CROSS-CUTTING] Surface team-best submission separately from this-branch's last submit

`tag: branch-lagging-sibling-by-46mu` (2026-05-23,
claude/strategy-framework-design-OyoYR).

State files (`state/current.md`) carry `current_submitted_agent` and
`team_peak_agent` fields, but the team-peak field was set 2026-05-17
to v15_banded and has not been refreshed since. Meanwhile a sibling
branch (`claude/extract-physics-trajectory-Vjaz9`) shipped
`baseline_joint_aggr_consolidated_orbitfix` at μ=1165.4 on 2026-05-22
— **46 μ above** the recorded team peak. The 5/23 session on the
strategy-framework branch ran an entire leaf-side rerun cycle on top
of v15 (μ≈1119.6) without surfacing that the team's strongest agent
was now on a sibling branch and 46 μ ahead. Session-start hook
prints `git log HEAD..origin/main` which DOES contain the sibling's
submit commits — the information was visible and not processed.

**Fix:** two coordinated changes —

1. Add a `team_best_submission` block to `state/current.md` schema,
   distinct from `current_submitted_agent` and `team_peak_agent`.
   Format:
   ```yaml
   team_best_submission:
     sub_id: <int>
     agent: <name>
     branch: <claude/...-slug>
     mu: <float>
     refreshed_utc: <ISO>
   ```
   Updated by every WRAPUP that pushes a new submit, and by the
   session-start hook if it can prove the previous value is stale.

2. Extend the session-start hook (per comp's
   `bootstrap.sh` / SessionStart hook) to: (a) run
   `kaggle competitions submissions <comp> --csv | head -20`,
   (b) parse out the top `publicScore` across all submissions in
   the last 7 days, (c) report `TEAM BEST: <agent> @ μ=<x>` if it
   differs from `state/current.md`'s `current_submitted_agent`.
   Same gate as the existing `git fetch origin` step.

**Cost evidence:** an entire session (5/23) optimised against a
foundation 46 μ behind team-best. The leaf-side rerun itself was
+0 μ; the strategic miss was the cost. Promotion ratified by PI
2026-05-23 in postmortem.

### [ ] [CROSS-CUTTING] Read state docs + recent audits before proposing subsystem edits

`tag: wrong-file-recon-skipped-state-md` +
`tag: crn-symmetry-broken-without-reading-prior-audits`
(both 2026-05-18, claude/reverse-engineer-seat-geometry-BPJKs).

Same-session double recurrence: proposed edits to "our agent" twice
in one session without first reading the state docs that index the
agent (`state/current.md`) or the audit notes that document the
subsystem's design history (`audit/2026-05-17-state-function-
principled-fix-results.md`). First recurrence caught by PI:
"is that really our submission? check again." Second cost ~30 min
compute + one full panel slot: asymmetric Tier-1 chooser change
violated the CRN symmetry the v11→v12→v13 line had specifically
fixed; panel returned 0/32, reverted.

Both frictions have the same shape as `agent-introspection-skipped-
bootstrap` (2026-05-13) and `fix-not-validated-against-real-failing-
state` (2026-05-14) — the bootstrap-side equivalents already landed
as Rule 38 + SessionStart hook (above). The **recon-side** equivalent
has not been codified.

**Fix:** add CLAUDE.md Rule 41 — "Before proposing edits to a
subsystem in tree, (a) `cat state/current.md` to confirm which file
implements it, and (b) `grep -l "<subsystem-name>" audit/2026-05-*.md`
to check recent design audits. Mandatory if the edit modifies the
agent's behaviour at submission time."

Cost evidence: 2x recurrence within one session; ~30 min eval burned
on a CRN-broken change; one panel slot consumed by a 0/32 result.

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
