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

### [ ] [CROSS-CUTTING] Session-start Kaggle-submissions reconciliation when HANDOVER claims a champion

`tag: handover-staleness-vs-kaggle-state` (2026-05-19,
claude/reverse-engineer-seat-geometry-BPJKs).

Third recurrence of the "framework didn't catch that the baseline
truth wasn't being verified" pattern, alongside
`wrong-file-recon-skipped-state-md` (5/18) and
`agent-introspection-skipped-bootstrap` (5/13). HANDOVER.md framed
v15_banded as the current champion; Kaggle showed submission
52784853 (commit `82df5b8`, μ=1121.2) from ~20 commits of
architectural progress on origin/main. The bootstrap "ahead 2 /
behind 1" output understated the divergence — the "+1" was a merge
commit pulling in the entire trajectory-chooser lineage.

**Where to insert:** bootstrap.sh / SessionStart hook output; also
reference in CLAUDE.md Rule 32.

**What to add:** when HANDOVER.md is >24h old OR the branch is
"behind N" origin/main with N≥3, run
`kaggle competitions submissions <comp> | head -10`
and require the agent to reconcile the most recent submission's
description against HANDOVER's "current champion" framing BEFORE
forming any h2h plan. Display the most recent 2-3 submissions in
the bootstrap summary so the gap is impossible to miss.

**Why:** Rule 32 catches code-tree freshness; this catches
narrative-freshness against the comp's ground truth. HANDOVER is
load-bearing; if it's wrong about the champion, every downstream
plan is wrong. Cost this session: ~30 min planning vs the wrong
opponent before PI override.

### [ ] [CODE-COMP-DISCOVERED] Bundle-vs-focal name collision in `fast.py eval --vs <path>`

`tag: bundle-filename-collides-with-focal-name` (2026-05-19).

`scripts/bundle_agent.py` defaults to `submissions/<agent_dir>.py`.
`fast.py:152::resolve_agent_spec` returns `Path.stem` for arbitrary
file paths. When the focal agent is `baseline` and we A/B against
a bundle of `agents/baseline/` at a different commit, `fast.py`
silently skips all games via the `same agent as focal`
short-circuit. ~9 min wasted this session.

**Where to insert:** `fast.py:152` `resolve_agent_spec`.

**What to add:** when the resolved opponent stem equals the focal
name AND the path resolves outside the repo, suffix the stem with
a short hash of the path (`<stem>@<sha8>`). Or extend
`scripts/bundle_agent.py` with a `--name` override and document
the A/B convention to always pass it.

**Why:** Silent failure (0 games + one-line SKIP, no error). The
A/B-vs-bundle workflow is the standard Rule 27 gate; anyone
repeating it will hit this.

### [ ] [CODE-COMP-DISCOVERED] A=A control variance characterisation in `fast.py eval`

`tag: same-config-runs-diverge-9pp` (2026-05-19).

Same config at `BASELINE_ROI_DENOM_FLOOR=1.0` produced 15.6% (5/32)
in one A/B run and 6.2% (2/32) in a back-to-back run vs the same
opponent on the same geometry panel. 9-point swing on identical
setup — Wilson intervals don't model this noise.

**Where to insert:** new `fast.py eval --self-control` flag, or a
test in `tests/test_fast_eval_determinism.py` that runs a known
A=A pair twice and reports the empirical variance band.

**What to add:** characterise the noise floor empirically so we
don't over-interpret borderline gates (Wlo near 0.55). Document
the source (likely kaggle_environments random tie-breaks; verify
via fixed-seed control).

**Why:** Single occurrence didn't block, but the implication is
that any 32-seed A/B has ~±5pp implicit noise above what Wilson
reports. Decisions near the gate threshold should use ≥64 seeds
or two independent 32-seed runs.

### [ ] [DOCS] Clarify `--max-seeds` vs `--geometry-panel` interaction in `fast.py eval --help`

`tag: max-seeds-overridden-by-geometry-panel` (2026-05-19).

`fast.py eval --geometry-panel --max-seeds 16` ran 32 seeds per
variant, not 16. Help text mentions auto-bump to 128 when
`--max-seeds` is "still the default" but is silent on the floor
when a smaller value is explicit.

**Where to insert:** `fast.py` argparse help text for `--max-seeds`
and `--geometry-panel`.

**What to add:** clarify the interaction. If geometry-panel
enforces a minimum cell-coverage seed count, say so. If
`--max-seeds N` is honoured strictly, document that. Match the
actual behaviour.

**Why:** Documentation gap; single occurrence; cost was ~24 min
extra compute (the wider scan was harmless, just budget-mismatched).

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
