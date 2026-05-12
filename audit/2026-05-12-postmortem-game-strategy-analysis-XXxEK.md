# Postmortem — 2026-05-12 game-strategy-analysis-XXxEK

## Session summary

PI asked a "mathematician + battlefield specialist" persona to
bootstrap, understand the geometry of Orbit Wars, and advise
strategic options. Outputs:

- `audit/2026-05-12-battlefield-geometry-report.md` — empirical
  geometry over 100 seeds; six load-bearing facts (4-fold symmetry on
  93% of seeds, 100% diagonal homes, 20% sun-blocked pair fraction,
  35% / 75% closest-approach savings on orbiting targets).
- `scripts/geometry_report.py` — regenerable.
- 8-option strategy menu in `/root/.claude/plans/...` (PI approved).
- Option A (recapture wire-up) — **attempted, regressed 36% in 200-game
  A/B, REVERTED**. Documented in
  `audit/2026-05-12-recapture-wireup-ab.md`.
- Option C (orbital phase-lead targeting library) — landed as
  `lib/orbit_lead.py` + 10 tests, all pass. Pure-library, no agent
  change, no submission.

Three commits on `claude/game-strategy-analysis-XXxEK`:
- `307ee8b` Day-3+1: battlefield-geometry report + initial recapture
  wire-up bundle.
- `f55bc77` Revert recapture wire-up after A/B fail (Wilson lo 0.297).
- `034b756` Add lib/orbit_lead.py — keystone for Options C/D/E/G.

No submissions consumed. Budget: 5/5 still available.

## What went wrong

- **Bad decision (decision-quality terms):** Plan agent's HIGH prior
  (+100-150 μ) on Option A (recapture wire-up). Given pre-run info,
  this prior was wrong, because audit/friction.md ALREADY contained
  the same-day entry
  `multi-mission-stack-regresses-even-with-conditional-gates` flagging
  that settle_plan is NOT mission-class-additive. I read the friction
  during planning but weighted the games-analysis comeback-gap evidence
  too heavily and the planner-non-additivity evidence too lightly. Not
  a hindsight-only call — the priors were in the right column at
  decision-time.

- **No PI overrides this session.** PI gave directional guidance (full
  geometry notebook first; concrete first step; "go") and ratified the
  plan. The recapture revert was self-detected via the plan's own
  Wilson gate, not PI-corrected.

- **Rule-bypass:** none. Rule 18 (claim ISSUES.md leaf for probes ≥10
  min) didn't apply — the geometry script ran in ~30 s and the 200-
  game A/B with 4 workers ran in ~5 min. Rule 27 (pre-submit prediction
  diff) didn't apply because no submission was made. Rule 6
  (heuristics before heavy compute) was followed: orbit_lead is
  closed-form, not learned.

- **Rule-gap:** the plan agent had access to today's friction file but
  no explicit constraint to weight planner-arbitration frictions when
  scoring new mission classes. This is the promotion candidate below.

## Frictions logged this session

Three new entries in `audit/friction.md` under
`## 2026-05-12 PM (game-strategy-analysis-XXxEK)`:

- `recapture-prior-too-high-ignoring-planner-arbitration` — load-
  bearing; same family as the existing
  `multi-mission-stack-regresses-even-with-conditional-gates`.
- `env-step-0-vs-step-1-zero-rotation-quirk` — small test-time
  footgun, no promotion.
- `drive-by-validation-picked-wrong-home-planet` — small footgun;
  suggests a tiny `lib/intent.my_home(world)` helper next time
  someone touches mission code.

## Promotion candidates (PI ratified: **NO — kept in audit only**)

PI declined promotion this session: the lesson stays in
`audit/friction.md` (tag `recapture-prior-too-high-ignoring-planner-
arbitration`) and in this postmortem. If the pattern re-fires a fifth
time without a rule, re-propose promotion next session.

### [ ] `.claude/skills/kaggle-comp/improvements.md` — cap mission-class wire-up priors at MED (NOT PROMOTED)

**Tag:** `mission-class-wire-up-prior-cap` (decision-quality lesson
from the recapture wire-up A/B failure)

**Where to insert:** new entry under the Orbit Wars / code-comp
section. Same file location as prior promoted entries.

**What to add:**

```markdown
### Mission-class wire-up: cap Plan-agent priors at MED until ablation passes

When Plan agent (or any planning step) scores a strategic option that
ADDS a new Mission class / proposer to `settle_plan` (or any per-source
greedy planner), the prior MUST be capped at MED (30-100 μ) regardless
of the underlying gap evidence. HIGH priors require either:

  - the new class has already cleared a 16-seed Wilson lo ≥ 0.55
    ablation gate (variant ⊕ baseline-best vs baseline-best alone), OR
  - the existing planner is provably mission-additive (no shared
    same-source contention with the existing proposers).

**Why:** observed at least 3 times in 2026-05 (v3.3 blanket eta fix,
v3.4 NEUTRAL_BONUS, v3.5 full-stack, and Option A recapture wire-up
all regressed 28-42% in A/B despite priors of +50-150 μ). settle_plan's
per-source greedy is not mission-class-additive; adding ONE proposer
shifts the proposal distribution enough to displace higher-EV
existing picks.

**Cost evidence:** 200-game A/B (Option A 2026-05-12, μ=36%), 32-seed
A/B (v3.5 2026-05-12, μ=39.1%), 32-seed A/B (v3.4 2026-05-12,
μ=28.1%), 32-seed A/B (v3.3 2026-05-11, μ=42.2%). Total compute cost
~30+ minutes per cycle, plus reset of the strategic plan on failure.

**Citation:** audit/2026-05-12-recapture-wireup-ab.md;
audit/friction.md tags `recapture-prior-too-high-ignoring-planner-
arbitration`, `multi-mission-stack-regresses-even-with-conditional-
gates`.
```

**Why:** same lesson has now fired four times in three days. The
existing friction entries document the pattern but the Plan-agent's
prompt does not yet enforce a prior cap, so each new session re-
discovers it. Encoding it in `improvements.md` (which the kaggle-comp
skill loads on session start) closes the loop.

## PI additions

PI: "Nothing to add — ship as-is." No additional frictions, rules, or
decision-quality flags raised.

## Framework version at session-end

- Commit SHA: 034b75619965f73e250550637e3522e5e8835118
- Branch: claude/game-strategy-analysis-XXxEK (up-to-date with origin)
- Active rules: CLAUDE.md rules 0-36 (170 lines, within cap).
- Skills loaded this session: `kaggle-comp`, `postmortem`,
  `claude-code-guide` (none invoked beyond `postmortem`).
- Plan file: `/root/.claude/plans/you-are-a-mathematician-purring-fiddle.md`.
- New artifacts:
  - `audit/2026-05-12-battlefield-geometry-report.md`
  - `audit/2026-05-12-battlefield-geometry-data.json`
  - `audit/2026-05-12-recapture-wireup-ab.md`
  - `audit/2026-05-12-postmortem-game-strategy-analysis-XXxEK.md`
    (this file)
  - `audit/tournaments/20260512T075340Z.json`
  - `scripts/geometry_report.py`
  - `scripts/run_recapture_ab.py`
  - `lib/orbit_lead.py`
  - `tests/test_orbit_lead.py`
