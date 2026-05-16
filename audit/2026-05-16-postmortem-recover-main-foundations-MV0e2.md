# Postmortem — 2026-05-16 recover-main-foundations-MV0e2

Branch: `claude/recover-main-foundations-MV0e2` (ahead 38 / behind 2)
Session focus: deep dive on 213tubo loss; v13 reactive-opp fix; submit.

## What went wrong

- **Commit-before-panel on lite_greedy-neutral-fix.** I made the
  correctness fix to `lib/opp_model.py` (don't accrue neutrals),
  verified Felipe 2/2, and pushed (commit d13ead0) before running
  the panel. Panel showed v7_0 Wlo 0.700 → 0.483 — hard regression.
  Reverted (commit 1c5e059). Decision-time priors: I knew panel is
  the actual gate, the single Felipe seed proves nothing. The
  Stop-hook pushing for commit was a distraction, not a reason.
  Should have used `git stash` or just left the working tree dirty
  while panel ran. Cost: 1 commit + 1 revert (recoverable, low).

- **Bumped MAX_HORIZON without tracing whether candidates would
  use it.** Tried MAX_HORIZON 30 → 60 thinking it would expose
  counter-attacks. But candidates use `horizon = max(eta+SETTLE,
  MIN_HORIZON)` clipped to MAX_HORIZON; the wait-11-tgt0 candidate
  had horizon=18 regardless of the cap. Discovered empirically via
  re-trace. Cost: ~10 min misdirection.

- **First-instinct restriction-tuning bias.** When the lite_greedy
  fix regressed, I considered MIN_FLEET_SIZE bump as a quick fix
  before PI re-articulated: "model should understand that it should
  not send suicide fleets" — i.e., emergence via better modeling,
  not via restriction. This is the 3rd-4th time on this branch I've
  reached for a constant bump before considering the modeling fix
  (prior: MAX_WAIT, MIN_FLEET_SIZE earlier sessions, MAX_HORIZON
  this session). Pattern is rule-gap-worthy.

## Frictions logged this session

- `tag: restriction-tuning-before-modeling-fix` — when the easy
  fix is a constant bump (MAX_*, MIN_*, threshold filters), my
  default is to propose the bump first. PI repeatedly redirects
  toward modeling improvements. The recurring nature suggests this
  isn't an individual lapse but a structural bias in how I
  approach symptoms vs root causes.

- `tag: stop-hook-pressure-commits-speculative-WIP` — Stop-hook
  presses for committing uncommitted changes; encourages
  committing-before-verifying. Triggered commit-and-revert on
  lite_greedy-neutral-fix. Mitigation: use `git stash` for
  intermediate states; only commit verified work.

(Both will be appended to `audit/friction.md` as one-liners on
next session-start.)

## Promotion candidates (PI ratified: both APPROVED)

### [ ] CLAUDE.md — add Rule 40: prefer modeling-correctness over restriction-tuning

**Tag:** `restriction-tuning-before-modeling-fix` (recurring across
v10/v11/v12/v13 iterations on this branch)

**Where to insert:** CLAUDE.md `## Operating rules — concise`,
after Rule 39.

**What to add:**

```
40. **Prefer modeling-correctness over restriction-tuning.** When
    a failure mode can be addressed by either (a) a constant bump
    (MAX_*, MIN_*, threshold filter, hard-cap) or (b) a fix to the
    underlying model (better opp model, better leaf scoring, better
    target prediction), prefer (b). Restrictions are band-aids on
    a model that misvalues actions; the right behaviour should
    emerge from a correct model, not from artificial caps. Origin:
    repeated PI corrections on the v10-v13 iteration line where my
    instinct was to bump MAX_WAIT / MAX_HORIZON / MIN_FLEET_SIZE
    before considering the modeling fix that made the symptom
    emerge naturally.
```

**Why:** PI has had to re-articulate this principle at least 3
times across the past 2 sessions on this branch. Each correction
costs 10-30 min of "go back, fix the model instead." Writing it
as a rule makes it bind without PI intervention.

### [ ] kaggle-comp/improvements.md — verify-before-commit under stop-hook pressure

**Tag:** `stop-hook-pressure-commits-speculative-WIP`

**Where to insert:** `## Pending — promotion needed` after
SessionStart hook entry.

**What to add:**

```markdown
### [ ] [CROSS-CUTTING] Stop-hook should not force commit-before-verify

`tag: stop-hook-pressure-commits-speculative-WIP` (2026-05-16,
v13 session).

Stop-hook `~/.claude/stop-hook-git-check.sh` warns on every
turn with uncommitted changes. Pattern: agent commits speculative
work to silence the hook, then has to revert when verification
reveals regression. Cost: 1 wasted commit/revert pair per
session in the v13 line.

**Fix:** when a change is being VERIFIED (panel-running, tests-running),
use `git stash` to silence the stop-hook without committing.
Stash, run verification, pop+commit only on PASS. Document this
in CLAUDE.md or kaggle-comp skill so the pattern doesn't recur.
```

**Why:** Same-session recurring pattern (cost: 1 commit/revert).
Worth surfacing because it's a structural pressure, not a one-off.

## PI additions (from step 4)

PI: "Nothing to add — proceed." Both promotion candidates approved.

Applied: CLAUDE.md Rule 40 added; improvements.md gets the
stop-hook-pressure entry.

## Framework version at session-end

- Commit SHA: `afbecbb` (HEAD), v13 submitted as sub_id 52704189
- Active rules: CLAUDE.md Rules 1..39 (Rule 39 added 2026-05-14)
- Loaded skills this session: kaggle-comp, postmortem
