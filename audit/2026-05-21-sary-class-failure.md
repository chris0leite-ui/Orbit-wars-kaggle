# 2026-05-21 — Sary-class Panel Anchor: FAILED (insightful negative)

**Branch**: `claude/audit-workflow-performance-btjeK`
**Plan**: `/root/.claude/plans/so-now-research-and-zany-widget.md`
**Predecessor**: `audit/2026-05-21-ledger-validation.md`

## Verdict

**Don't add `agents/sary_class/` to the panel.** The whole approach
(build a simple aggressive agent to catch under-emission regressions
locally) doesn't work because the failure class requires a DEFENSIVELY
COMPETENT opponent, not just an aggressive one. Every simple/older
anchor we tested loses to the failed ledger; only the current
production baseline catches the regression.

**New workflow:** candidate vs current production at n≥8 IS the
under-emission gate. No separate sary-class anchor is needed (or
buildable in a reasonable afternoon).

## What I tried

### Phase A — built `agents/sary_class/main.py`

Two iterations:
- **Selective** (MIN_FIRE_SHIPS=15, sized to outnumber defenders +
  production-during-flight): 0/4 vs current production.
- **Nuke** (MIN_FIRE_SHIPS=8, drain all available ships to top-ROI
  target): 0/8 vs current production. Cadence 0.86/turn (under the
  1.5 target) with tiny ~7-ship fleets that mostly bounce.

Two consecutive same-axis variants both fail — per Rule 37 lock the
axis. Do not try a third sary_class hand-coded variant without new
data.

### Phase B — test EXISTING anchors as regression detectors

Run each panel anchor vs `agents/_ledger_on/main.py` (the failed
ledger that lost 2/16 to current production in the 2026-05-21
validation):

| anchor | vs led_on (n=4 × 2 dirs = 8 games) | beat led_on rate |
|---|---|---|
| `agents/simple/roi.py` | 1/8 | 12.5% |
| `agents/sary_class/main.py` (new) | 0/8 | 0% |
| `submissions/v7_0_drop_one.py` | 0/8 | 0% |
| `submissions/baseline.py` (current production) | 14/16 | **87.5%** (prior data) |

**Only current production is strong enough to catch the regression.**
roi, sary_class, and v7_0 all LOSE to led_on. Adding any of them as
a "regression detector" would give false confidence — they'd pass
a regressed candidate that current production would catch.

## Why simple anchors can't catch this regression

led_on is built on the SAME advanced infrastructure as current
production (proposer, chooser, value head, lite_greedy opp model,
defensive reactive policy) — minus the wait_N reservation. It's an
"almost-prod-with-a-bug" agent that crushes anything below
prod-class strength.

The under-emission failure mode only manifests when the opponent has:
- Enough defensive capability to PUNISH the drained source reserves.
- Enough offensive coordination to exploit weakened defenses.
- A long-game value head that doesn't get fooled by short-term
  ship-count advantages.

Simple agents (roi, sary_class, v7_0) have none of these. They're
crushed by ANY agent that has the production-class infrastructure,
regressed or not.

## What this changes about workflow

Previous (broken) workflow:
1. Build candidate.
2. Local A/B against panel (roi, v7_0, v4_planner, v3.5.1).
3. If panel clears, submit.
4. Hope live h2h doesn't surface regressions.

New (correct) workflow:
1. Build candidate (must be env-gated by default).
2. **FIRST gate**: candidate vs current production at n=8. Wilson-LB
   must clear 0.40+ to proceed.
3. If gate fails, candidate is regressed against the very class of
   opponent we care about (similar-strength). Iterate or kill.
4. If gate passes, run extended panel (roi/v7_0/etc.) for OTHER
   regression classes (defensive over-reaction, weak-opponent
   scaling) at n=32.
5. Only then bundle + submit.

Current production is slow (~80s/game) but is the only valid
under-emission detector. Accept the cost.

## Codebase changes

- `agents/sary_class/main.py` — kept as a research artefact / future
  starting point. Not added to the panel. (~130 lines.)
- No changes to production agent.
- No changes to existing panel scripts (the new workflow is a process
  change, not a code change).

## Future work (out of scope this cycle)

Mechanisms that COULD produce a usable panel anchor:
1. **Distill current production into a faster surrogate.** Train a
   small policy network on production's actions; deploy as fast
   anchor. ~3-5 days; non-trivial.
2. **Tournament-bench top-LB replay-imitation policies.** Pull
   top-100 agents' replays; train an imitation per playstyle.
   Larger but well-defined effort.
3. **Build a defensive-reserve emitter** that hoards then strikes,
   mimicking sary's actual ship-size profile (avg 55/launch vs our
   handcoded 7/launch). The defensive sense is what sary_class is
   missing.

All of these are deferred. Prefer to spend cycles on improving the
PRODUCTION agent rather than building synthetic opponents.

## Wrappers that stay in the repo

- `agents/_ledger_{on,off,hard}/main.py` — for re-runnable h2h.
- `agents/_mpc/main.py` — for re-runnable mpc comparison.
- `agents/sary_class/main.py` — failed-but-instructive research code.
- `scripts/_ledger_ab_driver.py` — A/B harness with Wilson-LB.

All env-gated or test-only; production behaviour unchanged.

## Provenance

- Sary-class development output:
  `/tmp/claude-0/.../tasks/{bxskrdxon,bngk7ch42,b0uhpy709}.output`.
- vs led_on results: `/tmp/.../{bhrl6jjjj,bluwtrccd,bikrcwoq6}.output`.
- Replays of failed games: `audit/replays/20260520T080{340,447,647,807,957}Z/`.
- New agent: `agents/sary_class/main.py` (~130 lines).
