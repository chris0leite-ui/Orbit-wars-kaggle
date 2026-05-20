# 2026-05-20 — cross-branch consolidation pass

Branch: `claude/review-skills-improvements-moKOR`. Doc-only.

## Setting

Eight active `claude/*` branches building parallel Orbit Wars agents:
- Analytical chooser line (OyoYR + rebased)
- Hybrid-sim production line (btjeK + EpMVP + lO4mm foundation)
- Verify-first + goal-directed planning (PFhzM + precision-physics substrate)
- Closed axes (phase7 chain-bonus, OyoYR value-head)

State docs on each branch had diverged. `state/current.md` on `moKOR`
still claimed v15_banded was live (5 days stale, 5 submissions behind).
`audit/friction.md` had per-branch frictions not promoted up. Live
rolling pair was 320 μ below team peak because five sequential pushes
from one branch had evicted strong agents from sibling branches without
coordination — a friction (`cross-agent-push-coordination-gap`)
recorded on btjeK 14h before this session.

## What landed

Single source of truth across branches: `state/MULTI_BRANCH.md` (live
Kaggle, three-track registry, closed-tracks list, push claim board,
per-branch sync table). Companion `state/TOOLS.md` enumerates A/B
harnesses, single-game diagnostics, validation suite, and a 6-step
consolidation-merge gate. CLAUDE.md gained rules 41-47, with btjeK's
pre-existing Rule 41 candidate (confound-sweep) adopted as #41 and my
proposed cross-branch coordination gate moved to #42.

`improvements.md` rotated: 7 items promoted to rules, 2 superseded,
archive moved to `improvements-archive-2026-05-20.md`. Skill `SKILL.md`
+ `day-loop.md` got a step-0 / step-1 amendment to load MULTI_BRANCH
+ TOOLS first.

## What this changes for the team

The substrate tier split (closed-form Tier 1 vs simulation Tier 2)
borrowed from OyoYR-rebased's HANDOVER is now the canonical framing.
Each work-track sits on top of one or both tiers — they're orthogonal.
This dissolves the false binary "analytical OR simulation" that earlier
session prose had implied.

The three-track registry recognizes that PFhzM's "verify-first +
goal-directed planning" is methodologically distinct from both the
analytical-chooser line AND the hybrid-sim production line. Phase A
Test 3 PASS + wrap-baseline asymmetry (12/32 = 37.5%) are the only
positive signals from Track C; they hint at "augment baseline with a
portfolio veto layer" rather than "replace chooser entirely."

## What was deliberately NOT done

- No code consolidation (HANDOVER.md priority-1 for next session).
- No recovery submission (PI open question; 3 strong lineages identified:
  composite_a2 1149.2, trajectory v4 1143.7, hold-feasibility 1135.1).
- No SessionStart hook (improvements.md TOP PRIORITY but needs code).
- No promotion of this session's three workflow frictions to skill rules
  — PI explicit "do not promote" at wrap-up.

## What this session got wrong

Three underspecified inventory drafts (PI nudged thrice to surface
itemized content). Missed precision-physics-engine-ymJkA branch in
initial survey (recency filter excluded it). Authored Rule 41 without
cross-branch grep, collided with btjeK's existing proposal. Full
detail in `audit/2026-05-20-postmortem-review-skills-improvements-moKOR.md`.
