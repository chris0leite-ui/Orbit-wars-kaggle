# 2026-05-23 — open question: merge sibling orbitfix, or work this branch's foundation?

## The question

Sibling branch (`extract-physics-trajectory-Vjaz9`) shipped orbitfix
at μ=1165.4 (sub 52912707). This branch is on v15 (μ≈1119.6).
**Should the next session on this branch start with a merge / cherry-
pick of the sibling's orbital-safety stack, or work the chooser-side
candidates on the existing v15 foundation?**

## What I'd need to answer

- What's the diff between sibling's `baseline_joint_aggr_consolidated_orbitfix.py`
  and this branch's v15-style `agents/baseline/`? Is it a small
  cherry-pickable patch (commit 38372f4 + a chooser swap) or a
  branch-wide rewrite?
- Does today's capture-fix (28ce9f3) compose cleanly with sibling's
  chooser, or does it conflict with similar work (sibling's "leaf
  in-flight fate check" Phase 3b)? If the same axis was independently
  modified on both branches, the merge has to choose one.
- What's the EV ratio? Sibling's chooser-on-this-branch (best case:
  +46 μ if perfect transfer) vs chooser-side variants on v15 (best
  case: +5-15 μ per the 5/19 candidate analysis).

## Default if PI doesn't answer

Merge sibling's orbitfix stack. Higher EV ceiling, and the value-head
work this branch has been doing is more valuable on top of a stronger
chooser foundation anyway.

## When to revisit

Top of next session on this branch. Surface before claiming any new
ISSUES.md leaf — the leaf claim depends on which foundation we're
optimising against.
