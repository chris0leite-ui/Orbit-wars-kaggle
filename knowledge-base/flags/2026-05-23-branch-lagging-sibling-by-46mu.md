# 2026-05-23 — FLAG: branch lagging sibling by 46 μ

## The gap

| Branch | Submitted agent | Live μ |
|---|---|---|
| `claude/strategy-framework-design-OyoYR` (this) | v15_banded (sub 52710995) | ~1119.6 |
| `claude/extract-physics-trajectory-Vjaz9` | baseline_joint_aggr_consolidated_orbitfix (sub 52912707) | 1165.4 |

Sibling is **+46 μ** above this branch's foundation. This branch has
spent two sessions (5/19, 5/23) on value-head reshaping that landed at
tie-with-baseline both times.

## Why the gap matters

Per Rule 12 caveat, Kaggle's rolling-last-2 evaluation means a new
submit from this branch evicts the rolling champion. The current
rolling-2 are v15_banded + v20_dogpile. Any agent we push from this
branch has to clear v15's μ=1119.6 PLUS at least σ to be net-positive.
The sibling branch's orbitfix at μ=1165 is already above our local
ceiling.

## What this branch should be doing instead

Two options, neither of which is "continue on value-head shape":

1. **Merge / cherry-pick the sibling's orbital-safety stack** (commit
   38372f4 + baseline_joint_aggr_consolidated_orbitfix chooser) into
   this branch's foundation, then rebase the value-head fix (28ce9f3)
   on top. The fix is a permanent modeling improvement that should
   live wherever the future agent lives; today's TIE proved it doesn't
   matter on top of v15, but it might matter on top of the orbitfix
   chooser.
2. **Pivot to chooser-side on the v15 foundation** per the 5/19 audit
   (K-horizon sweep, opp-model strength, proposer dedup). This is the
   "fix the chooser, not the leaf" path. Has known ceiling around
   v15+ε; will not catch the sibling.

Option 1 is higher EV. Option 2 is what the 5/19 audit explicitly
listed and is mechanically simpler (no merge).

## Action

Next-session first-action ranking should put "investigate sibling's
orbitfix chooser, then merge or pivot" above any new value-head /
chooser-side variant on this branch's foundation. The branch is
working off a foundation 46 μ below where the team's best work
already lives.

## How this got missed

The 5/19 → 5/23 work cycle on this branch operated on v15 as if v15
were the team's best. It was — until 5/22 when sibling shipped
orbitfix at μ=1165. The fresh state in `state/current.md` lists v15
as `current_submitted_agent` (true for this branch's last submit) but
doesn't surface `team_best` separately. Result: this branch kept
optimising against the wrong baseline for ~4 days.

Should also surface as a friction tag if it recurs.
