# Open question: do failure modes (c) and (e) actually appear in live champion replays?

**Date raised**: 2026-05-19
**Branch**: `claude/ml-competition-strategy-PFhzM`
**Resolves by**: Phase 1a of the approved next-session plan
(replay-mining)

## The question

PI's five named failure modes are:
- (a) Recapture-loss
- (b) Drift-loss / sun-blocks-raycast
- (c) Garrison-counter (neutral adjacent to strong enemy)
- (d) Split-majority coordination failure
- (e) Distant-planet idleness

(a), (b), (d) are well-supported by audit history:
- (a) was the v2 motivation (`WorldModel.owner_at` dedup)
- (b) drift was the `geo_drift` / `secure` line (falsified Wlo 0.18)
- (d) is the bundle-vs-baseline ceiling pattern

(c) and (e) come from PI's live game watching but I haven't seen them
in audit data. **Open: do they appear in composite+A2 hybrid / v15
replays, or are they bundle-specific failure modes that the live
agents don't have?**

## Why it matters

Scenario priority order depends on it. If (c)/(e) only show up in
bundle (not the live champion), they're lower-priority — the ROI
agent should match the live champion before trying to exceed it on
PI-mental-model failure modes.

If (c)/(e) appear in composite+A2 / v15 replays, they're true
ceiling-limiting bugs and should be highest-priority in V0 scenarios.

## Resolution path

Phase 1a of next-session plan: run `scripts/replay_mine.py --recent
5` on the most recent COMPLETE submissions. Document in
`audit/2026-05-19-replay-mine-pre-roi.md` whether each pattern
shows up in live data, with approximate prevalence (fleets-per-game
or % of total fleets).

## Closing rule

Question closes when the audit doc explicitly states for each of (a-e):
"observed in live composite+A2 replays: YES / NO / N/A (no replays
available)" with a count or % per case.
