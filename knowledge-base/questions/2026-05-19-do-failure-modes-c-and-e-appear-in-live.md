# Open question: do failure modes (c) and (e) actually appear in live champion replays?

**Date raised**: 2026-05-19
**Branch**: `claude/ml-competition-strategy-PFhzM`
**Resolves by**: Phase 1a of the approved next-session plan
(replay-mining)

**STATUS: CLOSED 2026-05-19 AM** by replay-mine across 5 most recent
COMPLETE submissions (56,842 fleets, 501 episodes). Full audit at
`audit/2026-05-19-replay-mine-pre-roi.md`. Summary below.

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

## Resolution (5/19 AM)

Replay-mine ran across 5 most recent COMPLETE subs including
composite+A2 hybrid (sub 52744856, live peak μ≈1149.2). Per-failure
findings:

- **(a) recapture-loss**: YES, 0.97% prevalence (`arrived_but_lost`
  = 554 / 56,842 fleets across 5 subs).
- **(b) drift / sun-clip**: YES, 1.1% prevalence (`waste_trajectory`
  bucket: 599 fleets; comprises 453 oob + 94 vanished + 52 sun).
- **(c) garrison-counter**: YES, **13.9% prevalence** — biggest
  detectable failure mode (`bounced_enemy` 7913 fleets across 5
  subs; per-sub variance 13.7-17.6%). Bucket conflates true (c)
  with simple garrison-undershoot; both belong in G1 scenario.
- **(d) split-majority**: N/A from buckets (negative-space pattern;
  needs behavioural detector). INDIRECTLY corroborated by BPJKs'
  off-branch finding that we leave bigger garrisons at source
  universally (d=+0.82σ vs top-10).
- **(e) distant-planet idleness**: YES, **corroborated by BPJKs
  off-branch behavioural data** —
  `launches_per_turn` d=+1.26σ (we launch ~half as often as top-10)
  and `mean_garrison_at_launch` d=-0.82σ (we send a smaller
  fraction when we do) are universal deltas across archetypes.
  Quantified evidence of (e) at the team level. Priority-1 in V0
  scenario suite.

Priority-revised scenario order (deviates from plan's equal
priority): DI1 (e) > G1 (c) > SM1 (d) > R1 (a) > D1 (b) > sanity
S1-S3. Justified in
`audit/2026-05-19-replay-mine-pre-roi.md::Scenario priority ranking`.
