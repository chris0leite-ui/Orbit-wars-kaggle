# 2026-05-22 — flag: rolling-pair has coord (PENDING) as the older half-eviction risk

## State after sub 52927313 submit

Rolling pair will be:
- **Newest:** sub 52927313 (coord, PENDING — μ unknown for 12-24h)
- **Older half:** sub 52912707 (orbitfix, μ=1174.2 — team champion)

The next submission would evict orbitfix (μ=1174.2). That's a HIGH-VALUE
agent to put at risk. Until sub 52927313 settles, **do NOT submit
anything else** — every new push displaces a known-good agent for an
unknown one.

## What happens when coord settles

| Settled μ | Resulting rolling pair | Eviction risk on next submit |
|---|---|---|
| ≥ 1180 | coord + orbitfix (both ≥1170) | Either eviction loses ≥1170 — high stakes |
| 1100-1180 | coord + orbitfix | Submitting evicts orbitfix (μ=1174.2) — high stakes |
| < 1100 | coord + orbitfix | Submitting evicts orbitfix; we'd want to evict coord but can't (it's newer) |

## H44-fixed coord (commit 4c3c440) waiting on the source branch

The Day 12 H44 wait_N admissibility patch is committed and bundled
locally. **Not submitted** because resubmitting would evict the
champion orbitfix. Source-branch evidence says the fix doesn't break
the chooser-axis ceiling, so resubmitting for the fix alone is
likely neutral. Wait for sub 52927313's μ before deciding.

## Persistent flag

This flag is "active" until sub 52927313 settles or is explicitly
evicted. Reading the flag at session-start is the trigger to check
the Kaggle leaderboard before any other decision.
