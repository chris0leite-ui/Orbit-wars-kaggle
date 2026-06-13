# FLAG: a reject-only filter in front of the producer planner DELAYS, doesn't PREVENT

**Established:** 2026-06-13, shot-validator live probe (sub 53595717).
Full evidence: `audit/2026-06-12-shot-mlp-offline-counterfactual.md`.

The producer planner re-derives launches from scratch every turn and has
no memory that a wave was vetoed last turn. So any pass that only
*invalidates* candidate waves (shot-MLP veto, response veto in isolation,
etc.) does not remove the behaviour — the same wave is re-proposed next
turn until the board drifts enough to clear the gate, then it fires a few
turns later (often worse, against a moved target).

Live signature when this is happening: the targeted behaviour's rate does
NOT drop (here: low-P attack share stayed 33.4% vs 34.8% baseline) and the
launch count can even rise. μ moves only within noise.

**Consequence for design (Rule 40):** to actually change behaviour, a
mechanism must give the freed budget somewhere to GO (redirect to a better
target) or change what the planner PROPOSES (upstream value/target
reshaping), not just reject the planner's output. "Reject-only in front of
a re-proposing planner" is now a known dead pattern — the same lesson the
pre-producer mechanism-ledger learned about stacking scoring terms in
front of the K=10 rollout.

**Still potentially live:** shot-MLP veto COMPOSED WITH redirect
(`PRODUCER_PLUS_REDIRECT`) — re-aim the dropped ships at a target the
model likes. Unproven; referee blindness means it can only be tested with
a live submission slot.
