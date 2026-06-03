# Flag — VH-on-state-driven-K axis closed; 6 falsifications

**Date:** 2026-06-03
**Branch:** `claude/champion-ml-graft-majestic-storm`

The VH-on-state-driven-K integration axis is closed after 6 consecutive
falsifications (4 additive on the broken Jun-2 model + 1 additive on
the fresh Phase D3 model + 1 rerank on the fresh model). Reference
un-VH state-K = 62.5%; best VH result = 12.5%; rerank result = 6.2%.

Do NOT iterate further variants of "VH head predicts K-step ship-delta,
chooser adds or reranks." That pairing is now ruled out across both
broken and fresh models, both additive and rerank wirings, n=32.

A new VH-related axis is admissible only if the **target** differs
from K=10 ship-delta (see `knowledge-base/thoughts/2026-06-03-vh-axis-closed-target-mismatch.md`
for candidate targets that aren't redundant with the chooser's existing
PV-discounted scoring).

Persistence: this flag is durable knowledge until either
(a) someone finds a non-additive, non-rerank consumption surface, or
(b) a different target makes the existing VH-on-K10 pairing relevant
again. Until then, treat as Rule 37 closed.
