# HANDOVER.md — next-session brief

_Refreshed 2026-06-03. Read this first (Rule 15). Also read `state/MULTI_BRANCH.md`
(live rolling pair / track registry / region-mvp open track) per Rule 44._

## This session's outcome — region score-term FALSIFIED; region family parked

Branch **`claude/region-mvp`**. Built and tested the lever the *previous*
handover named as the top path from parity to lift: **region value as an additive
term in the chooser's final (post-rollout) score** (`BASELINE_REGION_SCORE`,
default OFF, byte-identical). The region desirability is the bias hook's own
`factor − 1.0` (shared via an extracted `_region_factor`), added to each
candidate's score scaled to the turn's mean Δ — a near-equal-Δ tie-breaker, never
an override (only Δ>0 candidates are re-ranked → reach-frontier guardrail holds).

**Result: NULL — falsified.** 3-weight sweep vs the table-ON champion, n=32 each,
process-isolated `clean_ab`: **0.10 → 51.6% [0.348,0.680]; 0.20 → 40.6%
[0.255,0.577]; 0.40 → 53.1% [0.364,0.691].** All Wilson-lows far below the 0.55
gate. Off-is-identical proven (100-turn behavioral parity vs HEAD champion, 0
divergence). **No submission** (Rule 42/43).

**Mechanism (why no weight works):** gentle bonus = parity (the look-ahead score
gaps between real candidates dwarf the tie-breaker, so it rarely flips the pick);
moderate bonus = regression (now strong enough to override the look-ahead into
worse moves — reach-frontier in miniature); no sweet spot. **The rollout already
prices whatever the region heuristic was trying to say.** Region-as-a-signal
fails at BOTH the enumeration layer (last session's bias hook → parity) and the
scoring layer (this session → parity/regression). **The whole region family is
parked** (see mechanism-ledger + MULTI_BRANCH "Region/chunk-aware track — PARKED").

**Still-live side-finding (NOT falsified):** idle-source probe
(`scripts/probe_idle_sources.py`, 1922 rows) — the champion leaves **~90% of
eligible planets idle/turn** even in close mid-game. This **refutes the
"source-saturated" premise** that closed the joint-coordination axis on
2026-06-02 (that null was measured only in blowout wins). The idle capacity is
real; the region advance pass (one redeploy heuristic) was net-neutral, leaning
"mostly correct," but only one heuristic was tried.

## NEXT-SESSION PLAN

**Do NOT re-open the region family** — both placements (enumeration + scoring) are
falsified; bolting a hand-built spatial heuristic onto the rollout's inputs or
outputs doesn't add signal the forward-sim lacks.

**Where lift has to come from (the session's takeaway):** either make the
**rollout itself** see further / cheaper, or give it a signal it genuinely lacks —
**opponent intent / multi-turn coordination** — not another static board
heuristic. The idle-source finding is the live thread here: ~90% idle capacity +
the refuted "source-saturated" premise reopens **joint-coordination** on better
evidence than the null that closed it. Strongest next bet: re-examine the
coordination axis (the `BASELINE_JOINT_SYNC` team-up family) **with the table ON**,
sized against the idle-capacity evidence — but BOTE with the PI first (Rule 26).

**Other flag-flip candidates (table ON, untested in isolation):** opening MILP
`BASELINE_OPENING_MILP`; composite value head. Horizon-decay
(`BASELINE_HORIZON_DECAY`) was only ever tested stacked on region (→ parity);
isolated A/B is cheap if a slot is free, but low priority.

**Method reminders:** one lever at a time (Rule 37); A/B **bundle-vs-bundle with
HARD-SET headers** (`os.environ[k]=v`, not setdefault — that was the `clean_ab`
contamination root cause); never pipe a live A/B through `grep` (block-buffers →
blind; redirect raw, filter on read); champion control = identical config minus
the new flag; submit only on Wilson-lo ≥ 0.55 panel + n≥32 h2h + Rule 42 eviction
check.


**Deferred — reassess WITH THE PI after the three above (do NOT start this session):**
- **2-hop redeploy** (shuffle forward so a follow-up can capture) — reverted
  (`5ec6a0d`); needs rebuild from spec `727e1bf`.
- **Reach-frontier chooser** (from-scratch value chooser) — separate agent
  (`agents/reach_frontier/`), had a hold=0 bug; it *replaces* the chooser, so
  evaluate whole-agent after the fix, don't stack.
- Cheap re-checks: H41 pv_horizon floor (`9ebd311`), PV_ETA tuning.

**Do NOT re-open** (dead for non-table reasons): H44 "fleets die in flight" (false —
sun/OOB only), 4p-cushion (4/32), b5 reward-axis (0/32), flat expansion-credit
(targeted the hoarding loss mode we refuted; real loss mode is the expansion race).

Full provenance (commit hashes per feature, the two confound mechanisms, deferred
detail) is in the session plan, mirrored into git history of this file.
