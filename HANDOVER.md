# HANDOVER — morning brief (written 2026-06-12 ~06:30 UTC, end of night shift)

## Decision waiting on PI (Rule 42 sign-off)
Submit `producer_plus_vetorf4p_sync_garval_on.py` (390,518 B) as the next
live experiment? Targets the now-significant 4P bleed (26.5% vs 53.8%
baseline) with the garrison-value mechanism. CAVEAT: the rolling pair is
53577315 (1258.3) + 53564198 (1286.8); a new submit evicts 53564198 — our
CURRENT BEST. Options: (a) accept the eviction, (b) burn 2 slots (garval +
re-anchor the best), (c) hold. Budget fresh: 5 slots today.

## Live state (06:00 UTC, n=78)
- 53577315 (coalitions): mu 1258.3 (peak 1317). 2P 65.9% — lift confirmed
  at scale. 4P 26.5% — significantly below baseline; the 1260-band 4P
  pool out-reinforces us (war-ledger law: whoever reinforces more wins).
- 53564198 (backstop best): 1286.8.

## The night's build: PRODUCER_PLUS_GARRISON_VALUE (default OFF)
Chooser-internal proactive-garrison credit from local balance of force
(threat from uncommitted enemy reserves vs garrison + production +
routable help), capped at one credited target per living rival. Variant
`vetorf4p_sync_garval` = full live stack + SOURCE_SAFETY=0.5 +
GARRISON_VALUE=12. Final battery (final bytes): 4P panel final share
36.4% vs 25% even, rank-1 14/32; 2P mirror 7/12, +20.2%@250 paired.
Panel seeds 6-7 wipes are the standoff-mirror artifact (control: pure
incumbent 4-mirror plays 25/25/25/25 for 250 steps; deviator gets
carved) — not extrapolable to the field. 22 unit tests; smoke max 157 ms.

## Other night threads (all in audit/2026-06-11-night-shift-garrison-value.md)
- 4P leader-avoidance: fully diagnosed = wave-size feasibility vs leader
  floors (valuation + reach eliminated). Coalitions are the existing bet.
- Defense vs counter-attack (PI observation): we already counter-attack
  65% of inbound threats; defended exchanges perform best (+120 mean) but
  selection-biased. No untapped counter-attack alpha found in our replays.
- Economy-credit thread CLOSED (3 refutations); superseded by garrison
  value, which addresses the same holdability defect from the defense side.

## Tools added this session
- scripts/decision_trace.py usage on live replays (mining workflow:
  fleet-vanish target attribution; owner sampled at decision step t0-1;
  endgame doomstack games are a contaminating specimen class).
- PPNSX namespacing recipe: sed PRODUCER_PLUS_ -> PPNSX_ on a built
  bundle => submissions/_ns_vetorf4p_sync.py (gitignored, rebuild).

## First actions next session
1. Rule 32 fetch; rebuild bundles (vetorf4p_sync_garval, vetorf4p_sync,
   _ns_ copy via sed).
2. Get PI verdict on the garval submit + eviction question.
3. Re-pull 53577315 episodes; check the coalition watch item (leader
   strikes when behind) on the larger 4P sample.
