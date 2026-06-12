# Night shift 2026-06-11→12 — live reads, war ledgers, garrison value

## Live experiment read (sub 53577315, coalition strikes), ~3h in
- mu 1317.4 — above backstop 53564198 (1281.6) and the old best (1244).
- 2P 15/18 = 83% (baseline 50%, n=34) — statistically meaningful jump;
  the coalition capability converts the 2P stalemates.
- 4P 9/24 = 38% (baseline 53.8%) — NOT significant (z≈1.2) given the
  stronger pool at 1317; no regression claim. Coalition implication
  cleared directly: 4 of the 4P losses had ZERO coalition strikes; wins
  used 1-3 modestly.
- Watch item from the leader-feasibility diagnosis stands for the
  morning read.

## The 4P loss mechanism, named from live war ledgers
Dominant loss shape: we PEAK at 35-45% material mid-game, then get
carved to elimination. Allocation during the carve is sane (44-74% of
ships at the aggressor). The deficit is exchange EFFICIENCY, and its
core is reinforcement mass:
- Blu3s siege (ep 79566423): their reinforcement 15,868 ships vs our
  942 (17x). Their waves 42/42 captures (overkill margin ~265); ours
  31/39 with 904 ships donated to 295-garrison walls.
- Across all 23 4P games: wins = we out-reinforce the top rival 58%/33%;
  losses = they out-reinforce us 61%/46%. Matches the top-ladder mining
  (1600+ agents: most launches are internal circulation).
- Engine cause: the flow scorer values reinforcement ONLY against
  in-flight waves (too late vs avalanches). Nothing prices proactive
  garrisoning against UNCOMMITTED reserves.

## Mechanism built: PRODUCER_PLUS_GARRISON_VALUE (default OFF)
Chooser-internal (per the three-falsifications friction note): an
own-target send earns lambda_g * prod_t when the planet's local balance
vs the enemy's uncommitted reserve (same balance-of-force model as
SOURCE_SAFETY: threat margin vs garrison + production + routable help)
is negative at/after arrival AND the send covers the deficit.
Iteration within the night:
1. v1 (every deficit credited): 4P panel mean final share 42.4%
   (even 25%), but turtle-wipe on seed 6 (36 reinforce vs 4 expand
   launches, stalled at 3 planets).
2. v2 R-cap (enemy reserve = one resource per rival; credit at most R =
   living-rival targets, ranked by what the enemy would take):
   modeling-correct fix for the over-insurance.
3. Step-gate 50 tried, changed nothing on the wipe seeds — knob kept in
   the engine, default 0, NOT in the variant.
4. CONTROL run explained the remaining seed 6-7 wipes: the incumbent
   4-way mirror plays a PERFECT 25/25/25/25 standoff there for 250
   steps. Any deviator (garrison value, or anything) gets carved by
   three copies of our own reactive stack — the known mirror-punishes-
   divergence artifact, NOT a field prediction. On the 6 non-standoff
   seeds the mechanism dominates (seeds 2-3: 8/8 rank-1 at 94-100%).

## Final measurement (final bytes: lambda=12, R-cap, no step gate)
4P panel n=32 (8 seeds x 4 seats, focal vs 3x live-sub copies):
final share 36.4% (even 25%), rank-1 14/32 (even 8/32), mean rank 2.03,
12/32 truncation-eliminations (8 of them the two standoff-mirror seeds).
2P mirror: 7/12, paired lead +2.6%@80 / +11.7%@120 / +20.2%@250 —
parity-or-better, the live 2P edge is protected.

## Morning ladder read (06:00 UTC 2026-06-12, n=78)
Sub 53577315: mu 1258.3 (peaked 1317 at ~3h, sagged as the pool
strengthened). Overall 38/78 = 48.7%. 2P 29/44 = 65.9% — the coalition
lift holds at scale. 4P 9/34 = 26.5% vs baseline 53.8% (n=26): NOW
significant (z ~ -2.2, unlike the n=24 read). The 4P bleed is real at
the 1260+ band and is exactly the reinforcement-mass war the garrison
value mechanism targets. Eviction picture for a garval submit: rolling
pair is 53577315 (1258.3) + 53564198 (1286.8); a new submit evicts
53564198 = our current best — Rule 42 PI sign-off mandatory.

## Tests / safety
22 unit tests green (hold value 8, source safety 7, garrison value 7);
smoke seed 7 max 141-157 ms << 1000.
