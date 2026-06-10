# 2026-06-10 (evening) — decision rules of the winners, and the fast-feedback unlock

## The PI's hypothesis, confirmed harder than stated

PI: "if we do a really good job we should see results with few games and
within the first 250 steps." Mined every corpus we have (top-3 teams' 120
replays + our own 78 live 2-player games): the total-ship lead stops
changing hands at median step 30–54, 90th percentile ≤ 95 — in EVERY
corpus, including ours. The top teams' median game still runs the full 500
steps; they win on holding a material lead, not on elimination. So the lead
at step ~100 already names the winner in ≳90% of games.

Instrument built on this: `scripts/margin_ab.py` — seat-paired ship-share
lead at steps 40/80/120/250 per seed. A continuous margin per game carries
far more information than the win bit, and seat-pairing cancels the map
draw, so 8–16 games triage a mechanism that used to need 32. (The Rule 45
submit gate stays at n ≥ 32 binary wins — this is for iteration speed, not
for shipping.)

## What the winners actually do (new tool: mine_decision_rules.py)

Reconstructed every fleet's target across the corpora (a fleet row carries
no target — track the id to its vanish step, snap to the nearest planet):

- **Overkill is class-dependent.** Neutrals get ~1.3× their garrison —
  cheap, and front-loaded (67–89% of neutral grabs before step 60). Enemy
  planets get 2.6–4.6× at median, 7.5–10× at p75, with 60–89-ship median
  fleets. Our ratios are close but our absolute enemy strikes are small
  (40 median) and launched from farther (eta 7 vs their 4–5) — and since
  big fleets fly FASTER, close+massive compounds.
- **Defense at the top is deterrence, not heroics**: ~30% hold rate when
  actually attacked at the 2000 level; 8% deliberate evacuations. Our
  reinforcements hold 0.59 vs Jake/TonyK 0.74–0.85.

## Mechanisms now in the tree (all default OFF)

1. **Terminal production value** (`PRODUCER_PLUS_TERMINAL_PROD_VALUE=λ`) —
   the expansion fix. The flow scorer truncates a captured planet's payoff
   at the horizon, so neutral captures scored ~0 and never cleared the roi
   threshold (seed-7 probe: dozens of valid neutral candidates per opening
   turn at best-score 0..1 while the bank hit ~300). Now the scorer credits
   production owned at the horizon's final step for λ more steps. Probe
   with λ=12: 8 planets by step 31, 12 by 47, bank held at 100–200.
2. **Class-split overkill** (`PRODUCER_PLUS_OVERKILL_FACTOR_ENEMY`) —
   sizing menu per target class: 1.3× neutral / 4× enemy, straight from
   the mining table.
3. Composites: `mass_termval12_splitovk` is the full master-agent
   candidate (mass + expansion fix + strike sizing).

## Open questions

- Does the margin triage agree with the 32-game binary verdict on the same
  candidate? (First read: convoy40 pool vs its margin profile.)
- Terminal value in 4P (H=13 truncates worse) — measure on the FFA pool.
- Attack-distance discipline (eta 7 → 4-5): does it emerge from bigger
  fleets being faster, or does the shortlist need a reach change?
- Reinforcement hold rate 0.59 vs 0.74+: re-judge reinforce_deficit against
  the NEW incumbent with margins (its null was measured on the producer
  pool with the old yardstick).
