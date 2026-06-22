# 2026-06-22 — PI 4P observations: premature attack + uncoordinated reinforcement

> Append-only (Rule 35). Transcribed from PI during live-ladder 4P replay watching.
> Plain English. Three 4P games (rating deltas visible): seeds 219030400, 1920255508,
> 55472333 (ChrisLeiteScha placing 2nd/poorly).

## The observations (PI's words, lightly cleaned)

> In two games I observed the same pattern of **premature attack which leaves us
> exposed**. In another game, **planets sending fleets to support each other and
> exposing them by uncoordinated fleet sending** — we lose the big planets for a couple
> of rounds, which is fatal.

## Why this matters

Live-ladder split (read this session): 2P **71%** (dominant) but 4P **21%** (below the
25% fair share; the backstop is 30% vs a stronger field). 4P is ~half the ladder and the
entire μ drag. These two failure modes are 4P-specific.

## Diagnosis (code-grounded, two Explore passes)

Both are one disease: **we over-commit ships (to attack OR to reinforce) without keeping
each planet enough garrison to survive the rest of the 4-player field.**

1. **Premature attack → exposed.** Attack/contest sizing (decisive capture,
   main.py ~1805-1819) sizes vs the **max single** enemy, never the coalition; the greedy
   plan-builder uses the producer net-ship-delta scorer (no holdability) so the 2-ply leaf
   can only trim a bad plan, not repair it; and the only 4P-aware path (the robust
   ensemble) is bypassed when the native leaf is on. Net: we drain a planet to hit rival A
   and rival B/C punishes the thin source.
2. **Uncoordinated reinforcement → lose big planets.** `LR_DEFEND` (~1891-1932) drains
   donor planets nearest-first by raw availability with **no source protection** and no
   coordination; `LR_GARRISON_FLOOR` (the per-source reserve, ~1676-1724) is default-OFF
   and even when on uses single-enemy threat. So a big donor gets emptied and lost.

## The fix being built (this session)

A **4P-only, coalition-aware per-source garrison reserve** (`LR_RESERVE_4P`, default-OFF)
applied before candidate generation (caps `available`). Threat = strongest reachable enemy
+ w·(sum of the other reachable enemies); support credited only as each donor's SURPLUS
(ships − its own coalition threat), so big threatened donors aren't over-lent. One reserve
protects both attack sources and reinforcement donors (both spend from `available`). 2P is
untouched (it wins at 71%). Calibrated `w` to avoid the 2026-06-19 Goldilocks paralysis.
Verify by rendering the three PI seeds in 4P before/after + a 4P winrate sanity check.

## PI follow-up (same session) — reserve looks good, defense coordination is the gap

PI watched the reserve-ON 4P replay (seed 219030400): "looks good, only our defense is
not good enough. at the bottom we seem not to be coordinated well enough to survive red's
attack." So the reserve fixed the holding/over-extension (validated by eye), but DEFENSE
COORDINATION is the remaining 4P weakness.

Diagnosis of LR_DEFEND (main.py ~1929-1966):
- Donor draw uses the RESERVE-CAPPED `available` -> with LR_RESERVE_4P on, neighbours
  hoard their reserve and won't concentrate to save the attacked planet (the reserve, a
  per-planet hoard, is the OPPOSITE of coordinated defense).
- NO timing: it sends `take` from nearest donors without checking the reinforcement
  ARRIVES before the attack lands -> fleets arrive late, the planet flips anyway
  ("uncoordinated fleet sending").
- threat = sum of enemy FLEETS within a fixed 35u range -> ignores arrival time and
  enemy planets that can launch.

Fix (this session): a 4P coordinated + TIMED defense (gated with LR_RESERVE_4P). For a
planet under imminent attack: size defense to win AT the attacker's arrival (garrison +
production accrued + reinforcement that ARRIVES IN TIME); donors lend their surplus over
their OWN imminent threat (so we concentrate to save the attacked planet but don't strip a
donor that is itself about to be hit) -- bypassing the attack-reserve cap, because saving a
real planet outranks the speculative reserve. 2P path byte-identical.
