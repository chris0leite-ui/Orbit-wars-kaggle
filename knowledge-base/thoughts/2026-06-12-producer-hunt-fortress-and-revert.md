# 2026-06-12 — The Producer hunt: a day of mechanisms, an accidental fortress, and an honest revert

PI directive was the head-on hunt: beat the vanilla Producer with the
ledger via a concentration rebuild.

## What we learned that stays true

1. **The Producer wastes nothing.** 98-99% of its landed tonnage ends
   on planets it owns after the landing. Our agent: ~50%. That gap —
   not expansion economics, not opening speed — was the measured core
   of the matchup. (Opening sinks and capture counts were at parity;
   the old "cheap-garrison-first" story was wrong.)

2. **Combat being exactly 1:1 means waste is the whole war.** The
   differential moves only through production flow, neutral sinks, and
   landings that don't stick. Every mechanism that cut my
   non-sticking landings extended games (90-170 steps -> 150-177)
   without flipping them.

3. **The accidental fortress.** Under heavy CPU contention the agent's
   time budget truncated shopping to near zero mid-game — and THAT
   build beat the Producer 8/10: fat garrisons everywhere -> its
   stacks found no profitable landing -> 220-tick freeze at score
   parity -> our stalemate-gambit endgame dismantled the frozen
   Producer 27-planets-to-1. The Producer has no stalemate-breaking
   logic. The win condition exists; encoding it deterministically
   failed six different ways (gates throttle openings or leak).

4. **Load-dependence is a real and dangerous property**: wall-clock
   budgets make agent strength depend on machine load. Batteries above
   ~3 workers measure a different agent.

5. **Every gate that throttles "dangerous" spending also throttles
   expansion somewhere** — and under-expansion loses to EVERYONE on
   score, not just the Producer (clean: v7_0 5/12, bundle 1/8 for the
   14-mechanism build). Response-pricing pessimism is a poison with a
   narrow therapeutic window; the only honest dosing is per-mechanism
   bisection with full panels.

## Where the matchup stands

Vanilla Producer vs ledger remains 0-for-everything at honest compute.
The path that actually won (fortress-freeze-gambit) is real but needs
the freeze to start from expansion parity, and parity-then-freeze has
not been reproduced deliberately. Next concrete step: bisect from
c42c9fc one mechanism at a time against a fixed 3-opponent panel, and
consider the opponent-adaptive response propensity design (exact
landing attribution) before any new gates.

## Status of the agent

agents/ledger/main.py restored to the ledger_v1_4 state (the proven
build: v7_0 ~70-75%, bundle ~75-85%, 4P leader objective 9/16,
Producer 0/16). The session's diagnostic tooling and the full mechanism
graveyard are preserved in git history and the audit.

---

## CORRECTION APPENDED SAME DAY (append-only folder — original above
## kept for the record)

The fortress narrative in this entry was disproven hours after it was
written. The 8/10 was the Producer timing out under 9-way battery CPU
contention (torch thread thrashing past the engine's 1 s act
timeout) — its 220-tick "deterred passivity" was no-op turns, and a
deliberate defense-only-from-t40 experiment at full compute showed the
real Producer eats fortress boards faster, not at all deterred (3/3).
Points 1, 2, 4 of "what we learned" stand; point 3 (the accidental
fortress) is an artifact. Full proof in
audit/2026-06-12-concentration-rebuild.md (CORRECTION header).
