# Synchronized arrivals / delayed launches — the architecture escalation

Date: 2026-06-11. Context: the referee-panel re-judgment confirmed the
mirror's verdicts (no hidden value in the shelved candidates) AND showed
the local referee pool saturates below our level. Conclusion adopted: live
slots are the only honest instrument against the real field; local
measurement = safety gating. The build budget therefore goes to the
qualitatively new capability with the most convergent external evidence:
synchronized multi-source arrivals ("the key winning strategy" of the 2010
Planet Wars champions; also the convergent lesson of our own week —
holding ships is undervalued, staggered waves die piecemeal to the 1:1
garrison trade).

## What was built (PRODUCER_PLUS_SYNC, default OFF)

1. The engine already had two-source coalitions but gated to pairs whose
   fleets NATURALLY arrive the same tick (eta_strict) — and the live stack
   doesn't even run that path (coalitions off). Sync pairs live in the
   multi-size path the live stack actually uses.
2. Pair candidates on targets neither source cracks alone, joint floor
   read at the LATER leg's arrival tick. The nearer leg is scored with the
   far leg's eta — EXACT under the flow scorer (arrival credit lands at
   ceil(eta)); the tick-0 source debit means held ships are priced as
   already gone, so the score is conservative, never flattering.
3. Chosen near legs are never launched now: post-veto they become memory
   holds (ships reserved from the planner budget, opening-searcher
   pattern), and fire on the LAST turn that still makes the shared arrival
   date, re-aimed fresh so orbit drift can't desynchronize them. Date
   missed by >1 tick → release; source lost/drained or target flipped →
   cancel; far partner vetoed → near leg dropped (its size only makes
   sense jointly).

## Why this shape and not alternatives

- Stateless emergence ("far fires under-floor, near catches up next turn")
  can't work: under-floor singles are invalid by construction, so the
  first move never happens.
- Scheduling by ARRIVAL DATE instead of by delay makes execution robust to
  rotation drift — the fresh aim each turn decides when "now" is the last
  makeable turn.
- Mirror/replan planning passes get no sync sink, so the opponent model
  never hallucinates holds and OFF stays byte-identical.

## First observations

In-process probe vs v7_0 (seeds 0/3/7): about one hold per game, executed
or cleanly released, all wins, max turn 116 ms. The Fix-A-style gate
(neither source clears alone) is intentionally narrow — this is the
capture channel for targets the incumbent literally cannot take, so even
a low fire rate can matter. The same-tick (no-hold) pairs also ride along
as plain coalitions in the live path, which had none before.

## Open questions

- Should pair sizes shrink to floor+overkill instead of full safe_drain?
  (Full drain = more committed capital, the week's known disease.)
- Three-source sync? The greedy/budget machinery supports L>2 in
  principle, but evidence first.
- While a hold is pending the projection doesn't know the future launch —
  the planner can double-target the same planet. Rare at one hold/game;
  revisit if probes show waste.
