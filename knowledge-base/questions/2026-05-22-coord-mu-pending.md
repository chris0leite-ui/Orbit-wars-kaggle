# 2026-05-22 — open question: what will coord (sub 52927313) settle to?

The single fact that picks the next major direction. Will be answered
in ~12-24h by `kaggle competitions submissions orbit-wars`.

## Predicted range

1100-1250 μ based on 4W/2L vs orbitfix (μ=1174.2) over n=6 unswapped
local seeds. Wilson 95% [0.30, 0.90] — wide.

## Branch decisions

- **μ ≥ 1180**: Lagrangian IS the breakthrough. Next move: compound
  with opening + endgame phase bonuses (~1 day) + multi-turn portfolio
  planning (~5 days). Both lean on coord's substrate.

- **μ ∈ [1100, 1180]**: Competitive but not breakthrough. The structural
  ceiling extends to the Lagrangian variant. Biggest swing remains
  multi-turn portfolio planning (Mission Renaissance's intended fix,
  finally on the right substrate). Defer phase bonuses until that lands.

- **μ < 1100**: Lagrangian alone doesn't break the ceiling. Time for
  the structurally different approaches in the strategic menu:
  opening book + cluster classifier (EDA-confirmed 4 archetypes),
  shot validator MLP, or opponent modeling beyond `lite_greedy_policy`.

## Secondary questions

- **Does the H44 wait_N fix (Day 12) materially change the μ?**
  Source branch's A/Bs landed at parity vs orbitfix even with this
  fix included — so the answer is probably "no". But coord's 2
  losses in n=6 could partly be in-flight deaths the bypass missed.

- **Do coord's "extra captures" vs minimal (Gate 1 showed coord
  launches when minimal doesn't) translate to live wins or are they
  net-noise?** If extras = wins, the no-hard-cheap-reject design is
  validated. If net-noise, add a CHEAP_REJECT_THRESHOLD-style filter.
