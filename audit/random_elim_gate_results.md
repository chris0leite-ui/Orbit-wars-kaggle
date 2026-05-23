# Random-elim gate results — baseline_joint_aggr_consolidated_orbitfix

## 2026-05-23 22:50 UTC — vs `random` (n=16)

Command: `python scripts/random_elim_gate.py submissions/baseline_joint_aggr_consolidated_orbitfix.py --n 16 --workers 4 --opp random`

**Result: 16/16 ELIM, GATE PASS** (elapsed 163s).

Range of outcomes:
- Fastest elim: seed=89048 in 81 steps / 6.5s wall
- Slowest elim: seed=97775 in 266 steps / 62.4s wall
- Median game: ~120 steps, ~25s wall
- My planet count at elim: 15-36 (board has 40 total; most games elim before full board owned)

## 2026-05-23 22:55 UTC — vs `nearest` (n=16)

Command: `python scripts/random_elim_gate.py submissions/baseline_joint_aggr_consolidated_orbitfix.py --n 16 --workers 4 --opp agents/simple/nearest.py`

**Result: 16/16 wins, 14 ELIM, 2 WIN(score) — ❌ GATE FAIL** (elapsed 324s).

Two failures both hit the 500-step episode cap with one stubborn enemy planet:

| seed | seat | my_planets | opp_planets | steps |
|---|---|---:|---:|---:|
| 98438 | P1 | 27 | 1 | 500 |
| 31448 | P0 | **35** | **1** | 500 |

35 own vs 1 opp at step 500 = the "midgame filter over-rejects in dominant
endgame" failure mode documented on `claude/session-EqJuT` (friction tag
`midgame-filter-overrejects-in-dominant-endgame`, fix `68c24be` patched
only `agents/lagrange_simple` — `agents/baseline/proposer.py` still
carries the bug). B1 hold filter refuses to attack opp's last planet
because the (own!) neighbouring planets are flagged as potential
counters.

## Implications

- `submissions/baseline_joint_aggr_consolidated_orbitfix.py` (sub 52912707,
  live μ=1165.4) **fails the candidate Rule 48 gate** as the bundle stands.
- Bug doesn't surface on the ladder because real opponents don't reach
  the "1 stubborn planet" endgame state — it's a substrate-correctness
  failure, not a ladder-competitiveness failure.
- Fix is small: cherry-pick the endgame-filter relaxation from
  `claude/session-EqJuT` commit `68c24be` onto `agents/baseline/proposer.py`
  and re-bundle. Should not affect ladder μ (the relaxation activates
  only in dominant endgame, which competitive games rarely reach).
