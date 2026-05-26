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

## 2026-05-23 23:30 UTC — cross-branch verification: Vjaz9's v8 endgame-elim bonus

Tested `submissions/orbitfix_kt_p23.py` from
`origin/claude/extract-physics-trajectory-Vjaz9` commit `02cbcb8` (the v8
endgame-elimination bonus, shipped ~30 min before this test). Extracted
via `git show ... > /tmp/orbitfix_kt_p23_v8.py`; same gate harness.

### v8 vs `random` (n=16)

**Result: 16/16 wins, 6 ELIM, 10 WIN(score) — ❌ GATE FAIL** (elapsed 371s).

10/16 games went to step 500 with opp holding 1-6 planets. Much worse than
the plain orbitfix bundle's 16/16 ELIM vs random.

### v8 vs `nearest` (n=16)

**Result: 11/16 wins, 2 ELIM, 9 WIN(score), 5 LOSSES — ❌ GATE FAIL** (elapsed 302s).

v8 was **eliminated by nearest** in 5/16 games (seeds 65865, 15613, 31448,
14514, 12851). All ended with v8 owning 0 planets vs nearest's 28-36 — full
elimination, not score-loss.

### Summary table

| agent | vs random ELIM | vs nearest wins/ELIM/losses |
|---|---:|---:|
| `baseline_joint_aggr_consolidated_orbitfix` (sub 52912707, μ=1165.4) | **16/16** | 16W / 14E / 0L |
| `orbitfix_kt_p23` v8 (Vjaz9 commit 02cbcb8, not yet submitted) | **6/16** | 11W / 2E / **5L** |

## Implications

- `submissions/baseline_joint_aggr_consolidated_orbitfix.py` (sub 52912707,
  live μ=1165.4) **fails the candidate Rule 48 gate vs nearest** (14/16 ELIM)
  but passes vs random (16/16 ELIM).
- Vjaz9's v8 endgame-elimination bonus does NOT solve the elim-failure
  pattern on the kt_p23 lineage. v8 is dramatically weaker than the plain
  orbitfix bundle on both gates — and v8 actually loses 5/16 games to
  simple nearest-planet behaviour. The kt_p23 lineage carries an underlying
  weakness the endgame bonus can't compensate for.
- Direction signal for Vjaz9: the v7→v8 commit message says "addresses
  rung-1 6/16 ELIM and rung-2 3/16 ELIM" — my measurement shows v8 vs
  random is STILL 6/16 ELIM. The bonus didn't move the random gate. Their
  Phase-4 rollout horizon, leaf in-flight fate, or attack_pull head
  switch is the regressor; the endgame bonus is masking that more central
  problem.
- For my branch: the cleanest path forward is the session-EqJuT
  `68c24be`-style filter relaxation on `agents/baseline/proposer.py`,
  re-bundle, re-test nearest gate, then optionally re-submit. v8's
  approach is verified NOT to help.
- For the team: the kt_p23 lineage that ate 4 live submission slots today
  (52949672, 52959167, 52963659, 52965748 — μ 971-984) is structurally
  weaker than the orbitfix bundle that hit μ=1165 yesterday. Recommend
  pausing kt_p23 iteration and either rolling back to plain orbitfix or
  cherry-picking the endgame bonus onto orbitfix instead of running on
  the kt_p23 substrate.
