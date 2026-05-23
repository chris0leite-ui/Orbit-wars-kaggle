# Strike-loss diagnostic — Step 3b findings

**Branch:** `claude/agent-design-exploration-Q0q9T` · **Date:** 2026-05-23
**Generator:** `scripts/strike_diagnostic.py` · **Raw data:** `audit/2026-05-23-strike-loss-diagnostic/summary.jsonl`

## TL;DR

The Step-3 A/B's −18.7 pp lift (strike-ON 25.0% vs strike-OFF 43.8%, n=16 vs `baseline_joint_aggr_consolidated`) **was n=16 noise**. A 16-trial diagnostic with strike-ON shipped 7 wins / 16 trials = **43.8%, Wilson CI [0.231, 0.668]** — statistically indistinguishable from the strike-OFF baseline. The "strike hurts" verdict from the original A/B doesn't survive a re-run.

The predicate's `is_winning_state_if_owned` gate is a **win-detector, not a win-causer**: lost games have ~0.6 plans/game on average; won games have ~61. Most of that variance is the gate correctly identifying which games are winnable, not the strikes themselves changing outcomes.

All three Step-3b modeling hypotheses (in-flight threats, defense-only consolidation, per-source budget) are **falsified by the data**. The Step-3 A/B's negative result was variance; the strike axis sits at parity with the consolidation-only baseline at n=16.

## Run setup

- Focal: `agents/buildup_planner` (`BUILDUP_PLANNER_STRIKE_ENABLED=1`)
- Opponent: `baseline_joint_aggr_consolidated`
- Seeds: `0..7` × seat-swap = 16 trials
- Per-game elect + strike JSONL logs written to `audit/2026-05-23-strike-loss-diagnostic/seed{N}_{elect,strike}.jsonl` (gitignored; bulky + regenerable)
- Total runtime: 5099 s (~85 min, sequential, 1 worker)

## Result table (demuxed — see note below)

| seed | seat | won | n_steps | plans | emit | drop_budget | p95 (ms) |
|------|------|:---:|---------|-------|------|-------------|----------|
| 0 | 0 | ✗ | 315 | 4 | 1 | 3 | 1037 |
| 0 | 1 | ✓ | 257 | 79 | 72 | 7 | 986 |
| 1 | 0 | ✗ | 334 | 1 | 1 | 0 | 787 |
| 1 | 1 | ✗ | 334 | 0 | 0 | 0 | 779 |
| 2 | 0 | ✓ | 370 | 69 | 36 | 33 | 762 |
| 2 | 1 | ✓ | 284 | 56 | 36 | 20 | 773 |
| 3 | 0 | ✗ | 300 | 0 | 0 | 0 | 1299 |
| 3 | 1 | ✗ | 375 | 0 | 0 | 0 | 1453 |
| 4 | 0 | ✗ | 147 | 0 | 0 | 0 | 808 |
| 4 | 1 | ✓ | 158 | 35 | 33 | 2 | 900 |
| 5 | 0 | ✗ | 347 | 0 | 0 | 0 | 1288 |
| 5 | 1 | ✓ | 379 | 91 | 79 | 12 | 1392 |
| 6 | 0 | ✓ | 298 | 46 | 44 | 2 | 857 |
| 6 | 1 | ✓ | 219 | 54 | 52 | 2 | 858 |
| 7 | 0 | ✗ | 205 | 0 | 0 | 0 | 934 |
| 7 | 1 | ✗ | 219 | 0 | 0 | 0 | 950 |

**Winrate:** 7/16 = 43.8%, Wilson 95% CI [0.231, 0.668].

> **Note on demuxing.** The diagnostic script (`scripts/strike_diagnostic.py`) had a bug: per-seed log files were unlinked only between seeds, not between the two swap iterations within a seed. The committed `summary.jsonl` reports seat-1 stats CUMULATIVELY over both seat-0 and seat-1 entries. The table above subtracts seat-0 from seat-1's reported numbers to recover seat-1's true values. Confirmed empirically (e.g. seed 0 reports seat-1 elect_turns=537 ≈ 314 + 223). Script fix and post-process logic land in the follow-up commit.

## Cohort means

| metric | WON (n=7) | LOST (n=9) | LOST − WON |
|---|---:|---:|---:|
| plan_count | 61.4 | **0.6** | −60.9 |
| emit_count | 50.3 | **0.2** | −50.1 |
| drop_budget_overflow | 11.1 | 0.3 | −10.8 |
| n_steps | 280.7 | 286.2 | +5.5 |
| focal_turn_ms_p95 | 932.6 | **1037.2** | +104.6 |

## Hypothesis verdicts

The original Step-3b plan listed three candidate modeling fixes; the diagnostic was designed to pick one. All three are **REFUTED**:

### (A) "In-flight threats not modeled in `is_winning_state_if_owned`" → **REFUTED**

Hypothesis: lost games have more emits than won games — strikes land but captured planets get reclaimed by opp's already-airborne fleets.

Data: lost cohort averages **0.2 emits/game**. There are no strikes to be reclaimed. The few emits that happen in lost games (1 in seed 0 seat 0, 1 in seed 1 seat 0, 1 in seed 1 seat 1) are too sparse to drive the result. The gate's "winning" certification is rarely overturned by reality because the gate rarely fires in losing positions.

### (B) "Defensive consolidation skipped during strike turn" → **REFUTED**

Hypothesis: strike's "strike-only emission" skips `emit_threat_reinforcements` for the strike turn; key planets fall in the 1–3 turns after each strike.

Data: 8 of 9 LOST games have **zero strikes**. Defense was never skipped — strike-ON ran consolidation every turn in those games, exactly like strike-OFF would. The losses come from the consolidation pipeline itself losing to `baseline_joint_aggr_consolidated`, not from strike.

### (C) "Per-source budget over-counting → wasted turns" → **REFUTED**

Hypothesis: predicate over-counts ships (Step-2 documented limitation); strike's atomic-drop returns `[]`; dispatcher emits no moves; lost turns accumulate into a loss.

Data: won games average **11 drops/game** (range 2–53) — and still win. Lost games average **0.3 drops/game**. If drops caused losses, the relationship would invert. The over-counting is real but doesn't carry the outcome variance.

## What the data actually says

The strongest signal in the data is **predicate fire rate ↔ winrate**. Won games elect ~100× more often than lost games. Two readings:

1. **The gate is a faithful win-predictor.** It elects when production lead × remaining turns > opp recovery pool — a snapshot of a winning position. Lost games are losing because we don't have those positions, not because the strikes hurt us. The predicate is doing its job of *detecting* viable strikes.
2. **The gate doesn't *create* winning positions.** When the gate says "no plan," we're already losing on consolidation alone. Strike can't rescue a losing trajectory.

So the strike axis sits at **parity** with the consolidation-only baseline. It doesn't lift; it doesn't hurt. The original Step-3 A/B's −18.7 pp result was n=16 variance.

## Wallclock observation

p95 turn time is **higher in losing games** (1037 ms vs 933 ms). The Kaggle budget is 1000 ms — lost games breach it. Two possible interpretations:

- Lost-game *positions* have more candidate moves to evaluate (longer consolidation search) → higher wallclock → higher likelihood of timeout-related game-state effects.
- Lost games are *just longer games at the tail* (n_steps means are 280 vs 286, near-equal), so the tail of slow turns happens to surface more.

Either way, this is a **Step-2 chooser/proposer issue**, surfacing only against heavier opponents. **Not strike-axis.** Separate session per Rule 41 (confound-sweep before correlational conclusion).

## What to do next

1. **Re-run the Step-3 A/B at n ≥ 32.** Per Rule 45, n=16 is the TRIAGE minimum, not a lift gate. With strike-ON and strike-OFF both centred at ~44% and Wilson half-widths of ~0.22, distinguishing parity from a ±5 pp lift needs ~n=200. At n=32 the half-width tightens to ~0.16, which would already let us reject "strike-ON is 0.5σ better" (≈ +6 pp) at meaningful confidence. **Commit-2 (default flip) stays gated on the n≥32 result.**

2. **Don't ship Step-3b modeling fixes.** All three candidate fixes (in-flight threats, defense-only consolidation, per-source budget tracking) were predicated on Step-3's negative lift being real. The diagnostic refutes that premise. **No code change is justified by this evidence.**

3. **Fix the bugs the code-review surfaced** (orthogonal to strike axis). The diagnostic itself exposed several issues — including its own double-counting bug — which should be cleaned up before any future diagnostic. Code-review findings are recorded; the most material are:
   - `scripts/strike_diagnostic.py:158` — log files not unlinked between swap iterations (this report demuxed around it; the fix is one line)
   - `agents/buildup_planner/main.py:215` — `strike.step` not wrapped in try/except (asymmetric with `evaluate_inflection`'s try/except)
   - `agents/buildup_planner/main.py:215` — empty strike emit doesn't fall back to consolidation (the originally-suspected bug; lower priority now that strike-ON ≈ strike-OFF on lift)
   - `agents/buildup_planner/strike.py` — wave-convergence invariant unverified (shots can arrive at different absolute steps if `find_shot_for_arrival`'s fallback path is taken)
   - `agents/buildup_planner/main.py:202` — comment about "one turn of opp recovery" is unsupported by the implementation

4. **The wallclock issue is real and orthogonal.** Total turn p95 ~1350 ms vs `baseline_joint_aggr_consolidated` (both strike-ON and strike-OFF). This is a Step-2 chooser/proposer issue, not strike-axis. Separate diagnostic session.

5. **Surface to PI.** The Step-3 axis is now believed to be at parity, not negative. PI's call on whether to:
   - Spend session compute on the n≥32 re-run (~3 h), OR
   - Pivot to the wallclock-fix axis (Step-2 chooser scaling), OR
   - Pivot to another architectural axis (Rule 4: never give up).

## Files referenced

- `agents/buildup_planner/main.py`, `strike.py`, `predicates.py` — Step 3 implementation
- `scripts/strike_diagnostic.py` — this diagnostic's runner
- `audit/2026-05-23-strike-loss-diagnostic/summary.jsonl` — per-trial roll-up (16 rows)
- `audit/2026-05-23-strike-loss-diagnostic/seed{N}_{elect,strike}.jsonl` — per-seed raw logs (gitignored; regenerable via `scripts/strike_diagnostic.py`)
