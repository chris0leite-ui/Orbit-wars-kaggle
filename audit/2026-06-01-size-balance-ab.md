# SIZE_BALANCE (A+D) — Phase 2 triage A/B + single-game trace (2026-06-01)

**Branch:** `claude/champion-strategy-rules-00JzI`
**Build:** commit `f4564d9` (default-OFF `BASELINE_SIZE_BALANCE`; solo-path
unified arrival-correct capture floor (D) + source-keep clamp (A) in
`proposer.enumerate_ship_counts`).
**Status:** TRIAGE only (Rule 45 — n=16 is never a submit gate). Evidence
for whether to proceed to n≥32 + panel. NO submission.

## Win-rate A/B (two-arm isolation, mirrors `_run_hold_ab.sh`)

Same focal (live branch baseline, carries the default-OFF fix) run twice
vs the frozen champion bundle (`baseline_launch_rules_universal` — immune,
its inlined proposer predates the flag). Arms differ ONLY in
`BASELINE_SIZE_BALANCE`, so the delta isolates the fix and cancels any
focal-vs-champion branch drift. Script: `scripts/_run_size_balance_ab.sh 8`.

| Arm | Focal wins | Win % | Wilson 95% | elapsed |
|---|---|---|---|---|
| A — fix OFF (control) | 7/16 | 43.8% | [0.231, 0.668] | 1278s |
| B — fix ON (A+D) | 12/16 | **75.0%** | [0.505, 0.898] | 975s |

**Delta = +31.2 pp.** Fix-ON lower bound 0.505 clears the 50% parity line
but is below the 0.55 high-confidence triage-proceed bar (Rule 45); the
two-arm delta is the stronger read and is decisively positive at this n.
Control at 43.8% (not 50%) suggests the live branch is marginally behind
the May-30 frozen bundle, or n=16 noise — the delta controls for it.

## Single-game behavioral trace (`scripts/_size_balance_single_game.py 42`)

One full game (fix ON) vs cheap v7_0 (WIN, 460 steps); every launch
decision point re-examined OFF vs ON:

| Behavior | Count | Detail |
|---|---|---|
| SUPPRESS (doomed solo launch dropped, D) | 17,073 (66%) | one source can't win at arrival |
| UPSIZE (lean column raised, D) | 6,067 (24%) | avg +1.1 ships (arrival-correct + margin) |
| CLAMP (full-send capped, A) | 96 (0.4%) | avg −76.6 ships (source under threat) |

**Hypothesis the trace raised:** the 66% suppression also drops the
partial-budget columns the multi-source BUNDLE solver uses (the budget
column is emitted precisely so several sources can gang up on a big
target — `enumerate_ship_counts` docstring / `test_proposer_bundling.py`).
Conflating "this one source can't solo-capture" with "doomed" could
disable team-ups.

**A/B refutes the harm reading:** despite 66% suppression, fix-ON won
+31 pp more. Net effect of pruning doomed solo launches > cost of any lost
team-ups, at least vs this opponent in 2P+4P-pooled seeds. UPSIZE (D) and
CLAMP (A) behave exactly as designed.

## Open questions before any submission decision
1. **Confirm at n≥32** (Rule 45) — n=16 lower bound is only 0.505.
2. **3-opponent panel** (Rule 43a) + 2P/4P split (Rule 41/48) — the
   team-up concern is most likely to bite vs bundling opponents and in 4P.
3. **No-suppression variant worth testing:** keep the bundle column (only
   ADD the arrival-correct lean column + CLAMP the full-send, drop the
   blanket `cap_arr > max_sendable → []` early-return). If team-ups carry
   value, this could lift further than the current build. The trace says
   the suppression is the highest-leverage knob to ablate.
4. **Failure-mix diagnostic re-run** on variant replays — confirm the
   lost-episode D/A shares actually drop (the plan's evidence deliverable).

## Rule-44 note
Distinct axis from the closed JOINT_SYNC size-to-hold lever: solo path,
unified A+D, arrival-garrison floor (not the nulled pessimistic
counter-recapture model). This A/B does not touch `BASELINE_JOINT_SYNC*`.

**Artifacts:** `/tmp/size_balance_ab.log` (transient; re-run via the
script). Scripts: `scripts/_run_size_balance_ab.sh`,
`scripts/_size_balance_single_game.py`.
