# v7.1–v7.6 stack iteration — session wrap (2026-05-12 PM)

> Branch: `claude/game-ai-lookahead-3ucqH`.
> Plan: `/root/.claude/plans/reflective-dazzling-flask.md`.
> Approval: "do it all carefully and test your improvements."
> Outcome: **v7.1 maximin FAIL, v7.5 combined FAIL, v7.6 bisect
> PENDING at session end. Best local candidate remains v7_0_drop_one
> (sha256 `bb7ab23a75bc5865`).** No new submission this session.

## What was built (committed `34420b2`, pushed)

Five ablation agents + v7.5 combined + v7.6 bisect, all bundled:

| Bundle | sha256 | Composition |
|---|---|---|
| `v7_0_drop_one.py` | `bb7ab23a75bc5865` | drop-one + fast_sim (no σ-equiv) |
| `v7_1_minimax.py`  | `006d48572fc937b1` | σ-equiv + symmetric + 2×N maximin |
| `v7_2_recapture.py`| `8a8a43f4395bfb55` | v7.1 + recapture |
| `v7_3_prodhead.py` | `0a2c4d247fa39489` | v7.2 + evaluate_value head |
| `v7_4_4p_aware.py` | `4d65d144b677912a` | v7.2 + 4P drop-one rollout |
| `v7_5_combined.py` | `dcff85263dabbf34` | drop-one + σ-equiv + recapture + 4P-aware (no maximin) |
| `v7_6_no_recapture.py` | (bundled, A/B in flight) | v7.5 minus recapture |

Library changes (committed):
- `lib/geometry.py::sym_hypot` — bit-symmetric hypot.
- `lib/planner.py::_tb` + `SCORE_ROUND=6` — σ-equiv tie-break.
- `lib/missions/snipe.py` — uses sym_hypot for src↔target distance.
- `lib/lookahead.py::score_joint_action(_symmetric)` — seat-flipped scorer.
- `lib/missions/recapture.py` — ported + calibrated
  (`RECAPTURE_SCORE_DENOM_MATCHES_SNIPE=1`, `RECAPTURE_TOPK_PER_TURN=5`).
- `lib/lookahead_planner.py::evaluate_value` — production-share head.
- `lib/v7_search.py`: added `score_candidate_symmetric`,
  `score_joint(_symmetric)`, `_drop_smallest`, `_opp_incumbent_action`,
  `choose_maximin`, `score_candidate_4p`, `choose_4p`, `choose_with_4p`,
  `choose_simple_2p`, `choose_simple_with_4p`. `value_fn` parameter
  plumbed through every scorer.
- `scripts/bundle_agent.py::DEFAULT_LIB_ORDER` += `missions/recapture`,
  `lookahead_planner`.

Test additions (16 new, all green):
- `tests/test_v7_1_sigma_equiv.py` — 11 tests for sym_hypot,
  planner σ-equiv invariance, score_joint_symmetric, drop_smallest,
  4P fallback.
- `tests/test_v7_4_4p.py` — 5 tests for 4P rollout machinery.

## A/B results

### v7.1 (σ-equiv + symmetric + 2×N maximin) vs v7_minimax + v7_0

```
v7_1_minimax vs v7_minimax: 15/24 = 62.5%  Wilson lo 42.7%  p95 1105ms  FAIL
v7_1_minimax vs v7_0       :  6/24 = 25.0%  Wilson lo 12.0%  p95  985ms  FAIL
v7_0          vs v7_minimax: 19/24 = 79.2%  Wilson lo 59.5%  p95  816ms  PASS
```

Wallclock: 1309 s. Artifact:
`audit/tournaments/20260512T145154Z.json`.

**Verdict:** maximin overlay regresses badly. Root cause: 2×N matrix
× symmetric scoring (2× per-cell cost) = 4× rollouts per turn →
700 ms watchdog truncates → maximin defaults to incumbent
(conservative pick) → -54pp regression vs v7_0.

### v7.5 (drop-one + σ-equiv + recapture + 4P-aware)

```
v7_5_combined vs v7_minimax: 14/24 = 58.3%  Wilson lo 38.8%  p95 831ms  FAIL
v7_5_combined vs v7_0       : 10/24 = 41.7%  Wilson lo 24.5%  p95 772ms  FAIL
```

Wallclock: 1141 s. Artifact: A/B run 2 JSON (after bundle fix).

**Verdict:** -8.3pp regression vs v7_0. In 2P A/B the 4P-aware path
doesn't activate, so regression is from σ-equiv (library-level) or
recapture (mission class).

### v7.6 (drop-one + σ-equiv + 4P-aware, recapture OFF)

**PENDING at session end.** Bisect: if v7.6 PASS vs v7_0, recapture
was the regression (even after the audit's #1 + #2 calibration
fixes — #3 "premature commit on infeasible recaptures" still
unaddressed). If FAIL, σ-equiv interacts regressively with our
incumbent.

## What was learned

1. **The maximin overlay's theoretical guarantee isn't worth its
   compute cost at K=10.** The Phase 2 audit predicted this: "Sim<K>
   measures predictive power not strategic discrimination." Symmetric
   scoring × 2×N matrix exhausts the 700 ms watchdog; conservative
   tie-break to row-0 (incumbent) eats the value.

2. **σ-equiv layer's +45 μ attribution (per the structural audit)
   doesn't transfer through choose_simple_2p when stacked with
   recapture.** Likely an interaction: recapture's missions appear in
   the incumbent's settle_plan, get σ-equiv tie-broken alongside
   snipe, but the calibrated recapture score still over-weighs
   close-to-tied snipe scores. Bisect (v7.6) will confirm.

3. **The original recapture revert audit's hypothesis #3
   ("premature commitment on infeasible recaptures") was not
   addressed by my calibration.** Score-scale fix + top-K cap
   covered #1 + #2 but not #3. v7.6 result will tell us if #3 is
   load-bearing.

4. **Bundle silent failure modes are insidious.** v7.5 A/B run 1
   returned 0/24 because `DEFAULT_LIB_ORDER` didn't include the new
   lib modules. NameError at runtime → kaggle_environments catches
   → empty action → loss by elimination. **No log.** Fixed; flagged
   in friction.md as a promotion candidate (bundle should pre-check
   imports).

## Best local candidate at session end

**v7_0_drop_one** (committed `2bed6b3` earlier; sha256
`bb7ab23a75bc5865`, 121 KB). Local A/B against the live μ=1063
v7_minimax bundle: 19/24 = 79.2% (Wilson lo 59.5%). Predicted live
μ if submitted: 1080–1100 (TrueSkill math).

**Live ladder state:**
- v7_minimax #52568317 — converged at μ=1063 (parallel branch).
- v3.5.1 #52565976 — PENDING (this branch's earlier push).
- Rolling-last-2 = [v3.5.1, v7_minimax].

**Submission decision (deferred):** PI hold on v7_0 maintained
pending v3.5.1's μ result. If v3.5.1 converges below 1063, v7_0
becomes the obvious submit. If above 1063 — also probably submit
(v7_0 likely lifts further).

## Next-session work

Ranked by expected lift / cost:

1. **v7.6 result.** Will land within ~20 min of session end. If
   PASS → bisect points at recapture; build v7.7 = σ-equiv + 4P
   only, ship. If FAIL → bisect points at σ-equiv interaction;
   build v7.7 = v7_0 + 4P-aware only, ship.

2. **Recapture hypothesis #3 fix.** Add feasibility check to
   `propose_recapture_missions`: require `model.ships_at(target,
   eta) < base_ships - 1` so we only commit fleets that will
   actually retake the planet. ~30 min.

3. **v7_0 + σ-equiv re-bundle.** Trivial: the v7_0 source agent
   already calls `lib.v7_search.choose(enumerator_mode="drop_one")`
   which goes through the now-σ-equiv-enabled `settle_plan`.
   Re-bundling produces "v7_0 with σ-equiv" automatically. A/B that
   vs the legacy v7_0 to isolate the σ-equiv contribution.

4. **Pure 4P drop-one overlay** on top of v7_0 (no recapture, no
   σ-equiv). Tests whether the 4P search alone lifts. Bypasses the
   2P regression.

5. **Depth-2 minimax with narrow beam** (deferred from this
   session's plan). Width-3 beam at K=5 leaf. Now that maximin at
   K=10 is known to blow budget, narrow + shallow is the next
   form to try.

## Submissions used today

1 (v3.5.1 #52565976, 05:20 UTC, PI-approved this morning).
4/5 daily slots remaining.

## Files committed this session (post-foundation)

See git log between `2bed6b3` (foundation final) and HEAD:
- `05df9f9` — v7.1-v7.5 stack build
- `575b5ac` — v7.5 pivot to choose_simple_with_4p
- `34420b2` — bundle fix (recapture + lookahead_planner inlined)
- `e0a4e14` — v7.6 bisect agent + bundle
- `<this-commit>` — wrap-up state/ISSUES/friction/audit
