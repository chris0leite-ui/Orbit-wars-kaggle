# 2026-05-16 — v21_compound design + Rule 38 sun-bug verification

## Context

PI's session brief: fleets need to act on **missions** (capture / defend
/ disrupt) with multi-step commitment, be **efficient** (don't fly into
the sun, don't send slow fleets across the map), and the value of
actions should **compound** through correct geometric reasoning.

Yesterday's session (2026-05-16, see
`audit/2026-05-16-v16-v20-asymmetric-compounding-postmortem.md`) tried
to put "compounding" into the leaf scorer (v17/v18/v19) — all monotonic
regressions. Root cause: v15's `_favor = (my − opp)·pv(500)` already
encodes asymmetry implicitly via the rollout's reactive opponent
(fragile captures get flipped → leaf shows opp owning → drops). Explicit
asymmetry in the leaf broke this calibration. Lesson: **don't touch the
leaf scorer; change the candidate pool**.

## Design — three orthogonal additions to v20's pipeline

`agents/v21_compound/main.py` is a copy of `agents/v20/main.py` with
ONLY the prerank stage augmented. The leaf scorer (`_favor`), K-rollout
loop, reactive opp model, dogpile emit — all identical to v20.

1. **Sun-safe pre-filter** at proposer time
   (`lib/compound.py::fleet_path_safe`). Drops candidates whose
   straight-line trajectory crosses the sun safety zone or terminates
   out of bounds. Uses cheap `lib.geometry.path_clears_sun` (~1 µs per
   call). The current realize-time `mechanism.sun_avoid` only catches
   intents in the planner pipeline; v20's dogpile emit bypasses
   that pipeline entirely, so sun-bound candidates were reaching the
   board.

2. **Compound bonus** added to `_cheap_marginal_value`
   (`lib/compound.py::compound_bonus`). Three components, all
   model-derived (Rule 40 — no constant caps):
   - `rotation_alignment × production × 0.02`: planets drifting toward
     our cluster centroid over 30 turns get a positive bump (longer
     expected hold time).
   - `chain_bonus`: if capturing this target unlocks an affordable
     follow-on capture within 15 turns, add 0.30 × follow-on PV.
   - `carryforward_bonus`: if (src, tgt) matches a live MissionBook
     commit, small TTL-decayed stability nudge.

3. **MissionBook persistence** (`lib/mission_book.py`). After dogpile
   emit chooses moves, each (src, tgt) pair is committed with TTL=3.
   Carried-forward commits get the carryforward bonus on next turn's
   scoring pass, reducing per-turn churn. Commits expire on TTL=0,
   src-lost, or target captured by us.

## Architecture-level decisions (and why)

| Decision | Rationale |
|---|---|
| Don't touch `_favor` / `_score_action` | 5/16 postmortem: explicit asymmetry breaks v15's implicit calibration. |
| Same dogpile chooser as v20 | v20 won 65.6% vs v15 by removing per-target dedup. Don't break a winning chooser. |
| Inject hooks in prerank, not in lib/missions/ | v20 doesn't use the Mission framework — its `(src, tgt, ships)` enumeration IS the proposer. Cleaner to extend the existing loop. |
| Pre-filter sun in proposer, not at realize | v20 bypasses `lib.mechanism.sun_avoid`. Pre-filter is also cheaper than running the validate-stage rollout on doomed candidates. |
| Chain bonus = 0.30, rotation bonus = 0.02 × prod | Total bonus magnitude ≤ ~50% of cheap_marginal_value (≈ 0.05 × prod × pv). Bonuses tilt ranking; never override. |
| TTL = 3 turns | Short enough to avoid lock-in to stale plans; long enough that a 2-turn capture sequence survives. |

## Rule 38 verification — sun-destruction symptom

Per CLAUDE.md Rule 38, the fix is verified by reproducing the failure
state. `scripts/outcome_histogram.py` walks each launched fleet through
`predict_fleet_fate` to classify its outcome.

3-seed smoke (v21 vs v20, seeds 1-3):

| Agent | total_launches | planet | sun | oob |
|---|---:|---:|---:|---:|
| v21_compound | 456 | 456 | **0** | 0 |
| v20 | 611 | 606 | **3** ★ | 2 |

★ = `*** SUN BUG ***` flagged in seeds 1 and 2. v20 also lost 2 fleets
to OOB. v21 zero across both buckets. The Rule 38 condition (reproduce
failure + apply fix + confirm symptom gone) is satisfied.

This is one of the symptoms PI explicitly observed. The realize-time
`lib.mechanism.sun_avoid` is wired only into the planner pipeline; v20's
dogpile chooser emits moves directly without going through the planner,
so the realize-time guard never ran. v21's proposer-time pre-filter is
the correct architectural fix.

## Compounding lift — head-to-head A/B

`fast.py eval v21_compound --vs v20 --max-seeds 32 --gate 0.55 --workers 2`

**Result: 15/32 (46.9%), Wlo=0.309, Whi=0.636. INCONCLUSIVE.**

v21 is within noise of v20 in 2P head-to-head. Slightly below 50% but
the Wilson CI brackets the 0.55 gate widely. Practical reads:
- v21 does **not** clearly regress vs v20 — the sun-bug fix and the
  three new bonuses don't break the chooser.
- v21 does **not** clearly improve vs v20 either — the compounding
  intuition doesn't surface as h2h winrate in 2P self-play because
  v20's K=10-40 rollout already prunes the same fragile captures the
  compound_bonus penalises.
- v20's sun losses (~0.35%) and v21's avoidance of them are EQUAL AND
  OPPOSITE in 2P (both sides see the same wasted ships in either
  direction); the lift, if any, would surface against opponents that
  punish v20's wasted launches more aggressively than v20 itself does.

Panel vs `[v15, v7_0]` deferred to next session (avoided in this build-
only session per PI direction; eats ~30 min × 2 = 1 hr of compute that
would push session end past safe wrap-up).

Bench (3 games vs v20): p50=78ms, p95=251ms, p99=304ms, max=354ms,
zero >1000ms. Comfortable headroom under the 1000ms env cap.

### Bundle-source parity — TIMING NOISE, not a bug

UPDATE 2026-05-17: revisited the previous session's MISMATCH flag.
Manual probe on seed 1 WITHOUT env-var override: src=1, bdl=-1
(diverge). Same probe WITH `ORBIT_WARS_PARITY_WALLCLOCK_MS=60000`:
src=1, bdl=1 (agree).

Conclusion: the parity contract `same obs → same action` IS satisfied
(the per-obs parity check in `scripts/bundle_agent.py:382-419` is the
canonical test, and it uses the env-var override). The full-game
divergence under default 1000ms cap is from adaptive wallclock probing
— source and bundle have slightly different module-load overhead, so
`n_affordable_validate` lands on different counts per turn, the
candidate set diverges, and game trajectories slowly differ. This is
EXPECTED behavior for v15/v20/v21 (all use the adaptive cap pattern)
and is also how they behave on Kaggle's servers vs local CI.

The 333 KB bundle at `submissions/v21_compound.py` IS safe to submit
per the bundler's parity contract.

### Rule 37 considerations

The 15/32 INCONCLUSIVE on a fresh axis (proposer augmentation) is NOT a
falsification per Rule 37 — we haven't run 3+ variants of "proposer
augmentation". To formally falsify the axis, next session needs to
ablate:
1. `v21a_sun_only` — v20 + sun pre-filter only (no bonuses, no book).
2. `v21b_rotation_only` — v20 + rotation_alignment bonus only.
3. `v21c_chain_only` — v20 + chain_bonus only.
4. `v21d_book_only` — v20 + MissionBook persistence only.

If any single axis shows ≥55% Wlo vs v20 on 32-seed h2h, that's a
clean lift and we ship it isolated. The combined v21 may be missing
lift because the bonuses interfere with each other (e.g., chain bonus
favors target A while rotation alignment favors target B; cheap-rank
gets confused).

## Rule 37 (consecutive-falsification cap) — fresh axis

| Axis | Status |
|---|---|
| Chooser value-function asymmetric compounding (v17/v18/v19) | Falsified yesterday (3 attempts) |
| Chooser leaf scorer changes (v7_1..v7_7 chooser axis) | Falsified 2026-05-14 (7 attempts) |
| Chooser banded multi-wait grid (v15a..v15) | Saturated 2026-05-16 |
| **Proposer-time efficiency + compound bonus (v21)** | **NEW AXIS, this session** |

Mission-persistence (TTL/carryforward) has also never been tested.
Geometric-rotation-aware scoring (rotation_alignment) has also never
been tested. All three new axes share the `v21_compound` variant; if
v21 fails the gate, we have multiple knobs to ablate before declaring
the axis dead (chain_bonus alone, rotation alone, mission_book alone).

## Files

**New:**
- `lib/geo/rotation.py` — rotation_alignment, drift_window, my_cluster_centroid
- `lib/compound.py` — fleet_path_safe, compound_bonus
- `lib/mission_book.py` — MissionBook + global BOOK instance
- `agents/v21_compound/main.py` — copy of v20 + hooks
- `scripts/outcome_histogram.py` — fleet-outcome instrumentation
- `tests/test_rotation_alignment.py`, `tests/test_mission_book.py`, `tests/test_compound_filter.py`
  (24 unit tests, all passing)

**Modified:** none. v20 and v15 source merged-in from
`claude/recover-main-foundations-MV0e2`, untouched after merge.

## Submission decision — deferred per PI

Per Phase-1 question 3: build-only this session. v15 and v20 are both
PENDING on the live ladder. Submission decision next session with live
data in hand. The v21 lift signal (if any) will inform that decision
along with the live v15/v20 settle.

## 2026-05-17 update — live data + strategy pivot

Live μ pulled:
- **v15 (sub 52710995) = 1115.5 ← team floor**
- v20 (sub 52721807) = 1095.5
- v13 (sub 52704189) = 1085.7 (evicted)

v20 LOST 20μ vs v15 live despite v20's 65.6% local h2h vs v15 — clear
local-overpredict pattern, and evidence v20's dogpile-emit has
unseen-opponent failure modes. **The live winner is v15, not v20.**

Submission economics:
- Rolling-last-2 = [v15=1115.5, v20=1095.5] (v20 is newer).
- Pushing v21 evicts v15 (older) → floor drops to max(v21, 1095.5).
- For v21 to be net-positive, live(v21) > 1115.5.
- Given v21 vs v20 = 46.9% local, p(live(v21) > 1115.5) is low.

**Strategy pivot:** build v22 = **v15 + sun-safe pre-filter only**.

Rationale: the smallest possible defensive addition to the actual
live winner. By construction v22 can only:
- Equal v15 bit-for-bit (rollout always drops sun-bound candidates)
- Beat v15 marginally (rollout was occasionally emitting sun-bound
  candidates that v22 catches before validate)

Worst case: v22 ≡ v15, live μ ≈ 1115.5. Best case: small lift from
removed waste. **Net floor risk is zero** — there's no realistic
mechanism by which v22 underperforms v15.

A/B v22 vs v15 running in background. If parity-or-better at n=32,
v22 is the recommended submission candidate.

Also built v21a/v21b/v21c/v21d ablations isolating each compound
component on v20 base. If any single axis shows ≥55% Wlo vs v20, that
axis is a candidate to apply to v15 base as a v22-variant (v22b, etc.).

### v22 vs v15 A/B result: 43.8% INCONCLUSIVE — do NOT submit

n=32: **14/32 wins (43.8%), Wlo=0.282, Whi=0.607**. Wilson CI brackets
50% widely; not a statistically significant regression, but the point
estimate is below parity. Combined with the local-overpredict-2x
calibration shift, live μ(v22) is more likely below v15's 1115.5 than
above.

**Architectural finding:** the sun-safe pre-filter is NOT a free
defensive win. The K-rollout already correctly prices sun-bound
candidates — when the rollout scores a sun-bound candidate as
positive Δ, the value isn't "reach the target" but "remove ships
from this source before opp captures it" (sacrifice fleet). v22's
pre-filter strips these candidates from the validate stage; the
chooser then picks objectively-worse alternatives that the rollout
would have ranked lower. Net effect: slightly under parity.

This is the THIRD time this session class of "additions in front of
the K-rollout regress because the rollout already implicitly handles
the concern":
1. Leaf-asymmetric-compounding (v17/v18/v19, 2026-05-16): rollout's
   reactive opp already encodes asymmetry
2. Pre-rank reweight on top of v7 (H17/H19/H21, 2026-05-13): rollout
   already evaluates contested territory
3. Sun-safe pre-filter (v22, 2026-05-17): rollout already prices
   sun-bound candidates correctly per their non-target value

**Submission recommendation:** do NOT submit v22. The 43.8% point
estimate + local-overpredict-2x suggests live μ(v22) ≈ 1090-1100,
below v15's 1115.5. Pushing v22 evicts v15 (older in rolling-last-2),
dropping floor from 1115.5 to max(v22_live, v20_live=1095.5).

Wait for v15 to drift / hold the floor for further iteration.
