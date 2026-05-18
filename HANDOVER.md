# HANDOVER.md — next-session brief

> Last written: 2026-05-18 (late session wrap-up) by
> `claude/audit-workflow-performance-btjeK`. **15 bugs catalogued,
> 1 critical bug (#15) discovered via synthetic oracle testing.**
> Next session: fix #15 first — it's likely the highest-leverage
> single change available.

## Live state

| Submission | μ (settled) | Status | Role |
|---|---:|---|---|
| 52766596 | **1094.1** | Active rolling pair | joint v3 (2P-only-gate) — UNDERPERFORMED |
| 52754310 | 1141.0 | Active rolling pair | trajectory champion |
| 52744856 | 1149.2 | Evicted | composite_a2 (was the floor; lost when we pushed 52766596) |

**Net for the session**: floor dropped from 1148 → 1094 (-54 μ). The
joint v3 submission underperformed — its capture candidates don't
register positive Δ in the leaf (bug #15 explains why).

Daily submission budget: 5/18 used 1 (52766596). 4 unused.

## The headline insight

**Bug #15** (discovered late-session): `composite_capture_value`
checks `pred_owner == my_id` at fleet arrival eta — but the
WorldModel's prediction INCLUDES this fleet's arrival, so the
check returns "already ours" precisely BECAUSE OF this fleet.
The composite then skips capture-bonus credit. Result: every
capture the agent makes gets ZERO leaf credit. Captures appear
as pure ship-loss in Δ scoring → emit gate (Δ > 0) rejects them.

This is THE root cause of:
- Bug #13 (chooser stalls in dominant positions)
- Major part of bugs #4 (drain-frontier) and #14 (asymmetric rollout)
- The recurring "chooser is too risk-averse" perception (it's not
  a choice; captures don't register as positive)
- The systematic 4P regression pattern across spatial leaf,
  joint v1, etc. (all change emit pattern but capture bonus
  doesn't help any of them)

## Synthetic oracle testing methodology (PI proposal, validated)

PI proposed (2026-05-18): backward-solve from situations with
OBVIOUS correct answers. Build a regression test suite of
synthetic scenarios. Each tests a specific planner property.

**The methodology worked**: building the first oracle (100 ships
vs 5 ships) immediately surfaced bug #15. No replay analysis
had revealed it in this depth.

Test file: `tests/test_planner_oracles.py` — 5 oracles defined:
- `test_oracle_sanity_trivial_capture` (currently FAILS — bug #15)
- `test_oracle_cleanup_capture_last_opp_planet` (xfail — bug #13)
- `test_oracle_coordinated_capture_two_sources` (xfail — bug #14)
- `test_oracle_solo_capture_but_loses_source` (xfail — bug #14)
- `test_oracle_defense_against_incoming_multi_fleet` (xpass currently)

Concept doc: `knowledge-base/concepts/coordination-oracle-testing.md`.

## Bug catalog summary (audit/2026-05-18-bug-catalog.md)

15 bugs identified. Status:

| # | Bug | Status | Severity |
|---:|---|---|---|
| 1 | Wait-N=1 emit-when-armed | ✅ FIXED | medium |
| 2 | Backward grid bare-capture | ✅ FIXED (v3 budget) | medium |
| 3 | Asymmetric reinforce sizing | ❌ NOT FIXED | medium |
| 4 | Drain-frontier blindness | ❌ NOT FIXED | high (linked to #14/#15) |
| 5 | Banding dedup ship variants | ⚠️ WORKAROUND | low |
| 6 | Lite_greedy smarter opp | 🔄 REVERTED (retry after #15) | (retry candidate) |
| 7 | Joint v3 4P regression | ⚠️ 2P-only gate | high |
| 8 | Spatial leaf 4P regression | ⚠️ 2P-only gate (off by default) | high |
| 9 | H1 forced emissions | ⚠️ DISABLED by default | low |
| 10 | Orbital wait variants | 🔍 NOT INVESTIGATED | low |
| 11 | Orbital ray-cast in fleet_target_planet | ✅ JUST FIXED (commit c300a31) | high |
| 12 | enemy_inflight window narrow | ❌ NOT FIXED (exposed by #11 fix) | medium |
| 13 | Chooser stalls in dominant positions | ❌ NOT FIXED (caused by #15) | high |
| 14 | Asymmetric rollout (no self-defense) | ❌ NOT FIXED | high |
| 15 | composite doesn't credit captures | ❌ NOT FIXED 🚨 | **CRITICAL** |

## Ranked priorities for next session

### Tier 1 — Critical fixes (do first)

**1. Bug #15 fix (HIGHEST PRIORITY)** — composite_capture_value
   counterfactual logic.

   Implementation sketch:
   ```python
   # In lib/value_heads.py composite_capture_value:
   pred_owner_with_us = model.owner_at(target.id, eta)
   if pred_owner_with_us != my_id:
       # We'd lose combat — bounce penalty
       delta -= waste_weight * ships
       continue
   # Check counterfactual: would target be ours WITHOUT this fleet?
   counterfactual_arrivals = [a for a in model.ledger.get(target.id, [])
                              if a[1] != my_id  # exclude OUR contributions
                              or a[0] != eta or a[2] != int(f.ships)]
   counterfactual_owner = simulate_planet_timeline(
       target, counterfactual_arrivals, eta + 1
   )["owner_at"][eta]
   if counterfactual_owner == my_id:
       # Would be ours anyway → over-reinforcement, no credit
       continue
   # WE cause the capture
   delta += capture_weight * production * time_remaining
   ```

   Cost: ~30-50 LOC + 2 unit tests + re-run oracle suite.
   Expected impact: trivial-capture sanity oracle should PASS.
   Many other oracles likely flip XFAIL → PASS.

**2. Re-run oracle suite + replay analysis** after #15 fix.
   Several xfail tests should xpass. Document the improvement.

**3. Local A/B + bench** with #15 fix:
   - vs hybrid bundle (n=32, then 64 if positive)
   - 4P sub-panel
   - Expected: significant 2P improvement; 4P may also improve
     because joint and other "more aggressive" patterns now have
     working leaf credit

### Tier 2 — After #15 lands

**4. Bug #14** (asymmetric rollout) — IF #15 alone doesn't unlock
   coordination, implement mirroring lite_greedy for ME in the
   rollout. With #15 fixed, our captures get credit → fixing #14
   might let coordination emerge naturally.

**5. Bug #3 + #12** (reinforce sizing fixes) — clean math fixes,
   target the asdf-game pattern. Quick wins.

**6. Bug #11 A/B validation** — orbital ray-cast fix LANDED but
   A/B never completed cleanly this session. Re-validate.

**7. Retry Bug #6** (lite_greedy vulnerability) — with #15
   fixed, smarter opp model should complement (captures get credit
   even if rollout shows opp counter-attacks).

### Tier 3 — After cleanup

**8. Synthetic oracle expansion** — extract more oracle scenarios
   from replays (Roman game step 50, asdf game step 37, more
   fall-then-recapture events). Each becomes a regression test.

**9. Submission decision** — once #15 fixed AND oracles + A/B
   pass AND 4P ≥ 25%, consider new submission. Goal: replace
   1094 floor with something ≥ 1141.

## What this session shipped

Commits to `claude/audit-workflow-performance-btjeK`:
- 4 fix attempts (spatial, H1, joint v1/v2/v3) — all submitted/
  documented (1 live, others off-by-default)
- 2 actual fixes landed: #1 backward wait grid + #11 orbital ray-cast
- 15 bugs catalogued (`audit/2026-05-18-bug-catalog.md`)
- Coordination oracle methodology defined
  (`knowledge-base/concepts/coordination-oracle-testing.md`)
- Synthetic oracle test suite started
  (`tests/test_planner_oracles.py`)

## Lessons learned

1. **Single-step Δ scoring + composite_capture_value's
   over-reinforcement check = systematic under-emission**. This
   was hidden by many surface patterns (drain-frontier, can't
   finish, 4P regression). Bug #15 is the underlying cause.

2. **Synthetic oracle testing surfaces bugs replay analysis
   misses**. The trivial 100-vs-5 case caught #15 in one test.
   Methodology adopted going forward.

3. **A/B alone is too noisy** for diagnosing structural issues.
   Use oracle tests for structural validation; A/B for
   final-go/no-go.

4. **The chooser at μ=1141 is closer to the limit of its
   architecture than expected.** Reaching top-of-LB (~1300+)
   requires architectural change, not parameter tuning. Bug #15
   fix could unlock the next plateau.

5. **PI's intuition about "the chooser should arrive at obvious
   solutions" was right** — and the data showed it. We catalogued
   FOUR situations (dekaineko, asdf, Roman, synthetic) where the
   correct move is obvious but the chooser fails. All trace to
   #15 (capture credit) + #14 (no self-defense in rollout).

## How to start next session

1. **Read this file + bug catalog** (`audit/2026-05-18-bug-catalog.md`)
2. **Read concept doc** (`knowledge-base/concepts/coordination-oracle-testing.md`)
3. **Read oracle tests** (`tests/test_planner_oracles.py`)
4. **First action**: implement bug #15 fix per sketch above.
   Single-PR change, ~50 LOC.
5. **Validate**: run `pytest tests/test_planner_oracles.py -v`
   — trivial-capture sanity should PASS.
6. **A/B**: bundle, test vs hybrid reference.
7. **Submit IF AND ONLY IF**: oracles + A/B + 4P all positive.

## State of files

- `lib/world_model.py` — orbital ray-cast fix (bug #11) landed
- `lib/value_heads.py` — needs bug #15 fix
- `agents/baseline/proposer.py` — backward wait grid (default ON)
- `agents/baseline/main.py` — joint v3 default ON, idle-drain default
  OFF, value head default `hybrid`
- `submissions/baseline.py` — bundle of joint v3 (live as 52766596)
- `tests/test_planner_oracles.py` — oracle test suite (5 tests)
- `audit/2026-05-18-bug-catalog.md` — 15 bugs catalogued

## Rule reminders

- Rule 1: submissions are PI-approved.
- Rule 12: rolling-last-2 = [52754310 (1141.0), 52766596 (1094.1)].
  Next push evicts 52754310 OR 52766596 depending on age. Submit
  carefully.
- Rule 38: fix-verification reproduces failure state. Use oracle
  tests as the new reproduction harness — they're deterministic
  and surgical.
- Rule 40: prefer modeling correctness over restriction tuning.
  Bug #15 fix IS the modeling correctness target. The 2P-only
  gates / drain-restrictions we added are band-aids that #15
  fix could dissolve.

## Next-session first commit suggestion

Implement bug #15 counterfactual capture credit. Single focused
PR. Validation: oracle suite + 2P A/B + 4P sub-panel.
