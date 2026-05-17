# Direction B — scoping note for next session (2026-05-17)

PI directive at session end: trajectory chooser v4 + wait_N is
shipped (52754310 PENDING). Next session goes into Direction B —
joint candidate evaluation.

## Why Direction B is the next real lift

Current chooser (trajectory v4 / composite) scores each candidate
launch INDEPENDENTLY. Then greedy emit with no-dogpile dedup. This
misses interaction effects:

1. **Defense+attack coupling.** Launching A→X (offense) AND B→A
   (defense for A's depleted garrison) is a JOINT decision. Scoring
   them separately means the defense Δ is computed under "A still
   has all its ships" but the offense already committed those ships.
2. **Split-force optimisation.** 60 ships from one source might be
   better split 30→X, 30→Y than concentrated 60→X. The chooser
   currently picks ONE target per source per turn.
3. **Opp dilemma.** Two simultaneous threats force opp to pick one
   to defend. The cost of the 2-target attack is more than 2× the
   cost of either alone — opp's response is upper-bounded.

The "wait_N+1" reservation mechanism is the closest we have to joint
scoring (reserve src+tgt for a future launch), but it's still single-
candidate. Direction B explicitly evaluates K-tuples.

## Concrete first-step plan

1. **Enumerator.** Top-K (K=5 first cut) independent candidates form
   seed pool S. Joint candidate set J = S ∪ {(s_i, s_j) | i<j} for
   pairs only first. Total = K + C(K,2) = 5 + 10 = 15 candidates.
2. **Scorer.** For each joint candidate J_k, run fast_sim with ALL
   constituent launches injected at their wait_N steps. Read favor at
   max(horizon over constituents). Subtract idle baseline at same h.
3. **Emit.** Pick best Δ joint. Emit its fire-now (wait_N=0)
   constituents. Reserve src+tgt for wait>0 constituents. NO further
   dedup needed within a joint set (constituents already non-conflicting
   by enumeration).

## Wallclock estimate

K=5 joint with pairs: 15 candidates × ~30ms each = 450ms. Tight on
600ms budget but fits.

K=10 with pairs: 55 candidates × ~30ms = 1650ms. Doesn't fit. Skip.

K=5 with pairs + triples: 5 + 10 + 10 = 25 candidates × ~30ms = 750ms.
Doesn't fit at default budget; could fit at extended budget (900ms).

Decision: start with K=5 + pairs only. Measure. If lift is real and
wallclock allows, extend.

## What could go wrong

1. **Joint scoring needs the right baseline.** If we use the same idle
   baseline for all, joints will systematically score higher than
   singles (more action = more Δ favor in expectation). Need to
   either (a) include the union of constituents in the comparison —
   not just the best — or (b) compare joints against the best single
   (relative lift).
2. **Greedy seeding misses joint optima.** The K=5 seed pool is the
   top-K INDEPENDENT candidates. A joint where neither constituent is
   in the top-5 might score higher than any top-5 pair. First-cut
   acceptable; investigate if results are borderline.
3. **wait_N joint interactions.** Two wait>0 constituents from same
   src conflict at the wait expiry — can't do both. Filter joints by
   src compatibility before scoring.
4. **Bundler edge cases.** The enumerator + joint scorer adds ~80-120
   LOC to chooser_trajectory.py. Each new symbol needs to survive the
   bundler's import-stripping regex. Friction tag
   `bundler-modular-agent-namespace-access-breaks-bundle` (2026-05-17)
   applies; keep imports single-line and explicit.

## What we know from prior sessions

- Composite emit was `1-per-src, 1-per-tgt`. Was that load-bearing?
  v20 tried removing the per-target dedup (dogpile) and got 65.6% h2h
  vs v15 at n=32 — within noise but not a clear lift. So bare dogpile
  doesn't help; the OPTIMISATION inside the dogpile (joint scoring)
  is the load-bearing piece.
- v4 with multi-launch budget (no per-src dedup) didn't beat the
  1-per-src variant. Reverted. Joint scoring + emit constituents
  together is the missing piece, not just relaxing emit.

## Verification plan for next session

1. Unit tests on enumerator (K=5 seed → 15 joint candidates).
2. Joint scorer parity test (single-candidate joint = score_candidate_v4
   result on that candidate).
3. Smoke 8/8 with `BASELINE_CHOOSER=trajectory_jointB`.
4. A/B vs v15 at n=64. Target: ≥70% (≥45/64), Wlo ≥ 0.58.
5. If A/B passes, A/B vs composite_a2 at n=32 (target: ≥55%, head-
   to-head sanity check vs the rolling-pair partner).

## What NOT to do

- Do NOT replace v4 with the joint scorer — add as opt-in via env
  var (BASELINE_CHOOSER=trajectory_jointB) until A/B confirms lift.
- Do NOT submit until local A/B passes AND head-to-head vs
  composite_a2 passes. The current live submission (52754310) is
  still settling.
- Do NOT extend K or add triples until pairs ablate cleanly.

## Cross-references

- `knowledge-base/concepts/probability-of-winning-framework.md` —
  Direction B description.
- `knowledge-base/concepts/trajectory-first-architecture.md` —
  architectural reframe motivation.
- `audit/2026-05-17-trajectory-chooser-shipped.md` — what shipped.
- `agents/baseline/chooser_trajectory.py` — extend here.
