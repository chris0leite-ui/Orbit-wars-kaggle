# Engine audit #3 (2026-06-10 night) — three lenses, three mechanisms queued

PI directive: re-examine the architecture strategically (game expert /
mathematician / software engineer), accumulate evidence, implement the
strongest lift thoroughly. No submissions left today (5/5 used), so tonight
is build + measure; tomorrow has 5 fresh slots.

## Lens findings

**Game expert — defense sizing is structurally absent from the size menu.**
When a wave's target is our own planet, `capture_floor` returns 1 (correct:
arrivals just add to the garrison). But the multi-size grid derives its three
candidate sizes from that floor: {ceil(floor·overkill) ≈ 1–2, 2× that, full
safe_drain}. The one size that matters for a defense — the projected deficit
at flip time — is not in the menu unless safe_drain happens to coincide.
This is the modeling shape of the measured hold-rate gap (ours 0.59 vs top
teams 0.74–0.85). `_apply_reinforce_deficit_floor` fixes exactly this and
already exists, but its only verdict came from the 4P vs-producer panel —
the yardstick that failed three live checks. Re-judging under the corrected
referees (champion + attribution vs the live stack).

**Mathematician — the veto is a filter where the model wants a fixed point.**
Pass 1 plans against the opponent's do-nothing-conditioned launches; the
veto predicts the reply to pass 1 and can only DROP waves. Three losses of
value: (a) vetoed ships idle instead of taking the next-best action; (b) the
defensive lane never sees the predicted counter (friendly_flip_targets
accepts a background — the reply is never fed to it on our side); (c) kept
waves are never re-sized/re-timed under the reply. The one-ply replan runs
the whole planner a second time with the reply as background and the roi
threshold re-normalized by do-nothing-under-reply (the same paralysis fix
the veto's mirror needed). Infrastructure was already in place:
plan_lite_waves takes background; the mirror takes base_background=mine.

**Software engineer — the turn budget makes the second pass nearly free.**
Live p50 71 ms / max 141 ms against a 1000 ms gate. Replan adds ~1 planner
pass + 1 mirror per opponent; with the veto verifying pass 2 the full stack
is ~5 planner-scale passes — still well inside budget (to be confirmed by
the Rule 46 smoke before any submit).

## Implemented tonight

- `PRODUCER_PLUS_REPLAN` (+ `_2P_ONLY` gate), `_apply_replan` in
  producer_plus/main.py; reply prediction extracted to `_predict_reply`
  (shared with the veto — veto tests 12/12 green, refactor behavior-neutral).
  Skips when pass 1 fired nothing or the reply is empty. 10 new tests
  (tests/test_replan.py) — gating, skip conditions, roi re-normalization.
- Bundle variants: `vetorf_replan` (replan + veto verify on the live stack),
  `replan_rf` (replan replaces veto — subsumption test), `vetorf_deficit`
  (deficit re-judge), `veto_rf_nq` (survivor stacking: does the quota's
  +31% @80 early lead convert when the reactive floor guards the frontier?).

## Measurement queue (margin harness, n=8, truncated at step 150, one job at a time)

1. veto_rf_nq vs champion + attribution vs veto_rf  (RUNNING)
2. Rule 46 timing smoke for vetorf_replan
3. vetorf_replan vs champion
4. vetorf_replan vs veto_rf (attribution — the submit-relevant gate)
5. replan_rf vs veto_rf (does replan subsume the veto?)
6. vetorf_deficit vs veto_rf (attribution)

Two-gate protocol: champion win AND positive attribution vs the live stack.

## Result: survivor stacking (veto+rf+nq), measured 2026-06-10 ~22:30 UTC

- vs champion: **6/8 wins, lead@80 +41.1% (ahead 8/8 games), @120 +54.7%**
  — strongest champion leg of the night (rf alone: 6/8, +28.3% @80).
- attribution vs veto_rf: **paired mean 0.0% at every checkpoint, 1 win /
  2 losses / 2 exact-mirror draws** in 5 clean games (3 games hit in-game
  timeouts — concurrent bundling/pytest during the leg, lesson re-learned:
  NOTHING runs alongside a measurement leg, however small it looks).
- Verdict: **fails gate 2 — not promoted.** The quota's early expansion
  adds nothing the rf stack wasn't already converting; its champion-leg
  dominance is another instance of why champion wins alone don't gate
  (upsize precedent). Revisit only with a late-game conversion fix.
