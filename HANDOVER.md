# HANDOVER — next session brief (written 2026-06-11 ~22:50 UTC)

## Live state
- **Sub 53577315** (vetorf4p_sync) submitted 17:44 UTC 2026-06-11 — live
  experiment: current best stack + same-tick two-source coalitions (holds
  OFF). TrueSkill warm-up ~24h; do NOT read μ before ~2026-06-12 18:00.
  Rolling pair: 53577315 + 53564198 (best, settled ~1244-1280 band).
- Daily budget resets midnight UTC; 2026-06-11 used 4/5 slots
  (2 parallel-branch ledger subs, our 53564198, our 53577315).

## First actions next session
1. Rule 32 git fetch; rebuild gitignored bundles as needed
   (`python scripts/bundle_producer_plus.py --variant veto_rf` etc. +
   PPNSX-namespaced referees via sed).
2. `python scripts/live_episode_summary.py 53577315 --pull` once ~20+
   episodes exist: compare 2P/4P winrates and coalition-fired games vs
   53564198's baseline (51.7% overall, 2P 50.0% n=34, 4P 53.8% n=26).

## The build queue (evidence-ranked)
1. **Holding-time-priced terminal value** — the open foundation fix.
   Diagnosis PROVEN (decision_trace on the Gregor Lied loss: every capture
   scores +0.0 vs threshold +1.5 for 3 straight turns; wins are
   production-ahead @40 in 16/17, losses behind 9/17). Flat λ=12 REFUTED
   (mirror 2/12; champion 4/8 +20% < control 6/8 +51%) — invested capital
   gets punished before payback. Build: per-target credit = production ×
   expected holding time given opponent's feasible retake (see ledger
   branch's capture pricing, `git show origin/claude/elegant-dijkstra-uae6p0:submissions/ledger_v1_2.py`).
   Components on hand: recapture_penalty (orbit_lite), reactive margin,
   capture_floor.
2. Sync holds redemption (SYNC_DMAX>0) only if a live observation suggests
   the field doesn't punish telegraphed splits like our mirror does.
3. Tuner v2 (more seeds, smaller steps, commit-cost eps knob).

## Key tools
- `scripts/decision_trace.py REPLAY SEAT STEP...` — planner internals on a
  live episode step (shortlist/floors/scores/veto). Use on every PI replay
  observation FIRST.
- `scripts/sync_probe.py` — in-process hold lifecycle probe; surfaces
  swallowed agent crashes (env plays on after agent exceptions — flipped
  outcome + zero mechanism activity ⇒ suspect a crash).
- `scripts/margin_ab.py` / `scripts/margin_ffa.py` — truncated margin
  triage, dead-seat guards in both.

## Doctrine (unchanged)
Local referees saturate below our level; mirror punishes any divergence.
Local measurement = safety gate (crash/timeout/rout detection) + mirror-
domination detection. The ~55 remaining live slots are the only honest
instrument vs the 1300+ field. One observation → one mechanism → one push.
