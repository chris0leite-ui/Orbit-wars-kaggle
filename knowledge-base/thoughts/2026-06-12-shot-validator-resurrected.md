# 2026-06-12 — The shot validator finally got trained

PI asked: of everything we tried before the Producer era, what was
promising but never thoroughly implemented? The review landed on four
candidates; PI said "Go" on the top one — the May shot-validator plan
(learned launch filter), with the denial/opening weight probe as the
cheap parallel. The weight probe turned out to be already answered:
awesome-clarke ran it 2026-06-10 at calibrated weights and banked nulls
(16/32, 15/32, 18/32 vs the base's 24/32). So the session went all-in on
the filter.

## What exists now

- 286,618 labeled launches from 634 live episodes of 7 recent subs
  (all seats — ours and the field's), rebuilt with an engine-exact
  fleet-speed curve (the May labeler missed the 1000-ship speed cap).
- A trained 24-32-16-8-1 MLP: val AUC 0.871, episode-grouped split,
  near-diagonal calibration. Weights live as base64 inside
  `agents/producer_plus/shot_mlp.py`; pure-numpy inference in-agent.
- A reject-only veto on attack waves (`PRODUCER_PLUS_SHOT_MLP=<thr>`),
  running after the response veto in the vendored producer_plus stack
  (pinned at the live submission's commit `be71c97`).

## The offline headline

Scoring our OWN 107,762 live-ladder launches: 25% have model
P(hold +10 turns) < 0.30 and succeed 14.5%; the kept 75% succeed 77.7%.
The current live submission (53577315) wastes 23.7% of its launches on
13.6%-success shots. The model sees the waste the 2026-06-10 live mining
measured (~321 ships/game into anticipated parries).

## Two integration traps worth remembering

1. **Flat-bundle name shadowing.** The bundler concatenates modules into
   one namespace; shot_mlp's float `fleet_speed` silently replaced
   orbit_lite's tensor version and crashed every bundled game at step 1.
   Fix: prelude modules are exec'd into a synthetic module object and
   only listed exports are re-bound.
2. **Baked env vars leak across same-process opponents** (2nd recurrence
   of the 2026-05-22 friction). A `setdefault` threshold in the bundle
   header turned candidate-vs-control into mirror-vs-mirror. Fix: the
   bundler hardcodes the threshold into the gate function and keeps it
   out of the env header entirely.

## Open

- A/B verdict (n=32 per arm vs vanilla producer) — see
  `audit/2026-06-12-shot-mlp-offline-counterfactual.md` for results as
  they land.
- Mirror games tell this mechanism nothing: in-family, the response
  veto's 1-ply mirror is an exact opponent model, so the MLP has little
  to add; its edge should be against opponents the mirror mispredicts.
  Local referees beyond vanilla producer worth considering for the
  confirm run: the old champion bundle, 4P pools via play4p.
- Threshold 0.30 was the first probe; 0.15 is the high-precision
  alternative (rejects 17% of our live launches at 7.9% success).
