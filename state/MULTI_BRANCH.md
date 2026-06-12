# state/MULTI_BRANCH.md — push-claim board (Rule 42)

> Slimmed 2026-06-03 to the push-claim board only. The pre-strategy-lock
> version (track registry, closed tracks, per-branch sync table, substrate
> tier split) is preserved at `state/_archive/MULTI_BRANCH-pre-adaptive-k.md`.

## Live Kaggle — read directly

Scores (μ) and the rolling pair are NEVER transcribed here — they go stale
within hours. The leaderboard is the single source of truth:

```
kaggle competitions submissions orbit-wars
```

- **Rolling pair** = the **two most-recent** submissions in that list
  (Kaggle auto-keeps exactly these two for final evaluation).
- **Daily submission budget:** 5/day.
- **Deadline:** 2026-06-23 23:59 UTC (fixed).

Before any submit, fill the push-claim board below using the freshly-read pair.

## Push-claim board (Rule 42 — fill before every `kaggle competitions submit`)

Most recent claim at top. Required fields: all 6 columns. The PI-signoff
column is mandatory if evicted-μ > predicted-μ (Rule 42).

| Timestamp (UTC) | Branch | Agent | Predicted μ | Will evict (sub_id, μ) | PI signoff |
|---|---|---|---:|---|---|
| 2026-06-12 08:43 | blissful-cray-tiusnw | `producer_plus_vetorf4p_sync_shotmlp015_on.py` (sub **53595717**, sha256 `03ce7fe729ba64a6…`, 410 225 B) — LIVE PROBE: live vetorf4p_sync stack + learned shot-success filter at threshold 0.15 (24-feature MLP, 286k live-episode launches, val AUC 0.871; drops attack waves below 15% predicted hold probability — the 7.9%-success tail, ~17% of live launches). Local A/B parity by referee blindness; live-only mechanism (audit/2026-06-12-shot-mlp-offline-counterfactual.md). | ≈1240–1300 (anchor: identical base config sub 53577315 settled ≈1258 peak 1317; filter direction unproven live — that is the probe). | **53588922** garval (warming, ~1173 at read; claim predicted ≥1280). Older pair half 53577315 (μ≈1240) already evicted at 08:12 by 53594710 `oracle_rw` (elegant-dijkstra). | ✅ PI explicit "Submit" 2026-06-12, re-confirmed "Submit now" after the board change (garval-eviction context surfaced). Rule 46 GREEN: bundler OK / tests 3-3 / full game max turn 767 ms. |
| 2026-06-03 10:37 | champion-strategy-rules-00JzI (built via worktree at commit `9985e98` from the `claude/champion-ml-graft-majestic-storm` session) | `baseline_adaptive_k` **resubmit** (sub **53324164**, bundle sha256 `6c0419dc20`, 608 844 B) — champion (`launch_rules_universal`) full config + adaptive horizon K baked ON (`BASELINE_ADAPTIVE_K = 1`, K_OPEN = 20 → floor 10 by step 30). Same agent as evicted sub 53265480 which settled at live μ = 1170.4. Designated as the **main strategy** going forward (`state/STRATEGY.md`). | ≈ 1170 based on prior live settle of identical agent. High confidence (same code; only stochastic ladder noise). | **53304016** `baseline_launch_rules_universal` μ = 1131.1 (older half of rolling pair; was itself a 2026-06-02 21:09 resubmit of the all-time-champion bundle that did NOT reach its historic 1183.7 peak — settled 52 μ below). Backstop position 2: **53316984** `baseline_state_k_orbital_lead` μ = 1109.0 (still settling, ~4 h on ladder at submit time). Evicted-μ 1131.1 < predicted 1170 → **Rule 42 GREEN**: rolling-pair floor rises ≈ 40 μ. | ✅ PI explicit "research submit this solution: baseline_adaptive_k …" (2026-06-03). Rule 46 GREEN: bundle 608 844 B / `tests/test_bundle.py` 15/15 / `fast.py play` vs `v7_0` seed = 7 → 281 steps p0_win, max turn 639 ms. |

## Pointers

- `state/STRATEGY.md` — the strategy itself.
- `CLAUDE.md` — process rules (incl. Rule 42 gate semantics).
