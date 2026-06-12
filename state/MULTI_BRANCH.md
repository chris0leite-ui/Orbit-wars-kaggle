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
| 2026-06-12 (pending PI sign-off) | claude/elegant-dijkstra-uae6p0 | `oracle_rw.py` (691 291 B, sha256 `0e72d157eb18e07b...`) — imitation-learning agent: policy net behavior-cloned from 1,110 top-ladder replays (23.5M decisions, defense-dense + natural-frequency reweighting) ranks candidate waves on an engine-exact future ledger; exact capture sizing at true arrival tick; verified flights; rate-integrated cadence; value-net blunder veto. Full story: audit/2026-06-11-oracle-track-day1.md + state/ORACLE_STRATEGY.md. | 1150-1300 (uncertain: new architecture, live mix unknown). Local n=16 panels: v7_0 14/16, ledger_v1_4 11/16, Producer 8/16, champion-rebuild 5/16; **4P 42/64 first-place (65.6%, parity 25%)**. Gates: parity 2/2, bench p99 8 ms max 221 ms zero-overage, 500-step self-play validation clean. | **53577315** `producer_plus_vetorf4p_sync_on` mu = 1241.6 (older half, parallel branch). Backstop: **53588922** `garval` (1199.9, climbing) stays. Evicted 1241.6 INSIDE predicted band -> **Rule 42 BLOCKED pending explicit PI sign-off**. | ⏳ AWAITING PI |
| 2026-06-11 (later) | claude/elegant-dijkstra-uae6p0 | `ledger_v1_2.py` (sha256 `35658905ea3a290b...`) — ledger_v1 + coalition rescue (multi-helper defense vs avalanche waves, from live-loss diagnosis of sub 53556728) + shopping-commitment response scaling (enemy garrisons discounted as responders while their stock is committed to buying neutrals; mild calibration 0.5x/floor 0.55). | ≈ ledger_v1's settle or better (v1 at 1094 still climbing at submit time). Local: paired live-1300.9 bundle 15/16 (= pre-change), v7_0 pool 9/12 (was 8/12), producer 4/4, live losses' root cause (single-helper rescue) fixed. | **53547475** `producer_plus_vetorf2p_ffa` μ = 1291.9 (older half — the proven safety net; PI explicitly accepts the eviction for live feedback). Backstop position 2: **53556728** `ledger_v1` (1094 climbing) stays. | ✅ PI explicit "Submit now. We need the feedback from the actual leaderboard" (2026-06-11); **submitted as sub 53558897**. Rule 46 adapted: artifact == source byte-identical; forecast parity 2/2 GREEN; play smoke vs v7_0 seed 13 GREEN; bench p95 39 ms max 62 ms (v1_1-era, code path unchanged in cost). |
| 2026-06-11 (this session) | claude/elegant-dijkstra-uae6p0 | `ledger_v1.py` (48 402 B, sha256 `d5928036010533f7...`) — the from-first-principles single-file agent: engine-exact per-planet future ledger + value-priced captures/snipes/defense/evacuation, coalition strikes with arrival-consistent shares, response-priced flow duration, pressure-scaled liquidity tax (lead-gated), rollout veto vs reactive opponent, FFA posture for 4P, dominance-gated banking. Full story: audit/2026-06-10-ledger-agent-from-first-principles.md. | ≈ 1250-1350 (uncertain: new architecture). Local: 3/4 distinct fresh maps vs the LIVE 1300.9 agent (sha-verified rebuild), 28/32 vs v7_0, 48/48 vs vanilla Producer (pre-banking-fix numbers; post-fix re-confirmation kicked off alongside this submit). | **53542171** `producer_plus_veto2p_ffa` μ = 1244.1 (older half). Backstop position 2: **53547475** `producer_plus_vetorf2p_ffa` μ = 1300.9 stays. Evicted 1244.1 ≤ predicted floor → Rule 42 GREEN with PI order. | ✅ PI explicit "Submit to test on the live ladder" (2026-06-11); **submitted as sub 53556728**. Rule 46 (adapted, no bundler): artifact == agent source byte-identical; tests/test_ledger_forecast.py 2/2 GREEN; fast.py play artifact vs v7_0 seed 7 full game max turn 44 ms << 1000 ms; self-play validation clean (earlier today). |
| 2026-06-03 10:37 | champion-strategy-rules-00JzI (built via worktree at commit `9985e98` from the `claude/champion-ml-graft-majestic-storm` session) | `baseline_adaptive_k` **resubmit** (sub **53324164**, bundle sha256 `6c0419dc20`, 608 844 B) — champion (`launch_rules_universal`) full config + adaptive horizon K baked ON (`BASELINE_ADAPTIVE_K = 1`, K_OPEN = 20 → floor 10 by step 30). Same agent as evicted sub 53265480 which settled at live μ = 1170.4. Designated as the **main strategy** going forward (`state/STRATEGY.md`). | ≈ 1170 based on prior live settle of identical agent. High confidence (same code; only stochastic ladder noise). | **53304016** `baseline_launch_rules_universal` μ = 1131.1 (older half of rolling pair; was itself a 2026-06-02 21:09 resubmit of the all-time-champion bundle that did NOT reach its historic 1183.7 peak — settled 52 μ below). Backstop position 2: **53316984** `baseline_state_k_orbital_lead` μ = 1109.0 (still settling, ~4 h on ladder at submit time). Evicted-μ 1131.1 < predicted 1170 → **Rule 42 GREEN**: rolling-pair floor rises ≈ 40 μ. | ✅ PI explicit "research submit this solution: baseline_adaptive_k …" (2026-06-03). Rule 46 GREEN: bundle 608 844 B / `tests/test_bundle.py` 15/15 / `fast.py play` vs `v7_0` seed = 7 → 281 steps p0_win, max turn 639 ms. |

## Pointers

- `state/STRATEGY.md` — the strategy itself.
- `CLAUDE.md` — process rules (incl. Rule 42 gate semantics).
