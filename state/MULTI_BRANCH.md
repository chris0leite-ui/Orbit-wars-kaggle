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
| 2026-06-03 10:37 | champion-strategy-rules-00JzI (built via worktree at commit `9985e98` from the `claude/champion-ml-graft-majestic-storm` session) | `baseline_adaptive_k` **resubmit** (sub **53324164**, bundle sha256 `6c0419dc20`, 608 844 B) — champion (`launch_rules_universal`) full config + adaptive horizon K baked ON (`BASELINE_ADAPTIVE_K = 1`, K_OPEN = 20 → floor 10 by step 30). Same agent as evicted sub 53265480 which settled at live μ = 1170.4. Designated as the **main strategy** going forward (`state/STRATEGY.md`). | ≈ 1170 based on prior live settle of identical agent. High confidence (same code; only stochastic ladder noise). | **53304016** `baseline_launch_rules_universal` μ = 1131.1 (older half of rolling pair; was itself a 2026-06-02 21:09 resubmit of the all-time-champion bundle that did NOT reach its historic 1183.7 peak — settled 52 μ below). Backstop position 2: **53316984** `baseline_state_k_orbital_lead` μ = 1109.0 (still settling, ~4 h on ladder at submit time). Evicted-μ 1131.1 < predicted 1170 → **Rule 42 GREEN**: rolling-pair floor rises ≈ 40 μ. | ✅ PI explicit "research submit this solution: baseline_adaptive_k …" (2026-06-03). Rule 46 GREEN: bundle 608 844 B / `tests/test_bundle.py` 15/15 / `fast.py play` vs `v7_0` seed = 7 → 281 steps p0_win, max turn 639 ms. |
| 2026-06-15 11:31 | claude/festive-knuth-roggck | **A/B PAIR** — sub **53708787** `seq_strength` (fresh resubmit of the 1280 flag set; field-drift baseline) + sub **53708789** `seq_strength_opening` (1280 base + opening beam-search `OPENING_SEARCH=40`/beam-64 — the **wide step**, spending early-game headroom to attack the real early-death loss cluster ~step 100, ⅓ of producer_plus's actual ladder losses; opening-search OFF in every shipped variant so never ladder-tested on the strong base). | baseline ≈ 1220 (drifted from 1280); opening ≈ 1280 ± the opening-search effect — **UNKNOWN, that is the experiment**. | evicts 53595717 `shotmlp` μ 1230 + 53618099 `rl_v7` μ 927 (rolling-pair churn for A/B; kept-pair curated at deadline per PI "ladder is for A/B now"). | ✅ PI "Go, fire both" (2026-06-15). Rule 46 smoke PASS (full game, max 199 ms < 1000). |
| 2026-06-15 13:12 | claude/festive-knuth-roggck | sub **53711823** `seq_strength_wideshortlist` (1280 base + `NEUTRAL_SHORTLIST=20`). **Fixes a CONFIRMED bug from a PI replay observation** (2P loss to CPMP, seed 641308308): we left high-value garrison-41 CORNERS neutral while the opponent took the symmetric one — the far corners fall outside the nearest-K neutral shortlist so are never candidates. Reproduced in the mirror on that seed: baseline leaves 2 garrison-41 corners neutral @step95, `NEUTRAL_SHORTLIST=20` grabs both (fc/opening/overkill/hold/denial did NOT). The most-grounded wide step of the session. | ≈ field 1220 ± the fix's effect — UNKNOWN. Early warm-up read: opening 1177 ≈ baseline 1188 @~2 h (noisy, Rule 12). | evicts 53708787 `seq_strength` baseline (frozen ~1188 @2 h, premature). New rolling pair = `{wideshortlist 53711823, opening 53708789}`. | ✅ PI "Go fire it now" (2026-06-15). Rule 46 smoke PASS (216 turns, max 229 ms < 1000). |

| 2026-06-15 14:35 | claude/festive-knuth-roggck | sub **53714433** `seq_strength_expand` (1280 base + `NEUTRAL_SHORTLIST=20` + `HORIZON_2P=30` + `HORIZON_4P=18`). **Grounded fix for the #1 loss driver** from the 46-loss replay analysis: early under-expansion (78% of losses trail by step 30; 5-6 planets vs winners' 8 by step 60). Deeper horizon + wider shortlist together lift planets@60 6->7-8 (winner rate); either alone partial. | ≈ field 1220 ± the fix's effect. | evicts 53708789 `opening` (warm-up 1139, tracking below field). New rolling pair `{expand 53714433, wide-shortlist 53711823}`. | ✅ PI "Go act now" (2026-06-15). Rule 46 smoke PASS (2P/4P full games, midgame max ~464ms < 1000). |

## Pointers

- `state/STRATEGY.md` — the strategy itself.
- `CLAUDE.md` — process rules (incl. Rule 42 gate semantics).

| 2026-06-16 06:14:49 | claude/affectionate-newton-19kqrp | sub **53733475** `champion_strongest` (`vetorf4p_seq_strength` 1280 champion + `FFA_WEIGHTS=strongest`: one-hot opponent weight on the current LEADER instead of strength-weighted average). 2P byte-identical to the champion (FFA inactive < 3 players); 4P (60% of ladder) optimises ships relative to the SINGLE strongest rival = "focus the leader". PI's relative-ship-count idea, applied to the strong base. bundle sha256 `72b9fa4663`, 398668 B. | ≈ 1150 (2P == champion ~1200; 4P leader-focus is the experiment, ±). 2P-byte-identical to a ~1200 agent floors the downside. | **COLLISION:** at read-time the pair was {pp_positional 1062.7, pp_expand 963.5}; I predicted evicting pp_expand. But another line submitted `pp_pos4p` (53733424) at 06:12 — 2 min before mine — evicting pp_expand first. So MY submit actually evicted **53722697 `pp_positional` μ 1062.7**. New pair = {champion_strongest 53733475, pp_pos4p 53733424}. Evicted 1062.7 < predicted 1150 → **Rule 42 still GREEN**. | ✅ PI "Go" (2026-06-16 06:09 UTC). Rule 46 smoke PASS: bundle 398668 B; 2P full game max 98 ms; 4P full game max 241 ms, 0 turns > 1000 ms. **Lesson:** re-read the rolling pair IMMEDIATELY before submit (not before the smoke) — a concurrent branch slipped in. |

| 2026-06-16 06:52 | claude/affectionate-newton-19kqrp | sub **53734450** `champion_holdval` (champion `vetorf4p_seq_strength` [inverse opp-projection + multi-size + veto + reactive-floor + reply-seq + FFA-strength] + `HOLD_VALUE=12`: holdability-discounted production value — credit post-horizon production ONLY for captures the opponent can't retake). The PI's "maximize production + ship count" fix for the banking/idle pathology (replay obs seed 1394215446: champion hoarded → 6 planets; +HOLD_VALUE took+held 28 planets, 2527 vs 498 ships). Holdability discount avoids the over-expansion that capped flat termval (pp_positional 1062.7). bundle sha256 `9c6e593abc1a`. | ≈ 1100 (changes BOTH 2P+4P unlike strongest; idle-seed strong, confounded seeds mixed — ladder is the judge). | evicts **53733424** `pp_pos4p` (other line's flat-production 4P test, still warming ~600-700). New pair {champion_holdval, champion_strongest 53733475}. Evicted ~warming < predicted 1100 → Rule 42 GREEN. | ✅ PI "Submit" (2026-06-16 06:51). Rule 46 smoke PASS: 4P full-game max 225 ms, 0 turns > 1000 ms. |
