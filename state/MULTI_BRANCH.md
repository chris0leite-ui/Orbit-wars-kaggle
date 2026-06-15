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

## Pointers

- `state/STRATEGY.md` — the strategy itself.
- `CLAUDE.md` — process rules (incl. Rule 42 gate semantics).
