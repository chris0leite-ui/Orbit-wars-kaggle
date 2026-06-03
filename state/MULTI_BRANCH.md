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
| 2026-06-03 14:?? | `claude/champion-ml-graft-majestic-storm` (commit `0025c67`, on top of strategy-lock commit `ccec9f8`) | `champ_computeByShips_on.py` (sha256 `53bf813b...`, 697 927 B). Same all-time-champion config + adaptive horizon K + **NEW: compute_by_ships lever** (`BASELINE_COMPUTE_BY_SHIPS=1`). Lever does two things proportional to a planet's ship surplus: (1) enumerates 4..16 target options per planet (log-scaled vs avg fleet), (2) raises the launch-arrival K cap by up to +50% for high-ship planets. Both default OFF; champion byte-identical when env unset. | ≈ 1170 (parity with adaptive_k sibling; local 8-seed/16-game A/B vs same-source lever-off = 7/16 wins, Wilson [0.231, 0.668] — INCONCLUSIVE, point estimate slightly below parity but CI overlaps both sides). Diagnostic probe showed mechanism is NOT activating the original rear-stockpile case (zero launches from sources >150 dist to opp in either bundle), but lever DOES shift the launch mix in 100% of games. Submitting to test against the broader Kaggle opponent panel (not just sibling). | **53316984** `baseline_state_k_orbital_lead` μ = 1114.1 (older half of rolling pair). Backstop position 2: **53324164** `champ_adaptiveK_on` μ = 1185.2 (our current live champion — stays in the rolling pair as new #2). Evicted-μ 1114.1 < predicted 1170 → **Rule 42 GREEN**: rolling-pair floor rises ≈ 56 μ; live champion NOT at risk. | PENDING — awaiting explicit PI submit command (2026-06-03). Rule 46 GREEN: bundle 697 927 B parses + has lever + has no cross-agent imports / `tests/test_bundle.py` 15/15 / `fast.py play` vs `v7_0` seed = 7 → won 192 steps, max turn 744 ms. Unit tests `tests/test_compute_by_ships.py` 12/12. |
| 2026-06-03 10:37 | champion-strategy-rules-00JzI (built via worktree at commit `9985e98` from the `claude/champion-ml-graft-majestic-storm` session) | `baseline_adaptive_k` **resubmit** (sub **53324164**, bundle sha256 `6c0419dc20`, 608 844 B) — champion (`launch_rules_universal`) full config + adaptive horizon K baked ON (`BASELINE_ADAPTIVE_K = 1`, K_OPEN = 20 → floor 10 by step 30). Same agent as evicted sub 53265480 which settled at live μ = 1170.4. Designated as the **main strategy** going forward (`state/STRATEGY.md`). | ≈ 1170 based on prior live settle of identical agent. High confidence (same code; only stochastic ladder noise). | **53304016** `baseline_launch_rules_universal` μ = 1131.1 (older half of rolling pair; was itself a 2026-06-02 21:09 resubmit of the all-time-champion bundle that did NOT reach its historic 1183.7 peak — settled 52 μ below). Backstop position 2: **53316984** `baseline_state_k_orbital_lead` μ = 1109.0 (still settling, ~4 h on ladder at submit time). Evicted-μ 1131.1 < predicted 1170 → **Rule 42 GREEN**: rolling-pair floor rises ≈ 40 μ. | ✅ PI explicit "research submit this solution: baseline_adaptive_k …" (2026-06-03). Rule 46 GREEN: bundle 608 844 B / `tests/test_bundle.py` 15/15 / `fast.py play` vs `v7_0` seed = 7 → 281 steps p0_win, max turn 639 ms. |

## Pointers

- `state/STRATEGY.md` — the strategy itself.
- `CLAUDE.md` — process rules (incl. Rule 42 gate semantics).
