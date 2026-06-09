# Champion failure-mode mix — verify-first gate (2026-06-01)

**Branch:** `claude/champion-strategy-rules-00JzI`
**Purpose:** PI-mandated verification (2026-06-01) of the fleet-sizing
premise *before* building. The 46% A+D figure that motivated the build came
from sub 52827111 (μ≈1137); the PI asked whether it still holds on the
current, stronger champion. This re-measures it.

## Method

- Agent: `baseline_launch_rules_universal`, live sub **53182323**, μ≈1183.7
  (team `ChrisLeiteScha`). Strongest agent we have shipped.
- Pulled **120 live tournament replays** (`scripts/live_episode_summary.py
  --pull`): 50 two-player, 70 four-player, 1 self-match. Real-field winrate
  **46.7%** (2P 50.0% / 4P 44.3%) — sanity-confirms these are genuine ladder
  games, not self-play.
- Two-stage diagnostic, recovered verbatim from commit `9994b62` (the
  F-flag-corrected versions):
  1. `large_to_small_audit_v2.py` → per-launch failure set (11,391 launch
     rows; 1,204 failed landing-captures) → `2026-06-01-champion-failures.jsonl`
  2. `h44_landing_capture_diagnostic.py` → classifies each failure
     A/C/B/D/E/G/other → `2026-06-01-champion-h44.jsonl` (1,125 diagnosed).
- Analyzer processes the 70 four-player FFA episodes only — same format as
  the original H44 study (Rule 41 confound discipline preserved).

Failure modes:
- **D** under-delivered cleanly — math was right, chooser sized too small;
  defender grew in transit / enemy co-arrived and won combat.
- **A** source lost pre-landing — the launch drained the source; an opponent
  took the undefended source while our fleet was away.
- **E** prediction off-by-one (landed within 5 steps after the predicted
  window) — timing noise, not a sizing error.
- **C** third-party flip · **B** target production accrual · **G** near-tie
  combat · **other** no flag fired.

## Result — DECISION NUMBER

**Lost episodes (n=644 diagnosed failures):**

| mode | n | share |
|------|----|------|
| **D under-delivered** | 234 | **36.3%** |
| **A src lost pre-landing** | 142 | **22.0%** |
| other | 121 | 18.8% |
| E off-by-one | 80 | 12.4% |
| C third-party flip | 32 | 5.0% |
| B tgt accrual | 25 | 3.9% |
| G near-tie | 10 | 1.6% |

**A + D = 376 / 644 = 58.3% of lost-episode capture failures.**

Won episodes (n=481) by contrast are E-heavy (28.7%) and D-light (19.5%) —
the off-by-one timing noise shows up when we're capturing a lot; it is not a
loss driver. A is also much lower in wins (13.5% vs 22.0% in losses),
confirming source over-drain specifically correlates with losing.

Overall (won+lost, n=1125): D 29.2%, A 18.4%, E 19.4%, other 22.4%, C 5.2%,
B 3.4%, G 2.0%.

## Verdict — GATE PASSED (decisively)

- Gate criterion (plan §0.4): lost-episode A+D ≥ ~35% → proceed to build.
- Measured: **58.3%**. Premise **holds and is stronger** than the original
  46% — it did *not* decay on the better agent, it grew.
- **D dominates A** (36.3% vs 22.0% in losses). The build should **lead with
  Step D** (arrival-correct capture sizing) as the primary lever, with Step A
  (`source_keep_floor`) secondary. This re-weights the plan, which had
  treated them as co-equal.
- No single mode ≥40% → still a genuinely multi-pronged failure surface; the
  D+A pair is the actionable mass but neither alone is the whole story.

## Flagged for follow-up (not blocking)

- **`other` = 22.4%** of all failures (18.8% in losses): launches that failed
  capture but matched none of the A/C/B/D/E/G flags. This is a real
  unexplained slice. Not the build target, but worth a later look — the
  classifier precedence may be missing a mode. Logged as an open question.
- The A↔D tension is real and opposite-signed (deliver more vs keep more),
  which is exactly why a constant bump can't fix it and a per-launch model is
  warranted (Rule 40).

**Artifacts:** `audit/2026-06-01-champion-failures.jsonl`,
`audit/2026-06-01-champion-h44.jsonl`. Replays in
`audit/live-episodes/53182323/` (gitignored; re-pull with
`python -m scripts.live_episode_summary 53182323 --pull --team ChrisLeiteScha`).
