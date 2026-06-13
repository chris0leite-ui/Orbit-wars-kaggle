# FLAG: rolling-pair endgame — final-eval floor is a sinking RL agent (cross-session)

**Raised:** 2026-06-13 ~17:40 UTC. Deadline: 2026-06-23 23:59 UTC (~10 days).

**The mechanic (CLAUDE.md Rule 12):** Kaggle keeps the **two most-recent
submissions by time** for final evaluation — not PI-selected. At the
deadline, whatever the last two submissions are (across ALL sessions) is
what we're scored on.

**Current rolling pair (read 2026-06-13 17:38 UTC):**

| sub | agent | settled μ | submitted |
|---|---|---|---|
| 53618099 | `rl_v7_selfplay_league` (RL session) | **944.5** (sinking: 963→944) | 06-12 23:33 |
| 53595717 | `producer_plus…shotmlp015` (this branch) | 1257.6 | 06-12 08:43 |

**The problem:** the RL self-play agent is the newest submission, so it
sits in the final-eval pair, and it is **~300 μ below our best** and still
dropping. If the competition ended now, half our final score is a 944.

**Our strongest settled agents (NOT currently in the pair):**
- 53564198 `vetorf4p_seq_strength` μ **1280.0**
- 53547475 `vetorf2p_ffa` μ 1291.9 (peak; older)
- 53595717 `…shotmlp015` μ 1257.6 (in pair)

**Why this is NOT a now-fire and needs coordination (not unilateral action):**
- It only bites AT the deadline; multiple sessions keep submitting.
- A single resubmit evicts the OLDER of the two (the 1257 probe) and
  leaves the 944 → makes the pair *worse* short-term. Flushing the 944
  needs TWO submits, and a resubmit restarts at μ≈600 (needs ~24 h to
  settle).
- The 944 is the RL/happy-babbage session's experiment — evicting it is
  cross-session (Rule 42 push-claim board).

**Recommended endgame play (PI / cross-session to orchestrate):** in the
last ~2 days before 06-23, deliberately make the final two submissions our
two best **settled** agents (e.g. rebuild the 1280 config + one more
strong distinct agent), submitted ~24 h apart so both settle before the
deadline. Until then, every session should be aware that sub-1000 probes
left as the newest submission are a final-eval liability.
