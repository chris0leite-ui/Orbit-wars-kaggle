# 2026-06-14 — Cross-branch state review + consolidation decision

Written after a full sweep of every branch, the live leaderboard, the
submission history, and the public-notebook meta. The on-main docs
(`HANDOVER.md`, `state/STRATEGY.md`) were ~9 days stale and described a
single-strategy lock at roughly 1170; reality had moved a long way past
that. This note records what is actually true on 2026-06-14.

## Where we actually stand

- **Public leaderboard: rank #137 of 4,447** (top ~3%), score **1261.7**.
- Top of the board is ~1750. The top-10 prize cut is ~1540 (≈280 above us).
- Deadline 2026-06-23 23:59 UTC — about 9.4 days out as of this note.

## The four branches that ran after the "single strategy" lock

Work did NOT stay on one strategy. Four branches ran in parallel through
June 12 (best settled rating on the live ladder in brackets):

- **awesome-clarke** — the heuristic "producer-plus" line. Best agents:
  the 2-player-tuned veto build (**~1291**, our peak) and the
  4-player-strength build (**~1280**). This is our strongest line by a
  wide margin. Built on the vendored public "Producer" agent
  (Slawek Biel, MIT) with our own gated mechanisms on top.
- **happy-babbage** — reinforcement-learning transformer, self-play
  league. Best ~937. Self-assessed as capped (behaviour-cloning aux
  refuted 0/16; moved to a clone-opponent league but still well behind).
- **elegant-dijkstra** — imitation learning cloning the Producer
  ("oracle", ~1018) plus a from-scratch rewrite ("ledger", ~980). Branch
  conclusion in its own audit: "offline self-play fine-tuning is
  structurally capped; reinforcement learning must be online or
  feature-driven."
- **blissful-cray** — diagnostics + a small shot-validator network (now
  closed as no-benefit). Raised the rolling-pair floor flag (below).

Read: the **heuristic producer-plus line is our best bet by ~270–340
rating points**. Two independent learning tracks each concluded the
offline approach is capped here — and a new public notebook literally
titled "Why Cloning the #1 Bot Loses to Greedy" says the same thing.

## The rolling-pair floor situation (the one operational risk)

Kaggle keeps the **two most-recent submissions by time** for final
evaluation — not our best. The current pair is:

- newest: a **937 reinforcement-learning experiment** (sub 53618099,
  06-12) — would rank ~#1,174 on its own.
- older: the **1261 producer-plus agent** (sub 53595717, 06-12).

Our two best agents (1280, 1291) are evicted and unprotected. The public
leaderboard shows 1261 (the better of the active pair), which implies the
displayed/final score is the best of the active two rather than a blend —
so the dead 937 is more an opportunity cost (~30 below our peak) than an
active drag, but the safe play covers both readings. **PI decision: do
NOT burn a submission just to reclaim the slot — the next producer-plus
improvement we ship reclaims it for free.** Lock the deliberate
two-best-agents pair in the last ~2 days before the deadline.

## New public agents / the meta we're matched against

The public notebook meta is dominated by Producer / ProducerLite
variants. Standouts: "The Producer V2" (Slawek Biel, 196 votes, 06-12) —
the dominant new public agent; ProducerLite logistics variants; a
self-contained 1266-Elo "V44"; a 1200+ "V2+Light". We have only ever
A/B-ed locally against **Producer V1**, and the awesome-clarke strategy
doc warns that the vs-V1 yardstick "steered us into the dribble meta"
(eleven straight mechanisms measured null on it). **Producer V2 is a
better, current yardstick and is unmeasured** — vendoring it (it's a
public MIT-style notebook) is a candidate next step, pending PI nod.

## The real competitive frontier: the 4-player axis

From awesome-clarke's own diagnosis: 4-player games are ~60% of ladder
volume, and our 4-player win rate is ~29% vs ~63% in 2-player. 82 of 83
4-player losses end with us eliminated around step 120, carved by 2+
opponents. It's "an extermination meta won by whoever wins the brawls,
not by farming"; the war-ledger law is "whoever reinforces more wins."
Several 4-player mechanisms have already measured null/regression
(strength-weighted objective, multi-tick projection, deficit
reinforcement). The most recent investigation concluded "FFA
re-weighting falsified as the lever." This axis is the open weakness AND
the hardest problem — any real gain likely needs a genuinely new model of
the mid-game brawl, not another knob.

## PI direction (2026-06-14)

1. **Consolidate on producer-plus.** Make it the canonical submission
   line; relegate the reinforcement-learning and imitation tracks to
   local research only (stop them submitting and evicting our best
   agents).
2. **Wait for a real improvement** before submitting — the next ship
   reclaims the floor slot.
3. **Attack the 4-player weakness** against the live meta.

## Environment fixes made this session (Rule 38)

The local box could not run anything on session start:

- **Simulator was dead:** `cffi`/`_cffi_backend` missing → cryptography's
  Rust binding panics → `kaggle_environments` (which eagerly imports the
  LLM werewolf env) fails to import → `make("orbit_wars")` dies. Fix:
  added `cffi` to `requirements.txt`. Verified `make("orbit_wars")` and a
  full game both work.
- **Producer-plus could not run locally:** the Producer engine needs
  `torch`, which the repo never declared. Installed CPU-only torch for
  this session. **TODO: fold torch into the bootstrap when producer-plus
  is consolidated onto main** (kept out of main's requirements for now
  because the producer-plus code isn't on main yet — main is 337 commits
  behind awesome-clarke).

Verified end-to-end after the fixes: built the canonical best config
(`vetorf2p_ffa`, 398 KB), played it vs vanilla Producer (seed 7) → win in
103 steps, per-turn timing p50=45ms / max=126ms (well under the 1000ms
budget).

## Open decisions queued for the PI

- Merge/consolidate the producer-plus line onto main (337-commit catch-up)
  so there is one canonical branch + bootstrap, or keep awesome-clarke as
  the working branch?
- Vendor "Producer V2" as the new A/B yardstick (replaces the stale V1)?
- Next 4-player lever to try, vs. accepting the line is near saturation
  and spending the remaining days on floor/endgame discipline.
