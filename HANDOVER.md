# HANDOVER.md — next-session brief

> Refreshed 2026-06-14. The prior version (and `state/STRATEGY.md`) were
> ~9 days stale — they described a single-strategy lock at ~1170. That is
> NOT where we are. Full cross-branch review:
> `knowledge-base/thoughts/2026-06-14-cross-branch-state-and-consolidation.md`.

## State of play (2026-06-14, ~9 days to the 06-23 deadline)

- **Public leaderboard: rank #137 / 4,447 (top ~3%), score 1261.7.**
- The canonical line is **producer-plus** (heuristic, built on the vendored
  public Producer agent). It lives on branch **`claude/awesome-clarke-ixy57v`**,
  NOT on main — main is 337 commits behind. Best agents: the 2-player veto
  build (~1291, our peak) and the 4-player-strength build (~1280).
- Three other branches ran learning tracks (reinforcement learning,
  imitation) and a diagnostics branch; all the learning agents settled
  ~937–1018 and were self-assessed as capped. **PI decision: producer-plus
  is the line; the learning tracks are local research only, no more
  submissions from them.**

## The one operational risk: the rolling-evaluation pair

Kaggle keeps the **two most-recent submissions by time** for final scoring.
Right now that pair is a **937 reinforcement-learning experiment** (newest)
+ the **1261 producer-plus agent**. Our best agents (1280, 1291) are
evicted. **PI decision: do not burn a submission just to reclaim the slot —
the next producer-plus improvement reclaims it for free. Lock the two best
settled agents as the pair in the last ~2 days before the deadline.**

## Working with the producer-plus line

It lives on `claude/awesome-clarke-ixy57v`. Use a git worktree (the bundler
is worktree-safe):

```
git worktree add --detach /home/user/awesome-clarke-wt origin/claude/awesome-clarke-ixy57v
cd /home/user/awesome-clarke-wt
python scripts/bundle_producer_plus.py --variant vetorf2p_ffa   # canonical best
python fast.py play submissions/producer_plus_vetorf2p_ffa_on.py --vs producer --seed 7
```

Read `state/STRATEGY.md` ON THAT BRANCH for the full producer-plus spec,
the variant list, the 9 rejected 2-player mechanisms, and the 4-player
loss anatomy. The competitive frontier is the **4-player axis** (60% of
volume, our win rate ~29% vs ~63% in 2-player) — hard and heavily mined.

## Environment (fixed this session — Rule 38)

- `cffi` added to `requirements.txt` — without it the simulator import
  panics and `make("orbit_wars")` dies. This is the universal fix; it is
  now persistent.
- **Producer-plus needs `torch`** (Producer engine), which the repo never
  declared. Install CPU torch in any producer-plus session:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
  Fold this into the bootstrap when producer-plus is consolidated onto main.

## Decisions queued for the PI

1. Consolidate producer-plus onto main (337-commit merge) for one canonical
   branch + bootstrap, or keep awesome-clarke as the working branch?
2. Vendor "Producer V2" (new public agent, 196 votes) as the A/B yardstick,
   replacing the stale Producer V1 the old A/B was steering against?
3. Next 4-player lever, or accept near-saturation and spend the remaining
   days on floor/endgame discipline?

## Pointers

- `knowledge-base/thoughts/2026-06-14-cross-branch-state-and-consolidation.md`
  — the full review (branches, leaderboard, meta, floor, env fixes).
- `state/STRATEGY.md` (on `claude/awesome-clarke-ixy57v`) — the producer-plus spec.
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42), fill before any submit.
- `CLAUDE.md` — process rules.
