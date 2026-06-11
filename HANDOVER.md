# HANDOVER.md — next-session brief

## Mode (this branch: claude/happy-babbage-6j46p6)

**RL track** — PI directive 2026-06-11: build a top leaderboard agent
through reinforcement learning, iterating autonomously on Kaggle GPU.
Full context + morning checklist:
`knowledge-base/thoughts/2026-06-11-rl-track-kickoff.md`.

NOTE: the repo-wide docs below this branch (STRATEGY.md etc.) describe
the older `baseline_adaptive_k` strategy; other branches
(awesome-clarke producer_plus family, elegant-dijkstra ledger family)
have since moved the live pair to ~1280 μ. Read
`kaggle competitions submissions orbit-wars | head -5` for truth.

## Live status (RL track)

- Kaggle GPU kernel `chrisleitescha/orbitwars-rl-train` v5: 8.2 h PPO
  mirror self-play run pushed ~22:10 UTC 2026-06-11. Checkpoints land
  in the kernel output (download with
  `kaggle kernels output chrisleitescha/orbitwars-rl-train -p <dir>`).
- Code+pool dataset: `chrisleitescha/orbitwars-rl-code`.
- Morning command: `bash rl/morning_pipeline.sh` (download → learning
  curve → panel eval of final checkpoint).
- Export any checkpoint to a submission file:
  `python -m rl.export_agent <ckpt.pkl> <out.py>`.
- League continuation (anti-self-play-collapse) is built and smoked:
  add `--league` to the kernel args (see
  `rl/kaggle_infra/train_kernel.py`), include the resume ckpt in the
  dataset (`RESUME_CKPT=<path> bash rl/kaggle_infra/push_code_dataset.sh
  version "resume"`).

## Submission discipline (unchanged)

Rules 1/12/42/45/46 in CLAUDE.md still bind. RL agent submits only on
n≥32 Wilson-lo ≥ 0.50 vs the live-pair rebuild + Rule 42 board claim +
Rule 46 smoke. 5 submissions/day shared with the other branches.

## Pointers

- `knowledge-base/thoughts/2026-06-11-rl-track-kickoff.md` — RL track
  state, hard-won facts, morning checklist.
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42).
- `CLAUDE.md` — process rules.
