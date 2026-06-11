# 2026-06-11 — RL track kickoff (overnight autonomous session)

PI directive (verbatim intent): take time to think and research, then
iterate all night to create a top leaderboard agent through
reinforcement learning; use Kaggle GPU and private datasets to save
data if necessary; iterate all night autonomously.

## What was built tonight (branch claude/happy-babbage-6j46p6)

A complete PPO self-play stack on top of the existing parity-tested
JAX game engine (`lib/game/jax/`):

- **Action design**: per owned planet, pick a target planet (or hold)
  plus a ship fraction (25/50/75/100%). A lead-aim intercept solver
  (closed-form orbits, comet path tables, ship-count-dependent fleet
  speed) converts choices to launch angles. Sun-crossing and
  comet-expired targets are masked before sampling.
- **Network**: 3-layer transformer (width 64, 120k params) over planet
  tokens + a global token; pointer-attention target head with a
  pairwise-geometry bias MLP; fraction head; value head. Runs in pure
  numpy at eval time (~64 ms/turn, budget is 1000 ms).
- **Training**: PPO + GAE, mirror self-play across all seats (2P and
  4P mixed 3:1 from a 1024-seed init pool), terminal win/loss reward
  + potential-based material-share shaping (planet+fleet ships +
  20×production, mine minus best rival over total).
- **League mode** (built, not yet used): half the envs play against
  frozen past snapshots or a scripted greedy bot to prevent self-play
  cycling. Tomorrow's lever.

## Hard-won facts

- The engine rotation uses the PRE-increment step counter: position
  after t ticks from step S is theta0 + omega*(S+t-1). Off-by-one here
  silently ruins every launch (caught by test before any training).
- Kaggle datasets auto-decompress uploaded tar.gz files; the kernel
  must self-locate the code root (mount path includes
  /kaggle/input/datasets/<user>/<slug>/ on script kernels).
- T4 OOM trap: anything param-independent (feature building, the
  48-step arrival scan, the aim solver) MUST be computed outside
  value_and_grad or XLA saves its activations for backward — 14.8 GB
  at batch 256 vs ~3 GB after the fix, bit-identical metrics.
- GPU throughput (T4, jax 0.7.2): 3.55 s per 4096 env-steps at batch
  128 including update → an 8 h run ≈ 30-45 M env-steps ≈ 60-90 k
  games. Local CPU: ~40× slower; fine for smokes only.
- kaggle_environments 1.30.1 import crashes on a broken system
  cryptography; `pip install --ignore-installed cffi cryptography`
  fixes it (session hook does NOT cover this yet).

## Live state at session pause

- Kaggle kernel `chrisleitescha/orbitwars-rl-train` **v5** running
  8.2 h (pushed ~22:10 UTC Jun 11): batch 256, rollout 32, minibatch 16,
  epochs 2, lr 3e-4, entropy 0.01, mirror self-play, ckpt every 20 min,
  greedy-bot eval probe every 25 iters. Expect COMPLETE ~06:30 UTC.
- Private dataset `chrisleitescha/orbitwars-rl-code` v3 carries code +
  1024-state init pool (+ later: resume checkpoints).
- Local CPU canary run in /tmp/rl_local_canary (3 h), old code path,
  stability watch only.
- Leaderboard context: our live pair 1281.6 / 1254 (producer_plus
  family, other branches); top-10 prize cutoff ≈ 1517; #1 = 1721.
  4 of 5 daily submissions used 2026-06-11; none used by this branch.

## Morning checklist (next session)

1. `bash rl/morning_pipeline.sh` — downloads kernel output, prints the
   wr_vs_greedy learning curve, evals final ckpt vs v7_0 / Producer /
   ledger_v1_4 (n=32 each side).
2. If learning is real but not champion-level: push dataset version
   including ckpt_latest.pkl (RESUME_CKPT=... bash
   rl/kaggle_infra/push_code_dataset.sh version "resume"), flip kernel
   args to --league, push v6 for the day run.
3. Only consider submission if Wilson-lo >= 0.50 vs the live-pair
   rebuild at n>=32 (Rule 45) + Rule 42/46 gates. RL exports go
   through rl/export_agent.py (340 KB single file, kaggle-env verified
   2P + 4P).
4. GPU quota: ~8.5 h used of 30 h/week after tonight.
