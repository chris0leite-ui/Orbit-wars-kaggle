# 2026-06-13 — the distilled-opponent league (PI-chosen RL direction)

## The diagnosis that drove the choice

Pure self-play + scripted-anchor league plateaus against a *self-
generated* metagame. Across v6→v7 the agent stayed perfect vs its
anchors (rusher/greedy 100%) but DROPPED vs the producer rebuild
(4/16→2/16). It learned aggression (peak fleets way up) but never
learned to DEFEND producer-style coordinated waves, because nothing in
its opponent distribution plays that way. BC-as-aux (v8) pulls toward
producer's style but can't teach beating it — you don't exceed a
teacher by imitating it.

## The unlock (PI picked "distilled-opponent league")

AlphaStar's "supervised agents seed the league": distill producer into
a FAST neural clone (BC, same 120k-param arch), freeze it, and put it
in the RL opponent pool. The learner then trains to BEAT producer-
style pressure at GPU speed. Solves the speed blocker (real producer is
Python ~80-400 ms/turn — can't be in the batched loop).

## Clone validation (local, 300 CPU BC steps)

- BC cross-entropy 4.01 → 1.18 (fits producer's target+frac choices)
- clone vs greedy 39.6% AND **peak fleets 149.7** — the clone launches
  ~150-fleet coordinated waves, the exact producer signature
  greedy/rusher (peak ~20-40) never produce. Style captured; 8000 GPU
  steps will sharpen imitation. The clone need not BEAT greedy — only
  PLAY like producer so the learner gets exposure.

## Engineering notes

- bc_loss featurizes ONCE outside the grad (ppo.featurize_bc +
  bc_loss_feats): the aim solver / 48-step arrival scan are param-
  independent; inside value_and_grad they recompile huge and would OOM
  (same trap as the PPO fix). BC step is now just net fwd+bwd.
- Two GPU kernel slugs now: orbitwars-rl-bc (clone pretrain, ~40 min)
  and orbitwars-rl-train (RL). They don't block each other.
- train.py league menu is a weighted draw over {scripted (greedy_frac),
  clone (clone_frac), snapshots (rest)} with empty-bucket fallbacks.

## Pipeline state (Sat ~10:15 UTC)

- Clone-pretrain kernel orbitwars-rl-bc v1 RUNNING (8000 steps).
- Next: download bc_net.pkl → verify (winrate + peak fleets + quick
  probe vs REAL producer to confirm style transfer) → ship into dataset
  → launch league kernel v9 (resume v7, --bc-opponent, clone-frac 0.35,
  + 0.1 BC aux).
- Live sub 53618099 (v7) warming ~955 μ.
- v8 (league+BC-aux) probe vs producer/live_garval running as the
  BC-aux data point.
- GPU quota fresh (Sat reset); BC (~0.7h) + league (~8h) fit easily.

## v8 negative result (BC-aux refuted, Sat ~10:30 UTC)

v8 = league + BC-aux (coef 0.3), resumed from v7. Sequential probe:
**0/16 vs producer, 0/16 vs live_garval** — WORSE than v7 (2/16).
Imitation as an auxiliary loss didn't help and likely HURT: pulling the
policy toward producer's action distribution diluted the self-play
strength without teaching it to win. Confirms the thesis — the lever is
"learn to beat producer" (clone opponent), not "act like producer"
(BC-aux). v9 keeps BC-aux at only 0.1 (light anchor); the engine is the
frozen clone in the opponent pool.

## Fallback levers if the clone-league stalls

1. Stronger/more-diverse clones: pretrain 2-3 clones on different
   producer variants (vanilla / garval / shotmlp) → multi-style league.
2. Value-reranked inference on top of producer candidates (RL-on-top).
3. RL-tune producer's ~12 numeric knobs (lowest-risk, improves the
   agent we'd actually submit to win).
