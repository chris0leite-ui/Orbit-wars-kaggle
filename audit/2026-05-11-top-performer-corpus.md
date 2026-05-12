# 2026-05-11 — top-performer analysis corpus index

Companion source manifest for `knowledge-base/concepts/top-performer-strategies.md`.
Lists every source consulted, with the URL and the on-disk path.

## Leaderboard snapshot

- **URL**: `https://www.kaggle.com/api/v1/competitions/orbit-wars/leaderboard/download`
  (Bearer auth via `KAGGLE_KEY` env var)
- **Pulled**: 2026-05-11 20:05 UTC
- **On-disk**: `audit/2026-05-11-lb-top30-snapshot.csv` (also raw zip at
  `audit/external/lb.zip`)
- **Rows**: 2528 teams; top-30 fully analysed.
- **Top-10 names + μ**: bowwowforeach (1682.9), flg (1598.6),
  Vadasz (1556.3), Isaiah @ Tufa Labs (1553.3), Ebi (1548.8),
  Shun_PI (1525.3), Erfan Eshratifar (1485.0), kovi (1468.2),
  sash (1440.9), 3Comets (1439.5).

## Submission-ID crawl

BFS frontier across the Kaggle API `/api/v1/competitions/submissions/{id}/episodes`
endpoint. Bootstrap from our v2 submission ID 52532938 (64 episodes,
102 unique opponents). Hop-2 hit 20 top-30 teams across 1861 unique
submissions. Hop-3 hit all top-10 teams via crawl through Vadasz +
Galatea + Erfan submissions. Final result: latest submission ID per
top-10 team known.

- `audit/external/episodes/crawl-frontier.json` (top-30 + rank-1 hop)
- `audit/external/episodes/crawl-hop2.json` (rank-2 BFS)
- `audit/external/episodes/crawl-hop3.json` (rank-3 BFS, full top-10)
- `audit/external/episodes/sub-<id>.json` (per-submission episode lists)

## Replay corpus

50 replays for top-10 (5 each: 3 × 2P-wins + 2 × 4P-wins) + 10 midpack
control replays from our v2's live ladder. Total ~305 MB.

- **URL pattern**: `https://www.kaggle.com/api/v1/competitions/episodes/{episode_id}/replay`
- **On-disk**: `audit/external/replays/r<rank>-<team>-<2P|4P>-<W|L>-<episode_id>.json`
  and `audit/external/replays/midpack-2P_vs_v2-<size>P-<episode_id>.json`
  (both gitignored)
- **Schema**: kaggle_environments standard — `steps[step][player_idx] =
  {observation, action, reward, status}`; obs has `planets`, `fleets`,
  `comets`, `comet_planet_ids`, `step`, `angular_velocity`,
  `initial_planets`. Action format `[from_pid, angle, ships, ...]`
  (only first 3 used for fingerprint).

## Behavioural fingerprints

- **15-feature base** (lib/fingerprint.py): `audit/2026-05-11-top-performer-fingerprints.json`
- **10-feature extended** (scripts/extended_features.py): `audit/2026-05-11-top-performer-extended.json`
  Features: first_launch_step, comet_capture_rate, fleets_lost_to_sun,
  fleets_lost_unknown, gang_up_rate, recapture_rate, flip_count,
  early_launches, mid_launches, late_launches, n_launches_total,
  n_steps_total
- **Per-team aggregates**: `audit/2026-05-11-top-performer-profiles.json`
- **Conversion script**: `scripts/fingerprint_external.py` (KE replay
  → flat fingerprint format → 15-feature matrix)

## Public kernels pulled

All via `https://www.kaggle.com/api/v1/kernels/pull?userName=X&kernelSlug=Y`.
Stored as JSON-wrapping-notebook at
`audit/external/kernels-pulled/<user>_<slug>.json` (gitignored).

| Author | Slug | Notes |
| --- | --- | --- |
| @konbu17 | orbit-wars-rule-base-ml-shot-validator-hybrid | **#1 in 50-agent tournament (85.4%).** ML shot-filter on top of rule-base. |
| @nina2025 | orbit-wars-two-bot-combine-v3 | #2 tournament (84.4%). Two-bot ensemble. |
| @emanuellcs | orbit-wars-meta-optimized-spoofing-agent | #3 tournament (83.3%). Adversarial 1-ship spoofing every 18 turns. |
| @emanuellcs | orbit-wars-ffa-mode-aware-strategist | FFA-specific tuning multipliers. |
| @yuriygreben | orbit-wars-physics-aware-architect | 5-layer architecture; WorldModel + simulate_planet_timeline. |
| @yijue1 | 1103-peaking-bot | Mission types: Capture/Reinforce/Snipe/Swarm. Self-described, 1103 peak. |
| @djenkivanov | orbit-wars-agent-ow-proto-passed-1-000 | 1080 peak; "max 1 fleet/tick/planet"; avoids comets completely. |
| @ykhnkf | distance-prioritized-agent-lb-max-score-1100 | 1100 peak; frontier_keep + opponent_priority weights. |
| @woosungyoon | how-to-create-a-baseline | (Roche Overflow #26 team member) Tutorial-style 5-mission builder. |
| @rahulchauhan016 | orbit-wars-target-score-2000-4 | Aspirational 2000 target. |
| @marcodg | marco-dg-v3-3-top-score-1060-5 | 1060 peak; #6 tournament. |
| @thisisn0mad | orbit-wars-rl-pipeline-public | Replicates konbu17 hybrid plus extras. |

Previously analysed (see `audit/2026-05-10-public-kernel-teardown.md`):
- @romantamrazov / orbit-star-wars-lb-max-1224 (μ=1224 published)
- @pilkwang / orbit-wars-structured-baseline
- @sigmaborov / lb-928-7-physics-accurate-planner
- @debugendless / orbit-wars-sun-dodging-baseline

## Discussion threads (Kaggle official forum)

Pulled via `https://www.kaggle.com/api/v1/competitions/orbit-wars/topics/{tid}/messages`.
Saved at `audit/external/discussions/topic-<tid>.json`.

| Topic ID | Title | Strategic content |
| --- | --- | --- |
| 697413 | Orbit Wars top-10% daily episode replay datasets | Bovard publishes daily replay datasets `bovard/orbit-wars-top10-episodes-YYYY-MM-DD`. Methodology: top-10% episodes by sum of post-episode rating. ~20 GiB cap. |
| 697725 | Sharing our RL lessons so far | JAX env rewrite at 10000 SPS. Entity transformer architecture. +1/-1 reward enough for 2P. PPO fails ~0% winrate against tier3+ public agents after 1000 updates. |
| 696219 | Day 1 Autonomous Research & Development Summary | "Aggressive Expansion Wins" — concurrent launches biggest performance jump. |
| 696043 | Fleet-tunneling-planet bug | Two-phase collision check has a gap: 13 tunneling events observed in one game. Exploitable physics quirk. |
| 698614 | 50-Agent Mega Tournament | Definitive panel ranking of all 50 strongest public agents. konbu17 hybrid wins. |
| 693755 | RL vs. Rule-Based — Which Will Dominate? | Forum debate; RL practitioners report failing ceilings vs strong rule-bases. |
| 692800 | Baseline method: Planet Wars AI Competition | Historical precedent (2009-era Galcon-style competition). |
| 698395 | Caution: Agent metadata may be publicly readable | Submission-id-based episode crawling is publicly accessible (we used this). |
| 692695 | observation inconsistency: initial_planets differs by player | Env quirk in some seeds. |
| 694188 | remainingOverageTime 60s | Overage budget = 60s total per game. |

## External narrative content

- **`dev.to/diven_rastdus_c5af27d68f3/your-ai-agent-evaluation-is-lying-to-you-why-10-test-runs-prove-nothing-1ij2`**
  — TrueSkill methodology post by an Orbit Wars competitor. Sample
  ladder: v22_timeline=907, v21_capture=842, romantamrazov=823.
  Persist ratings across runs; σ-floor 15; report confidence intervals.

- **`digitado.com.br/anyone-participating-in-orbit-wars-on-kaggle-50k-in-prize-money`**
  — Bovard (comp creator) confirms "The action space is HUGE, but I
  think very prune-able." Reddit r/reinforcementlearning thread
  (inaccessible to WebFetch in this sandbox; URL preserved for
  PI direct browsing).

## Tooling created

- `scripts/fingerprint_external.py` — converts KE replay format
  → flat fingerprint-compatible format → 15-feature matrix.
- `scripts/extended_features.py` — computes 10 extended features
  (opening tempo, comet rate, sun-deaths, gang-up rate, recapture rate)
  directly from KE replays.

## Tooling NOT created (deliberate)

- No re-pull of full 1860-submission BFS frontier into a graph. The
  3-hop pull was enough to locate top-10.
- No live submission of a probe agent (would consume a daily slot
  and was unnecessary given the BFS path worked).
- No re-runs of public kernels in a local panel. The 50-agent
  @marcodg tournament already published in topic 698614 is the
  comparison we needed.
