# Secure-variants probe — 2026-05-14/15 wrap

Branch: `claude/fix-weak-game-starts-NhDQ3`.

## What this session investigated

PI directive: "weak starts of geo v3.1." Live geo v3.1 (#52643676)
settled at **985.5 μ live**, ~77 μ below v7_pv (1062.2 μ, still on
rolling-last-2). Local A/B predicted +7pp 2P / +31pp 4P — local-
overpredict-2x flag fired (5/12 v3.5.1 −150 μ; 5/14 geo v3.1 −80 μ).

## Diagnostic (7-step framework)

### Loss-mode classification on 52 live replays

`scripts/classify_losses.py 52643676` →
`audit/2026-05-13-loss-modes-52643676.csv`:

| Bucket | geo 5/14 | v7_0 5/13 | Δ |
|---|---:|---:|---:|
| `opening_lost` | 33 % (8/24) | 68 % (34/50) | **−35 pp** |
| `mid_economy_lost` | 67 % (16/24) | 32 % (16/50) | **+35 pp** |

**Reframe.** geo's openings are *not* weak. The opening_boost tilt
(steps 0–15, ×2.0) cut opening_lost share by 35 pp vs v7_0. The
bottleneck rotated to steps 30–200.

### Capture-and-secure cross-tab

| | Won (n=27) | Lost (n=25) |
|---|---:|---:|
| Captures per game (median) | 45 | 22 |
| `lost_back_rate` (captured planets re-flipped to enemy) | 30 % | **100 %** |
| Median turns held before flip-back | — | **13** |

In every lost game geo loses **~100 %** of captured planets back to
the enemy within ~13 turns. The mid-game ship collapse is a
*consequence*: no production base because we don't hold ground.

### PI's two additional observations

1. **Bounced fleets** (ships sent at targets we don't capture):
   ships_per_capture 98 in 2P wins vs 109 in 2P losses → **+11 % gap**.
2. **Orbital-drift losses** (planets sweep into enemy halves): 67 %
   of planets orbit; orbital share of cap-then-lost events is 73.7 % →
   **1.10× over-representation**, modest.

Both real, both ~10 % effects compounding into the dominant
"capture-then-lose" pattern. Underlying mechanism in code: geo's
mission scoring (`opening.py` `(remaining)^1.5/(d+1)`, snipe ROI)
uses **static current-step geometry**. No proposer or scoring path
in `lib/missions/*.py` or `lib/geo/sense.py` looks up
`omega + initial_planets` to forecast position, even though
`lib/aim.py` already does that math for fleet targeting.

### Missing wirings in geo

- `lib/missions/recapture.py` — built, defaults corrected after
  the 5/11 v3.5 revert (`RECAPTURE_SCORE_DENOM_MATCHES_SNIPE=1`,
  `RECAPTURE_TOPK_PER_TURN=5`), **not imported by geo**.
- `lib/missions/drain.py` — H12 source-emptying, **not imported by
  geo**.
- Reinforce IS wired but reactive-only: only fires when WorldModel
  predicts a flip.

## Variants built + tested (3-opp panel, gate Wlo ≥ 0.50)

Panel: v7_0, v4_planner, v3.5.1 (`DEFAULT_PANEL`).

| Variant | vs v7_0 | vs v4_planner | vs v3.5.1 | Verdict |
|---|---|---|---|---|
| **geo_recap** | **64.1 % PASS** Wlo .518 | 56.2 % INCONCL Wlo .441 | **62.5 % PASS** Wlo .503 | **2/3 PASS, mean 60.9 %** |
| geo_garrison | 42.2 % INCONCL Wlo .309 | 56.2 % INCONCL Wlo .441 | 56.2 % INCONCL Wlo .441 | flat ~52 % |
| geo_drift | 32.8 % **FAIL** Whi .450 | 31.2 % **FAIL** Whi .486 | 43.8 % INCONCL Wlo .323 | **panel FAIL (worst .180)** |
| geo_all | 26.6 % **FAIL** Whi .385 | (running) | (running) | **panel FAIL on v7_0** |

`geo_all` failed vs v7_0 with 26.6 % — combining recap+garrison+drift
is **worse than any single axis**. The drift component poisons the
other two; this is a confound, not an independent axis-falsification
(Rule 37 still at 1, not 3).

## Falsified

- **`drift-discount` scoring axis.** Multiplying base mission scores
  by a `hold_prob` (predicted future-Voronoi share) regressed 32 % on
  v7_0, 31 % on v4_planner. Same family-of-regression as the v7_1–v7_7
  chooser-axis variants and v3.0's composite-value-head: adding scoring
  terms *in front of* the K=10 lookahead consistently hurts. The K=10
  rollout already prices keepability via its own value head.
- **`geo_all` combined.** Drift's drag dominates. Not retestable as
  written; if recap+garrison without drift is wanted, that's
  `geo_recap_garrison` — not built this session.

## Confirmed

- **`geo_recap`** (`agents/geo_recap/main.py`) — wiring
  `propose_recapture_missions` into the base mission pool. 2-of-3
  PASS on the 3-opp panel (v7_0, v3.5.1), v4_planner at 56 % point
  estimate (CI brackets 50). Mean 60.9 % across 192 games.
- The corrected recapture defaults are validated — they were never
  ladder-confirmed before; the 5/11 v3.5 revert came from the
  pre-correction defaults.

## Submission verdict

**Did NOT submit this session.** The reasoning:

1. `geo_recap` is the only PASS-2-of-3 single-axis variant. Mean panel
   60.9 %; minus the calibrated local-overpredict-2x discount (~6–7 pp),
   expected live ~54 %. Likely above geo v3.1's 985.5 μ floor but
   not *decisively* above v7_pv's 1062.2 μ.
2. **Every push evicts v7_pv** (rolling-last-2 rule for code-comps,
   CLAUDE.md Rule 12 caveat). Eviction of our best ladder slot for
   an agent unlikely to clearly exceed it is negative-EV.
3. Calibration warning still active: panel passes don't translate to
   ladder µ without the discount. 32 seeds vs each opponent is the
   minimum local sample for a v4_planner-class CI, but v4_planner
   itself was the INCONCL opp — i.e., the closest call.

## Pointers

- `audit/2026-05-13-loss-modes-52643676.csv` — bucketing data
- `audit/live-episodes/52643676/summary.json` — winrate / opponents
- `audit/2026-05-14-secure-variants/panel.log` — full panel log
  (gitignored; in-progress on branch)
- `audit/2026-05-14-secure-variants/resume_panel.sh` — detached
  launcher (survives session boundaries)
- `agents/geo_recap/main.py` — the PASS variant
- `agents/geo_drift/main.py`, `agents/geo_garrison/main.py`,
  `agents/geo_all/main.py` — falsified / inconclusive; kept on
  branch for record, **do NOT merge to main** without the JAX
  speedup that would let drift's hold_prob be re-priced against a
  cheaper rollout.

## Merge-to-main decisions (recommend to PI)

**Merge:**
- `agents/geo_recap/main.py` — PASS variant; future iteration base.
- `audit/2026-05-13-loss-modes-52643676.csv` — diagnostic backing.
- `audit/live-episodes/52643676/summary.json` — already gitignored-
  with-exception; size negligible.
- `audit/2026-05-14-secure-variants/resume_panel.sh` — reusable
  detached-launcher pattern.
- `audit/2026-05-15-secure-variants-wrap.md` — this doc.

**Do NOT merge:**
- `agents/geo_drift/main.py` — panel FAIL (worst Wlo .180). Falsified
  axis. Keep on branch for postmortem reference; orphan from main.
- `agents/geo_all/main.py` — panel FAIL on v7_0; drift-contaminated.
  Orphan from main; the "all without drift" variant would be a fresh
  build, not this code.
- `agents/geo_garrison/main.py` — 3× INCONCLUSIVE around 50 %. Not
  strong enough to clutter main; revisit when recap is on the ladder
  and we have live-replay data on whether garrison would *add* on top.
- `audit/2026-05-14-secure-variants/panel.log` — already gitignored.

## Next-session first-action (ranked by EV / cost)

1. **Bundle and submit `geo_recap`.** Single-file bundle via
   `scripts/bundle_agent.py`. 3-opp panel parity gate before push
   (the resume_panel.sh pattern, single-variant). Eviction call: it
   will replace v7_pv. Push when PI gates it explicitly. ~20 min.
2. **JAX-port `score_candidate` inside the winner.**
   `lib/game/jax/jax_score.py` has `score_candidate_jax_pure_jit`
   (~6 ms after JIT vs ~80–100 ms Python; 30–70× claim from the
   5/13 sub-phase 1 audit). The repo has `agents/jax_v7_0/main.py`
   as an integration template. CPU-only on the ladder (the JIT-fused
   pipeline is the win, not GPU). Costs: ~1–2 h port + Rule 2 + Rule 30
   two-tier smoke; risks: first-turn cold-compile, ~150 MB bundle
   size (kernel-push path, not single-file), float-parity drift
   (`1e-3` tolerance in JAX phase tests vs scalar env). Payoff:
   per-turn drops from ~500 ms to ~10–80 ms, frees budget for K=15+
   search and depth-2 maximin. **Parked all session because the
   capture-and-secure diagnostic was higher-EV; this is the right
   next move once recap is on the ladder.**
3. **Re-test garrison on top of recap as `geo_recap_garrison`.**
   geo_garrison was tested in isolation against unchanged geo; the
   combined "recap captures lost ground, garrison holds new ground"
   has plausible additivity. Single-axis A/B vs the recap-only
   baseline. ~30 min.
4. **Investigate WHY drift discount regressed.** Hypothesis: the
   hold_prob is correct but the K=10 rollout already prices it
   through ship-delta — pre-discounting double-counts and starves
   legitimate captures of score. Probe: instrument hold_prob
   distribution + which missions get dropped on live geo_drift
   self-play and compare to which the rollout would have validated.
   Cheap (~1 h). Either resurrects the drift axis with a better
   shape, or buries it.
