# state/hypothesis-board.md — open agent-design hypotheses

## Open — consolidation branch wrap (2026-05-12 EVE)

### Lead hypothesis: a 100%-accurate pure-Python game rebuild is the next-tier substrate

> Plan target: research-first session before any code lands.

`lib/fast_sim.py` already bypasses ~99% of the Environment overhead but
still calls `kaggle_environments.envs.orbit_wars.orbit_wars.interpreter()`
for physics. The next phase replaces that call with our own
re-implementation, parity-tested against the recorded live episodes in
`audit/live-episodes/`. Expected wins:

- 2-4× another speedup over `fast_sim` (no `Struct` boxing, no package
  import overhead).
- Vectorisation hook: a numpy-batched version can roll N independent
  rollouts in parallel inside one process, opening the door to wider
  candidate enumeration in `lib/v7_search.py`.
- Independence from `kaggle_environments` updates — current package
  releases have already broken parity once via `Planet` namedtuple
  changes.

**Not yet started.** First session task is a design doc + parity rig,
not code. See `HANDOVER.md` for the research questions.

### Tactical hypothesis: v7_0_drop_one's σ band tightens further over the next 24 h

v7_0_drop_one is at 64 evaluation episodes (Score 1094.9). TrueSkill σ
shrinks ∝ 1/√N. By the next session the σ band should fall from ~6 to
~4 Score points. **Do not push a new submission that would auto-evict
v7_0_drop_one** until the next agent's expected gain is convincingly
above that band.

### Falsified this session

- **Aggressive snipe sizing (v3.5.1) generalises to live ladder.**
  Local 32-seed 68.8% Wilson lo 56.6% PASS. Live: 945.6, **regression
  of −60 vs v3_snipe**. Lesson: σ-equiv-base agents draw tightly
  against v3_snipe but lose against the broader ladder. Future local
  gates should panel against ≥ 3 distinct opponent classes, not just
  the in-family baseline.
- **σ-equivariance helps in drop-one regime.** v7.6 bisect: σ-equiv
  layer regresses drop-one architecture by −54 pp. σ-equiv stays in
  the v3_snipe / v7_minimax / v4_planner lineage but is REVERTED out
  of v7_0_drop_one.

---

## Older / archived

### 2026-05-10 — Phase 1 manifold hypothesis: partial refute

> Plan: `/root/.claude/plans/read-the-handover-next-imperative-whisper.md`.
> Audit: `audit/2026-05-10-phase1-manifold-verdict.md`.
> Reports: `audit/manifold/20260510T141114Z/` (7-class),
>          `audit/manifold/20260510T141409Z/` (5-class — gate target).
> Capture: `audit/replays/20260510T132957Z/` (1568 games, gitignored).

The user's hypothesis "competitor strategies live on a small-dim
manifold so a short prefix is informative enough to identify a class"
**partially confirmed** at 32 seeds × 5-strategy zoo with 15
hand-designed features:

- `weakest` (89.7%), `enemy_first` (83.4%), `baseline` (95% in 7-class)
  sit in their own basins — broad-class routing works.
- `nearest`, `production`, `roi` form a single "production-aware-
  greedy" basin with mutual confusion 12-17%; our 15-feature
  fingerprint can't separate them at K ≤ 200.
- Best 5-class score: RF 80.5% / LR 80.6% at K=100. Gate target was
  90%; **gate ❌ NOT cleared.**

**H-coarsen-labels (open, unranked):** merging the ROI-family into
a single class `production_aware_greedy` likely lifts RF to ≥92%
at K=100. Lets a 3-class meta-router proceed (broad-class routing
is what the panel actually needs — there's no submission incentive
to distinguish ROI-family members because ROI dominates them all).

**H-richer-fingerprint (open, queued behind H-coarsen):** adding
target-distance/production distribution-shape features + early-vs-
late split + target-id Shannon entropy plausibly separates the
ROI-family at K ≤ 100. Bumps `FEATURE_VERSION` to 2.

**H-learned-embedding (parked):** Grover et al. ICML 2018 protocol —
last resort if H-coarsen and H-richer-fingerprint both fail.

### 2026-05-10 — simple-strategy panel (target-selection ablations)

Five strategies under `agents/simple/` share v1.1's mechanism stack
(`[validate, arrival_size, lead_aim]`); they differ only in the score
function for picking a target. Run via
`python -m scripts.strategy_panel --seeds 32` for confidence;
`--seeds 8` for quick iter. Plan:
`/root/.claude/plans/read-the-handover-next-imperative-whisper.md`.

**8-seed smoke results (audit/tournaments/20260510T123059Z.json):**

| Strategy      | Hypothesis (one-liner)                                                  | Mean panel winrate | vs v1_orbitfix | Verdict (8-seed) |
| ------------- | ----------------------------------------------------------------------- | ------------------ | -------------- | ---------------- |
| `roi`         | production / distance is the right travel-adjusted ROI signal           | 96.9%              | 100% (16/16)   | ✅ strong        |
| `production`  | highest-production target beats nearest                                 | 75.0%              | 69% (11/16)    | ✅ confirmed     |
| `nearest`     | (control) reproduces v1's distance-greedy under the shared stack        | 56.2%              | 19% (3/16)     | ≈ tied with v1   |
| `enemy_first` | pressure-on-opponent beats economy                                      | 32.3%              | 12% (2/16)     | ❌ refuted       |
| `weakest`     | cheap snipes dominate                                                   | 15.6%              |  0% (0/16)     | ❌ refuted       |

**Open verdicts pending 32-seed confirmation:**
- H-roi-32: confirm `roi`'s 100% beat over v1_orbitfix holds at 32 seeds.
  If Wilson lo ≥ 0.6 over 32 seeds, `roi` is a v1.2 submission candidate
  (subject to roadmap submission economy: rolling-last-2 means do NOT
  push until v1.1's live μ has settled).
- H-production-32: same for `production`'s 69% beat — narrower margin,
  needs the seed bag to confirm or invert.
- H-nearest-vs-v1: nearest using DEFAULT_MECHANISMS is statistically
  the same agent as v1_orbitfix; the 19/81 split observed at 8 seeds
  is within the seat-asymmetry noise floor (sp 2/1/5 in own self-play
  cell). Confirm at 32 seeds — if it does diverge, dig into RNG seed
  ordering or whether `propose_intents` mirrors v1's exactly.

### 2026-05-10 — research-driven extensions (game-strategy-research-8w7EO)

> Source: `docs/strategies/heuristics-research.md` (universal heuristics
> + 15-item brainstorm). Plan: `audit/2026-05-10-research-driven-next-experiments.md`.
> All six hypotheses are pre-registered with explicit decision gates;
> H9 in particular is the immediate-priority Axis 0 (Phase 1 path-A
> meta-router) and operationalises §F (compete-relative) on top of the
> Phase 1 fingerprint infra already on main.

- **H4 (research-note §E.3 — Axis 3 in prep doc):** Multi-source
  simultaneous-arrival timer beats `roi` standalone by ≥55% over 24
  seeds × both sides. Combat resolver groups same-owner same-step
  arrivals; timing N source planets to land on the same step gives
  combat mass without speed loss. Predicted to become v3 mission class
  `gang_up` (roadmap mechanism 4) if H4 holds.
- **H5 (research-note §B.2):** The production-dominance lock predicate
  fires before step 200 in <10% of self-play games but, when it does,
  switching to defense-only retains the win in ≥95% of cases.
  Diagnostic-only at v2; policy-actionable at v3.
- **H6 (research-note §F.3):** A spoiler-vs-leader rule in 4P FFA
  improves μ by ≥30 vs the always-expand baseline over a 60-game
  panel. Parked until 4P-FFA panel infra exists
  (`scripts/strategy_panel_4p.py`).
- **H7 (research-note §D.1 — Axis 4 in prep doc):** Front-loading
  neutral capture by re-weighting ROI in steps 0–60 by
  `(500 − step − eta)^1.5` (vs linear) gains ≥3% winrate over `roi`
  baseline.
- **H8 (research-note §G.6):** Replacing greedy per-source-best with
  Hungarian-assignment solver gains ≥2% winrate at <100 µs added
  per-step cost. Future axis (heavier than current panel agents).
- **H9 (NEW — Axis 0 in prep doc):** A 3-class meta-router (Phase 1
  path-A relabel: nearest+production+roi → `production_aware_greedy`)
  with the BR table from `heuristics-research.md` §K.5 beats `roi`
  standalone by ≥3 μ on a 60-game zoo panel including
  `[weakest, enemy_first, baseline, roi, v1_orbitfix]`. Builds on the
  existing `lib/fingerprint.py` v1 (15 features) and
  `scripts/manifold_check.py` already on main; one-line `--label-merge`
  re-run; predicted RF ≥ 92% at K=100. **Recommended path A on §F's
  rank-aware-override grounds** — see `heuristics-research.md` §K.4
  for the rationale vs paths B and C.

### 2026-05-10 evening — physics-correctness + look-ahead north star

> Source: PI voice-dump captured in
> `knowledge-base/thoughts/2026-05-10-pi-direction-physics-then-lookahead.md`.
> Two halves of the same plan.

**Half 1 — accuracy (ships don't die to wrong physics):**

- **H-lead-eta (LANDED, neutral A/B):** lead_aim ETA was overestimated
  by `r_src + r_target + 0.1` because the fleet spawns just outside src
  and captures when entering target.radius. 32-seed A/B vs live: 47/0/53
  (Wilson lo 0.35, hi 0.59 — tied within noise; 0 draws). Commit
  `cbf142b`. Expected lift surfaces against varied ladder opponents,
  not self-vs-self.
- **H-sun-arrival (punch #7, open):** Re-promoting `sun_avoid` to
  DEFAULT_MECHANISMS regressed in two ablation attempts because the
  mechanism checks `path_clears_sun(src.center, target.xy_current)`
  but the fleet flies to the lead-predicted arrival point. Fix: pass
  the same `predict_relative(target, omega, eta)` arrival point.
  Mechanism + strategy-side pivot share one helper.
- **H-lead-3iter-with-eta (punch #8, open):** 3 fixed-point iterations
  alone regressed 42/52 in 32-seed A/B. With the ETA correction in
  place the fixed point is different — quick re-test queued.
- **H-capture-success-probe (open):** Instrument a roi run; count per-
  fleet outcome (declared target reached / died in sun / out of
  bounds / unreached at episode end). The first direct measure of
  whether physics fixes change game outcomes at scale.

**Half 2 — look-ahead + global decisions (the ceiling):**

- **H-fleets-in-flight (open, high priority):** Today's strategies all
  ignore `obs.fleets`. First use case: don't double-commit a source's
  garrison to a target that already has our fleet arriving with enough
  ships. Lives in a new `arrival_ledger` mechanism (propose only if
  `mine.ships > target.ships_at_arrival − our_already_arriving`).
  Second use case: predict enemy fleet arrivals; defend or counter.
- **H-roi-threshold (open):** Pre-filter the target list (top-K
  globally and/or per-source absolute threshold) before any later
  solver runs. Reduces branching for the joint solver and any search.
  Cheap to ship; isolates the action space.
- **H-joint-assignment (open):** Replace per-source greedy with
  bipartite matching `(sources × targets)` maximising total ROI
  subject to garrison constraints. Hungarian / LP relaxation; fine at
  24 planets. Subsumes earlier H4 (gang-up timing) when paired with
  multi-source same-arrival-step coordination.
- **H-lookahead-search (deferred):** Beam search or mini-MCTS over
  pruned target set, scoring via short-horizon rollout against a
  fixed-policy opponent model. Only after the pieces above are stable.

### 2026-05-11 — top-performer replay analysis (H10-H15)

> Source: `knowledge-base/concepts/top-performer-strategies.md`. 50
> top-10 replays + 10 midpack replays fingerprinted. The fingerprint
> features where top-10 differ ≥1.5× from midpack are the H10-H15
> targets. See doc for fingerprint table + per-team profiles.

- **H10 (high-EV, ~3 days):** **Enemy-target multiplier in v3_snipe ROI.**
  Top-10 picks enemy targets at 32% vs midpack 14% (×2.3 gap). Multiply
  `target.value` by 1.3-1.5 when `target.owner ≠ ourselves AND
  target.owner ≠ -1`. Decision gate: 32-seed 2P vs v3_snipe ≥55%
  Wilson-lo; 4P FFA parity-or-better.
- **H11 (medium-EV, ~5 days):** **Opening-only first-fleet rule.**
  Top-10 first-launches at step 4.1; midpack at step 10.5. The 6-step
  gap forfeits ~30 ships of production. New Mission class
  `opening_landgrab` fires once if `step <= 5 AND ours.ships > 8` from
  every near-home planet. Decision gate: 32-seed 2P winrate vs v3_snipe
  ≥55%; first-launch-step measured ≤3.
- **H12 (medium-EV, ~5 days):** **Source-emptying `drain` mission.**
  Top-10 mean garrison-at-launch = 11; we leave ~25. Trigger when
  `ours.ships > 30 AND no incoming enemy fleet within ETA+5`. Decision
  gate: mean_garrison_at_launch drops from ~25 to ≤15 without
  fleets_lost_to_enemy_recapture rising by >2 per game.
- **H13 (high-EV, larger build, ~10 days):** **Multi-source same-turn
  arrival as Mission class (`swarm` / `gang_up`).** yijue1 and
  yuriygreben both ship this; v3_snipe is per-source greedy. In
  settle_plan, after per-source scoring, run second pass on pairs whose
  ETAs to the same target are within ±2 turns; bonus if combined ships
  > predicted-defenders-at-arrival. Decision gate: gang_up_rate rises
  from ~35% to ≥50%; 32-seed 2P winrate ≥55%.
- **H14 (high-EV, ML workstream, ~15 days):** **konbu17-style shot
  validator.** A small numpy MLP (24-dim input, 32-16-8-1, sigmoid)
  drops shots predicted to fail. Train from our replay corpus, label
  by "target was ours 10 turns later." Conservative: only reject,
  never propose. konbu17's hybrid wins 84% vs 65% pure rule-base
  (+19pp panel). Decision gate: +5pp panel winrate over v3_snipe;
  no regression vs Roman-1224.
- **H15 (cheap probe, ~½ day):** **Drop comet chasing entirely.**
  Top-10 comet-capture rate = 3.4%; midpack = 13.4%. emanuellcs
  formalises a break-even filter (`eta + cost/prod < remaining_path`).
  Hard filter on comet targets. Decision gate: panel winrate
  parity-or-better with v3_snipe; comet_capture_rate ≤ 5%.

### 2026-05-13 — discussion + dataset re-read (H16-H29)

Source: `/root/.claude/plans/taking-the-role-of-buzzing-rossum.md`
(approved round-2). Each hypothesis maps to an `Idea X` letter in that
plan; full file:function adapt points + test routes are there.
A-G = scalar pipeline tweaks; H/N = read-only audits already cleared;
I-M = data-driven; K = realism check already cleared.

- **H16 [A] (½ day):** **Present-value target valuation.** Replace
  the linear `(500 − step − eta)` horizon factor in
  `lib/missions/snipe.py:7-9` (and the mirror in
  `lib/missions/reinforce.py:94`) with a geometric series
  `γ^eta · (1 − γ^(horizon−eta+1)) / (1 − γ)`, γ=0.99.
  Source: TID 699003 author claims this shape alone hits ~1000 μ.
  Decision gate: `ab_variants.py --candidate pv` vs
  `{v3_snipe, v4_planner, v7_minimax}` Wilson-lo ≥ 0.55 each.
- **H17 [B] (½ day):** **3-closest-planet hardcoded danger map.**
  `_danger_score(pid, world) = Σ_{p ∈ 3-nearest} w[p.owner]/(d+1)`,
  `w = {ally: +1, neutral: 0, enemy: −1}`. Multiplicative penalty on
  snipe + reinforce score. Source: TID 699003 (16-0 vs gradient
  variant). Decision gate: 32-seed multi-anchor as above.
- **H18 [C] (~1 day inc. audit):** **Comet arrival synchronization.**
  Audit `lib/trajectory.py:predict_fleet_fate` for comet motion
  awareness; extend ray-cast to advance comet path-index per step.
  Source: TID 697397 Day-2 finding. Cross-check with
  `scripts/lookahead_probe.py` on spawn boundary turns
  (50/150/250/350/450).
- **H19 [D] (½ day):** **1.1× fleet-speed over-commitment.**
  Post-WorldModel scale step in `lib/mechanism.py:arrival_size`:
  bump non-reinforce intents to
  `max(ships_needed + margin, ceil(1.1 · ships_needed))`. Pair-eval
  with H12 source-emptying — they're in tension. Source: TID 697397.
- **H20 [E] (½ day):** **4P kingmaker multipliers.** Extend
  `lib/missions/snipe.py:_leader_pid` to flag leader (×1.5) and
  non-leader (×0.8) targets in 4P. FFA panel gate via
  `scripts/ffa_panel.py`. Source: TIDs 697397, 698659.
- **H21 [F] (½ day):** **Pre-reinforce against visible enemy
  arrival.** Extend `lib/missions/reinforce.py` to detect
  `enemy_arrival = our_capture + 1` patterns via
  `world_model.ledger.get(planet_id)` and propose a
  `pre_reinforce` mission at `eta = enemy_eta − 1`. Source: TID 698478.
- **H22 [G] (½ day, LANDED 2026-05-13):** **3-anchor Wilson gate.**
  Per-anchor Wilson-lo ≥0.55 instead of pooled. Catches non-transitive
  A>B>C>A loops at high μ. Done — `scripts/ab_variants.py
  --candidate NAME --gate-threshold 0.55`. Sources: TIDs 698478/698512
  + our own v3.5.1 live regression (`state/current.md:171`).
- **H23 [H] (audit, CLEARED 2026-05-13):** Y-axis convention audit.
  Result: clean. No render-style `+y down` flips anywhere in scalar
  engine, JAX engine, or orbit math. See
  `audit/2026-05-13-day-1-audits.md`.
- **H24 [I] (~3 days):** **Konbu17-style shot-validator MLP** trained
  on Bovard's CC0 top-10% replays. Uses existing
  `scripts/label_shot_outcomes.py` (24-d features, LABEL_BUFFER=10).
  Inline as `lib/missions/shot_validator.py`; weights as static numpy
  arrays in the bundle. Offline gate: val-AUC ≥ 0.92.
- **H25 [J] (~1 day post-data):** **Opening classifier** trained on
  `(initial_planets, angular_velocity)` + first-10 winner actions.
  Apply only at `obs.step ≤ 5` to set a cached opening recipe.
- **H26 [K] (probe, CLEARED 2026-05-13):** Eval-cost cgroup probe.
  Result: v7_0_drop_one single-threaded, scales linearly with CPU
  share. Extrapolated to eval's 0.6 CPU: p99 ≈ 444 ms, max ≈ 676 ms.
  Zero overage. See `audit/2026-05-13-day-1-audits.md`.
- **H27 [L] (~1 day):** **Arrival-window candidate enumerator.**
  Add `_enumerate_arrival_windows` mode to `lib/v7_search.py` —
  sweep ship-counts s.t. arrival ∈ [enemy_eta − 2, enemy_eta + 2];
  prefer smallest fleet landing ≤ enemy_eta. Combines with H18 / H21.
- **H28 [M] (~3 days post-data):** **Archetype meta-selector.**
  Cluster Bovard fingerprints to 3-5 archetypes;
  `lib/opp_model.classify_opponent_from_history(obs_history)` →
  archetype tag; `v7_search` picks K + value_fn per-archetype.
  Offline gate: 3-class accuracy ≥ 0.92 at turn 50.
- **H29 [N] (audit, CLEARED 2026-05-13):** Sun-gravity + fog-of-war
  zeroing audits. Both clean. See
  `audit/2026-05-13-day-1-audits.md`.

### Pre-existing seeds (carried over from Day 1)

- H-search: A search-based agent (MCTS over short horizons) beats a
  hand-coded heuristic on the baseline-opponent panel.
- H-rl-curriculum: An RL agent trained on self-play overfits to
  symmetric strategies and loses to rule-based opponents — needs
  opponent-curriculum diversity.
- H-replay-mining: **PARTIALLY CONFIRMED.** Replay statistics from top
  public-LB agents reveal a load-bearing tactic that no public notebook
  documents — see top-performer-strategies.md §6 (within-top-10
  archetypes) and §8 (what unpublished top-10 likely add on top).

## Killed

(empty — `weakest` and `enemy_first` are leaning falsified at 8 seeds
but stay in **Open** until 32-seed confirmation. Falsified entries get
moved here with the audit-JSON path attached.)

## Hedge ladder

> Per CLAUDE.md R2: PRIMARY = best-current; HEDGE = next-best agent
> that regressed ≤ defined-bracket on the rank ladder. Populate during
> the final 3-day window.
