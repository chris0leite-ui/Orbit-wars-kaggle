# Plan — Producer-lite opponent model (build next session)

_Created 2026-06-04 (`claude/kaggle-submission-strategy-JzIAr`). Design locked, **not
yet built** (PI: lay out this session, start next). Read with
`audit/2026-06-04-producer-eval-observations.md` and
`knowledge-base/questions/2026-06-04-why-does-producer-beat-us.md`._

## Why

The vendored public **Producer** torch planner beats our line **~60% of the time**
(larger cross-branch measurement, 2026-06-04; the earlier n=16 triage read 13/16≈81%
but that was small-sample). It beat champion (μ~1170) and refine at the *same* rate in
triage → a **shared, chooser-independent loss mode** (our weak opponent anticipation +
opening hoarding).
Our chooser rolls out against a weak opponent model (Tier-0 `lite_greedy`). If it
instead anticipates a **Producer-like aggressive attacker**, our defensive timing and
capture-safety calibrate to the threat that's actually beating us → closes part of the
13/16. The same competitive-flow proxy is later reusable as our *offense* leaf score.

**Approach: fidelity first.** Build a faithful pure-python port of Producer's *attack
policy* as a drop-in opponent `Policy`; compact for speed only after it's faithful.

## Licensing (decided 2026-06-04, PI)

Code-reuse-with-attribution is cleared (comp rule §2.6 external public+equally-accessible
tools; notebook likely Apache-2.0). We re-implement (don't copy) and **attribute the
Producer notebook** (see `agents/producer/PROVENANCE.md`) in the module header + any
shared writeup. No licensing gate on this work.

## The seam (drop-in, no plumbing changes)

- **Policy type:** `Callable[[obs], list[[src_id, angle, ships]]]` (`lib/opp_model.py:55`).
- **obs schema** (rollout & live, dict-or-Struct, `lib/fast_sim.py:139`): `planets`
  (tuple `[id, owner, x, y, radius, ships, production]`), `fleets`, `angular_velocity`,
  `step`, `player`, `comets`, `comet_planet_ids`, `initial_planets`.
- **Gate:** add to `_select_opp_policy()` (`agents/baseline/chooser.py:34`) behind
  `BASELINE_OPP_TIER=2` (currently a stub → `top_tier_mirror`) or a new
  `BASELINE_OPP_MODEL=producer_lite`. Default OFF → champion bundle byte-identical.
- **Consumed:** `opp_actions_for_snap()` (`chooser.py:53`) calls the policy per
  opponent seat **inside the per-step rollout** → many calls/turn → must be ~Tier-0 fast.
- **CRITICAL:** `fast_sim.step` (`lib/fast_sim.py:364`) applies `[src,angle,ships]`
  through the **real env interpreter** → the **angle controls the trajectory**, a wrong
  angle makes the fleet miss. So producer_lite MUST emit accurate aim. Reuse
  `lib/aim.aim_orbiting(src_xy, src_radius, target_tuple, target_radius, ships, omega)
  -> (angle, arrival_xy, eta)|None`, `lib/aim.aim_comet(...)`, `lib/aim.estimate_eta`,
  `lib/fleet.speed(ships)` — all pure-python, no torch.

## Module layout

- **New `lib/producer_lite.py`** → `producer_lite_policy(obs) -> list`. Pure
  python/numpy, no torch. Attribution header.
- **Reuse:** `lib/aim` (aim_orbiting/aim_comet/estimate_eta), `lib/fleet.speed`.
- **Wire:** `agents/baseline/chooser.py:_select_opp_policy`.
- **Template to mirror:** `lib/opp_model.py:lite_greedy_policy` (155–233) — same
  dict-or-attr obs access (`p[0..6]`), same `moves.append([src_id, angle, ships])` return.

## The policy — faithful port of Producer's attack phase

**Config (mirror Producer 2P; 4P overrides in parens):** `H=18(13)`,
`max_sources=12(6)`, `max_offensive_targets=12`, `max_defensive_targets=4(2)`,
`max_waves_per_turn=6`, `roi_threshold=1.5`, `min_ships_to_launch=4`,
`capture_overhead=1.0`. (Producer `main.py:53-76`.)

Per call (`me = obs.player`):
1. **Parse.** sources = my planets (owner==me, ships≥4); offensive targets =
   enemy/neutral; defensive targets = my flip-risk planets.
2. **Incoming-threat per planet** from in-flight enemy fleets (cheap fleet→target
   assignment; reuse `world_model.fleet_target_planet` if affordable, else straight-line
   nearest-along-bearing). Feeds safe_drain + defender projection.
3. **`safe_drain(s)`** (compaction of Producer's exact recurrence,
   `planner_core.py:587`): `≈ ships_s − max_{t∈[1,H]}(incoming_hostile_s(t) − prod_s·t)`,
   floored 0, capped `ships_s`. The aggression signature — send all you don't need to hold.
4. **Shortlist:** top-12 sources by ships; top-12 offensive targets by min straight-line
   distance to any source; top-4 defensive (my planet with `incoming > ships+prod·eta`).
5. **Per (s,t):** straight-line `eta = estimate_eta` (cheap, for scoring); 
   `capture_floor(t,eta) = ceil(defenders_now + prod_t·eta + 1.0)` (Producer
   `planner_core.py:186`). Viable iff `safe_drain(s) ≥ floor` and `eta ≤ H`.
6. **Competitive proxy score** (compaction of Producer's exact `competitive_score` =
   `Δnet_me − ΣΔnet_opp`, `planner_core.py:75`):
   `score(s→t) = prod_t·(H−eta)·flip_mult − defenders_cleared`, `flip_mult = 2` if t
   enemy-owned (production swings from them to me) else `1` (neutral). Defensive targets
   scored by flip-urgency.
7. **Greedy top-6** (Producer `_greedy_select`): argmax score over viable & untaken
   targets & funded sources & role-mutex (don't reinforce a drained source); fire iff
   `score > 1.5`; debit source budget; mark taken/used.
8. **Exact aim on the ≤6 FIRED waves only:** `aim_orbiting`/`aim_comet`/`atan2(static)`
   → angle, eta. Emit `[src_id, angle, int(ships)]`. (Straight-line eta for scoring;
   exact aim only here → ~6 aim solves/call, not ~190.)

**Compaction flags (where we approximate the exact torch recurrence):**
- `safe_drain`: closed-form vs exact multi-player combat recurrence.
- **`competitive_score`: production-integral proxy vs exact sparse flow delta — drops
  cascade/combat-timing. THE main fidelity risk; the winrate-transfer test measures it.**
- eta-for-scoring: straight-line vs continuous intercept (exact aim only on fired waves).
- **Skip** regroup/pressure-gradient *defense* in v1 (opponent model predicts attacks;
  add later if defense-prediction proves to matter).

## Acceptance gates — done only when ALL pass

**Fidelity (vs full Producer):**
- **Strong vs our `lite_greedy` opponent model (cheap primary gate, PI-required):**
  producer_lite-as-agent must **beat Tier-0 `lite_greedy_policy` by a clear margin**
  (target Wilson-lo ≥ ~0.65, n≥32). Full Producer trounces `lite_greedy`; a faithful
  lite port must too. This is the fastest "did we keep Producer's strength" check — run
  it first; if producer_lite can't beat our *weak* opponent model, it isn't faithful.
- **Move-agreement ≥ ~70%** on shared boards: matching (source, target, ±20% ships).
- **Winrate-transfer:** producer_lite-as-agent beats our champion at **≈ full Producer's
  measured rate (~60% wins; NOT the n=16 triage's 81%)**, n≥32, Wilson-lo reported. If
  far lower → the exact flow recurrence is load-bearing → add a cheap combat-loss term
  and re-measure before wiring.

**Performance (does the opponent model HELP us — the actual point):**
- champion **with producer_lite as rollout opponent** vs **full Producer**: winrate
  **rises** vs champion-with-`lite_greedy`-opponent, n≥32, Wilson-lo improvement.
- **No regression** vs the weak panel (v7_0/v4_planner/v3.5.1); default-OFF →
  byte-identical champion.

**Speed (binding — runs on the rollout hot path):**
- **≤ ~3 ms/call** standalone (within ~1.5× Tier-0 `lite_greedy`; under Tier-1 5–10ms).
- **Turn budget intact:** champion + producer_lite rollout max turn < 1000 ms
  (Rule 46c); `fast.py bench` p95<800ms, zero≥1000ms.

## Validation harness (commands)

- **Speed:** new `scripts/bench_producer_lite.py` — time `producer_lite_policy` on N
  real boards; assert mean < 3 ms.
- **Strong-vs-lite_greedy (run FIRST — cheap primary fidelity gate):** wrap both
  producer_lite and `lite_greedy_policy` as standalone agents, then
  `python scripts/clean_ab.py <producer_lite_agent> <lite_greedy_agent> --seeds 32 --workers 4`
  → require a clear win (Wilson-lo ≥ ~0.65). Fast, no full-Producer dependency.
- **Fidelity (as agent):** wrap producer_lite as a standalone agent (mirror
  `agents/producer/producer_agent.py`), then on matched seeds:
  `python scripts/clean_ab.py <producer_lite_agent> submissions/baseline.py --seeds 32 --workers 4`
  and `python scripts/clean_ab.py agents/producer/producer_agent.py submissions/baseline.py --seeds 32 --workers 4`
  → compare wins (winrate-transfer); diff per-board launches (move-agreement).
- **Does it help us:** build a champion bundle with `BASELINE_OPP_TIER=2`, then
  `python scripts/clean_ab.py <champ_lite_opp_bundle> agents/producer/producer_agent.py --seeds 32`
  vs `<champ_default_opp> agents/producer/producer_agent.py --seeds 32`.
- **Turn budget:** `python fast.py bench <champ_lite_opp> --vs producer --games 4`.

## Build sequence (next session)

1. `lib/producer_lite.py` — faithful, fidelity-first; attribution header.
2. `scripts/bench_producer_lite.py` — speed gate on real boards.
3. Standalone agent wrapper for the as-agent A/Bs (producer_lite + a lite_greedy wrapper).
4a. **clean_ab producer_lite-agent vs lite_greedy-agent (n≥32) — cheap primary gate;
   must win clearly (Wilson-lo ≥ ~0.65) before bothering with full Producer.**
4b. clean_ab producer_lite-agent vs champion (n≥32) + full producer vs champion (n≥32,
   matched seeds) → fidelity (move-agreement + winrate-transfer ≈ ~60%).
5. If fidelity passes → wire `_select_opp_policy` (`BASELINE_OPP_TIER=2`), build
   champion+producer_lite-opp bundle, A/B vs full producer (does it help), Rule 46 smoke.
6. If winrate-transfer fails → add a cheap combat-loss correction to the competitive
   proxy, re-measure, then wire.

## Risks / open items

- **Competitive-proxy fidelity** (main risk) — winrate-transfer decides; fallback =
  cheap combat-loss term or a reduced exact recurrence on touched planets only.
- **Incoming-threat without a World object** — need a cheap fleet→target map for speed;
  `world_model.fleet_target_planet` may be affordable, else straight-line approx.
- **Aim cost on fired waves** — 6 `aim_orbiting`/call should be cheap; if rollout call
  count explodes, cache aim per snapshot.
- torch is installed + in `requirements.txt` (done 2026-06-04) → full Producer usable as
  the fidelity oracle.

## Attribution

`lib/producer_lite.py` header must credit the upstream Producer Kaggle notebook
(`agents/producer/PROVENANCE.md`) per the 2026-06-04 licensing decision.
