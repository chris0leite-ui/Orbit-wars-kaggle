# Frontier / gateway value — spec (2026-06-15)

A new leaf-scorer term for `producer_plus` that values a capture not only by
the planet's own income, but by the **new options it opens up** — the
production-weighted set of neutral planets that become newly reachable once we
own it and can launch from it. Default-OFF, env-gated, byte-identical champion
when unset.

## Why (the modeling cause, not a band-aid)

The #1 loss driver (≈76% of losses) is **under-expansion**: we trail on planet
count by ~step 30 and hold 5–6 planets vs winners' 8 by step 60, and we
disproportionately leave **far, high-value planets** (corners) neutral.

The current ladder fixes (`wideshortlist` = consider 20 neutrals; `expand` =
wider shortlist + deeper horizon) are *restriction-tuning* in the Rule-40
sense: they widen the menu and stretch the scoring window so a far planet
*survives* to be scored — but it is still scored on **its own production
alone**. A corner planet's own income looks mediocre, so even when surfaced it
loses to a nearer planet of equal production. We are hoping a bigger menu plus a
longer ruler *accidentally* surfaces gateways.

The **model fix**: score the thing that actually makes a gateway valuable — the
cluster it unlocks. A human wants the corner not for the corner, but for
everything the corner reaches. Formally, the strategic value of acquiring a
planet `a` includes the **option-set delta**: owning `a` as a launch base
brings a whole new set of cheap captures into reach. Our scorer (a net-ship
flow over a short horizon) never computes this. This term adds it.

This is the "reach / option leverage" factor from the positional-game framing
(`knowledge-base/thoughts/` 2026-06-15): an asset's worth ≈ (own income + new
options it opens) × expected tenure × denial. We model the first factor's
near-term slice today; this term adds the **new-options** part.

## Definitions (rigorous)

Let `A` = our currently-owned, alive planets. Let `V` = candidate frontier
targets: alive, **neutral**, non-comet planets (enemy targets optional, default
off — they are already valued via the denial/competitive terms; including them
risks double-counting).

**Reach predicate.** A planet `v` is reachable from base `u` within a turn
budget `R` at nominal fleet speed `c` iff there exists an arrival tick
`k ∈ [1, R]` with

    cross_dist[k, u, v] ≤ c · k            and v is alive at tick k

where `cross_dist[k, u, v] = dist(u@0, v@k)` is the cross-time distance from the
distance cache (exact for static planets; for orbiting targets it already
accounts for the target's motion to `v@k`). Define the best arrival tick

    eta(u, v) = min such k    (+∞ if none in [1, R]),   reach(u, v) = isfinite(eta).

This mirrors the planner's own reachability shape (`surf/k ≤ speed` in
`reachable_mask`) and reuses the per-turn `DistanceCache`, so no new geometry
and no extra per-turn cost beyond a few `[P, P]` reductions (P ≤ 40).

**Marginal frontier of `a` (the new options).**

    reach_from_A(v) = OR over u in A of reach(u, v)          # already reachable today
    newly(a, v)     = reach(a, v) AND NOT reach_from_A(v)    # a uniquely unlocks v

`newly` is the option-set delta `ΔF(a)`: targets we could *not* cheaply reach
before, that owning `a` puts in reach. (Diagonal `newly(a,a)` is forced False —
a base is not its own frontier.)

**Proximity discount.** Closer-unlocked targets are worth more (more likely to
be realized, and sooner):

    disc(a, v) = clamp(1 − eta(a, v) / (R + 1), 0, 1)        # 1 when adjacent, →0 at the budget edge

**Contest discount (ambition knob, default off).** A newly-unlocked target that
an enemy base reaches no later than `a` does is contested — we might lose the
race, so down-weight it:

    eta_enemy(v) = min over enemy alive bases e of eta(e, v)
    contested(a, v) = eta_enemy(v) ≤ eta(a, v)
    contest_factor(a, v) = (1 − w_contest) if (contested AND newly) else 1     # w_contest ∈ [0,1]

This is the "contested frontier" idea and the bridge toward option-denial /
sun-severing (a later term). At `w_contest = 0` it is a no-op.

**Per-base frontier value** (one value per potential base `a`):

    fv(a) = Σ over v in V of  newly(a, v) · prod[v] · disc(a, v) · contest_factor(a, v)

**Per-candidate bonus.** For a candidate `c` capturing target `a = tgt(c)`:

    future_h     = max(0, game_length_est − current_step − H)        # post-horizon turns left
    captures_c   = (send ≥ capture_floor at arrival) AND valid AND not-defensive
                   AND we don't already own `a` at arrival             # shared _compute_captures gate
    frontier_c   = captures_c · weight · future_h · fv(a)

`frontier_c ≥ 0` is **added** to the candidate's competitive score (same sign /
ship-unit convention as `denial_bonus` / `opening_bonus`).

**Units.** `prod` (ships/turn) · `future_h` (turns) · dimensionless discounts ·
weight → ships, commensurate with `competitive_score`. `future_h` makes a
gateway worth most early (when the #1 driver, early under-expansion, bites) and
zero at game end.

## Knobs (env, all default to the OFF / no-op path)

| Env var | Default | Meaning |
|---|---|---|
| `PRODUCER_PLUS_FRONTIER_BONUS` | `0` (off) | master enable |
| `PRODUCER_PLUS_FRONTIER_WEIGHT` | `0.05` | `weight`. Interpreted as "expected fraction of the unlocked options' full-hold production we realize." Calibrate on ladder. |
| `PRODUCER_PLUS_FRONTIER_REACH` | `12` | `R`, reach budget in turns (clamped to the distance-cache horizon) |
| `PRODUCER_PLUS_FRONTIER_SPEED` | `3.0` | `c`, nominal fleet speed (~a 30–50-ship early expansion fleet) |
| `PRODUCER_PLUS_FRONTIER_CONTEST` | `0.0` | `w_contest`, contest down-weight ∈ [0,1]; 0 = off |
| `PRODUCER_PLUS_FRONTIER_INCLUDE_ENEMY` | `0` | also count enemy planets as frontier targets |

Magnitude check: early game (`future_h ≈ 170`), a base that uniquely unlocks
~3 neutrals (prod ≈ 3, disc ≈ 0.7) gives `fv ≈ 6`, so the bonus ≈
`0.05 · 170 · 6 ≈ 50` ships — same order as a strong capture's own score (so it
can tip a gateway above a non-gateway, not dominate). The weight is the
single calibration dial.

## Integration

- New code lives in `agents/producer/orbit_lite/strategic_value.py`
  (`frontier_bonus`, helper `_reach_eta_matrix`) — single source of truth,
  reuses the existing `_compute_captures` gate and `_future_value_horizon`.
- Wiring (env getters + call site) in `agents/producer_plus/main.py`, mirroring
  `denial_bonus` / `opening_bonus`. The comet mask is computed at the call site
  (`is_comet_planet`) and passed in, keeping `strategic_value` dependency-free.
- Bundler variant `seq_strength_frontier` in `scripts/bundle_producer_plus.py`
  = the live 1280 base (`vetorf4p_seq_strength` flags) + frontier ON.
- Default-OFF byte-identity is structural (the call site only runs under the
  flag) and re-verified by a fixed-seed diff of the `vetorf4p_seq_strength`
  bundle before vs after this change.

## Verification plan

1. **Unit (synthetic):** a hand-built board with a gateway planet (the only base
   that reaches a back cluster) vs an equal-income non-gateway. Assert
   `frontier_bonus` is strictly larger for the gateway, zero for non-captures,
   zero when disabled, and respects the reach budget.
2. **Rule-38 reproduction:** on the real corner-neglect loss seed (2P vs CPMP,
   `seed 641308308`), confirm the baseline leaves the far corner/cluster neutral
   and frontier-ON now values + takes it (behaviour change in the right
   direction), without latency regression.
3. **Rule-46 smoke:** bundle builds + parses; full game max turn < 1000 ms.
4. **Ladder (Rule 45):** A/B `seq_strength_frontier` vs the field at n ≥ 32 — the
   only honest test. The local self-play is referee-blind and cannot reward
   fixing a flaw the self-opponent shares.

## Local verification results (2026-06-15) — read honestly

What passed cleanly:
- **Unit:** `tests/test_frontier_bonus.py` 9/9 green — gateway scores strictly
  above an equal-income dead-end (28.0 vs 0.0), reach budget + contest knobs
  behave, zero without a capture, zero when disabled.
- **Default-OFF parity:** the `vetorf4p_seq_strength` bundle is behaviourally
  byte-identical pre/post change (outcomes + step counts identical on seeds
  7/13/42 vs `v7_0_drop_one`). The gated path is dead when the flag is unset.
- **Latency:** full game max turn 144 ms (p50 89 ms) — well under the 1000 ms
  gate. The term is one `[P,P]` reach matrix + reductions per turn.
- **Active:** games diverge from base (different step/planet trajectories), so
  the term is wired and firing, not a silent no-op.

What did NOT verify — and the decisive reproduction (Rule 38):
After the PI asked to mine a real CPMP loss first, I downloaded the real loss
replays (`kaggle competitions replay`), pulled their seeds from `info.seed`,
and ran the symmetric `producer_plus` mirror (flags are global env, so
asymmetric in-process is impossible — separate-process symmetric self-play is
the valid tool, as the prior session used).

On the canonical corner-neglect seed **641308308** the mirror **reproduces the
failure exactly**: base leaves the two garrison-41, prod-4 corners (planet ids
5 and 6, distance 63 from centre) neutral at steps 60, 95 **and every step to
500** — never captured. The four-way comparison:

| agent (mirror, seed 641308308) | corners 5,6 captured? |
|---|---|
| base | never (neglected @60/@95/final) |
| **frontier alone** | **never — identical to base** |
| wideshortlist (generation fix) | yes, but only by ~step 499 |
| **frontier + wideshortlist** | yes, by **step 95** (much earlier) |

**Frontier alone does NOT fix corner-neglect.** Root cause, now proven:
corner-neglect is a candidate-**generation** truncation — the far corner falls
outside the nearest-K neutral shortlist, so it is never a candidate — while
`frontier_bonus` is a candidate-**scoring** term that can only boost candidates
that already exist. A scoring term cannot surface a target that generation
never proposes. This is a layer mismatch between the term and the failure it
was meant to fix.

But the reproduction is also **constructive**: once generation surfaces the
corner (wideshortlist), frontier **accelerates its capture from ~step 499 to
step 95** — exactly the early-expansion behaviour we want. **Frontier is a
multiplier on a generation fix, not a standalone fix.**

The other two real CPMP losses confirm the term is aimed at a phenomenon these
losses don't have: seed **1692894782** (fast 96-step loss) was an expansion-
*rate* race lost to a stronger CPMP with **no** far-HV-neutral neglect in the
mirror (p0=9 @60); seed **1506374610** was a pure **collapse** (both sides
reach 8 planets then die) — the tenure/durability driver, the proposed *second*
term, not frontier.

Verdict: the **mechanism is correct, fast, and parity-safe**, but **frontier
alone is the wrong layer for the corner-neglect failure** — it must be composed
with a generation fix (wideshortlist / the `expand` variant) to bite at all, and
its standalone value over that generation fix is the open question. Earlier I
also saw frontier-alone slightly *reduce* `planets@60` vs the bare `producer`
(12→10, 14→12, 11→8) — consistent with "scoring redirection with no new far
candidates surfaced." Recommended unit to test next: **`expand + frontier`**
(generation surfaces far targets; frontier + horizon take them early), ladder
A/B'd against `expand` alone to isolate whether frontier adds lift on top of
generation. Lift claims gated to n ≥ 32 (Rule 45); the single-seed step-95-vs-499
acceleration above is a mechanism demonstration, not a lift claim.

## Known approximations / risks

- **Sun not modeled in reach.** `cross_dist` is straight-line; a target behind
  the sun from `a` is counted reachable though a fleet would die crossing it.
  This over-credits some frontier (same approximation `recapture` uses). The
  `contest`/sun-severing extension is the documented follow-up; first cut keeps
  it simple.
- **Base orbital drift.** Reach uses `a@0`; we actually launch from `a` a few
  turns later (capture delay). Exact for static `a`, small drift for orbiting.
- **Optimistic option value.** Crediting full `future_h` per unlocked target
  assumes we realize the option; the small `weight` and `disc` are the
  deliberate discount. Calibrate on ladder; watch the long-game/collapse
  results (the tenure factor is the natural next term and the principled brake
  on over-extension this term could otherwise encourage).
