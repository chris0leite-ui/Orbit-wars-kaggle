# Tenure / durability value — spec (2026-06-15)

A default-OFF leaf-scorer term that discounts a capture by **how long we can
actually hold it** — the framework's *durability* factor: an asset's worth is
its income × **expected tenure** × denial, and the champion values income as if
tenure were always full. Targets the #2 loss driver (collapse). Same gating /
parity discipline as the frontier and recapture terms.

## Grounding — what collapse actually looks like (3 real CPMP-era losses)

Traced from downloaded replays (`/tmp/collapse_trace.py`):

- **seed 1506374610 (CPMP, peak 10→0):** thin-garrison **churn** at the
  contested mid-board (planet ids at distance 30–37, garrisons g5–g15, captured
  and lost and recaptured repeatedly). The opponent keeps **2–5× more ships in
  flight** (constant pressure) and grinds the thinly-held frontier down.
- **seed 645691379 (LuckyXC, peak 8→3):** we lose dist-49–67 planets, some
  well-garrisoned (g141, g41); opponent builds an **overwhelming ship count**
  (825→1462 vs our 39–96) and crushes us.
- **seed 2066324996 (Ebi, peak 17):** we hold planet *count* (15–17) but are
  **massively out-shipped** (3655 vs 508) and overwhelmed.

Two components of collapse:
1. **Unholdable captures / churn** — we spend force capturing/recapturing
   contested planets we cannot keep (clearest on 1506374610). **This term's
   target.**
2. **Losing the production/force race** — the opponent compounds an overwhelming
   ship advantage and overwhelms even well-garrisoned planets (645691379,
   2066324996). A deeper economic problem; **out of this term's scope** (noted
   honestly — partly delayed under-expansion).

The tenure term attacks (1): stop pouring ships into captures we'll immediately
lose, conserving force for keepable positions. It is **capture-selection
shaping, not global defense** — the distinction from `garval`
(garrison-value + source-safety), which over-defended globally and lost on the
ladder (1181 < 1280).

## Definitions (rigorous)

For a candidate capturing target `T` at arrival tick `e` with send `s` (so it
leaves `defender = s − capture_floor(T,e)` holding `T`), over a hold window `W`
turns past arrival:

**Enemy reachable force on `T`** (who can hit it):

    enemy_force(T) = Σ over enemy alive planets p that can reach T within W of
                     (1 − safety) · ships_p + prod_p · reach_time(p, T)

(identical in spirit to `recapture.recapture_penalty`'s threat; reach via
`cross_dist[1] / fleet_speed`, static-exact / orbit-conservative.)

**Our reinforcement reach on `T`** (the NEW part — can we hold it?):

    friend_reach(T) = Σ over OUR alive planets q (q ≠ the capture source) that
                      can reach T within W of  hold_fraction · ships_q

i.e. the ships we could plausibly send to defend `T` in time. `hold_fraction`
(default 0.5) is the share of a planet's garrison we'd spare for defense.

**Net exposure and expected tenure:**

    exposure(T) = max(0, enemy_force(T) − defender − friend_reach(T))
    # turns until we likely lose T, if exposed: nearest enemy arrival
    turns_lost  = max(0, W − reach_to_T_enemy_min)        # 0 if not exposed
    tenure_penalty = captures · (exposure > 0) · prod[T] · turns_lost · weight

`tenure_penalty ≥ 0` is **subtracted** from the candidate score (same sign as
`recapture_penalty`). A capture we can comfortably hold (defender + reinforcement
≥ enemy force) gets `exposure = 0` → no penalty → full value. A thin capture of a
contested planet with enemy force nearby and no reinforcement → large penalty →
deprioritized.

**Relationship to `recapture_penalty`:** this is recapture's enemy-threat idea
**plus our reinforcement reach** (net, not gross, exposure) — it answers "can we
*hold* it", not just "can they *reach* it". They overlap; **do not enable both**
(double-penalty). Tenure is the more complete form; recapture stays available for
A/B reference.

## Knobs (env, all default to the OFF / no-op path)

| Env var | Default | Meaning |
|---|---|---|
| `PRODUCER_PLUS_TENURE_PENALTY` | `0` (off) | master enable |
| `PRODUCER_PLUS_TENURE_WEIGHT` | `1.0` | overall penalty scale (ship units already) |
| `PRODUCER_PLUS_TENURE_W` | `8` | hold window `W` (turns past arrival), clamped to cache horizon |
| `PRODUCER_PLUS_TENURE_HOLD_FRACTION` | `0.5` | share of an owned garrison counted as reinforcement |
| `PRODUCER_PLUS_TENURE_SAFETY` | `0.5` | enemy-garrison fraction counted as threat (mirrors recapture) |
| `PRODUCER_PLUS_TENURE_FROM_STEP` | `0` | only apply from this step on (collapse is a mid/late phenomenon; lets us avoid taxing the opening) |

## Integration
- New module `agents/producer/orbit_lite/durability.py` (`tenure_penalty`),
  reusing `DistanceCache`, `fleet_speed`, and the `_compute_captures` gate
  pattern. Added to the bundler's `ORBIT_LITE_ORDER`.
- Wiring (env getters + call site) in `agents/producer_plus/main.py`, mirroring
  `recapture_penalty` (it is the closest sibling). Subtracted from `score`.
- Bundler variant `seq_strength_tenure` = the 1280 base + tenure on.
- Default-OFF byte-identity is structural and re-verified by a fixed-seed diff.

## Verification plan
1. **Unit (synthetic):** an unholdable contested capture (enemy force ≫
   defender + reinforcement) gets a large penalty; an identical capture with a
   friendly reinforcing neighbour in reach gets ~0 (reinforcement closes the
   exposure); a comfortably-held capture gets 0; disabled ⇒ 0; respects
   `from_step`.
2. **Rule-38 reproduction:** on the real collapse seed 1506374610, compare the
   base mirror vs the tenure mirror on **churn** — planets lost-then-recaptured
   after our peak, and ships spent on captures lost within `W` turns. Tenure
   should reduce the wasteful recapture churn.
3. **Rule-46 smoke:** bundle builds + parses; full game max turn < 1000 ms.
4. **Ladder (Rule 45):** A/B `seq_strength_tenure` vs the field at n ≥ 32.

## Verification results (2026-06-15) — read honestly

Passed cleanly:
- **Unit:** `tests/test_tenure_penalty.py` 6/6 — an unholdable contested capture
  is penalised; the SAME capture with a friendly reinforcer in reach gets 0
  (reinforcement closes the exposure — the tenure novelty); no-enemy /
  no-capture / disabled / zero-window all 0.
- **Default-OFF parity:** the `vetorf4p_seq_strength` bundle rebuilt with this
  code present is behaviourally byte-identical (106/98/114 on seeds 7/13/42).
- **Latency:** full game max turn 139 ms.

Could NOT verify the collapse fix at the game level — and the reason is
structural:
- **vs a fixed flag-agnostic opponent (bare `producer`), we have ZERO losses**
  on the collapse seeds (1506374610, 645691379: peak==final, 0 lost-planet
  transitions). The opponent is too weak to make us collapse, so tenure has
  nothing to act on — base and tenure are identical.
- **the symmetric mirror is confounded**: which side wins a base-vs-base or
  tenure-vs-tenure game is arbitrary, and it dominates the churn counts (a won
  377-step game has far more gains/losses than a lost 186-step one), so it
  cannot isolate tenure's effect on *our* side.

This is the same wall the frontier reproduction hit, now confirmed as a general
limit: **our remaining loss drivers (corner-neglect, collapse) manifest only
against top-tier opponents (CPMP ~1600+) that we cannot run locally.** Local
self-play and our weak local panel cannot reproduce them, so they cannot verify
fixes for them. The mechanism is unit-correct; whether it *helps* — and whether
it cedes ground (the key risk below) — only the ladder (Rule 45) can answer.

## Known risks
- **Ceding ground.** Down-weighting unholdable captures can tip into *passivity*
  (give up the contested frontier → cede the board → lose differently). This is
  the mirror risk of garval's over-defense and the key thing the ladder must
  check. `TENURE_FROM_STEP` and a modest `weight` are the safety dials.
- **Scope.** Does not address the "out-produced and overwhelmed" component of
  collapse (645691379 / 2066324996) — that is closer to the under-expansion /
  force-concentration problem.
- **Reach approximations** (sun not modeled, orbital drift, k=1 immediate
  distance) — same as `recapture`; lean conservative (under-estimate threat).
