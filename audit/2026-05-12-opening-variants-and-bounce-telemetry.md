# 2026-05-12 — Opening variants (A/B/C) + bounce-margin telemetry (D)

Branch: `claude/fix-early-game-strategy-YSClQ`.
Follows: `audit/2026-05-12-v3.6-opening-failed.md` (variant A FAIL).
PI directive: "test all" (three opener variants + bounce telemetry).

## Variants tested

| # | Name | Config | Hypothesis |
|---|------|--------|-----------|
| A | `v3.6_opening` | `window=5, min=8, reserve=2, allow_enemies=False` | Drain-source from step 0 |
| B | `v3.6_opening_B` | `window=5, min=14, reserve=7, allow_enemies=False` | Timing-matched bowwow (wait for production) |
| C | `v3.6_opening_C` | `window=5, min=8, reserve=2, allow_enemies=True` | Drain-source + enemy raids |
| D | postmortem extension | n/a | Per-fleet bounce-margin instrumentation |

## Panel results (8-seed canonical SEEDS_32[:8])

Artifact: `audit/tournaments/20260512T085822Z.json` (5 agents × 8 seeds).

| Agent | mean panel WR | head-to-head vs v3.5.1 (W/D/L of 16) |
|-------|---|---|
| **v3.5.1**            | **60.9 %** | self |
| **v3.6_opening_B**    | **51.6 %** | **5 W / 8 D / 3 L** (31 % raw, 56 % win-or-draw) |
| v3.6_opening (A)      | 35.9 %    | 2 W / 0 D / 14 L (12 %) |
| v3.6_opening_C        | 35.9 %    | 2 W / 0 D / 14 L (12 %) |
| baseline              | 0.0 %     | 0 W (0 %) |

**Headline:** Only variant **B (timing-matched bowwow)** is viable.
A and C decisively regress (12 % each, mostly losses) — confirms
that emptying home to 2 ships from step 0 is the wrong move,
regardless of whether enemies are targeted.

B vs v3.5.1 is **directionally positive in win-or-draw share (56 %)
but tied on raw winrate (31 %), with a high 50 % draw rate.** Signal
is not strong enough at 8 seeds.

### 32-seed B vs v3.5.1 confirmation

Artifact: `audit/tournaments/v36-opening-B-vs-v3.5.1-32seed-20260512T090341Z.json`.

| metric | value |
|--------|-------|
| P0 (B as seat 0) wins | 7 / 32 |
| P1 (B as seat 1) wins | 9 / 32 |
| total wins | 16 / 64 |
| draws | 28 (44 %) |
| losses | 20 |
| raw winrate | **25.0 %** |
| Wilson lo | **16.0 %** |
| verdict | **FAIL** |

With draws scored as half (TrueSkill-style):
`(16 + 14) / 64 = 46.9 %` — slightly **below** parity.

The 8-seed panel result (51.6 % mean WR) overstated B's strength;
at 32 seeds with 44 % draws, B is calibration-equivalent to v3.5.1
in non-decisive games and slightly weaker (44 % decisive winrate)
in decisive games.

**All three opener variants (A, B, C) now FAIL the 55 % Wilson-lo
promotion gate.** The "fix the weak start" workstream has been
falsified across three parameter settings. Going further requires
either (a) a fundamentally different opening mechanism (search-
based; opponent-aware fleet sizing), or (b) accepting that the
opening is already optimal for our agent class and pivoting to
the bounce work below.

## Variant D — bounce-margin telemetry

Implementation: `scripts/episode_postmortem.py::attribute_fleets`
now records `target_owner_before`, `target_ships_before`, and
`margin = ships_sent - target_ships_before` per launched fleet.
The aggregate `bounce_margin_hist` field (in `roll-up.json`)
buckets margins from −15 to +15 (clamped).

**4-seed v3_snipe self-play sample:**
- Total bounces: 64 (16/game, all 2P)
- Distribution:

```
margin -15:  27 (42 %)   heavy under-sizing
margin -11:   2
margin -10:   2
margin  -9:   2
margin  -8:   1
margin  -7:   2
margin  -6:   1
margin  -5:   1
margin  -2:   1
margin  -1:   3 (4.7 %)
margin   0:   3 (4.7 %)  tied (combat rule: > garrison to flip)
margin  +1:   7 (11 %)   one ship over but adversary stacking ate it
margin  +3:   3
margin  +4:   1
margin  +6:   1
margin  +7:   1
margin +11:   1
margin +13:   1
margin +15:   4 (6.2 %)  catastrophic adversary stacking
```

**±1 bounces (margin ∈ {-1, 0, +1}) = 13 / 64 = 20 %.**

**Adversary-stacking miss (margin ≥ +6) = 9 / 64 = 14 %.** These are
the bounces where we sent more than the visible defender count but
still lost — `WorldModel.ships_at` failed to predict same-turn enemy
launches.

**Catastrophic under-prediction (margin = -15 clamped) = 42 %.** Our
fleet arrived at a planet whose ships had grown well beyond what we
budgeted for. These are mostly LONG-eta fleets (the few `long` rows
in `bounce_margin_by_eta`).

## Strategic conclusions

1. **The opening cannot be fixed at the single-mission-class level
   for v3.5.1.** Three independent parameter settings (A: drain at
   step 0, B: bowwow-timing-matched, C: drain + enemy targets) all
   fail the 55 % Wilson-lo gate. The 8-seed panel intermediate
   result (B = 51.6 % mean WR) suggested a directional positive,
   but 32-seed confirmation collapsed to 25.0 % raw with 44 %
   draws. v3.5.1's `aggressive=True` already does the bowwow trick
   at steps 2-3 once `src.ships > 12`; adding an explicit opener
   class duplicates that work without adding edge.
2. **The PI's "weak start" observation is real, but the fix is
   elsewhere.** Either it lives in (a) a smarter opener that
   considers the opponent's likely first move (search-based; v7-
   minimax-style maximin in the opening), or (b) the bounce work,
   which the telemetry below quantifies as the higher-EV slot.
3. **Bounce telemetry is the deliverable that worked.** Per-fleet
   margin distribution from a 4-game self-play sample:
   - **20 %** of bounces at ±1 margin — immediately addressable.
   - **14 %** at margin ≥ +6 — adversary-stacking misses (need
     sharper `WorldModel.ships_at`).
   - **42 %** clamped at margin -15 — long-flight under-sizing
     where the defender garrison grew much beyond our estimate.

## Recommended next moves (PI-gated)

**Priority 1 — bounce cushion variant in `arrival_size`.** With the
telemetry now landed, design a *selective* cushion (not the v3.3
blanket fix that regressed):
- For static enemy targets with `eta > 15`: cushion = `+3` (catches
  catastrophic long-flight cases).
- For dynamic (orbit/comet) targets: keep `+1` (the targeted fix
  that's already shipped).
- Per-fleet margin telemetry stays on in the postmortem so we can
  validate that the new cushion converts ±1 bounces without
  over-sizing safe captures.

**Priority 2 — pull live replays for 52544634 and 52568317** so
the margin distribution can be measured on real opponents instead
of self-play. Self-play 64-bounce sample is small and may overstate
certain bucket sizes.

**Priority 3 — opener via v7_minimax integration.** v7 already
considers a pair of candidate actions per turn. Add an "opening
action" candidate to v7's enumeration so the maximin overlay can
evaluate it against the opponent's likely response. This is
plausibly the only way to make the bowwow archetype net-positive,
because it requires opponent-awareness that a single-pass
proposer can't have.

**Archive (FAILED, do not re-attempt without new data):**
- A: drain-source from step 0 (32-seed: 50 %, 8-seed: 12 %)
- B: timing-matched bowwow (32-seed: 25 %, 44 % draws)
- C: drain + enemy targets (8-seed: 12 %)

