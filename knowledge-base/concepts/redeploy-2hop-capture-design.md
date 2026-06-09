# Forward-redeploy, redesigned: score the *capture it enables*, not the spatial drift

> Written 2026-05-30 on `claude/champion-strategy-rules-00JzI`.
> Supersedes the SEU7P spatial-coupled redeploy generator
> (`_enumerate_redeploy_candidates` + `cheap_marginal_redeploy`,
> commit `fa696d0`) as the *design* to pursue if forward-redeploy is
> revisited. Status: SPEC ONLY — not implemented, PI-review-gated.

## Why the SEU7P redeploy generator cannot stand alone

The SEU7P forward-redeploy is two coupled pieces:

1. `cheap_marginal_redeploy` (pre-rank filter) — a self-contained
   spatial-gain proxy that mirrors `value._positional_ship_value`. It
   decides which own→own candidates *enter* the K-step rollout. It does
   not call the value head.
2. The rollout leaf (the value head) — decides whether a redeploy beats
   the idle baseline (Δ vs `build_idle_baseline`).

An own→own redeploy **captures nothing and changes no production**. Under
the plain `favor_hybrid` head the leaf scores production + ships-at-
planets + capture-credit, so a redeploy's Δ is **zero or slightly
negative** (ships spend ticks in flight: exposed, not defending). The
*only* term that flips Δ positive is the additive spatial pull in
`favor_hybrid_spatial`:

    base + SPATIAL_WEIGHT · _positional_ship_value

and the cheap filter was hand-matched to that term (same
`1/(1+d/decay)` shape, same `0.05` coefficient). **The spatial head is
not a companion signal — it is the entire reason a redeploy is ever
selected.** Remove it and the generator still emits candidates (wasted
enumeration + rollout compute) that all die at the leaf. Live
corroboration: the combined `baseline_redeploy_gangup` push (sub
53177486) settled at μ=971.1, ~190 below the 1163 anchor.

Conclusion: redeploy falls *with* the spatial head. Salvaging it is a
**redesign**, not a decoupling.

## The redesign: value the redeploy through the downstream capture

The plain head already rewards captures (`cheap_marginal_value` tail:
`0.05 · tgt.production · pv_horizon(now, arrival)`; the leaf rewards the
production-time integral of the captured planet). So instead of valuing
"ship-mass moved toward the frontier" (spatial, dead), value the **2-hop
plan**:

> Interior planet *I* reinforces frontier planet *F*, so that *F* can
> then make a **capture *C*** that *I* could not have reached in time
> on its own.

The value being claimed is a *real capture* the plain head already
optimizes — not a positional drift. This is why it can stand alone.

### Eligibility (generator)

For each peaceful interior source *I* with spare garrison:

1. **Find the enabled capture first, then the redeploy** (reverse the
   SEU7P order — don't enumerate redeploys and hope a capture follows).
   For each candidate capture target *C* (neutral or opponent planet
   that the existing capture/reinforce generators would *want* but
   currently can't fund or reach in time):
   - Identify own frontier planets *F* from which a launch at *C* lands
     by the needed arrival tick (use `aim_and_eta`, the same primitive
     the capture generator uses).
   - *F* lacks the ships to fund *C* now, but *F* + an inbound redeploy
     from *I* would. (Check against `model.ships_at(F, eta_redeploy)`
     along the timeline — `ships_at` already accounts for friendly
     in-flight fleets, so the redeploy's arrival is visible to the
     follow-on capture leg within the same rollout.)
2. **Reachability gate:** *I*'s own direct launch at *C* must arrive
   *strictly later* than the F-relay path (else just launch I→C; no
   redeploy needed). This is the precise "I could not reach it in time"
   condition and it replaces the SEU7P `src_d ≥ 1.5·tgt_d` spatial-
   ratio gate.
3. **Source-safety:** reuse `_source_survives_launch(I, …)` unchanged
   (the existing Bug-#4 pre-cut — don't drain an exposed interior).
4. **Target-safety on F:** F must not itself be threatened before the
   relay completes (reuse `model.time_to_enemy_threat(F)`); otherwise
   the reinforce path owns that case.

### Scoring (pre-rank — replaces `cheap_marginal_redeploy`)

Score the **redeploy leg by the capture it unlocks**, using the existing
capture-credit shape so it composes with the plain head:

    cheap_marginal_redeploy_2hop(I, F, C, ships, world, me)
      = cheap_marginal_value-style capture credit for C
        as launched from F at the relay arrival tick
        MINUS the in-flight exposure cost of the I→F leg
        (the -0.5·ships idle-penalty shape, scaled by the extra
         ticks the redeployed ships spend in transit vs sitting at I)

Key property: when no capture is unlocked (no eligible *C*), the score
is ≤ 0 and the candidate is dropped at `CHEAP_REJECT_THRESHOLD` — so
with the plain head, redeploy emits **nothing** unless it genuinely
funds a capture. That is the "stands on its own" guarantee.

### Rollout leaf

No value-head change required. The K-step rollout must have horizon
≥ (relay arrival + capture arrival + SIM_SETTLE_TURNS) so the leaf
actually *sees* the captured planet's production accrue — otherwise the
plan is invisible and Δ collapses to the in-flight penalty again.
**This horizon requirement is the single biggest implementation risk:**
verify per-turn that the chosen `horizon` covers both legs before
trusting any A/B (Rule 47 — trace one game through
`predict_fleet_fate` / the timeline and confirm the relayed capture is
within the leaf window). If the two-leg plan exceeds the rollout
horizon, this design is falsified for *substrate* reasons, not
strategy — and that must be ruled out before spending A/B slots.

## Build sequence (hard gates between steps)

1. **Physics/horizon verification (Rule 47), ~30 min, no A/B.** Hand-
   construct one 2P state with a known I→F→C relay capture. Confirm:
   (a) `aim_and_eta` gives the relay legs; (b) `ships_at(F, t)` shows
   the redeploy arrival; (c) a horizon covering both legs makes the
   leaf value the captured planet. **Exit if the relay can't fit the
   horizon** — the design is dead on substrate; do not proceed.
2. **Generator + scorer (~150 LOC), env-gated `PROPOSER_REDEPLOY_2HOP`,
   default OFF.** Mirror the `_enumerate_reactor_candidates` extend-
   prerank pattern. Probe per-turn emission counts with a
   `probe_new_generators.py`-style script: emissions must be **0 on
   states with no fundable capture** and **>0 only when a relay unlocks
   one**. This is the falsifiable Phase-0 check the SEU7P version never
   had (it always emitted on spatial gradient).
3. **n=16 triage A/B vs the current champion** (Rule 45 triage tier).
   Multi-opponent panel, not single-opponent (Rule 43). Proceed to
   step 4 only at μ≥0.75 / Wilson-lo≥0.55.
4. **n=32 confirmation A/B** (Rule 45 submit gate) + `--vs-panel`
   Wilson-lo≥0.55 per opponent + `--vs <rolling_champion>` Wilson-lo
   ≥0.50. Only then is it a submit candidate.

## Hard exit conditions

- Step-1 horizon doesn't fit the relay → STOP (substrate-dead).
- Step-2 emits on no-capture states → scorer is still a spatial proxy
  in disguise → STOP, fix the scorer.
- Step-3 triage < μ=0.75 → this is the 2-hop axis; under Rule 37 you
  get ≤3 variants total on the *scorer/eligibility* axis before the
  axis is closed and we stop iterating.

## Relationship to closed knowledge

- The SEU7P spatial-coupled redeploy is retired with the spatial head;
  do not re-enable `PROPOSER_REDEPLOY` as-is.
- `gang_up` support is a **confirmed independent regressor** (SEU7P
  isolation A/B, 7/16 vs 11/16) — do NOT bundle it with this.
- This design does not touch the value head, chooser, or `lib/`
  (Rule 44 scope: proposer-only), so it does not collide with the
  closed analytical-chooser / reach-frontier / chain-bonus tracks.
