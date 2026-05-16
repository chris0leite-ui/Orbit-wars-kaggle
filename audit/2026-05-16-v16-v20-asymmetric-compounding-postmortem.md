# 2026-05-16 — Asymmetric-compounding axis postmortem + v20 dogpile fix

## What was attempted

Session started with PI's intuition: "compounding value of production —
asymmetric per planet, to the power of how quickly we reach and how
long we can defend." Goal: lift v15's late-game under-emission
(Forrest 2P loss episode 76727858: v15 emits 1-3 launches/turn while
opp Forrest emits 5-11; we dominate to step 70, eliminated by 255).

Five variants attempted in sequence (4 null, 1 win):

| Variant | Hypothesis | Result |
|---|---|---|
| v16 | Global multiplier `1 + 2 × (n_neutrals / n_total)` on F2 | Null — multiplier symmetric, factored out of `(my_prod − opp_prod)`. Identical to v15 in funnel and h2h. |
| v17 | Per-planet `prod × pv(hold_eta_me)` in `_favor` (turn-0 hold) | Regression — F2 magnitude universally shrunk (pv(hold) < pv(500)); F1 ship balance over-weighted; chooser became MORE conservative. H2H vs v15 = 40.6% FAIL. Wallclock max=2001ms. |
| v18 | Proportional-split: `prod × pv(500) × hold_me/(hold_me+hold_opp)` | Worse — distant-safe neutrals (both holds long) get 50/50 split = half v15 credit. H2H vs v15 = 34.4% FAIL. Wallclock max=1753ms. |
| v19 | Me-policy (lite_greedy) in baseline rollout instead of me-idle | Catastrophic — under realistic baseline, candidates that match lite_greedy default have Δ≈0 and don't emit; chooser becomes near-passive. vs v7_0 = 12.5% FAIL. |
| **v20** | **Remove per-target dedup in chooser emit loop** | **WIN — h2h vs v15 = 65.6% (21/32 n=32). Felipe 2/2 (was 1/2 v15). 213tubo 2/2. Forrest funnel step 120/190 emit: 2→4, 2→5. Submitted sub 52721807.** |

## Root cause of v16-v19 failures (consolidated)

**I changed the value function without measuring v15's calibration
first.** v15's F2 = `(my_prod − opp_prod) × pv(500)` over-credits both
sides equally; the difference cancels the over-credit. Any per-planet
asymmetry in `_favor` breaks this cancellation:
- v17 shrunk pv per planet → F2 magnitude collapse → F1 over-weighted
- v18 split pv between sides → same magnitude shrinkage for non-extreme
  holds → also F1 over-weighted

The asymmetry PI envisioned was **already implicit** in v15 via two
mechanisms:
1. Rollout's reactive opp catches fragile captures → leaf shows opp
   owning → my F2 contribution drops automatically (owner-flip).
2. v15's symmetric over-credit cancels in the (my − opp) difference.

Explicit asymmetry in `_favor` was redundant with the implicit handling
AND broke the calibration that made v15 work.

v19's me-policy approach failed differently: realistic baseline made
the chooser require strict-better-than-lite_greedy candidates,
under-emitting on most turns since chooser's best candidates often
≈ lite_greedy's choices.

## Root cause of Forrest under-emit (and the v20 fix)

**Diagnosed correctly in v20 only.** v15's chooser found 6-16
positive-Δ candidates per turn at the Forrest crisis steps but the
per-target dedup (`if sid in used_srcs or tid in used_tgts:` at the
emit loop) capped emits at 1-3 because top-Δ candidates clustered
on a few high-value planets.

Single-line fix: drop `or tid in used_tgts`. Each candidate's Δ was
already validated by the rollout, so dogpiling a target is
self-balancing — the 2nd candidate's Δ was scored as "this candidate
vs baseline" with the rollout's reactive opp simulating the first
candidate's effects. If 2nd-launch Δ > 0, the rollout judged it
net-useful (e.g. extra surplus for defense, or harder kill against
a heavy garrison).

Forrest funnel A/B (Forrest replay episode 76727858, seat 0):
| step | v15 emit | v20 emit | gain |
|---|---|---|---|
| 30  | 1 | 1 | 0 (early, only 4 my planets ready) |
| 80  | 3 | 3 | 0 |
| 120 | 2 | 4 | **+2** |
| 150 | 2 | 2 | 0 |
| 190 | 2 | 5 | **+3** |

Targeted lift at the under-emit hot zones (mid-late game where opp
out-cascades v15) without changing safe-zone behavior.

## What I should have done differently

Captured in the session reflection (full version in transcript). Top 5:

1. **First 30 min: instrument, not code.** I had
   `scripts/instrument_v12_chooser.py` but only used it to confirm
   changes — never to diagnose v15's behavior at the crisis step.
2. **Head-to-head vs v15 as FIRST gate, not LAST.** v17/v18 both
   panel-passed against weak agents (v7_0, v4_planner, v3.5.1) but
   failed h2h vs v15. Panel was a misleading green light.
3. **Trust the rollout's implicit handling.** v15's reactive opp +
   leaf owner-flip already encodes asymmetry. Explicit re-addition
   was redundant.
4. **Smaller changes, isolated.** The 1-LOC sanity check (cheap-rank
   capture branch hold-aware) should have been a full v17 candidate
   evaluated standalone before adding leaf-side changes.
5. **The chooser's STRUCTURAL emit cap, not its value function,
   was the bottleneck.** v20's win was a chooser architecture
   change, not a value function change. Notice this earlier.

## Bundled / submitted

- `agents/v20/main.py` — 1-LOC change from v15 plus comment update
- `submissions/v20.py` — 317 KB bundle; parity OK 858 turns (1 seed)
- Kaggle sub 52721807, PENDING

## Rolling-last-2

- v20 (52721807, PENDING) — newest
- v15 (52710995, PENDING from earlier session) — also pending
- v13 (1063.8 settled) — evicted by v20 push

Floor risk: if both v15 and v20 settle below v13's 1063.8, we lose
the floor. Mitigation: v20 won 65.6% vs v15 locally — confident it
will at least match v15 live.

## Next-session first action

Per the session reflection: instrument v20 vs v15 specifically at
the Forrest crisis steps to confirm the v20 emit-rate lift translates
to actual capture-rate lift in live games (or expose what doesn't).
Don't write code until that instrumentation is done.

Secondary candidates (not started this session):
- F1 position-awareness (790 home ships ≠ 169 in-flight; PI flagged)
- Adaptive cap floor tightening (`max(8, ...)` → `max(4, ...)`)
  to bound the 1090ms wallclock tail
- Wait-N reserve-but-don't-emit behavior may be cutting emits in
  opening — investigate after v20 live signal
