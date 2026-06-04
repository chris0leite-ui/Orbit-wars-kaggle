# 2026-06-04 — open question: at which migration step does `producer_plus` overtake our live champion locally?

## Background

`producer_plus` is currently bit-identical to Producer. Producer beats our
champion ~60% locally (n=32, 2026-06-04). Producer's live μ ≈ 1200 vs our
champion's live μ ≈ 1185 — gap of ~15.

The migration plan adds our pieces in cheap-first order: adaptive K (Step 2)
→ opponent projection (Step 3) → multiple sizes (Step 4) → multi-source
coalitions (Step 5, biggest expected lift) → wait-then-fire (Step 6) →
comet aim (Step 7, conditional).

## Open question

After which step does `producer_plus_stepN vs champ_computeByShips_on` at
n=32 produce Wilson-lo ≥ 0.55? That is the moment the hybrid is locally
provably better than the live champion, and the first valid submission
candidate (Step 8).

Predictions to test:
- Step 2 (adaptive K) alone: likely NOT enough — Producer already beats
  champion by ~10pp without adaptive K; one cheap delta probably doesn't
  flip it the other direction.
- Step 3 (opponent projection) added: maybe — Producer's biggest blind
  spot is opp-launch-blindness, which is exactly what champion's chooser
  fixes via mirror_self_policy. Could be the inflection point.
- Step 5 (multi-source coalitions): most likely inflection if 3 isn't
  enough — explicitly Producer's structural blind spot.

## Why it matters

If Step 3 already clears, we can submit earlier and let TrueSkill rate it
while we keep adding Steps 4-6 as further improvements. If Step 5 is
required, we have a longer dev window with the live champion as backstop.

## How we'll know

Each step's n=32 A/B includes a "vs champion" arm in addition to the
"vs previous step" gate. Vs-previous-step decides whether to keep the
delta; vs-champion decides when we have a submission candidate.
