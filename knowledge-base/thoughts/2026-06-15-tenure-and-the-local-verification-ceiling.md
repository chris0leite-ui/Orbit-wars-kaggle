# 2026-06-15 — Tenure term built; local verification is exhausted for our remaining drivers

Continued the positional-game program onto the **durability/tenure** factor
(the #2 loss driver, collapse). Built `tenure_penalty` (discount a capture by
whether we can HOLD it: enemy reachable force vs our defender + reinforcement
reach). Grounded it in three real CPMP-era collapse replays.

## What collapse actually is (3 real replays, traced)

- **1506374610 (CPMP):** thin-garrison **churn** at the contested mid-board —
  capture/lose/recapture planets at dist 30–37 with g5–g15 while CPMP keeps
  2–5× more ships in flight and grinds us down. (Tenure's target.)
- **645691379 (LuckyXC) / 2066324996 (Ebi):** we keep planet *count* (8, even
  17) but get **massively out-shipped** (opp reaches 825→3655 vs our 39–999) and
  overwhelmed — losing even g141/g200 planets. This is losing the *production/
  force race* (partly delayed under-expansion), NOT churn. Out of tenure's scope.

So "collapse" is two things; tenure attacks the churn half (stop pouring force
into captures we can't keep), as capture-SELECTION shaping — deliberately NOT
global defense (that was `garval`, which over-defended and lost 1181<1280).

## The term
`agents/producer/orbit_lite/durability.py::tenure_penalty`. Sibling of
`recapture_penalty`; the novelty is subtracting OUR reinforcement reach from the
enemy threat (net "can we hold it", not gross "can they reach it"). Gated
default-OFF; 6/6 unit; byte-identical off-path; 139 ms; `seq_strength_tenure`
variant. Committed, NOT submitted.

## The meta-finding (the durable lesson of the session)

**Local verification is exhausted for our remaining loss drivers.** Both things
we lose to — corner-neglect and collapse — happen only against **top-tier
opponents (CPMP ~1600+) we cannot run locally**:
- vs the bare `producer` (our strongest flag-agnostic local opponent), we win
  with **0 lost planets** on the collapse seeds — there is no collapse to fix,
  so tenure is provably inert there. Same for frontier (base already grabs the
  far planets vs weak opponents).
- the symmetric mirror reproduces the *failure* (e.g. corners left neutral on
  641308308) but **cannot isolate a one-sided fix**: both sides share the flag,
  and the arbitrary symmetric winner dominates any churn/holding metric.

Consequence: for these "lose-to-strong-opponents" drivers, **the ladder is the
only honest judge** — exactly what HANDOVER said. The reproduction loop is still
valuable for *falsifying mismatched fixes cheaply* (it killed frontier-for-corner
-neglect by showing the generation/scoring layer mismatch), but it **cannot
confirm a fix helps** when the failure needs a stronger opponent than we own.

## Where this leaves the program
Two framework terms built, gated, parity-safe, unit-verified, locally
unfalsifiable-as-helpful:
- **frontier (reach)** — mismatched to corner-neglect alone; only useful
  composed with a generation fix (`expand`), where it accelerates capture
  (turn 499→95 on 641308308).
- **tenure (durability)** — targets the confirmed #2 driver; real ceding-ground
  risk; mechanism sound.

The decision is now a ladder-spend question, not a build question. The remaining
under-built factor of the framework is **denial / option-severing** (the
differential/relative term). And the "out-produced and overwhelmed" half of
collapse points back at the economy/force-concentration problem the live
`expand` variant already targets.

## Reusable tools built this session (in /tmp, worth promoting if kept)
- `repro_mirror.py` — lists the exact neglected high-value far neutrals at
  steps 60/95/final (corner-neglect check).
- `collapse_trace.py` — from our peak, which planets we lose (dist/garrison) +
  enemy ships-in-flight (collapse mechanism).
- `churn_mirror.py` — per-side gains/losses/recaptures (churn signature).
- `replay_neglect.py` — ground-truth under-expansion/neglect from a real replay.
- Replay-seed loop: `kaggle competitions replay <ep>` → `info.seed` → reproduce.
