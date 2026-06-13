# 2026-06-13 — Referee blindness, and the ProducerLite monoculture

Two findings from fixing referee blindness, both with strategic weight
beyond the immediate task.

## 1. The strong public field is a ProducerLite monoculture

I pulled and inspected the top 5 public Orbit Wars kernels (V44 "1266
Elo", a GRU agent, exp51 "1400 Elo", "I'm Smarter", "veto-evacuation").
**Every one is a fork of Slawek Biel's Producer / `orbit_lite` base** with
different knob tunings — even the "GRU" agent is ProducerLite with a
disabled neural head. There is no rival architecture in the strong field.

Implications:
- Our `producer_plus` is on the correct base; we are not missing a
  superior architecture.
- The competition is a **knob/mechanism-tuning race within ProducerLite**,
  not an architecture race. Effort is best spent on better mechanisms on
  this base, evaluated faithfully.
- "Vendor a foreign architecture for diversity" is a dead idea — there
  isn't one to vendor.

## 2. Local referee blindness = response-veto mirror misprediction

The shot-MLP was inert locally but the live ladder draws ~33% low-P
attacks out of us. Cause: our response-veto predicts the opponent with a
1-ply ProducerLite mirror. Against ProducerLite opponents (which is what
our local panel was), the mirror is accurate, so it pre-filters our bad
attacks — they never happen locally. The low-P attacks on the ladder come
from opponents the mirror gets WRONG: weak, erratic, and the multi-front
chaos of heterogeneous 4P games.

Measured (our low-P attack share by setup): strong-ProducerLite-2P 0% →
non-producer-2P 4–9% → weak-2P 11% → **heterogeneous-4P ~38% ≈ ladder
33%**. A heterogeneous 4P panel reproduces the ladder. Fix is wired into
`state/TOOLS.md` + `scripts/play4p.py`.

**General rule:** our local panel's fidelity is governed by how badly the
response-veto mirror mispredicts the panel's opponents. To surface a
weakness, the panel must contain opponents the mirror gets wrong — NOT
more strong ProducerLite clones. This likely explains a chunk of the
historical "local A/B says X, ladder says not-X" friction across the
project: many of those A/Bs were strong-ProducerLite-2P, i.e. blind by
construction.

## What this unblocks

The shot-MLP — and any reject/redirect/over-extension mechanism — is now
locally testable for the first time, via the heterogeneous 4P panel,
before spending a live submission slot. The natural next step (PI's call)
is to re-run the shot-MLP filter against this panel and see whether it
helps where it actually fires; and, if pursuing the salvage, to test the
veto+redirect composition locally instead of burning a live probe.
