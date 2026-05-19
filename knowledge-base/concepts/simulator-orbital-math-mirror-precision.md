# Orbit Wars simulator: orbital math is NOT bit-exact mirror in float64

## Headline

The simulator places planets with 4-fold mirror symmetry and the
documented turn order treats both seats symmetrically. But the orbital
position update `(CENTER + r * cos(angle), CENTER + r * sin(angle))`
is **not bit-exact mirror under 180-degree rotation in IEEE-754
float64**. Mirror-pair planets drift apart by 1-3 ULPs from the first
orbital tick onward, capped by the magnitude's round-off ceiling
(stays bounded; does NOT accumulate).

## Verified mechanism

1. Initial planet placement IS bit-exact mirror (verified: every
   planet at (x, y) has a sibling at exactly (100 - x, 100 - y)).
2. `atan2(dy, dx)` for the mirror-pair returns values exactly π apart
   (verified at full precision).
3. After the first orbital step where the step counter is nontrivial,
   mirror-pair positions differ by 1 ULP in x or y. Example seed 1:
   ```
   pid 12 at (65.25412705784618, 66.29258953009081)
   pid 15 at (34.74587294215384,  33.70741046990919)
   exact 180-deg mirror of pid 12 would be
              (34.745872942153824, 33.70741046990919)
                       ^^^ off by 1.4210854715202e-14 (1 ULP at x≈65)
   ```
4. Root cause: `cos(α + π)` and `-cos(α)` are not bit-equal in
   float64 for arbitrary α (verified: ~66% of random α produce a
   mismatch up to |Δ| ≈ 1.67e-16). The simulator computes each of the
   four symmetric planet copies' positions INDEPENDENTLY via cos/sin
   of slightly-different angles, so the mirror identity isn't preserved
   at the last bit.
5. The gap does NOT grow over time — measured as oscillating in
   {0, 1, 2, 3} ULPs over the first 50 turns. IEEE-754 caps it at
   the round-off ceiling for the magnitude (~1.4e-14 at coordinate
   level ~65).

## Why it matters

For deterministic agents whose decisions are sensitive to a 1-ULP
distance difference (e.g., the comp's `data/main.py` baseline uses a
strict `<` for nearest-target selection), the mirror eventually breaks
at the AGENT-DECISION level. On seed 1 baseline self-play this
happens at turn 47; on other seeds the break-turn varies (seed 2:
167; seed 7: 66; seed 3: never breaks, draws).

For CRN-paired both-seats A/B evaluation in our pipeline: there is a
small built-in source of seat-conditional variance that doesn't
disappear with longer-run averaging on any single seed. It's bounded
and small (sub-ULP at the distance level on most board configurations),
but worth being aware of when interpreting tight Wilson intervals on
small effects.

## Replicate

Two probes live in `scripts/`:

- `scripts/probe_simulator_isolation.py` — runs two no-op agents
  (return `[]`), checks bit-exact planet mirroring each turn. First
  violation appears at turn 2 on seed 1.
- `scripts/probe_seat_mirror_break.py` — runs baseline-vs-baseline,
  walks turn by turn, reports first action-mirror break with
  full-precision distance dump. Finds turn 47 on seed 1.

Both depend only on `kaggle_environments` and the comp's `data/main.py`.
No internal-repo imports.

## Discussion-thread receipt

Posted publicly in response to the comment "The map is symmetric. You
should get identical game results if you swap seats and use a
canonical planet ordering" (May 2026). Verdict on the comment: right
at the math level, wrong about canonical ordering being a fix
(`obs.planets` is already canonical between seats; the float-precision
gap lives below the agent layer). See
`audit/2026-05-19-discussion-draft-seed-panel.md` for context on the
post that prompted the comment.

## Origin

2026-05-19 community-engagement session, Session C of
`claude/reverse-engineer-seat-geometry-BPJKs`.
