# 2026-06-13 — Referee blindness diagnosed and fixed (heterogeneous 4P panel)

**Context.** The shot-MLP probe (sub 53595717) showed our local A/B panel
cannot reproduce the conditions our agent faces on the live ladder: the
filter was inert locally (0% of attacks below threshold) yet the live
field draws ~33% low-probability attacks out of us. PI directive: "fix
referee blindness first." This note records the diagnosis and the fix.

## Step 1 — the strong public field has no architectural diversity

Pulled 5 of the top public Orbit Wars kernels and extracted their agents:

| kernel | votes | what it is |
|---|---|---|
| itzzomkar/…v44…self-contained | 10 | **ProducerLite** (writes its own `orbit_lite/`) |
| caoyupeng/v2-gru | 20 | **ProducerLite** + optional GRU head (disabled, no weights) |
| ramesh888/exp51…1400-elo | 2 | **ProducerLite** (broken as published: `max=1..0` syntax error) |
| romantamrazov/…i-m-smarter | 60 | **ProducerLite** |
| anthonytherrien/…veto-evacuation | 34 | **ProducerLite** (evacuation defense) |

**Every strong public agent is a ProducerLite fork** (Slawek Biel's
`orbit_lite` base, `ProducerLiteConfig` / `plan_lite_waves` /
`ProducerLiteRuntime`) with different knob tunings. The competition is a
knob-tuning race within one architecture, not an architecture race. So
"vendor a foreign architecture to diversify the panel" is moot — there
isn't one in the strong field. (Strategic corollary: our producer_plus IS
on the right base.)

## Step 2 — vendoring strong ProducerLite opponents did NOT un-blind

Vendored the two clean ones (`panel_smarter`, `panel_veto`) and measured
how often our 0.15 shot-MLP bundle (P0) emits attacks the model scores
below threshold (5 seeds each):

| opponent | attack waves | P<0.15 | P<0.30 | P<0.50 |
|---|---|---|---|---|
| producer (old referee) | 240 | 2.1% | 10.0% | 36.2% |
| panel_smarter (NEW, strong) | 97 | **0.0%** | 0.0% | 0.0% |
| panel_veto (NEW, strong) | 97 | **0.0%** | 0.0% | 0.0% |

Strong ProducerLite opponents elicit FEWER low-P attacks, not more.

## Step 3 — the inverted hypothesis, confirmed

Our response-veto models the opponent **as a ProducerLite planner**
(1-ply mirror). Against ProducerLite opponents that mirror is accurate, so
it pre-filters our bad attacks before they're emitted. The live ladder's
low-P attacks therefore come from opponents the mirror CANNOT model —
weak, erratic, or non-ProducerLite agents — and from multi-front 4P chaos.
Tested with agents we already have:

| local setup | attack waves | **P<0.15** | P<0.30 | P<0.50 |
|---|---|---|---|---|
| 2P vs v7_0 (non-producer) | 179 | 8.9% | 12.8% | 33.0% |
| 2P vs v3.5.1 (non-producer) | 152 | 3.9% | 7.9% | 22.4% |
| 2P vs nearest (weak) | 198 | 11.1% | 12.6% | 18.2% |
| **4P mixed [producer,v7_0,nearest]** | 309 | **36.6%** | 54.4% | 85.4% |
| **4P mixed [v7_0,v4_planner,v3.5.1]** | 276 | **38.8%** | 49.6% | 69.9% |
| — live ladder (sub 53595717, ref) | — | (~33%) | — | — |

**A heterogeneous 4P pool reproduces the live launch-quality distribution
(37–39% low-P ≈ ladder 33.4%).** Blindness scales with how well the mirror
predicts the field: strong-ProducerLite-2P (mirror perfect) → 0%;
non-producer-2P → 4–9%; weak-2P → 11%; heterogeneous-4P → ~38%.

## The fix (what future mechanism A/Bs should use)

Stop A/B-ing against a strong-ProducerLite monoculture. Use a
**heterogeneous 4P panel** that mixes architectures and strengths:

```
python scripts/play4p.py --focal <candidate> \
    --bg producer,v7_0,nearest --rotate-seats --seeds <list> --workers 4
```

Recommended backgrounds (rotate seats; both reproduce the ladder's
launch-quality spread): `producer,v7_0,nearest` and `v7_0,v4_planner,v3.5.1`.
Registered opponents now include `producer`, `panel_smarter`, `panel_veto`
(vendored publics) plus the existing non-producer lineage. See
`state/TOOLS.md`.

## Consequences

- The shot-MLP (and any reject/redirect mechanism) is now **locally
  testable** for the first time — re-evaluate it against the heterogeneous
  4P panel before spending another live slot.
- The two strong vendored publics (`panel_smarter`, `panel_veto`) are kept
  as panel members for behavioral variety, but they are NOT what un-blinds
  the panel — the 4P heterogeneity + non-producer/weak inclusion is.
- Reusable rule of thumb: our local panel's fidelity is governed by mirror
  misprediction. To surface a weakness, the panel must contain opponents
  the response-veto mirror gets WRONG.

## Harness caveat — torch contention corrupts multi-worker 4P eval

Validating the `play4p` winrate wrapper surfaced a contention trap (same
class as the elegant-dijkstra "torch thread-thrashing" finding). Same
config (focal 0.15 bundle, bg `producer,v7_0,nearest`, seed 7, 4 seat
rotations):

| run | focal turn-ms p50 / p95 / max | over-1000ms turns | result | verdict |
|---|---|---|---|---|
| `--workers 4`, threads unpinned | 2083 / 5090 / 6209 | 183 | 0/4 | FAIL-wallclock |
| `--workers 1` | 39–68 / ~120 / 237 | 0 | 2/4 | PASS |
| `--workers 4` + `OMP/MKL/OPENBLAS_NUM_THREADS=1` | — / 129 / 152 | 0 | 2/4 | PASS |

Four parallel worker processes each spawn default-thread torch agents and
oversubscribe the CPU; turns time out and the **winrate itself flips**
(0/4 → 2/4). Always pin torch threads for multi-worker local eval, or run
`--workers 1`. The un-blinding measurement above (the low-P attack tables)
is unaffected — it ran serially in one process, no cross-game contention.
