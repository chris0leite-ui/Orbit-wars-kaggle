# Trajectory chooser v4 + wait_N shipped — submission 52754310 (2026-05-17)

> End-of-session record. Direction A finished (richer leaf), wait_N
> port closed the remaining gap to composite_a2, wallclock budget
> brought max turn-ms under cap, submitted as production default.

## What landed

Single agent default flip — trajectory chooser is now the production
path in `agents/baseline/main.py`:

```python
os.environ.setdefault("BASELINE_VALUE_HEAD", "hybrid")
os.environ.setdefault("BASELINE_CHOOSER", "trajectory")  # NEW
```

Three commits this session:
- `da2473f` — wait_N>0 candidates threaded through v4 (was previously
  filtered with `if int(wait_N) != 0: continue` at the chooser
  iteration boundary).
- `10c9601` — wallclock budgeting (affordable_validate_cap +
  safe_deadline pre-bail, mirrors composite chooser).
- `f192cf4` — default-on BASELINE_CHOOSER=trajectory.

Submission `52754310 baseline.py` PENDING at 22:06 UTC.

## A/B receipts (n=64 vs v15, BASELINE_VALUE_HEAD=hybrid)

| variant | wins | rate | Wlo | max-ms |
|---|---:|---:|---:|---:|
| Pre-fix (no cap, sun-safety bug) | — | — | — | — |
| v4 fire-now only (pre-wait_N) | 31/64 | 48.4% | 0.36 | — |
| **v4 + wait_N (pre-wallclock-fix)** | **42/64** | **65.6%** | **0.534** | **2416** |
| v4 + wait_N + N_VALIDATE=60 cap | 37/64 | 57.8% | 0.456 | 1016 |
| **v4 + wait_N + deadline-only (shipped)** | **42/64** | **65.6%** | **0.534** | **1077** |
| composite_a2_hybrid (reference) | 40/64 | 62.5% | 0.503 | 1292 |

The N_VALIDATE=60 ablation matters: cap binding too tight (n_aff
floored to 8 on heavy turns) cost ~8pp of winrate. N_VALIDATE=200
with safe_deadline pre-bail preserves the full lift.

Bench post-fix: p50=138 p95=412 p99=566 max=623 over_1000ms=0.

## What the agent actually IS

Mostly v15-era composite pipeline with two surgical edits:

1. **Deterministic admissibility filter** (`predict_fleet_fate`):
   sun / oob / expired-comet / comet-collision-en-route candidates
   rejected before scoring. Replays of composite_a2 showed ~0.2pct
   of fleets dying to the sun — this should now be 0pct.
2. **Slightly different leaf eval routing**: v4's score_candidate
   runs fast_sim through the proposer's horizon (max(wait+eta+settle,
   25)) then computes `Δ favor = leaf − baseline_favors[horizon]`.
   Composite chooser's score_action does effectively the same with
   wait_N handling at line 60-73 of chooser.py.

Everything else — proposer, value head (favor_hybrid), emit rule
(1-per-src, 1-per-tgt, wait reservation) — is unchanged.

## What this is NOT

The "trajectory-first" architectural reframe filed in
`knowledge-base/concepts/trajectory-first-architecture.md` and
`probability-of-winning-framework.md` was DELIBERATELY scoped down:

- **Direction A (richer leaf)**: shipped via v4's favor-Δ scoring.
- **Direction B (joint candidates)**: NOT shipped. Each candidate
  still scored independently; emit is greedy non-dogpile.
- **Direction C (multi-turn plan)**: NOT shipped. Agent replans from
  scratch every turn; wait_N>0 is the closest thing to sequencing.

Rule 37 (3-strikes axis cap) bound us: v1/v2/v3 all hit 0/32 on the
"replace composite entirely" axis. v4 succeeded by making the
trajectory chooser LOOK LIKE composite_a2 with a smarter filter.

## Performance prediction

- Local A/B 65.6% vs composite_a2's 62.5% on the same A/B = within
  noise (both INCONCL at Wlo gate).
- Expected μ landing: **between 1140 and 1180** (composite_a2 at 1158.6
  settling, this should match or slightly exceed in 2P; 4P untested
  locally so uncertainty is asymmetric).
- Settling window: 6h / 50 games per the `early-trueskill-mu-unreliable`
  friction tag.

## Next session: Direction B

PI directive at session end: "let's go into direction B, wrap up,
we will start next session."

Direction B = joint candidate evaluation. The current chooser scores
each candidate independently (single Δ favor per launch). Direction
B scores combinations: "launch A→X AND simultaneously B→Y" as one
decision, capturing interaction effects.

Concrete first-action shape for next session:
1. Beam-search style enumerator: top-K independent candidates form
   the seed pool; pairs/triples drawn from the seed pool form the
   joint candidate set.
2. Score each joint candidate by running fast_sim with ALL launches
   in the set injected at their respective wait_N steps.
3. Emit: pick the highest-Δ joint set; emit its constituent fire-now
   launches; reserve src/tgt for wait>0 components.

Open questions to think through:
- Pair / triple enumeration cost vs wallclock. K=5 seeds → C(5,2)=10
  pairs + 5 singles = 15 candidates. Doable. K=10 → 55 candidates.
  Tight but might fit.
- Does joint scoring need a different baseline? Currently each
  candidate's Δ is against the SAME idle baseline. For joint, we
  probably want each constituent's marginal Δ given the others.
- Defense vs offense joints. A defense-only set vs an offense-only
  set vs mixed should all be evaluated.

## Friction filed today

- `default-on-restriction-tuning-binds-too-tight` — N_VALIDATE=60
  cap was the obvious mirror of composite's pattern but cost 8pp
  winrate. The cap should be permissive (200+); safe_deadline is
  the real binder. Rule 40 applies (modeling-correctness over
  restriction-tuning).

## Branch state

- `claude/audit-workflow-performance-btjeK`, 39 ahead of origin/main.
- Tip: `f192cf4` (default-on BASELINE_CHOOSER=trajectory).
- Submission: `52754310 baseline.py` PENDING.
- Rolling pair: [composite_a2 52744856 μ=1158.6, trajectory 52754310
  PENDING]. v20 (1082.4) evicted.
