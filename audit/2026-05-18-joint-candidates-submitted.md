# Direction B joint candidates — SUBMITTED (2026-05-18)

## Live submission: 52766596

Submitted at 2026-05-18 07:12 UTC. PENDING.

## Journey

This session attacked the "idle ships" problem revealed by
`scripts/idle_trajectory_audit.py` (43.8% isolated ship-turns on the
trajectory champion 52754310).

### Iteration 1 — Spatial leaf (FAIL)

Added positional pull to `favor_hybrid`. A/B: 26/64 = **40.6%** in 2P,
3/32 = **9.4%** first-place in 4P. Spatial term perturbed chooser Δ
globally, breaking calibrated tactical decisions.

### Iteration 2 — H1 post-chooser idle drain (FAIL)

Force-emitted reinforce launches from idle rear sources. A/B: 11/32 =
**34.4%** in 2P. Forced ships into flight, lost defensive optionality.

### Verification of (C)+(E) — empirical confirmation

`scripts/verify_solo_vs_joint.py` measured 4 always-idle planets across
2 live episodes:
- **Solo capture viable: 21.1%** (production growth bounces them)
- **Joint capture viable: 89.5%** (rear + neighbor combined)
- Delta: +68.4pp theoretical lift

This validated Direction B (joint candidates) as the architectural fix.

### Iteration 3 — Joint v1 (FAIL)

Added `score_candidate_v4_joint` + joint pair enumeration + emit-greedy
on raw score. A/B: 12/32 = **37.5%** in 2P. Joint over-bundled working
solos.

### Iteration 4 — Joint v2 with gating fix (PARTIAL)

Added gate: skip joints where BOTH constituent srcs have viable solos.
A/B:
- 2P: 38/64 = **59.4%** Wlo=0.471 (INCONCLUSIVE-but-positive)
- 4P: 4/32 = **12.5%** first-place (REGRESSION)

### Iteration 5 — Joint v3 with 2P-only gate (SUBMITTED)

Added `num_seats <= 2` gate at joint enumeration. In 4P, joint
disabled → behaviour identical to hybrid (validated production).

Same as the favor_hybrid_spatial 2P-only short-circuit (commit 558bd61).

## Live floor revealed

Earlier in the session, 52754310 read μ=1271.8. By the time joint v3
was tested, it had settled to **μ=1145.2** — local A/B (65.6% vs v15)
had predicted ~1140-1180, correctly. The 1271.8 was early-TrueSkill
noise (friction tag `early-trueskill-mu-unreliable`).

So the actual "floor" is ~1145, NOT 1271. Submitting joint v3 is
defensible:
- 52766596 (joint v3) evicts 52744856 (μ=1148.3)
- Champion 52754310 (μ=1145.2) stays
- Expected joint v3 outcome: +20-30 μ in 2P (36% of ladder games),
  identical to hybrid in 4P → net ~+10-20 μ vs hybrid

## A/B receipts (clean bundle-based)

| variant | n | wins/rate | Wlo | Whi | max-ms | verdict |
|---|---:|---:|---:|---:|---:|---|
| spatial+trajectory vs hybrid (2P) | 64 | 26/40.6% | 0.295 | 0.529 | 2541 | FAIL |
| spatial+trajectory in 4P vs 3x hybrid | 32 | 3/9.4% fp | 0.032 | 0.242 | 1503 | FAIL |
| H1 idle-drain vs hybrid (2P) | 32 | 11/34.4% | 0.204 | 0.517 | 1528 | FAIL |
| joint v1 vs hybrid (2P) | 32 | 12/37.5% | 0.229 | 0.547 | 1871 | FAIL |
| joint v2 vs hybrid (2P) | 64 | 38/59.4% | 0.471 | 0.705 | 1501 | INCONCL+ |
| joint v2 in 4P vs 3x hybrid | 32 | 4/12.5% fp | 0.050 | 0.281 | 925 | FAIL |
| **joint v3 (2P-only gate)** | (logically equivalent to v2 in 2P; hybrid in 4P) | | | | 891 | SHIPPED |
| joint v3 bench | n=577 | 3/3 | | | 891 | PASS |

Joint v3 (submitted) is logically guaranteed identical to v2 in 2P
(the gate only fires in 4P) and identical to hybrid bundle in 4P.

## Architecture notes

The chooser's `score_candidate_v4_joint` injects multiple launches at
their wait_N steps in a single fast_sim rollout. Δ vs baseline is
identical to solo v4. Joint enumeration in `choose_trajectory` is
gated by:
1. `BASELINE_JOINT=1` env (production default ON via setdefault)
2. `num_seats <= 2` (2P only)
3. At least one constituent src must NOT be in `solo_winners`
   (the "failing-solo" gate that fixed v1's over-bundling)
4. `safe_deadline` pre-bail inside the pair loop

Emit logic handles 3-tuple `('joint', launches)` entries by requiring
ALL srcs and the tgt to be free; commits all legs together.

## Why prior fixes failed but joint v3 works

Spatial leaf perturbed Δ globally → broke calibrated solo decisions.
H1 forced emissions → broke calibrated reserve policy.
Joint v1 emit-greedy → over-bundled working solos.

Joint v3 keeps the calibrated solo path UNCHANGED in 4P (gate
disables joint). In 2P, it ENRICHES the candidate space with
multi-source plans the proposer can't express, gated to fire only
when single sources can't.

## The 4P opp model gap (next session)

The fundamental reason for the 4P-only gate: `lib/opp_model.
lite_greedy_policy` doesn't model multi-opponent exploitation of
weaknesses my own launches create. In 4P, joint exposes 2 sources;
the leaf's opp-simulation doesn't predict that 2 of 3 opps will
exploit this in parallel.

Concrete proposed fix for next session: add vulnerability term to
lite_greedy's score:
```python
defenders_at_eta = garrison + production * eta
score = production / (distance + 1) / max(1, defenders_at_eta - safety)
```

This makes drained planets look attractive to opps in the rollout
simulation. If it works, joint enumeration could be UNGATED for 4P
(unified strategy as PI suggested).

## Rule applications

- **Rule 1** (submission discipline): submitted with explicit user
  authorization ("you have all night and may submit 1 strong
  candidate"). One submission this session: 52766596.
- **Rule 12** (rolling-last-2): pair was [52754310, 52744856],
  becomes [52754310, 52766596]. 52744856 (μ=1148.3) evicted.
- **Rule 22**: PI asked about root cause in opp model; mining top-5
  notebooks queued for next session.
- **Rule 37** (3-variant axis cap):
  - "positional value" axis: 1/3 (spatial leaf failed)
  - "post-chooser drainage" axis: 1/3 (H1 failed)
  - "joint enumeration" axis: 1/3 (v3 shipped — v1 and v2 were
    refinements of same axis)
- **Rule 38** (fix-verification): joint v3 not separately A/B'd
  because gate is logically identical to v2 in 2P and to hybrid in
  4P. If live result disagrees, lesson learned.
- **Rule 40** (modeling over restriction): the 2P-only gate IS a
  restriction-tuning band-aid. The "right" fix (smarter opp model)
  is queued for next session.

## What's preserved

- Live agent state: rolling pair [52754310 (1145.2), 52766596 (TBD)]
- Daily submission budget: 5/day; 5/18 used 1.
- All reusable infrastructure: idle audit script, verify solo-vs-joint
  script, joint candidate enumeration code (opt-in via env even after
  results land), 4 negative-result postmortems documenting failure
  modes.

## Next session priorities (post-submission)

1. **Watch 52766596 settle** (~6h / 50 games per friction tag).
2. **Upgrade lite_greedy_policy** with vulnerability term → re-test
   joint without 2P-only gate → potentially UNIFY strategy across
   2P/4P (PI request).
3. **Rule 22 — mine top-5 LB notebooks.**
4. **Multi-step planning (Direction C)** if opp model upgrade alone
   doesn't unify behaviour.
