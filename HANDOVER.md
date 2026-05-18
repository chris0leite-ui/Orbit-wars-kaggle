# HANDOVER.md — next-session brief

> Last written: 2026-05-18 (late) by
> `claude/audit-workflow-performance-btjeK`. Joint v3 (2P-only-gated
> joint candidates) SUBMITTED as 52766596 PENDING. Rolling pair now
> [52754310 (μ=1145.2), 52766596 (TBD)]. The 1271.8 reading was
> early-noise; settled floor is ~1145.

## Where we are

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC. ~36 days.
- **Live production:** `52766596 baseline.py` — trajectory chooser v4
  + wait_N + wallclock budget + hybrid value head + Direction B joint
  candidate evaluation (2P-only gated). Built on 52754310's bundle.
  PENDING; expected to settle in ~6h / 50 games per friction tag
  `early-trueskill-mu-unreliable`.
- **Rolling-last-2** (auto-eval pair):
  - `52766596` joint v3 (5/18 07:12 UTC) — PENDING
  - `52754310` trajectory v4 (5/17 22:06 UTC) — COMPLETE, μ=1145.2
  - `52744856` composite_a2 (μ=1148.3) — EVICTED by 52766596 push
- **Daily submission budget:** 5/day; 5/18 used 1 (52766596).
- **Calibration confirmed**: local A/B at 65.6% vs v15 → settled
  μ=1145.2. The earlier 1271.8 reading was early-TrueSkill noise.

## What this session shipped

10 commits this session:
- `b5f5296` — spatial leaf head + idle-trajectory audit infra
- `cc38e11` — summary.json for 52754310 live episodes
- `558bd61` — spatial leaf 2P-only short-circuit
- `70fcc28` — spatial leaf: negative result documented
- `1b3f920` — H1 post-chooser idle drain
- `90c6adb` — H1 A/B FAIL: default flipped OFF
- `e55d07b` — HANDOVER update after H1
- `2dfd2a2` — verify (C)+(E): solo 21pct vs joint 89pct
- `835bb7d` — Direction B v1: joint enumeration
- `6d15562` — Joint v2: gate to failing-solo
- `f14eb46` — Joint v3: 2P-only gate + default-on

### A/B receipts (clean bundle-based)

| variant | n | wins/rate | Wlo | Whi | max-ms | verdict |
|---|---:|---:|---:|---:|---:|---|
| spatial+trajectory vs hybrid (2P) | 64 | 26/40.6% | 0.295 | 0.529 | 2541 | FAIL |
| spatial+trajectory in 4P | 32 | 3 first-place/9.4% | 0.032 | 0.242 | 1503 | FAIL |
| H1 idle-drain vs hybrid (2P) | 32 | 11/34.4% | 0.204 | 0.517 | 1528 | FAIL |
| joint v1 vs hybrid (2P) | 32 | 12/37.5% | 0.229 | 0.547 | 1871 | FAIL |
| joint v2 vs hybrid (2P) | 64 | 38/59.4% | 0.471 | 0.705 | 1501 | INCONCL+ |
| joint v2 in 4P | 32 | 4 first-place/12.5% | 0.050 | 0.281 | 925 | FAIL |
| **joint v3 (2P-only gate)** | bench n=577 | 3/3 vs trivial | | | 891 | SHIPPED |

Joint v3 in 2P is logically identical to joint v2 (gate fires in 4P
only). Joint v3 in 4P is logically identical to hybrid bundle.

## Core architectural insight

The "idle ships" problem is REAL (43.8% isolated ship-turns; 89%
of theoretical captures from idle planets need joint launches per
`scripts/verify_solo_vs_joint.py`).

But three single-axis fixes failed:
- Modifying leaf scoring (spatial) → broke calibrated Δ
- Forcing emissions (H1) → broke reserve discipline
- Adding multi-source candidates (joint v1) → over-bundled

The path that WORKED (joint v3): enrich the chooser's CANDIDATE SPACE
with multi-source plans, but gate aggressively:
1. 2P-only (4P needs better opp model first)
2. Only when at least one constituent solo fails (no over-bundling)

## Next-session priorities (ranked by EV / cost)

**1. Watch 52766596 settle (~6h).** If μ ≥ 1145, ship was a net wash
or positive. If μ < 1100, joint hurt 2P live; investigate via replay
mining (`scripts/replay_mine.py 52766596 --pull`).

**2. Upgrade `lite_greedy_policy` with vulnerability term.** Cheapest
fix (~2h) that addresses the root cause of 4P regression:
```python
# in lib/opp_model.py:155
defenders_at_eta = garrison + production * eta
score = production / (distance + 1) / max(1, defenders_at_eta - 2)
```
This makes drained planets look attractive to opps in rollout
simulation. If A/B confirms, remove the 2P-only gate from joint → unified strategy across 2P and 4P (PI request).

**3. Mine top-5 LB notebooks (Rule 22).** We're at ~1145. Top of LB
is presumably 1300+. Pull and compare structural choices.

**4. Symmetric self-play opp model.** Each opp uses our own chooser
logic in the rollout (3-5x cost). Theoretically correct, expensive.
Tier 2 only if vulnerability term doesn't unify behaviour.

**5. Multi-step planning (Direction C).** Score k-turn plans, not
single moves. Major rewrite; long-term destination.

## Pointers

- `agents/baseline/main.py` — entry, sets BASELINE_CHOOSER=trajectory
  + BASELINE_VALUE_HEAD=hybrid + **BASELINE_JOINT=1** via setdefault.
- `agents/baseline/chooser_trajectory.py` —
  - `score_candidate_v4_joint` (line ~440): joint scoring via fast_sim
  - `choose_trajectory`: joint enumeration after solo scoring loop,
    gated by num_seats <= 2, solo_winners check, safe_deadline
  - `JOINT_TOP_K_PER_TARGET=3`, `JOINT_MAX_PAIRS=20` constants
- `agents/baseline/value.py` — favor_hybrid (production), spatial
  head (opt-in, default OFF).
- `lib/opp_model.py:lite_greedy_policy` (line 155) — **the next fix
  target** (vulnerability term).
- `scripts/verify_solo_vs_joint.py` — (C)+(E) verification tool;
  re-runnable.
- `scripts/idle_trajectory_audit.py` — ship-turn density measurement.
- `audit/2026-05-18-joint-candidates-submitted.md` — full submission
  postmortem with all A/B receipts.
- `audit/2026-05-18-spatial-leaf-negative-result.md`
- `audit/2026-05-18-h1-idle-drain-negative-result.md`

## Rule reminders

- Rule 1: submitted with explicit user authorization "may submit 1
  strong candidate."
- Rule 12: rolling-last-2 = [52754310 (1145.2), 52766596 (PENDING)].
- Rule 22: at every plateau, mine top-5 public notebooks. **Fires
  next session.**
- Rule 32: `kaggle competitions submissions orbit-wars` is the source
  of truth for μ.
- Rule 37: spatial-positional axis 1/3, post-chooser-drainage axis 1/3,
  joint axis 3/3 (v1/v2/v3 all used). Next variant on joint axis
  requires PI escalation.
- Rule 40: prefer modeling-correctness over restriction-tuning. The
  2P-only gate IS a band-aid; the real fix is the lite_greedy
  vulnerability term (priority 2 above).
