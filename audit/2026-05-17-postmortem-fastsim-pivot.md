# 2026-05-17 Postmortem — v8_analytic value-head pivot to fast_sim

Branch: `claude/space-fleet-physics-engine-lrLE6`.
Session: kill-or-keep investigation of v8_analytic (50% vs nearest
ceiling after 5 sessions of work) → diagnosed root cause → pivoted
value head to fast_sim → measured. Verdict: KEEP per PI.

## What went wrong

### Bad decisions

1. **K=8 inherited from JAX-budget config when porting to fast_sim.**
   I copied the K constant across substrates with very different
   per-step cost (JAX vmap ≈ 200 ms per chunk vs fast_sim ≈ 1 ms per
   step). K=8 was a budget-derived value for JAX; on fast_sim with
   ~10-30× lower cost, K=15-25 is affordable. The cost regime
   should have been re-derived from first principles, not copied.
   Captured as friction tag
   `copy-K-from-jax-budget-to-fastsim`. Cost: one full Probe 1 cycle
   (~25 min wallclock + the embarrassment of the
   2/8 vs nearest regression intermediate). Decision-quality on the
   pivot DESIGN was good; tactical K-selection on day-of was sloppy.

2. **Strict Wilson-LB kill threshold in plan-mode WRAPUP.** Plan
   said "Wilson 95% LB < 40% vs nearest → kill". At n=8 with 4
   wins, LB=21.5% (well below 40%), but the LB cannot clear 40%
   unless ≥7/8 wins. The strict-read gate would have killed an
   architecture that's actually viable as a baseline. PI overrode
   correctly. Lesson: at small n, Wilson LB encodes uncertainty
   not poor quality; supplement it with substrate-viability checks
   (knob responsiveness, timing headroom, predicted-from-microtrace
   outcome matches actual bench outcome).

### PI-overrides

- **"Fix the timing tail first"** (Probe 1 nearest came back at 4/8
  with seeds 0+3 bursting 1000ms 17-20% of turns). I had proposed
  proceeding to Probe 2 with a fragility patch. PI's choice
  decoupled timing from quality cleanly and made the next bench
  data unambiguous. Width=3 fix took 1 commit, restored timing
  with no quality change.
- **"Plan it first"** before fast_sim implementation. I had begun
  writing the new module directly after the diagnosis. PI forced
  re-entry to plan mode, which produced a tighter implementation
  (atom-tagging design choice, explicit pre_committed prefix
  contract, k/n knob placement decided up front).
- **"We do not need to win, we just need to know if we can use the
  architecture as a strong baseline."** Reframed the success
  criterion from "≥5/8 wins vs nearest" to "is the substrate alive
  and tunable?" My internal threshold was the original plan's
  strict 5/8 = signal. PI's reframe was correct given session goal
  ("can we build on this?"), not absolute-quality ("does it win?").

### Rule-bypass / rule-gap failures

- **Rule 37 (consecutive-falsification cap on axes) not invoked early
  enough.** Through this session we touched 7+ axes: cap=64/128,
  cheap-rank formula, single-wave mirror, multi-wave mirror,
  defensive window, width 4→3→4→3, fragility plan. Each axis hit
  3-variant-falsification before structural diagnosis became the
  focal investigation. Should have run the micro-trace earlier
  (after 2-3 axes failed, not 7+). Lesson: when 2 axes hit
  Rule-37-N=3 in the same component, the next move is
  ROOT-CAUSE-INSPECT, not "try a different axis."

## Promotion candidates

Drafted per `.claude/skills/postmortem/SKILL.md` step 3. Each meets
at least one threshold (≥1 LB slot cost, OR ≥1h compute waste, OR
required PI override).

### [ ] `.claude/skills/kaggle-comp/improvements.md` — Wilson-LB + substrate-viability dual gate

**Tag:** `rule-37-strict-kill-vs-pi-override` (this session)

**Where to insert:** new section after "Single-model-first" rule
block.

**What to add:**

```markdown
## Kill-or-keep gates at small n

Wilson 95% LB thresholds at n≤8 are very conservative; require
≥7/8 wins to clear 40% LB. Using LB as the sole kill signal will
falsely kill substrates that have viable per-seed signal but small
total n. Supplement LB with substrate-viability checks:

- Does a single tunable knob produce monotone improvement in a
  micro-metric (e.g., "atoms beating no-op" as a function of K)?
- Did a predicted-from-micro-trace outcome match actual bench
  movement on at least one seed?
- Is per-turn timing well under budget with headroom (>200 ms p95
  slack) for layering more work?

Kill ONLY if Wilson LB < threshold AND none of the above viability
checks pass. Otherwise the LB is encoding small-n uncertainty, not
poor architecture.
```

**Why:** Friction `rule-37-strict-kill-vs-pi-override` (2026-05-17).
PI override required. Cost evidence: would have killed v8_analytic
mid-pivot before the K-sweep showed clean monotone signal.

### [ ] CLAUDE.md Rule 38 amendment — root-cause-inspect after 2 axes hit Rule-37 cap

**Tag:** N/A (rule-gap; root-cause-trace skipped 5 sessions)

**Where to insert:** Rule 38 currently covers fix-verification-
reproduces-failure-state. Add a complementary 38b for axis-
exhaustion → root-cause:

**What to add:**

```markdown
38b. **Two-axis-exhaustion triggers root-cause-inspect.** When two
distinct design axes (Rule 37 boundaries) each hit their
3-variant-falsification cap within the same component, the next move
is NOT a third axis — it is a root-cause inspection of the
component's leaf representation. Cost evidence: v8_analytic spent
5 sessions tuning cap / mirror / width / defensive variants (7+
total) before /tmp/micro_trace.py revealed the K=8 leaf was
bit-equal to no-op for 38/40 atoms. Root-cause-inspect on day 1
would have saved ~4 sessions of axis churn. Origin: 2026-05-17
fastsim-pivot postmortem.
```

**Why:** Friction tag `K-shorter-than-launch-eta-makes-value-head-
blind` represents a structural blocker that was invisible until
explicit leaf-value introspection. 5 sessions of tuning never
discovered it because no tuning surfaced the leaf.

### [ ] `.claude/skills/kaggle-comp/improvements.md` — cost-regime port checklist

**Tag:** `copy-K-from-jax-budget-to-fastsim` (this session)

**Where to insert:** new short bullet in the "Substrate / runtime
porting" section if it exists; else create one.

**What to add:**

```markdown
- When porting an algorithm between cost regimes (e.g., JAX vmap →
  Python rollout, CPU → GPU, scalar → vectorized), do NOT copy
  budget-derived constants (K, batch size, candidate cap). Re-derive
  from the new cost regime's per-call timing. Worked example: K=8
  on JAX-vmap value head was a budget compromise; on fast_sim with
  ~10-30× lower per-step cost, K=15-25 was affordable AND necessary
  for correctness (K must exceed median action ETA). Origin:
  2026-05-17 fastsim-pivot.
```

**Why:** Friction tag `copy-K-from-jax-budget-to-fastsim`. Cost: one
full bench cycle of 2/8 regression before the K knob was reset.

## Decisions I would not retake (given the same priors)

- Trying multi-wave per-step opp mirror as my third Tier-1-mirror
  variant. Two single-wave variants had already shown 0 quality
  improvement; the multi-wave was speculative axis-stacking at
  Rule-37 cap and regressed v7_0 1/4 → 0/4. Reverted as `8893e9b`.
  Could have skipped this in favor of root-cause inspection.

## What worked

- The structural diagnosis (micro_trace.py) was decisive in <30 min
  once attempted. The exact "38/40 atoms tie no-op" output is
  unambiguous and pointed directly at the leaf representation as
  the blocker, not at any pre-leaf machinery.
- PI's "plan it first" instruction caught a sloppy implementation
  start. The plan-mode design produced cleaner code than my initial
  draft would have.
- The fast_sim pivot was completed and benched in a single session.
  Substrate is provably alive (K-sweep monotone) and timed well
  (~3× cheaper than JAX path).
