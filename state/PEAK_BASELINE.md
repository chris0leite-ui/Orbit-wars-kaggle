# PEAK_BASELINE — single source of truth for "build on top cleanly"

**Read this before proposing any change to the baseline agent.** The peak is
the strongest historical bundle we have ever submitted. Every recent
attempt to build on top of it has regressed; this document fixes the
common starting point and the protocol so the next iteration doesn't.

---

## What "the peak" is

| field | value |
|---|---|
| Git tag | `peak-1165` |
| Git commit | `458f663` (`fix: complete orbital-safety modeling pass (B1-B7)`) |
| Bundle SHA-256 | `9ec3af835a2aefcc91afa9fd586ca75246fc884cac035e3a00e83e5cbbcc6512` |
| Frozen anchor in tree | [`submissions/baseline_peak_1165_anchor.py`](../submissions/baseline_peak_1165_anchor.py) |
| Original submission | [`52912707`](https://www.kaggle.com/competitions/orbit-wars/submissions) (2026-05-22, μ=**1165.4**) |
| Byte-identical resubmit | `53013786` (2026-05-25, μ=**1144.6**) — same bundle, ~20μ rolling-pair noise |
| Current peak-restore | `53099429` (2026-05-28, pending) |

The frozen anchor is the **calibration reference** for every local A/B —
you compare a candidate against the actual byte-identical bundle that the
ladder rated at 1144-1165μ. Do not regenerate it; do not modify it.

---

## What is ACTIVE at peak (config that drives decisions)

The wrapper preamble at `agents/baseline_joint_aggr_consolidated_orbitfix/main.py`
sets 9 env vars. Of those, only the following are actually read by live
code paths:

| env var | value | effect |
|---|---|---|
| `BASELINE_JOINT_AGGR` | `1` | enable joint-target scoring in chooser |
| `BASELINE_JOINT_TOP_K` | `5` | top-K candidates per source for joint enumeration |
| `BASELINE_JOINT_MAX_PAIRS` | `60` | cap on joint pairs scored per turn |
| `BASELINE_REINFORCE_EMIT` | `1` | proposer emits reinforce candidates |
| `BASELINE_REINFORCE_ANTICIPATE` | `1` | reinforces respond to anticipated incoming threats |
| `BASELINE_ORBITAL_SAFETY` | `1` | B1-B7 fix: predict target/opp positions at our arrival |

## What is DORMANT at peak (declared but inert)

The wrapper ALSO sets the following — they look active but only feed the
dead `score_candidate` v2 scorer (line 250 of `chooser_trajectory.py` at
peak commit). They do NOT affect `score_candidate_v4`/`_v4_joint`, which
is what the chooser actually calls:

| env var | value | actual effect at peak |
|---|---|---|
| `BASELINE_NEUTRAL_BONUS` | `2.0` | **inert** — read only by dead v2 code |
| `BASELINE_NEUTRAL_EARLY_EXTRA` | `1.5` | **inert** — same |
| `BASELINE_NEUTRAL_EARLY_HORIZON` | `50` | **inert** — same |

**Critical lesson (2026-05-27).** Sub 53083109 "fixed" this by wiring
`NEUTRAL_BONUS_WEIGHT` into `score_candidate_v4` + `_v4_joint`. That
"fix" coincided with a ~20μ regression on the ladder (REVERT 53088099
landed at 1125.2 vs peak 1144.6). **An env var that looks dead may be
load-bearing precisely because it's dead.** Do not "wire up" dormant
env vars without an isolated n=32 A/B against the peak anchor first.

---

## Build-on-top protocol (mandatory checklist)

Every change layered on the peak goes through these gates. Skipping any
gate to "ship for early feedback" has, in our recorded history,
delivered a μ regression every single time.

### 0. Start from a clean tree

```bash
git checkout claude/<your-branch>
git diff peak-1165..HEAD agents/baseline/ lib/  # know what's already different
```

If the diff against `peak-1165` is non-empty, you are NOT building on the
peak — you are building on whatever else has been layered. Decide
explicitly whether to keep those layers.

### 1. Single env-var-gated change, default OFF

The change MUST be gated behind a `BASELINE_<NAME>` env var. The default
value MUST produce byte-identical behavior to the peak. Verify with a
unit test that imports the chooser/proposer at the default and asserts
identical scores against a synthetic fixture.

Why: the env-var gate is the only path that lets us submit the unchanged
bundle as a true control while testing the variant locally, and the only
back-out that doesn't require a code revert.

### 2. Local A/B against the peak anchor, NOT v7_0

Use `submissions/baseline_peak_1165_anchor.py` as the opponent. The
ladder evidence is unambiguous: winrate vs v7_0 does not predict winrate
vs peer-anchor (sub 53083109 was 48/64=75% vs v7_0 but 2/32=6% vs
peer-anchor, settled μ=921). v7_0 is too weak a sparring partner.

```bash
BASELINE_<NAME>=<value> python fast.py eval \
    <focal_agent> --vs submissions/baseline_peak_1165_anchor.py --n 32
```

### 3. Rule 45 gate — n ≥ 32 minimum

`n=8` does not predict ladder behavior. We have two days of evidence:

| sub | local A/B | live μ |
|---|---|---:|
| 53083109 | 48/64=75% vs v7_0 | **921** |
| 53099001 | 6/8=75% vs head_anchor | **680** |

Wilson 95%-lower-bound at n=8 is too wide; n=32 minimum, n=64 preferred.

### 4. Rule 43 gate — multi-opponent panel

```bash
python fast.py eval <focal_agent> \
    --vs-panel --require-h2h submissions/baseline_peak_1165_anchor.py \
    --geometry-panel --by-archetype --n 32
```

Pass criterion: per-opponent Wilson-lo ≥ 0.55 (the gate exists because
A/B loops — A beats B beats C beats A — are common in this game).

### 5. Bundle + Rule 46 parity smoke

```bash
python scripts/bundle_agent.py agents/baseline --force
cp submissions/baseline.py submissions/<your-bundle-name>.py
# prepend orbitfix preamble + your new env-var setdefault (see existing pattern)
python -m pytest tests/test_bundle.py -q
python fast.py play submissions/<your-bundle-name>.py --vs v7_0 --seed 7
```

### 6. Rule 42 push-coordination

```bash
kaggle competitions submissions orbit-wars | head -5
# Read what the rolling pair currently is; identify which slot will be evicted.
# If evicted-μ > predicted-μ → BLOCKED until explicit PI signoff.
```

Append a row to the push-claim board in `state/MULTI_BRANCH.md` with
branch, agent, predicted μ band, evicted (sub_id, μ), and PI signoff.

### 7. Submit (Rule 1)

One submission per approved go. Wait for it to settle. Update the
push-claim row with the settled μ.

---

## Anti-patterns to avoid

These are the recorded failure modes from the past two days:

- **"Just submit at n=8 for early feedback."** Sub 53099001 cost a μ=680
  rolling-pair slot. Sub 53083109 cost a μ=921 slot. Both were "early
  feedback" pushes that skipped Rule 45.
- **Wiring a dormant env var without isolation.** The NEUTRAL_BONUS-into-v4
  plumbing change (sub 53083109) is suspected to carry ~20μ regression.
  The change SHOULD have been gated independently from the other 3 fixes
  it was bundled with.
- **Stacking 3+ changes in one submit.** Sub 53083109 stacked Fix 1
  (NEUTRAL_BONUS), Fix 2a (holdability v2), Fix 3 (source-drain), Fix 4
  (follow-on). When it regressed, we couldn't attribute. The REVERT
  isolated 3 of the 4 (made them opt-in) but kept Fix 1 active —
  hence we're still ~20μ below peak.
- **Trusting the bundler at face value.** The orbitfix wrapper pattern
  (cross-agent import) is NOT directly bundled by `bundle_agent.py
  agents/baseline_joint_aggr_consolidated_orbitfix`; it requires
  bundling `agents/baseline` + hand-prepending the env-var preamble.
  See `scripts/bundle_agent.py` line 368 (`if agent_dir.is_dir():`) and
  the friction note in commit 458f663's message.

---

## How to use the peak anchor in scripts

```python
# Local A/B reference
PEAK_ANCHOR = "submissions/baseline_peak_1165_anchor.py"

# Frozen bundle SHA-256 — assert before A/B
EXPECTED_SHA = "9ec3af835a2aefcc91afa9fd586ca75246fc884cac035e3a00e83e5cbbcc6512"
```

If the anchor's SHA ever drifts from the expected value, somebody
modified the frozen bundle. Stop and restore from
`git show peak-1165:submissions/baseline_joint_aggr_consolidated_orbitfix.py`.

---

## Open questions worth answering before the next build-on-top

1. **What does removing the NEUTRAL_BONUS-into-v4 plumbing actually
   recover?** The REVERT (1125.2) kept that plumbing active; the peak
   anchor doesn't have it. The peak-restore submission `53099429`
   (pending) is the cleanest test we have run. If 53099429 settles
   ≥1140, we've confirmed the ~20μ gap was the plumbing — and we know
   to never add it back without isolated panel-clearance.
2. **Are any of the dormant env vars worth wiring up correctly?**
   `BASELINE_LEADER_FOCUS` (declared at peak, value 1.0 = inert) and
   the 3 NEUTRAL_BONUS knobs were designed as tilts. They may help if
   plumbed AND tuned at panel scale. The "fix-and-tune" path needs
   Rule 43 evidence before submit.
3. **Why does v7_0 winrate fail to predict peer-anchor winrate?** This
   is the single most important calibration question. The likely
   answer: v7_0 is a weak generalist; the peer-anchor's behavior is
   distinct enough that our local fixture (which often targets v7_0's
   specific weaknesses) generalizes poorly. Worth a dedicated
   investigation when ladder budget allows.
