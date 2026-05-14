# Postmortem — 2026-05-14 research-competition-analysis-2R8I3

## What went wrong

- **Skipped bootstrap.** Decided "Day-1 audits are read-only grep, no
  bootstrap needed" and went straight to changes. Worked for items
  H/N/K by luck; would have broken [A]'s A/B which needed
  `data/main.py`. PI caught it with "have you bootstrapped
  carefully?" — load-bearing override. **Prior-time defense:** "this
  is read-only" was true for the *next* item but the chain ahead
  included A/B work; I should have run bootstrap up-front anyway.
  See `agent-introspection-skipped-bootstrap` friction.

- **Silent broken bundle on first PV smoke.** Added `lib/scoring.py`
  but `bundle_agent.DEFAULT_LIB_ORDER` didn't list it. Bundler
  stripped `from lib.scoring import pv_horizon` and produced
  importable-but-broken bundles. `test_bundle.py` parity uses
  v1_orbitfix (no snipe missions), so the regression was invisible
  to the test suite. Cost: 1 wasted 8-seed smoke (~5 min) + ~10 min
  diagnosis. See `new-lib-module-silently-broken-bundle` friction.

- **HAV-1 binary drop launched without fixture sanity-check.** Plan
  said "if `expected_hold ≤ 0` drop the Mission." Should have asked:
  "how many Missions does that drop in a typical mid-game world?"
  Quick `propose_snipe_missions(fixture)` call would have shown
  near-total pruning. Instead burned a 5-min smoke to discover the
  0/16 result. Cost: ~5 min smoke + ~5 min diagnostic. Soft-floor
  fix landed within 15 min after; not catastrophic but avoidable.

- **Compute burned on Holding-tier smoke after architectural lesson
  was clear.** By the time HAV-1 failed three times, the
  K-rollout-dominance pattern was at 8 falsifications. I ran Holding
  tier "for completeness." Cost: ~10 min wallclock that produced a
  same-pattern data point. Better use of that budget: write the
  postmortem earlier and start the architectural pivot.

- **8-seed smoke wallclock budget tight for bigger-incumbent
  variants.** Holding-tier smoke timed out at the default 10-min
  budget because bigger incumbent → drop-one explores more candidates
  → per-turn cost spikes. Should have predicted this and set
  `timeout 1500` (25 min) up-front, not after the first kill.

## Frictions logged this session

Eight entries under `## 2026-05-13` in `audit/friction.md`:

1. `fresh-sandbox-no-deps-installed` — blinker RECORD conflict on
   `pip install`; `--ignore-installed` workaround.
2. `data-main-py-missing-on-fresh-clone` — three smoke tests fail
   when `bootstrap.sh` hasn't run.
3. `bundler-overwrites-tracked-submission` — `bundle_agent.bundle(out_dir=submissions/)`
   overwrote the tracked `submissions/v7_0_drop_one.py`; required
   `git checkout HEAD --` recovery.
4. `bootstrap-data-check-false-positive` — bootstrap's data-presence
   guard skips download when `data/shot_validator/` exists.
5. `agent-introspection-skipped-bootstrap` — self-callout, PI override.
6. `new-lib-module-silently-broken-bundle` — `lib/scoring.py` not in
   `DEFAULT_LIB_ORDER` → broken bundle, invisible to v1_orbitfix
   parity gate.
7. `ab_variants-hardcoded-v3_snipe-agent` — added `--agent PATH` flag.
8. `no-cgroup-v2-no-systemd-bus` — Docker is the right fallback for
   future cgroup probes.

## Promotion candidates (PI ratification pending)

### [ ] [CODE-COMP-DISCOVERED] bundle_agent.py: refuse to overwrite a tracked submission file

**Tag:** `bundler-overwrites-tracked-submission` (Orbit Wars 2026-05-13)

**Where to insert:** `## Pending` block in `improvements.md`.

**What to add:**

When `scripts/bundle_agent.bundle(agent_dir=X, out_dir=Y/)` would
produce `Y/X.py` and that path is a tracked git file, EITHER:
- (a) refuse with a clear error directing the caller to pass
  `out_name=...` or a temp dir, OR
- (b) write to `Y/_pending_X.py` and let the caller atomic-rename
  after a parity sanity check.

Today's incident: an inline bundling helper for `v7_pv` accidentally
deleted the live-reference `submissions/v7_0_drop_one.py`. Caught by
the stop-hook before push; would have been a real loss in a hurry.

**Why:** `audit/friction.md::bundler-overwrites-tracked-submission`.
Cost: 0 minutes (stop-hook caught it) but credibility hit + scary
diff to PI. Generalises: any auto-named output that collides with
a tracked artifact is a footgun.

---

### [ ] [CODE-COMP-DISCOVERED] bundle parity gate must cover the full mission proposer pipeline

**Tag:** `new-lib-module-silently-broken-bundle` (Orbit Wars 2026-05-13)

**Where to insert:** `## Pending` block, near the existing
`bundler-missing-block-e-modules` candidate (same family).

**What to add:**

`tests/test_bundle.py::test_bundle_outcome_matches_original_on_fixed_seeds`
currently runs only on v1_orbitfix. Add a parallel test that uses
**v3_snipe** (exercises full snipe + reinforce + recapture mission
proposer chain + WorldModel + settle_plan) as the parity agent. The
v1 fixture passes today even when snipe-only lib modules are missing
from the bundle — that's how the H16 broken-bundle bug slipped past.

**Why:** `audit/friction.md::new-lib-module-silently-broken-bundle`.
Cost: 1 wasted 8-seed smoke + 10 min diagnosis. The existing pending
candidate ("AST-based discovery") is the long-term fix; this is a
short-term net.

---

### [ ] [CROSS-CUTTING] CLAUDE.md addendum: bootstrap-first rule

**Tag:** `agent-introspection-skipped-bootstrap` (Orbit Wars 2026-05-13)

**Where to insert:** new rule in CLAUDE.md `## Operating rules`
section, OR addendum to existing Rule 32 (session-start git fetch).

**What to add:**

> **Rule 32a — Session-start bootstrap.** When the session will touch
> any test/A-B/submission code path (even if the immediate task is
> read-only audit), run `bash bootstrap.sh && python -m pytest tests/ -q`
> as the FIRST action after `git fetch`. The cost is ~5 min and the
> benefit is catching env / dep / data-shipping gaps before they
> blow up mid-A/B.

**Why:** `audit/friction.md::agent-introspection-skipped-bootstrap` +
`bootstrap-data-check-false-positive`. PI override required to
correct. Cost: ~5 min recovery + credibility hit.

---

### [ ] [CROSS-CUTTING] Heuristic-pre-A/B sanity-check protocol

**Tag:** `binary-drop-killed-all-targets-before-ab` (HAV-1 incident,
Orbit Wars 2026-05-13)

**Where to insert:** new bullet in CLAUDE.md `## Operating rules`,
OR adjacent to existing Rule 6 (Heuristics before heavy compute).

**What to add:**

> **Rule 6a — Smoke a heuristic on a fixture before paying for an
> A/B.** When the new heuristic is a *filter* (drops candidates) or a
> *cap* (compresses scores), run `propose_*(fixture_world)` once and
> verify the output size + score distribution looks sane. Spending
> 5 min on a fixture call to confirm "we still emit Missions" is
> strictly cheaper than spending 5 min on an A/B that returns 0%
> winrate because the proposer emitted nothing.

**Why:** HAV-1 binary drop returned 0/16 because the pruning was
near-total. A 30-second fixture call would have shown
`len(missions) == 0` and the bug.

---

### [ ] [CODE-COMP-DISCOVERED] Knowledge: K-step rollout dominates score-shape heuristics

**Tag:** `k-rollout-absorbs-scoring-shape-interventions` (cumulative,
Orbit Wars 2026-05-13)

**Where to insert:** new doc
`knowledge-base/concepts/k-rollout-dominance.md`, referenced from
improvements.md.

**What to add:**

> When an agent's chooser already runs a K-step rollout (e.g. v7's
> drop-one with K=10), heuristic additions at the proposer or scoring
> layer that REWEIGHT existing candidates **regress** the agent. The
> rollout has already evaluated the dynamics those heuristics try to
> add at proposal time, and pre-discounting double-counts. The only
> productive proposer-layer interventions are ones that change
> WHICH candidates exist — value-SHAPE reshapes (PV/H16: the only
> PASS) or candidate-FILTER prunings (DROP_COMET: directionally
> positive). Adding new Mission classes regresses unless paired with
> wider candidate enumeration (portfolio search) that gives the
> chooser alternative incumbents to compare across.

**Why:** Eight independent falsifications today (B/C/D/F/Renaissance/
HAV-1/HAV-2/Holding-tier), all with the same monotonic-regression
pattern. This is the load-bearing lesson of the session and should
inform tomorrow's architecture choice (portfolio search vs deeper K
vs opponent ensemble). Audit cross-links:
- `audit/tournaments/ab-20260513T19{53,58}*.json` (danger3)
- `audit/tournaments/ab-20260513T20{05,10}*.json` (FLEET_OVERCOMMIT)
- `audit/tournaments/ab-20260513T202{1,9}*.json` (PRE_REINFORCE)
- `audit/tournaments/ab-20260513T235144Z.json` (Renaissance all-on)
- `audit/tournaments/ab-20260514T00*.json` (Renaissance per-mission)
- `audit/tournaments/ab-20260514T01*.json` (HAV-1 + Holding)

## PI additions (from step 4)

(pending — see "Anything you'd add?" prompt below)

## Framework version at session-end

- Commit SHA: `3231bc9af6bf78ec79cd0faf7bf64078d71d54cf`
- Branch: `claude/research-competition-analysis-2R8I3`
- Active rules: CLAUDE.md `## Operating rules — concise` (1..36)
- Loaded skills this session: `kaggle-comp`, `postmortem`, `update-config`
