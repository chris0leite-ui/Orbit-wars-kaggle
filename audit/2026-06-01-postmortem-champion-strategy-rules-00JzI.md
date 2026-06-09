# Postmortem — 2026-06-01 champion-strategy-rules-00JzI

Decision-quality basis, not outcome basis.

## What went wrong
- **PI-override #1 — submit size-balance on n=16 single-opponent evidence**
  (Rules 43/45 overridden). Risk + μ=711 precedent were surfaced; PI chose to
  probe. Settled ~1136 (mild regression vs ~1183 champion). Decision quality:
  acceptable — gate-gap surfaced, PI decided with full information.
- **PI-override #2 — submit expansion-credit with no completed winrate A/B**
  (PI: "add your suggestion, then submit"). Settled ~1086. Acceptable as a
  PI-directed calibration probe; no-evidence state was logged on the claim
  board. Net: rolling-pair floor drifted ~1183 → ~1086/1138 across the two
  probes — a real cost of the probe-heavy cadence.
- **Rule-gap — wrong A/B instrument.** The expansion fix was A/B'd vs the
  champion mirror, which is itself a hoarder and therefore structurally cannot
  exercise (differentiate) an expansion gain. Produced a flat 40% "non-gain"
  that could have wrongly killed a live-relevant idea. No existing rule names
  this opponent-type/instrument-selection confound.
- **Operational — background-job mortality.** Long A/Bs (~30–50 min) were
  repeatedly killed by container reclaim across idle windows; one completed
  result was nearly lost when a relaunch truncated the log. Cost: ~2 lost or
  near-lost runs.

## Frictions logged this session
(`audit/friction.md` ## 2026-06-01)
- `n16-triage-misleads-again`
- `wrong-ab-instrument-champion-mirror`
- `bg-jobs-killed-by-container-restart`
- `submitted-on-weak-evidence-twice`

## Promotion candidates (PI ratified: NO — none promoted; improvements.md untouched)

### [ ] CLAUDE.md / guardrails — A/B opponent must exercise the mechanism under test
**Tag:** `wrong-ab-instrument-champion-mirror` (expansion fix A/B'd vs a hoarder mirror → false null)
**Where to insert:** new operating rule (or guardrails.md confound section), adjacent to Rule 41 / Rule 43.
**What to add:** Before accepting an A/B verdict on a mechanism, confirm the
opponent set can *exercise* that mechanism. Test aggression/expansion fixes vs
AGGRESSIVE opponents; defense fixes vs aggressive attackers. A passive/mirror
opponent yields a structural false null. Origin: 2026-06-01 expansion-credit
A/B (40% vs champion mirror = non-gain, but the fix targets aggressive-expander
opponents the mirror can't represent).
**Why:** near-false-kill of a live-relevant idea; 2 A/Bs spent on a blind instrument.

### [ ] operational-environment / kaggle-comp — long A/Bs need checkpointed, detached runs
**Tag:** `bg-jobs-killed-by-container-restart`
**Where to insert:** operational-environment notes / TOOLS.md eval section.
**What to add:** A/Bs expected to run ≥~25 min MUST (a) emit incremental
checkpoint lines (partial reads survive a kill), (b) launch detached, (c) be
verified not-already-running before relaunch to the same log path. Container
reclaim on idle kills background jobs regardless of nohup.
**Why:** ~2 runs lost/near-lost on 2026-06-01.

### [ ] CLAUDE.md Rule 45 — strengthen to hard-block, no triage exception
**Tag:** `n16-triage-misleads-again`
**Where to insert:** Rule 45 body.
**What to add:** n≥32 hard-blocks the submit path; remove the "triage may
proceed to confirmation" softener for any result feeding a submission. This is
the third small-n false-positive lift this comp.
**Why:** Rule 45 already exists but recurred; cost = rolling-pair slot drift.

## PI additions (from step 4)
- Nothing to add; no candidates promoted (PI, 2026-06-01).

## Framework version at session-end
- Commit SHA: (this commit)
- Active rules: CLAUDE.md Rules 1–48.
- Loaded skills this session: postmortem.
