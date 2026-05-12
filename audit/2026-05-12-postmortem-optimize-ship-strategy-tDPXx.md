# Postmortem — 2026-05-11/12 optimize-ship-strategy-tDPXx

## What went wrong

- **Bad decision: trusted Phase-0 idle-bucket proxy without correlating
  it to winrate first.** Phase-0 surfaced MECHANISM_DROP at ~96% of all
  idle classifications. I shipped four consecutive variants targeting
  it (airtime, endgame, affordability filter, gang-up) — every one
  *reduced* the bucket and *tied or regressed* at 64-seed Wilson vs
  v3.4 baseline. The first regression (v3.5 airtime+endgame at 32-seed,
  43.8% [32.3, 55.9]) was already evidence the proxy was wrong; I
  doubled down instead of pausing to validate. **Same priors at
  decision-time:** I had the v3.4 ladder rank (1055.5) and the
  Phase-0 numbers, but no evidence that bucket-reduction *causally*
  improved winrate. I treated the proxy as actionable when it was a
  hypothesis. Cost: ~12 hours of compute + iteration, reverted to
  identity defaults.
- **Bad decision: shipped v3.5 with non-identity defaults (AIRTIME=1.0,
  ENDGAME=1.5) BEFORE running a head-to-head A/B.** I committed and
  pushed `7f69c13` declaring v3.5 ready, then discovered at the first
  pair-level test that it regressed. Should have built the
  `scripts/ab_variants.py` harness first, validated AIRTIME=0.5 (the
  eventually-tied variant) at 64 seeds, THEN flipped defaults. Wasted
  one commit cycle + the reversion.
- **Bad decision: gang-up implementation didn't audit the full
  mechanism pipeline.** The Plan agent's design correctly placed
  `gang_up_size` before `validate`, but didn't check what
  `arrival_size` would do AFTER. `arrival_size`'s `intent.ships =
  max(intent.ships, needed)` silently re-inflated every throttled
  share to the full target garrison — Phase-0 evidence: validate
  drops -39%, arrival_size drops +31%, net total -2%. Gang-up wasn't
  even mechanically working end-to-end. A 5-minute pre-implementation
  pipeline trace would have surfaced this.
- **PI-override (productive):** "run through the 7 steps problem
  solving. understand what went wrong and what we need. then iterate
  again" — reframed three nights of variant-shipping into a structured
  diagnosis. The 7-step framework correctly identified the gang-up
  substrate gap. **One PI override this session.** Calibration: PI
  overrode after I'd already cycled through 3 failed variants — the
  intervention point was correct, mine was earlier.
- **Rule-gap: no rule about proxy-validation before targeting it.**
  Phase-0 was treated as actionable from day 1. CLAUDE.md has Rule 27
  (pre-submit prediction diff) but no analogue for "validate that a
  new metric correlates with the outcome metric BEFORE running ≥1
  variant against it." Surfaced as a promotion candidate.
- **Rule-gap: no rule on default-OFF for unproven decision constants.**
  v3.5 shipped with non-identity defaults; reverted same session. A
  rule "new decision constants default to identity until 64-seed
  Wilson_lo > 50% A/B" would have prevented the wasted cycle.
- **Rule-gap: 32-seed vs 64-seed promotion threshold.** AIRTIME=0.5
  was 54.7% [42.6, 66.3] at 32 seeds (point estimate flattering, lo
  below 50%) and converged to 52.3% [43.7, 60.8] at 64 seeds. The
  extra 32 seeds were exactly 32W/32L. Rule candidate: "32-seed
  Wilson_lo < 50% → require 64-seed retest before any ship/revert
  decision."

## Frictions logged this session

Six entries in `audit/friction.md ## 2026-05-11/12 (optimize-ship-strategy-tDPXx)`:
- `idle-bucket-reduction-is-misleading-proxy`
- `gang-up-substrate-bug-arrival-size-reinflates`
- `ab-variants-regex-rejected-inline-comments`
- `ab-variants-hardcoded-snipe-only`
- `bool-vs-int-constant-typing-for-ab-regex`
- `claude-bash-pipe-buffers-progress-output`
- `32-seed-point-estimate-noise-at-128-game-level`

## Promotion candidates (PI ratified: PENDING)

### [ ] [CODE-COMP-DISCOVERED] CLAUDE.md Rule 7b — proxy validation gate

**Tag:** `idle-bucket-reduction-is-misleading-proxy` (Phase-0
bucket reduction looked actionable but uncorrelated with winrate;
4 variants shipped, all tied/regressed).

**Where to insert:** CLAUDE.md, new Rule 7b under existing Rule 7
"Research before saturation."

**What to add:**
```
7b. **Validate new proxy metrics before targeting them.** Any new
    metric proposed as actionable (e.g., idle-source rate,
    bounce-rate, MECHANISM_DROP bucket) must be correlated with the
    OUTCOME metric (head-to-head winrate vs the current best on
    ≥32-seed A/B) BEFORE running ≥1 variant against it. If the
    correlation is < 0.3 or the sample doesn't allow the correlation
    to be measured, treat the proxy as a hypothesis, not a target.
    Cost evidence: 2026-05-11/12 spent ~12 hours iterating on Phase-0
    bucket-reduction; 4 variants all tied/regressed.
```

**Why:** Same pattern fired twice in this comp:
- 2026-05-11 evening: flat NEUTRAL_BONUS=1.5/COMET_BONUS=1.3 (motivated
  by "78.6% of comets sit neutral" Phase-0 finding) — 28.1% regression.
- 2026-05-11/12 overnight: airtime/endgame/filter/gang-up (motivated
  by Phase-0 idle-bucket decomposition) — all tied/regressed.
Both were action driven by an unvalidated proxy. The "if the finding
is local, the fix must be local" friction tag from yesterday is
adjacent but doesn't catch this — the new rule is upstream of it.

---

### [ ] [CODE-COMP-DISCOVERED] CLAUDE.md Rule 6b — default-OFF for new decision constants

**Tag:** `default-on-new-constants-requires-rollback`

**Where to insert:** CLAUDE.md, new Rule 6b under existing Rule 6
"Heuristics before heavy compute."

**What to add:**
```
6b. **New decision-criteria constants default to identity values
    until a 64-seed pair-level Wilson_lo > 50% A/B confirms
    otherwise.** Adding a constant ON-by-default is a change to the
    submitted-agent behaviour; treat it like a submission. Default
    to off (= identity / no-op); flip the default only after a
    confirmation A/B. Cost evidence: 2026-05-11/12 shipped v3.5
    with AIRTIME=1.0 + ENDGAME=1.5 ON; 32-seed A/B showed 43.8%
    regression; reverted same session.
```

**Why:** New constants are often ablation knobs not behaviour
changes; treating them as behaviour changes by default forces the
validation discipline. Cost: one wasted commit cycle + reversion this
session; will save many across future ablation campaigns.

---

### [ ] [CODE-COMP-DISCOVERED] CLAUDE.md Rule 12c — 32-vs-64 seed promotion threshold

**Tag:** `32-seed-point-estimate-noise-at-128-game-level`

**Where to insert:** CLAUDE.md, new Rule 12c under existing Rule 12
on submission discipline.

**What to add:**
```
12c. **32-seed Wilson_lo < 50% requires a 64-seed retest before any
     ship-or-revert decision.** 32-seed pair-level point estimates
     drift noisily in [-5pp, +5pp] across re-runs; the Wilson_lo
     is the load-bearing summary. Cost evidence: 2026-05-11/12
     AIRTIME=0.5+ENDGAME=1.5 looked +4.7pp at 32-seed (54.7% [42.6,
     66.3]) but converged to TIE at 64-seed (52.3% [43.7, 60.8]).
     The extra 32 seeds were 32W/32L.
```

**Why:** 32-seed Wilson_lo can sit at 42-46% even with a 54.7%
point estimate. Acting on the point estimate alone produces
ship-and-revert cycles. The rule defers the ship decision by ~10 min
of compute (64-seed completes in ~10 min on 8 workers) and prevents
many cycles like this one.

---

### [ ] [CODE-COMP-DISCOVERED] scripts/ab_variants.py infra — multi-file constant patching + comment-tolerant regex

**Tag:** `ab-variants-hardcoded-snipe-only` + `ab-variants-regex-rejected-inline-comments`

**Where to insert:** Already applied this session (commit `a8ae69a`).
Promotion target: surface as a known-good pattern in the kaggle-comp
skill so the next code-comp's A/B harness inherits both fixes.

**What to add:**
```markdown
## Multi-file constant-patching harness pattern

When the ablation surface spans multiple lib files, harness scripts
that mutate source code MUST:
1. Auto-discover the owning file per constant (scan a declared
   `PATCHABLE_PATHS` list, error on multi-file collisions).
2. Tolerate inline comments on constant declarations
   (`(?:#.*)?$` suffix in the regex; preserve via named capture).
3. Smoke-test on a known-good constant from each declared path at
   harness load time.

See orbit-wars `scripts/ab_variants.py` for a worked implementation.
```

**Why:** Both bugs fired in the same overnight session; harness
took ~20 minutes to diagnose + fix. Future code-comps will hit the
same pattern. Filing as a pattern, not just a per-comp tweak.

## PI additions (from step 4)

PI: "Looks complete — proceed." No additional frictions, rules, or
decisions to flag.

**Promotion ratification:** PI gave no preference on the four
promotion candidates. Per the postmortem skill protocol, candidates
are retained in this postmortem as DRAFTS but NOT applied to
`.claude/skills/kaggle-comp/improvements.md`. They can be promoted
later if the same patterns re-fire and the PI ratifies then.

## Framework version at session-end

- Commit SHA: `227a052` (HEAD of `claude/optimize-ship-strategy-tDPXx`)
- Active rules: 1..36 (CLAUDE.md `## Operating rules — concise`)
- Loaded skills this session: kaggle-comp, postmortem
- Branch: `claude/optimize-ship-strategy-tDPXx` (Day-3 PM)
- Pending PR: none today
