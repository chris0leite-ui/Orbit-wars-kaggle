# Postmortem — 2026-06-20 kaggle-dropout-strategy-improve-g57iln

Decision-quality based (not outcome-based). A clean-running session of
disciplined falsification: four strength ideas tested, all refuted, all banked,
nothing shipped on an unvalidated premise.

## What went wrong

- **Planned to ship on a sub-n-32 strength premise (Rule 45 near-miss).** The
  session opened by planning to "harden + ship" deep-search depth-3, on the
  strength of a prior `17/28` producer-wall claim. That number was n=28 binary —
  *below* the Rule 45 n≥32 bar — and was used as the planning basis for a ship,
  not just a probe. When finally re-measured at n=32 paired, it was dead parity
  (Δmargin −0.062, p=1.00). The plan *did* include a strength-preservation check,
  but framed as "preserve 17/28" rather than "first confirm 17/28 is real." Cost:
  most of a session hardening a lever whose premise evaporated. Caught correctly,
  but later than ideal. → promotion candidate below.

- **Mild over-autonomy across the "go" chain.** After the *second* refutation
  (scorer-myopia), the picture was already "marginal + compounding"; I launched a
  third investigation (step30→50) before surfacing the strategic fork to the PI.
  Each diagnostic was cheap and informative and the PI kept saying "go," so this
  is borderline — but the fork ("keep chasing tactical knobs vs. pause for the
  probe vs. pivot to win-equity") was PI-shaped and could have been raised a
  round earlier.

- **Process frictions (minor, self-inflicted, no LB/▢ cost):**
  - *Self-matching pgrep waiters.* Background "wait until process dies" commands
    whose own command line contained the pgrep pattern matched themselves and
    looped forever (killed twice). Don't grep a pattern your waiter command
    contains.
  - *Timeout undersized for the job.* The scorer diagnostic was given 540 s for a
    ~20-min (32-game) replay; timeout killed it mid-run, result lost, relaunched.
    Size timeouts to the actual workload.
  - *AskUserQuestion called without the required `question` field* (twice).

## What went right (decision-quality)

- **Never submitted during exploration.** Probe still warming, nothing cleared
  the bar → correct hold. Submission discipline intact.
- **Refuted cleanly and banked.** Each negative measured at the proper bar and
  written to the knowledge base (deep-search, scorer objective) so no future
  session re-walks them. The triangulation (not depth, not scatter, not objective,
  not raw neutral count) is itself a durable result.
- **Caught the small-n trap in real time.** d3_prod looked 7/11 ahead at n≈11,
  resolved to 18/32 parity at full n. Flagged, not shipped.

## Frictions logged this session
- (none pre-written in audit/friction.md; captured inline above. The two
  knowledge-base thoughts written this session — deep-search-refuted and
  scorer-objective-not-myopic — carry the substantive findings.)

## Promotion candidates (PI ratified: PENDING)

### [ ] .claude/skills/kaggle-comp/improvements.md — re-validate a sub-bar strength claim before planning to ship on it

**Tag:** `revalidate-before-ship` (a historical lift number below the n≥32 bar is
a probe result, not a ship basis)

**Where to insert:** alongside the existing small-n-overconfidence guidance.

**What to add:**
> Before building a *ship* plan around a prior strength claim, check the n it was
> established at. If it was below the Rule 45 bar (n<32, or a non-paired binary
> count), the FIRST step of the plan is to re-confirm it at n≥32 paired — not to
> harden/optimize around it. This session planned to ship deep-search depth-3 on a
> `17/28` (n=28) claim that proved to be dead parity at n=32; the hardening work
> was spent before the premise was checked.

**Why:** Cost ~most of a session's compute hardening a null lever; the premise was
a sub-bar number treated as solid. Same family as the already-promoted small-n
overconfidence rule, but distinct: that one warns against *reading* a small-n
lift; this one warns against *planning a ship* on one.

## PI additions (from step 4)
- (pending PI reply)

## Framework version at session-end
- Commit SHA: 00910fc7
- Active rules: CLAUDE.md Rules 0, 1, 12, 32, 35, 36, 38, 39, 40, 42, 45, 46
  (observation-driven single-strategy mode).
- Loaded skills this session: postmortem (and kaggle-comp context).
