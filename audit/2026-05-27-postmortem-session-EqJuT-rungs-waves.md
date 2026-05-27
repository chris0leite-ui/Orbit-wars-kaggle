# Postmortem — 2026-05-27 session-EqJuT (rungs 1-2 unblocked, wave-attacks null)

> Second postmortem of the day on `claude/session-EqJuT`. First was
> `audit/2026-05-27-postmortem-session-EqJuT.md` (H44 staleness fix).
> This one is the rungs-and-waves iteration cycle PI ran after.

## Session summary

Branch HEAD at session start: `ce70160` (claimed 14/16 ELIM but
`145ee8d` had re-verified 9/16 from a cold tree). PI directive:
*"what next? think hard, keep it simple. Pareto principle."*

Two iterations landed:

1. **`ba58caf` — `MAX_LAUNCH_TICK=15→3` speed fix.** Trace of the 2
   "stuck" random-ELIM seeds (31448, 80504) revealed the failure mode
   was NOT strategic ("opp pocket no single source can match") but
   `actTimeout` disqualification. `predict_fleet_fate` × 518 calls
   × 7 ms = 3.6 s/turn under the 1 s gate timeout. Capping the wait_N
   grid took p90 from 4.0 s to 0.7 s. Result: 9/16 → **16/16 ELIM**
   on rng-seed 2026, 11/16 → **16/16** on rng-seed 7, **16/16** on
   rng-seed 42. Same fix also lifted rung 2 (vs `starter`):
   9/16+1L → **16/16 ELIM**.

2. **`1da2652` — wave attacks behind `LAGRANGE_SIMPLE_WAVES` env
   var.** Allows ≤ 2 SOLO picks per target at distinct `launch_tick`
   values (structurally distinct from the Rule-37-closed same-tick
   dogpile). Predicted 2-4/16 lift vs baseline. Result: **0/16 wins
   on rung 3** (median LOSS step 150 → 170, no captures). Pre-declared
   threshold: 0/16 = null lift; code stays dark-launched.

End-of-session ladder posture:
- Rung 1 (random):  **16/16 ELIM** ✅ (Rule-48 candidate gate clear)
- Rung 2 (starter): **16/16 ELIM** ✅
- Rung 3 (baseline): **0/16** ❌ — structural gap (no opp model in
  picker, no positional/tempo awareness)

## What went wrong

Decision-quality flags (assessed against priors-at-decision-time per
`knowledge-base/concepts/decision-quality-vs-outcome-quality.md`):

- **Initial trace framing was right.** I described option #1 (trace
  the 2 stuck seeds) and PI selected it. The Pareto-cheap diagnostic
  surfaced the actual root cause (timeout-DQ), which a strategic
  redesign would never have reached.

- **Wave-attacks decision was sound, outcome was null.** Predicted
  2-4/16 lift, threshold pre-declared, falsification clean. The null
  result is information: it confirms the rung-3 gap is structural,
  not per-target firepower. Spending ~30 min for that signal was
  Pareto-justified.

- **Rule-bypass: not introduced this session but discovered.**
  `agents/lagrange_simple/main.py:14-17` advertises
  `KINEMATIC_TABLE_ENABLED=1` and tries to import
  `lib.kinematic_table.begin_turn` which **does not exist on this
  branch** (PFhzM-only). The try/except silently swallows ImportError.
  The accompanying comment claims "~50-100 ms / turn cache." That
  speedup never existed. Rule 47 (physics-primitive verification
  before agent design) was bypassed when the hint was authored in a
  prior session. The silent absence MASKED the timeout bug for
  several days because operators assumed the cache was cutting
  per-turn cost.

## PI overrides (calibration data)

Zero mid-session corrections. PI selected options cleanly three times:
- "do 1" (trace the seeds).
- "go" (climb rungs 2-3 after the speed fix landed).
- "go #1" (wave attacks after rung 3 0/16 confirmed).

Workflow was clean.

## Frictions logged this session

See `audit/friction.md` § "2026-05-27 (claude/session-EqJuT —
timeout-DQ root cause + wave-attacks null)":

- `tag: claimed-optimisation-silently-absent`
- `tag: postmortem-narrative-taken-as-diagnosis`

## Promotion candidates (PI ratified: no)

PI verdict: *"Nothing to add, nothing to promote."*

Drafted candidates (NOT promoted — recorded here for future
recurrence-pattern matching):

- **`claimed-optimisation-silently-absent`** — drafted as a Rule
  candidate ("Optional optimisations must be observable"). PI did
  not promote. Revisit if the same pattern (silent try/except hiding
  an absent optimisation that another file's comment claims is
  load-bearing) is observed in a future session.

- **`postmortem-narrative-taken-as-diagnosis`** — I flagged this as
  better covered by Rule 38 (fix-verification reproduces failure
  state) and recommended not adding a new rule; PI did not promote.

## PI additions

> "Nothing to add, nothing to promote."

No additions.

## Carry-forward for next session

- Rungs 1+2 clear at 16/16 with the speed fix. Rung 3 is the live
  question; wave-attacks falsified that single-target firepower
  closes it. Remaining axes from the menu:
  - **Port baseline's JOINT multi-source coalition into
    lagrange_simple** (structural; production-tested code in
    `agents/baseline/chooser_trajectory.py`; the same-tick coalition
    pattern that's distinct from the Rule-37-closed dogpile variant
    on this branch).
  - **Pivot to baseline-line directly** (Track B in MULTI_BRANCH.md
    has the strongest live ladder evidence; 27 days left).
- `LAGRANGE_SIMPLE_WAVES=1` flag remains dark-launched. Code path is
  tested + gated; flipping default would be a 1-line change IF a
  future session shows lift on a different opponent class.
- The kinematic_table claim in `main.py:14-17` is still misleading.
  Either delete the env-var-setdefault + comment, or land the actual
  cache substrate (PFhzM merge-up candidate per MULTI_BRANCH.md).
  Not promoting to a rule per PI, but worth a 2-line cleanup at
  some point.

## Framework version at session-end

- Branch: `claude/session-EqJuT` (HEAD will be at wrap commit after
  this artifact lands)
- Commit SHA pre-wrap: `1da2652`
- Active rules: CLAUDE.md Rules 1-47 (unchanged this session; the
  drafted candidates above are NOT promoted)
- Skills invoked this session: postmortem
