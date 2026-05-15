# Next-session prompt: build on the foundations

Paste this whole file (or its key paragraphs) at the start of the next
session. The point is to skip the rediscovery phase the previous
session burned 12 variants on.

---

## What you're starting from

Branch: `claude/bootstrap-competition-setup-6uU6k`. Latest commit
`5972247` (v7 lead-aim chooser). Local A/B vs `baselines/v7_0.py`
(n=48): **43.8 %**, Wilson lo 30.7 %, verdict TIE.

Read first, in this order:
1. `audit/2026-05-15-session-wrapup.md` — the full arc, including the
   three bugs that mattered, what was tried and failed, and the table
   of "still-hand-rolled" pieces with their main-branch equivalents.
2. `main.py` and `favor.py` — current 300-LoC chooser stack.
3. `lib/aim.py`, `lib/orbit.py`, `lib/fleet.py`, `lib/geometry.py` —
   the parity-tested physics primitives the previous session recovered
   from `origin/main`. THESE NEVER MISS. Trust them.
4. `lib/fast_sim.py` + `lib/game/interpreter.py` — parity-tested
   simulator (62/62 parity tests vs the kaggle env). THESE NEVER MISS.
   Trust them.

## The single lesson from last session

**Every breakthrough was about removing a hand-rolled heuristic and
trusting the proven foundations.** Every regression was about adding
one. v7's +25 pp lift came from deleting my hand-rolled
`_lead_aimed_angle` and using `lib.aim.aim_orbiting` instead.

Do not re-implement what `origin/main` already has. When you find
yourself writing math for orbital motion, fleet trajectory, swept-pair
collision, target prediction, capture-value scoring, ship-size
heuristics, or candidate enumeration — STOP and check `git ls-tree -r
origin/main lib/`. Almost certainly there's a parity-tested function
that does exactly that.

## What the previous session did NOT recover yet

These are the pieces in our current `main.py` that are still
hand-rolled and have proven counterparts in `origin/main`. Each is a
likely lift, ranked by expected impact:

1. **`favor.py`** (F1+F2 leaf eval) → `lib/value_heads.py`
   (`composite_capture_value` rewards predicted captures + penalises
   waste; was responsible for v7_4's 31 → 40.6 % lift over v7_0 in
   prior sessions).
2. **`_enumerate_candidates`** (per-source nearest-K + min-cap +
   multi-size) → `lib/v7_search._enumerate_drop_or_add_one` (richer
   candidate space; can both refine and extend the incumbent action).
   Combined with proper incumbent generation via `lib/missions/*.py`.
3. **`_capture_size_guess`** → `lib/missions/snipe.py` (proper
   size estimation that accounts for production growth and
   reinforcement).
4. **Greedy non-dogpile match** → `lib/planner.settle_plan` (proper
   joint planning instead of greedy local).

The natural next step: recover those, refactor `main.py` to compose
them, and re-A/B vs `baselines/v7_0.py`.

## How to find the next bug (the methodology)

The previous session's three breakthroughs all came from this loop:

1. Run `python eval.py --vs v7_0 -n 48 -w 4` (or whatever opponent
   you're chasing). Note the result.
2. Pick a seed where we lose. Run `python trace_game.py <seed> 0` to
   get the full per-fleet event ledger. Look for weird patterns:
   - launches at planets we already targeted (redundant)
   - launches that "MISS or DEFEAT" against beatable garrisons
   - long sequences of opp launches with no ME launches
   - my fleets going to inferred targets that are MY OWN planets
3. For each weird moment, run `python diagnose_chooser.py <seed>
   <turn>` to see exactly what the chooser was thinking — what
   candidates it considered and what Δfavor each scored.
4. When the chooser says "no candidates" but the situation begs for
   action, ask: which guard is rejecting them? Test by removing it
   manually in a single-state script (the way `diagnose_formula.py`
   was used to compare v1 vs v2 Δfavor side-by-side at seed 1003
   turn 30).
5. When you find a guard that looks suspicious, FIRST check whether
   `origin/main` has a tested equivalent. Recover it before
   rewriting.

## Submission state on the live ladder

Per `state/current.md` (last updated 2026-05-14):

```
rolling_last_2:
  - geo:    sub_id 52643676  μ=984.0
  - v7_pv:  sub_id 52630118  μ=1064.4
```

Submitting v7 (current branch tip) would evict `geo`. Estimated μ:
~1000-1100 based on local 43.8 % vs v7_0. **DO NOT submit without
explicit PI sign-off.** Comp rules: rolling-last-2, so a bad
submission costs the slot for ~24 h.

## The cleanup nit

`origin/claude/baseline-fix-port` exists as an empty branch (commit
`e242099` = `origin/main` tip, no new commits — the signing infra
failed during the previous session's worktree experiment). Safe to
delete:

```
git push origin --delete claude/baseline-fix-port
```

Don't run that without checking with the PI first.

## What "thoroughly" means

The previous session's mistake was building minimally and "staying
clean" — which meant rediscovering bugs `origin/main` had already
fixed. "Thoroughly" here means:

- Read the foundations FIRST. Skim `lib/value_heads.py`,
  `lib/v7_search.py`, `lib/missions/*.py`, `lib/planner.py`,
  `lib/lookahead_planner.py` before writing any new chooser code.
- When you write a function, ask "does main have this?" If yes,
  recover instead of re-implement.
- The `audit/` directory has session postmortems from prior work;
  `audit/2026-05-13-v7-0-loss-modes.md` etc. document specific
  v7 failure modes the current session's lessons echo. Read them.
- Run the full test suite (currently 109/109) before each commit.

## Your goal

Push from 43.8 % vs v7_0 toward 55 %+ (decisive win). The path is
visible: replace the hand-rolled pieces in our minimal `main.py` with
their tested main-branch equivalents, one at a time, A/B each step.
Don't invent. Compose.

When you're done — or stuck — write a wrap-up to
`audit/<date>-session-wrapup.md` in the same shape as
`audit/2026-05-15-session-wrapup.md`.

Good luck.
