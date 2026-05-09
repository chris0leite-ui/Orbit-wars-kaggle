# Do and don't — one-page checklist

## Do

- ✅ Read `comp-context.md` first thing every session.
- ✅ Check `kaggle competitions submissions` before recommending
  any "unprobed" candidate.
- ✅ Heuristic baseline before tree / NN / Optuna.
- ✅ Smoke + 1-fold time-probe before any multi-hour run.
- ✅ 4-gate leakage filter before every LB probe.
- ✅ Minimal-input meta sanity check on every stacking candidate.
- ✅ Ask PI before every submit. Single-shot.
- ✅ Use the full daily 5/day submission budget. Slots not used by
  Kaggle UTC midnight are forfeit. Don't sit on slots.
- ✅ Re-rank queue by *expected learning per slot* at every replan,
  not by speculative lift. Best slot reduces uncertainty about
  OOF→LB gap per family or about a pool member's behaviour.
- ✅ **Auto-recognize day-end from context.** When `submissions_used_today`
  hits 5, OR the PI says any of {"the day is done", "let's wrap up",
  "stop for today", "EOD", "we're done", or any close paraphrase},
  IMMEDIATELY execute the EOD wrap (loops.md Day-loop steps 5-7) in
  one batch without prompting. Don't ask "should I write the wrap?"
  — write it. Don't run slash commands; the PI doesn't use them.
- ✅ Run the Research-loop at every plateau (3+ nulls or 5+
  saturations).
- ✅ Persona-rotate when stuck. Subagent invocation = fresh context.
- ✅ Write an audit entry every day. `audit/YYYY-MM-DD-<topic>.md`.
- ✅ Update calibration ladder after every LB result.
- ✅ Cap files at 150 lines. Archive CLAUDE.md when bloated.
- ✅ Route models by task: Haiku for read-only, Sonnet default,
  Opus for hard reasoning.
- ✅ Pull-style updates: 1-2 sentences on demand.
- ✅ Log friction one-liners to `audit/friction.md` when something
  in the loop felt avoidable. Distill weekly into skill edits.

## Don't

- ❌ Don't wrap `kaggle competitions submit` in any retry / `until` /
  `while` / `for` loop. Ever.
- ❌ Don't recommend "lock final selection and stop" while LB
  budget remains.
- ❌ Don't write end-of-day audit until 5/5 slots are used or PI
  declares EOD. "Experiments done" is NOT a day-end trigger.
- ❌ Don't ASK whether to write the EOD wrap once a day-end cue
  fires. The wrap is non-LB-touching; Rule 1 doesn't gate it. Just
  write it, commit, push, then announce in 1 sentence.
- ❌ Don't propose slash commands or hooks as the automation
  mechanism. The PI explicitly does not run them. Recognize cues
  from natural-language conversation context and act on them.
- ❌ Don't pipe long-running scripts through `tail -N`; the pipe
  buffers all output until process exit. Use `> file 2>&1` and
  tail the file separately.
- ❌ Don't use `df.groupby(K).transform(lambda s: s.rolling(...))`
  on >1k groups. It calls the lambda per group. Use
  `df.groupby(K).rolling(W).mean().reset_index(level=K, drop=True)
  .reindex(df.index)` instead.
- ❌ Don't scope FE features without first checking
  `train.columns` and the data dictionary. 4 of 6 cross-comp-cited
  features for s6e5's RelState pack already existed in the dataset;
  re-deriving them was no-op.
- ❌ Don't spawn subagents that launch python via Monitor + early
  exit. The agent's completion event fires before its child
  process finishes; artifacts are half-written. Subagent contract:
  `python script.py > log 2>&1`; wait for exit; read log; summarise.
- ❌ Don't extrapolate 5-fold wall time from a downsampled probe
  for tree models with high-cardinality native categoricals. M2
  XGB probe predicted 481s; actual >1200s (~5× drift). Either
  multiply by 2-3× safety factor, or apply the new
  "1-fold-actual within 1h" gate per `loops.md`.
- ❌ Don't declare a "structural ceiling" without first running the
  Research-loop.
- ❌ Don't re-recommend a CSV that's already in the LB submissions
  list. Surface the prior result instead.
- ❌ Don't load CLAUDE.md > 50k tokens. Archive it first.
- ❌ Don't write a single file > 150 lines. Split into modules.
- ❌ Don't reach for Optuna / GPU / 5-fold-bagging before a
  heuristic baseline.
- ❌ Don't trust published "wall time" claims for GPU pipelines.
  Always 1-fold-probe on the same hardware.
- ❌ Don't grid-search for OOF-best hyperparameters on the final
  candidate. Use theory-only / LB-validated defaults.
- ❌ Don't stack on a saturated bank without the minimal-input meta
  test. Cross-component memorization will inflate OOF.
- ❌ Don't ask the same settled-once question twice. Write it to
  `comp-context.md` Day 1.
- ❌ Don't push minute-level chatter during long jobs. Wait for the
  human pull.
- ❌ Don't use top-tier model for routine `ls` / file-existence /
  LB-status grep calls. Haiku-tier is enough.
- ❌ Don't delete domain research before the DGP is understood. It
  may be a hypothesis seeder you'll need.
- ❌ Don't blend other people's LB submissions. (Banned per the
  reference comp's NEVER-SUGGEST-PUBLIC-CSV rule.)
- ❌ Don't append friction observations to CLAUDE.md. They go in
  `audit/friction.md` — one line per event, distilled weekly.
- ❌ Don't try to fix every friction in real time. Capture, then
  distill on the weekly schedule.

## Decision tree at session start

```
session_start:
  load comp-context.md
  check `kaggle competitions submissions`
  load last 3 audit/*.md
  ↓
  is there a queued experiment?
    yes → Experiment-loop
    no  → has a plateau triggered?
            yes → Research-loop
            no  → generate hypothesis (heuristic-first)
                  → Experiment-loop
```

## Decision tree before any LB submit

```
candidate ready?
  ↓
  G1-G4 all PASS? — no → reject
  ↓ yes
  minimal-input meta beats anchor? — no → reject
  ↓ yes
  not in `kaggle submissions` already? — no → surface prior result
  ↓ yes
  PI explicitly approves this submit? — no → wait
  ↓ yes
  single-shot kaggle submit (no loop)
  ↓
  log result, update calibration ladder
```

## When the human says "lock and stop"

If you're inside the final-3-day window, fine. Lock and stop.

If you're earlier than that, push back politely:

> "We're <N> days from deadline with <K> daily LB slots remaining.
> The current LB-best mechanism is <one-line>. Plateaus in similar
> tabular comps have been broken by <2-3 examples>. I recommend we
> keep exploring. If you're concerned about cost or attention, I can
> route to Haiku-tier and run the Research-loop on a fresh subagent
> before any more submits. Want to try that?"

The CLAUDE.md NEVER-LOCK-FINALS rule overrides anything else.
