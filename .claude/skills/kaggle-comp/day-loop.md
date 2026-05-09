# Day-loop — full spec

The outermost loop. Runs every session start, wraps Experiment-loop,
auto-fires Strategy-critic at EOD. See `loops.md` for the router and
loop-interaction context.

## Day boundary

- A "day" = a Kaggle UTC submission-quota day (10 submits/day, resets
  at UTC midnight). NOT a work-session boundary.
- Day ends when EITHER `submissions_used_today == 10` OR PI declares
  EOD. Default = drive toward 10/10.
- Compute may continue past EOD; LB submits cannot.
- A day is NOT done because experiments are conducted, even if the
  queue is empty. If slots remain and PI hasn't called EOD, pick a
  new hypothesis and keep going.

## Auto-trigger recognition (load-bearing)

The agent MUST recognize day-end from CONTEXT and execute steps 5-7
WITHOUT being prompted. PI does not run commands; the agent listens
and acts.

Day-end cues to catch:

- Most recent LB submit pushed `submissions_used_today` to 10.
- PI's natural-language EOD signals: "the day is done", "let's wrap
  up", "stop submitting today", "EOD", "let's call it", or any close
  paraphrase.
- Kaggle UTC midnight passes during an active session (slots reset).

When ANY cue fires, immediately execute steps 5-7 in one batch
without asking permission. The wrap is non-LB-touching and Rule 1
does not apply. After, give PI a 1-sentence "day-N closed; wrap
committed" notice and stop. Don't sit on artifacts. Don't ask
"should I write the wrap?" — just write it.

## Steps

```
1. Load state (Haiku): comp-context.md, last 3 audit/, lb_status.py.

2. Pick experiment (Sonnet): from queue or new hypothesis.
   *RE-RANK THE QUEUE BY EXPECTED LEARNING-PER-SLOT* at every replan,
   not by speculative lift. Best slot is the one that most reduces
   uncertainty about (a) OOF→LB calibration per mechanism family,
   (b) a pool member's behaviour, or (c) a structural-overfit
   signature. Heuristic-first if novel.

3. Execute Experiment-loop (see `experiment-loop.md`). If gate-passing
   candidate ready and slot remains, propose to PI for single-shot
   submit (Rule 1).

4. After each LB result lands: update calibration ladder; if slots
   still remain and PI hasn't called EOD, return to step 2.

5. End-of-day audit (auto-trigger; no prompting): write
   audit/YYYY-MM-DD-day-N-wrap.md with FOUR REQUIRED SECTIONS:
   (a) 3-bullet PI summary
   (b) Calibration ladder snapshot (today's submits + OOF→LB gaps)
   (c) Problems to address (load-bearing constraints surfaced today)
   (d) Hypotheses ranked by predicted-lift × CPU-feasibility +
       next-steps sequence (compute window + slot plan)

5.5 Run Strategy-critic-loop (see `strategy-critic.md`) if any
    auto-trigger cue is live. Output drives the (d) re-rank in
    step 5. Update CLAUDE.md state block (day, our_lb_best,
    headroom).

6. Append friction one-liners distilled from the day to
   audit/friction.md (NOT CLAUDE.md — see self-improvement.md).

7. Queue next session's first 3 experiments in CLAUDE.md hypothesis
   board AND rewrite HANDOVER.md (Rule 15) with the next-session
   prompt: read order, state recap, today's plan, workflow-rule
   reminders, anti-patterns, open PI questions. Then commit + push
   to feature branch AND merge to main (PI authorized 2026-05-04).
```

## Anti-patterns specific to Day-loop

- **Don't sit on EOD artifacts.** Wrap is auto-trigger. Asking
  "should I write the wrap?" wastes a turn and slips the commit.
- **Don't underspend the 10/day budget** without a written reason
  in the day-wrap. Slots are calibration probes; unused slots are
  unmeasured OOF→LB gaps.
- **Don't promote a candidate to PRIMARY based on OOF alone when
  the gap is widening.** Rule 14 strategy-critic fires on gap drift
  ≥2bp on consecutive submits in the same family — surface that
  signal in step 5 and let it re-rank step 5(d).
- **Don't skip step 7's HANDOVER.md rewrite** even if the next
  session is "obvious." The PI may start a fresh session on a new
  machine; the doc is the only handover surface.
