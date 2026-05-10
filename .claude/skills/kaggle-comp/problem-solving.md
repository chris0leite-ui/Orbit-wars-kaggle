# Problem-solving framework — 7 steps

Reference: Charles Conn & Robert McLean, *Bulletproof Problem
Solving: The One Skill That Changes Everything* (Wiley, 2018).

The kaggle-comp loops are organised around these seven steps.
When stuck or starting fresh, return to step 1.

## The seven steps

1. **Define** — Problem statement: question, decision-maker,
   criteria, constraints, time horizon, boundary.
2. **Disaggregate** — Logic tree (MECE: mutually exclusive,
   collectively exhaustive). Components, hypotheses, contributors.
3. **Prioritise** — 2×2 (impact × effort, or impact ×
   uncertainty). Prune low-impact branches; concentrate on
   high-impact / high-actionability.
4. **Workplan** — For each priority branch: hypotheses,
   analyses, owner, timing, milestone. Concrete commitments.
5. **Analyse** — Right tool per node. Heuristics first, advanced
   analytics later (mirrors guardrail #6).
6. **Synthesise** — So-what conclusions pulled back up the tree.
   Not "we found X"; "X means we should do Y."
7. **Communicate** — Pyramid principle. Governing thought →
   supporting arguments → evidence → decision/action.

## Problem-statement worksheet (Q3.5)

PI fills this at kickoff. One chat turn. Written to
`audit/<date>-problem-statement.md`. Re-read at every plateau and
every weekly distillation.

```yaml
problem:
  l1_question: <this comp's goal — e.g. "top-5% on s6e5 by 2026-05-31">
  l2_question: <framework reuse — e.g. "skill stable across future comps">
  l3_question: <autonomy — e.g. "AI runs loop, PI does only sign-off">
decision_maker: PI
criteria:
  - <e.g. private-LB rank percentile>
  - <e.g. PI hand-holding cost>
constraints:
  - 5 subs/day, 2 final, 1h GPU/day
  - 50k token CLAUDE.md cap, 150-line doc cap
  - <comp deadline>
boundary:
  - in: tabular Playground/Featured comps; PI sign-off; AI loop
  - out: <e.g. NLP/CV; solo finals; auto-submit>
```

## Logic-tree template (step 2)

```
<root question>
├── A. <branch>
│   ├── A1. <sub>
│   └── A2. <sub>
├── B. <branch>
└── C. <branch>
```

MECE check: do branches cover ALL paths (collectively
exhaustive)? Does any sub belong under two parents (not mutually
exclusive)? If yes, refactor.

## 2×2 prioritisation (step 3)

|                     | Easy / cheap     | Hard / expensive |
|---------------------|------------------|------------------|
| **High impact**     | DO NOW           | PLAN FOR         |
| **Low impact**      | OPPORTUNISTIC    | PRUNE            |

Re-apply at every plateau. The cheap-and-high-impact node on
Day 1 is rarely the same on Day 14.

## Mapping to existing kaggle-comp loops

| Skill loop / gate | 7-step phase |
|---|---|
| Kickoff Q1-Q4 (`kickoff-runbook.md`) | 1 + 2 |
| Q3.5 problem statement | 1 (explicit) |
| Pre-baseline gate (`pre-baseline-gate.md`) | 2 (data-side decomposition) |
| Day-loop hypothesis board (`CLAUDE.md`) | 4 (workplan) |
| Calibration-loop (`loops.md`) | 5 + 6 |
| Research-loop (plateau trigger, `loops.md`) | re-entry to 1 |
| Weekly distillation (`self-improvement.md`) | 7 + 1 (next week) |
| Final 3-day window (R8) | 7 (PRIMARY + HEDGE story) |

## When to re-enter step 1

A plateau is NOT a step-5 problem ("more analysis"). It is a
step-1 re-entry: "are we still solving the right problem? Has
L1 shifted? Is the criterion still private-LB rank, or is it now
'understand what the leak is'?" The discipline most often missing
in practice is questioning the problem definition when stuck.

## Anti-patterns

- Skipping step 1 because "we know the comp." (We rarely do
  until the pre-baseline gate clears.)
- Logic trees that aren't MECE — overlapping branches waste
  effort; missing branches become blind spots.
- Step 3 prioritisation done once at kickoff and never re-applied.
- Step 5 dominating sessions (analysis paralysis) with no step 6
  synthesis.
- Step 7 communicate-as-summary instead of communicate-as-decision.
