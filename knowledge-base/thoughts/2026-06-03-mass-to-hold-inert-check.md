# 2026-06-03 (PM) — mass-to-HOLD: inert-check before build

Branch: `claude/champion-strategy-rules-00JzI`. Session: PI said "go" to build a
consolidation/massed-strike proposer for `holdgrab`. The durable lesson of the
session is **the discipline, not the mechanism**.

## The move that mattered: recon flipped "build it" into "census it first"

The PI greenlit a build. Recon found the same axis (joint-coordination /
coalition / massed-strike) was built and falsified *the day before* on the
sibling `baseline` lineage — greedy-replace HURT, coalition-refiner INERT (0
candidates ever, because sources solo-capture). Rule 44 says: if your idea lives
in closed tracks, STOP. Rule 49 (CLAUDE.md) still lists joint-coordination as the
active thrust. That contradiction is exactly the AskUserQuestion case; the PI
delegated ("no preference"), so the judgment call was mine.

The resolution that kept faith with both the "go" and the falsification: target
the ONE documented residual the prior work pointed at but never built —
**mass-to-HOLD, not mass-to-capture** (2026-05-31: "the next ceiling is HOLD,
not capture"; 2/4 coalitions captured-but-didn't-hold). And gate it behind a
**cheap pre-registered inert-check** before building the mechanism, because the
strong prior is that it's inert too.

## Why mass-to-HOLD is narrow by construction (and the census is the right gate)

Orbit Wars combat is Lanchester **linear** (`sizing.py`): two 30-ship fleets vs
a 40 garrison leave 20 survivors — identical to one 60-ship fleet. There is NO
concentration bonus. So consolidation is **pure budget-pooling**: it only helps
when a single source's *spendable* can't reach `need_hold` on a double-value
enemy planet but two sources' combined budget can, arriving synced. That band is
inherently thin — which is why "does it ever arise?" must be measured before any
mechanism is built, not assumed.

## Early read

4-game smoke vs v7_0: 2 opportunities in 608 turns (0.33%), median 0/game →
NO-GO. The enumerator *fires correctly* (both hits were ~2600–2992-value enemy
planets), so the mechanism is sound; the opportunity is just rare. Full 192-game
panel census launched at wrap; tool committed + reproducible
(`scripts/probe_consolidation.py`), frozen thresholds in
`audit/2026-06-03-mass-to-hold-consolidation-step1.md`.

## The meta-pattern (compounding with this morning's lesson)

Two sessions in a row, a coordination/value lever died not because the idea was
dumb but because (a) the coordination seam is empirically small at the champion's
level and (b) local A/B can't distinguish neutral from small-lift at feasible n.
The cheap-census-first move generalises: **for any mostly-falsified axis, build
the instrument that measures whether the opportunity exists before building the
thing that exploits it.** A 4-game census killed a multi-day build for the cost
of writing one enumerator (which doubles as the proposer's core if GO). That is
the Rule-37 / Rule-47 discipline paying off concretely.
