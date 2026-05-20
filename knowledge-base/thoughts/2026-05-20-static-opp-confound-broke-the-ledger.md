# 2026-05-20 — Static-opp confound broke the ledger

The biggest lesson of this work-cycle: **what-if rollouts that drive
the opponent from a recorded replay are systematically biased toward
whichever agent diverges from the recording first.**

After divergence, the opp's recorded actions hit a state they were
never computed for — they miss, no-op, or target planets that have
already changed hands. The diverging agent then plays effectively
unopposed and racks up false-positive metrics (final planet count,
ship count, idle reduction). The non-diverging agent gets crushed by
the opp's actions that DO still apply.

The ledger looked great in what-if (+118 vs +15 final planets across
6 episodes) because what-if rewarded the agent that diverged most
from the original game. In real h2h, with the opp playing reactively,
the ledger drained reserves that the wait_N>0 reservation was
correctly hoarding for defensive use.

The corollary is uncomfortable: **most of the metrics I instrumented
on the what-if harness measure divergence, not lift.** Idle rate
drops, launch volume rises, planet count rises — all of these are
positively correlated with "the agent did something different from
the recording" rather than "the agent is better."

The corrected workflow:
- What-if is a chooser-behavior debugger. It answers "what does the
  agent do per-turn under policy P?" It does NOT answer "would
  policy P beat current production?"
- The latter question requires h2h vs current production. Even at
  n=8 it's a stronger signal than what-if at n=6 (the corpus we
  used for the ledger validation).

The principle generalizes: any harness that replaces an adversary
with a static / recorded / scripted policy is biased toward the
agent that diverges from the script. To detect this, every such
harness should include a parity test against a reactive opp on at
least 1 game, and flag if results differ.

This is a structural rule, not a code change. Drafted as Rule 42 in
today's postmortem; awaits PI ratification.

The deeper conceptual point: **the lite_greedy opp model has the
same problem at a different level.** The chooser's K=25 rollouts
use lite_greedy as the opp simulator. lite_greedy is also a
"scripted" opp — it doesn't actually adapt to our coordinated multi-
launch attacks the way real opponents do. So the chooser's plans
are scored against a static-opp confound INSIDE the chooser too.
This is why over-emission strategies (the ledger family) look great
in the chooser's leaf scoring but lose in real play. The fix isn't
just at the harness layer — it's at the value-head layer.

Implication: the highest-leverage next move on this branch is
probably the opp-model upgrade (lite_greedy → learned from corpus),
NOT another chooser-emit variant.
