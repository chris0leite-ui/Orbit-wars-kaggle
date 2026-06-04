# Forward-simulation scorer in our own agent (options A & B)

_Branch JzIAr, 2026-06-04. The "different approach from the sister branch"
the PI asked for: NOT host-on-Producer's-torch-engine, but bring the one
piece of Producer that actually matters — its forward-simulated value
function — into our own agent. This notes the diagnosis, the two build
options, the key composition insight, and next steps._

## The diagnosis — there is only one piece that matters

Producer's whole edge is a single thing: its value function **simulates
combat forward ~18 turns and scores the real net-ship swing** — `my net
gain − opponents' net gain`, accounting for who-captures-what, cascades,
and combat timing (`sparse_launch_flow_delta` in
`agents/producer/orbit_lite/garrison_launch.py`). Every other Producer
part — hold-reserve sizing (`safe_drain`), capture-floor, target
shortlists, greedy best-wave-per-target selection — we already have
equivalents of.

**Proof it is the whole edge:** our `producer_lite` port (`lib/producer_lite.py`)
kept all the cheap pieces faithfully and only swapped the forward-sim
scorer for a cheap "production × time-left" proxy. It wins **3%** as an
attacker where real Producer wins **78%** (`/tmp/transfer2.log`,
`audit/2026-06-04-producer-lite-build.md`). The ~45 points of win-rate
ARE the scorer; the cheap heuristics carry almost none of it.

**The gap, stated exactly:** our champion's value function (`favor` in
`agents/baseline/value.py`) is a *static one-shot board score* —
`(my ships − their ships) + (my prod − their prod)×discount`. It never
simulates combat forward. That is the whole difference between us and
Producer.

## The asset most plans overlook — we already built a forward-sim scorer

`lib/v7_search.py` + `lib/fast_sim.py`: enumerate candidate launch
bundles, roll each forward K turns through our own fast simulator
(0.12 ms/step) with a reactive opponent model, pick the best `us − them`
net-ship total at the rollout's terminal state. **Structurally the same
lens as Producer, pure-Python, no torch.** A dozen `v7_*` agents exist.

It was shelved — but not because the scorer was wrong. Because its
candidate *enumeration* was too narrow ("drop-one": incumbent minus each
launch, a strict subset that can't out-score the incumbent). The scoring
lens was never the problem; the menu it scored was.

## Two ways forward simulation goes INTO our agent

**Option A — Forward-sim as a re-ranker over our existing candidates
(rollout-based).** Our proposer already generates rich candidates
(coalitions, launches). Instead of scoring them with the static `favor`
leaf, forward-sim-score only the **top-K** shortlist and pick the
winner. Bounds cost — roll out the shortlist, not hundreds. The
infrastructure exists (`v7_search.score_candidate_symmetric`). Lowest new
code: revive + widen `v7`, don't rebuild. Risk: rollout-scoring each
candidate is pricier than Producer's vectorized analytic flow-delta, so
candidate breadth is capped by the turn budget.

**Option B — Reimplement Producer's analytic flow-delta on our substrate
(pure-Python, no torch).** producer_lite ALREADY builds the do-nothing
forward projection of every garrison (`_build_projection`). What it got
wrong was *scoring* candidates against that projection with a cheap
proxy. Swap the proxy for a real net-ship-swing computation against the
projection → Producer's scorer in pure Python, cheap enough to score many
candidates. Higher payoff, higher effort (reimplement the flow-delta in
numpy), but the projection substrate is already written, so it is NOT the
multi-week torch port the sister branch flinched from.

## The key reframe (PI, 2026-06-04) — DROP the opponent model entirely

Earlier in this note I made producer_lite "the linchpin — the aggressive
opponent inside the rollout." **Producer refutes that, and the PI caught
it.** Producer does NOT model the opponent's policy: its forward sim
freezes the opponent (they produce, in-flight fleets resolve, but they
launch nothing over the scoring window), then scores `Δnet_me − Σ Δnet_opp`
against that passive baseline. It is **opponent-agnostic** — and it beats
us, who DO run an opponent model in our rollout. So the right move is the
opposite of what I first wrote: not a better rollout opponent — **no
rollout opponent at all.**

Why opponent-agnostic is MORE robust, not less:
- An opponent model is a guess, and a wrong guess is worse than no guess.
  Rolling out vs `lite_greedy` (a patsy) when the field plays like
  Producer systematically under-defends. You can't be wrong about an
  opponent you refuse to predict.
- The do-nothing baseline measures the **first-order** value of a move —
  the territory/production it secures, captures that land, in-flight
  threats it answers. That term is opponent-independent (production
  accrues, territory compounds regardless of enemy policy). Opponent
  reactions are a second-order correction.
- Producer wins because it scores the first-order term *exactly* (forward
  sim) while we score it *approximately* (static leaf) AND burn effort
  modeling the second-order term *badly*. Score the big thing right; do
  not guess the small thing wrong.
- On a 3,700-team ladder no single opponent model is accurate. Modelling
  buys exploitation of one foe at the cost of fragility across the field.
  Robustness wins a ladder.

Opponent-agnostic is NOT opponent-blind: the projection still resolves
in-flight fleets, so it defends against committed attacks (Producer's
`friendly_flip_targets`). It is myopic only about launches not yet made —
exactly the speculative part to stay humble about.

Honest limits (second-order, fine to drop): a passive-opponent assumption
under-values **pure denial** moves and **capture races**. Both are partly
caught anyway (competitive score subtracts opp passive net gain;
`safe_drain` bounds overextension without a prediction). We trade a small,
fragile exploit edge for a large, robust calibration gain.

**Consequence — a simplification, not just a perf change.** Delete the
opponent model wholesale (`lib/opp_model`, producer_lite-as-rollout-
opponent). Less code, less cost (no per-tick opp policy), no torch. Then
feed the robust scorer RICHER candidates than Producer (coalitions,
wait-then-fire) — which an opponent-agnostic net-ship-swing scorer values
correctly. Same lens, better menu → plausibly BETTER than Producer.

Mechanically small in our code: `v7_search` already rolls out and scores
`us − them`; hand its rollout a **no-op opponent policy** instead of
`lite_greedy`. producer_lite's `_build_projection` is the analytic version
of the same opponent-passive forecast.

## Probe decision (PI, 2026-06-04): SKIPPED

The v7-vs-Producer triage A/B was NOT run — PI judged it a waste ("the
strategy will win"). We measure at the real gate instead: n≥32 + Rule 46
wallclock once the opponent-agnostic hybrid exists. (Kept for the record:
bare v7 would have confounded the scorer with a weak rollout opponent and
narrow drop-one enumeration, so a loss would not have refuted the idea
anyway.)

## The target architecture (opponent-agnostic hybrid)

- **Value head:** opponent-passive forward projection (produce + resolve
  in-flight fleets, NO enemy launches), scored by `Δnet_me − Σ Δnet_opp`.
  No opponent model anywhere.
- **Candidates:** our rich generators (multi-source coalitions, multiple
  sizes per (src,tgt), wait-then-fire, adaptive-K reach).
- **Selection:** greedy best-wave-per-target with role-mutex + hold-reserve
  (`safe_drain`) — Producer's structure, which we already mirror.

Two implementations of the same value head:
- **Option A (rollout):** `v7_search` re-ranks the top-K candidates by a
  fast_sim rollout with a **no-op opponent policy**. Lowest new code;
  cost-capped by top-K. Faithful to exact combat, pricier per candidate.
- **Option B (analytic):** reimplement Producer's flow-delta on
  producer_lite's `_build_projection`; cheap enough to score many
  candidates; numpy, no torch. Higher payoff, higher effort.

## Next steps (no opponent model in any of them)

1. **Switch the rollout to opponent-passive (Option A spike).** Point
   `v7_search`'s rollout at a no-op opponent policy + competitive net-ship
   delta score. Smallest change that makes our forward sim match
   Producer's lens. Sanity-check it plays sensibly (captures, holds).
2. **Widen the candidate menu** feeding that scorer: our coalitions +
   capture-floor sizes + adaptive-K reach + wait-then-fire.
3. **Option B parallel track (bigger bet):** reimplement the analytic
   flow-delta on producer_lite's projection; unit-test net-ship-swing
   against `env.step` ground truth before wiring as a value head.
4. **Measure at the real gate only** (PI skipped the triage probe): n≥32
   vs Producer + a multi-opponent panel, Rule 46 wallclock < 1000 ms max
   turn, top-K pre-prune by the cheap leaf before forward-sim re-rank.
5. **Delete on success:** `lib/opp_model`, producer_lite-as-opponent. The
   opponent-agnostic scorer makes them dead weight.

## Pointers
- `lib/v7_search.py`, `lib/fast_sim.py` — our forward-sim substrate.
- `agents/producer/orbit_lite/garrison_launch.py::sparse_launch_flow_delta`
  — Producer's exact scorer (the thing to mirror in Option B).
- `lib/producer_lite.py::_build_projection` — the projection substrate
  Option B reuses; the proxy at lines 301–312 is what Option B replaces.
- `audit/2026-06-04-producer-lite-build.md`, `/tmp/transfer2.log` — the
  3%-vs-78% proof that the scorer is the edge.
