# Forward-simulation scorer in our own agent (options A & B)

> **STATUS — REFUTED 2026-06-04 (n=32 clean A/Bs). DO NOT re-open as written.**
> Both levers were built (default-OFF, gated) and measured: the net-ship-swing
> lens loses to our champion (37.5%) and buys nothing vs Producer (21.9%), and
> the passive-opponent assumption ALONE loses to the champion (31.2%). The
> opponent-agnostic *assumption* is the culprit — our rollout's reactive
> opponent provides a load-bearing recapture penalty; dropping it overextends.
> Producer wins despite being opponent-passive because of its OTHER machinery,
> not its lens. Full record + standing "do not delete lib/opp_model"
> implication: `knowledge-base/thoughts/2026-06-04-opponent-agnostic-refuted.md`.
> The architecture *correction* below (our champion already forward-simulates)
> stands regardless. The rest of this doc is the original (pre-result) plan,
> kept for the reasoning trail.


_Branch JzIAr, 2026-06-04. The "different approach from the sister branch"
the PI asked for: NOT host-on-Producer's-torch-engine, but bring the one
piece of Producer that actually matters — its forward-simulated value
function — into our own agent. This notes the diagnosis, the two build
options, the key composition insight, and next steps._

## CORRECTION (2026-06-04) — our live champion already forward-simulates

An earlier version of this note (and the HANDOVER) said our champion uses
a "static one-shot board score" that "never simulates combat forward."
**That is wrong for the agent we actually ship**, and the mis-statement
made the gap to Producer look bigger than it is. The traced truth:

- The live champion is `BASELINE_CHOOSER=refine` → `choose_refine` →
  `choose_trajectory`, and its real scorer is `score_candidate_v4`
  (`agents/baseline/chooser_trajectory.py:600`).
- For **every** candidate launch it **clones the board and rolls it
  forward K ticks** through `fast_sim`
  (`chooser_trajectory.py:676-706`), then scores the leaf with `favor`
  and subtracts a me-idle baseline (`build_trajectory_baseline`,
  `chooser_trajectory.py:581`).
- The "static one-shot `favor`" is only the **leaf** of that rollout —
  not the scorer. (The composite chooser in `agents/baseline/chooser.py`
  IS a static-leaf path, but it is NOT the live chooser.)

So we already forward-simulate. The real difference from Producer is
narrower and lives in **two** places:

1. **The opponent model.** Our rollout has every opponent seat **actively
   launching** `lite_greedy_policy` at every tick (`opp_actions_for_snap`,
   `chooser.py:60`). Producer **freezes** the opponent — they produce and
   resolve in-flight fleets but launch nothing. This is the opponent-agnostic
   axis, and it is the lever this note is about.
2. **The leaf fidelity.** Our leaf is the linear `favor` board score;
   Producer's is the exact net-ship-swing combat delta
   (`sparse_launch_flow_delta` in
   `agents/producer/orbit_lite/garrison_launch.py`). Separate, bigger
   lever (Option B below).

**Proof the scorer is where Producer's edge lives:** our `producer_lite`
port (`lib/producer_lite.py`) kept all the cheap pieces faithfully and
only swapped the forward-sim scorer for a cheap "production × time-left"
proxy. It wins **3%** as an attacker where real Producer wins **78%**
(`/tmp/transfer2.log`, `audit/2026-06-04-producer-lite-build.md`). The
~45 points of win-rate ARE the scorer; the cheap heuristics carry almost
none of it.

## The opponent model is ONE function — `opp_actions_for_snap`

Every live call to the opponent model funnels through a single function,
`opp_actions_for_snap` (`agents/baseline/chooser.py:60`), called from
exactly three live sites:
- `build_trajectory_baseline` (the idle baseline) — `chooser_trajectory.py:581`
- `score_candidate_v4` (each solo candidate) — `chooser_trajectory.py:693`
- `score_candidate_v4_joint` (coalition candidates) — `chooser_trajectory.py:875`

That one chokepoint is the entire opponent-model surface. Making the
agent opponent-agnostic is therefore a **gate on one function**, not a
rebuild (see "The spike" below). `lib/v7_search.py` + `lib/fast_sim.py`
remain a SECOND, heavier forward-sim substrate (terminal `us − them` net
totals), shelved for narrow drop-one enumeration — useful for Option B,
but NOT needed to answer the opponent-model question.

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

**Consequence — a simplification, not just a perf change.** When the
passive rollout proves out, `lib/opp_model` (`lite_greedy_policy`,
`top_tier_mirror_policy`) and the producer_lite-as-rollout-opponent
become dead weight — less code, less cost (no per-tick opp policy
across every candidate × every tick), no torch. Then feed the robust
scorer RICHER candidates than Producer (coalitions, wait-then-fire),
which an opponent-agnostic scorer values correctly. Same lens, better
menu → plausibly BETTER than Producer.

## The spike — IMPLEMENTED 2026-06-04 (`BASELINE_OPP_PASSIVE`, default OFF)

The opponent model is ONE chokepoint function, so the spike is a gate on
it, not a v7_search rewire. `opp_actions_for_snap`
(`agents/baseline/chooser.py:60`) now returns all-empty opponent actions
when `BASELINE_OPP_PASSIVE=1`:

```python
if os.environ.get("BASELINE_OPP_PASSIVE", "0").strip() == "1":
    return [[] for _ in range(num_seats)]
```

`fs_step` with `[]` for a seat is passive-not-deleted
(`lib/fast_sim.py:373`): it calls the real env interpreter, so opponents
still gain production and their in-flight fleets still resolve combat —
only NEW enemy launches are frozen. Both the candidate rollout
(`score_candidate_v4`) and the me-idle baseline
(`build_trajectory_baseline`) read the SAME function, so they switch
together and the marginal-value framing stays symmetric ("my move's value
given opponents sit still"). Default OFF ⇒ champion bundle byte-identical.

**Behavioral consequence (the trade):** active `lite_greedy` opponents
penalize fragile captures (they counter-launch and retake inside the
window → `favor` drops → delta shrinks). Passive opponents remove that
recapture correction — exactly Producer's behavior. We give up a
correction computed against a patsy (`lite_greedy` plays nothing like the
3,700-team field) for robustness + a real speed win. The one risk —
overextension into recapture — is guarded by `hold_need` / launch-rules
discipline (our `safe_drain` equivalent).

## Probe decision (PI, 2026-06-04): SKIPPED

The v7-vs-Producer triage A/B was NOT run — PI judged it a waste ("the
strategy will win"). We measure at the real gate instead: n≥32 + Rule 46
wallclock once the opponent-agnostic hybrid exists.

## Next steps

1. ✅ **DONE — passive-opponent flag** (`BASELINE_OPP_PASSIVE`, above).
   Sanity smoke confirms it plays a full game (captures, holds, wallclock
   under budget). It does NOT yet change the leaf.
2. **Leaf fidelity (Option B, the bigger lever):** the leaf is still the
   linear `favor` score, not Producer's exact net-ship-swing combat
   delta. Reimplement the flow-delta on producer_lite's `_build_projection`
   (`lib/producer_lite.py`); unit-test net-ship-swing against `env.step`
   ground truth before wiring as a value head.
3. **Widen the candidate menu** feeding the passive scorer: our coalitions
   + capture-floor sizes + adaptive-K reach + wait-then-fire.
4. **Measure at the real gate only** (PI skipped the triage probe): n≥32
   vs Producer + a multi-opponent panel, Rule 46 wallclock < 1000 ms max
   turn.
5. **Delete on success:** `lib/opp_model`, producer_lite-as-opponent.

## Pointers
- `agents/baseline/chooser.py::opp_actions_for_snap` — the one opponent-model
  chokepoint; gated by `BASELINE_OPP_PASSIVE`.
- `agents/baseline/chooser_trajectory.py::score_candidate_v4` — the live
  forward-sim scorer (clones + rolls K ticks + `favor` leaf).
- `lib/v7_search.py`, `lib/fast_sim.py` — heavier forward-sim substrate
  (terminal `us − them`), reserve for Option B candidate breadth.
- `agents/producer/orbit_lite/garrison_launch.py::sparse_launch_flow_delta`
  — Producer's exact scorer (the thing to mirror in Option B).
- `lib/producer_lite.py::_build_projection` — the projection substrate
  Option B reuses; the proxy at lines 301–312 is what Option B replaces.
- `audit/2026-06-04-producer-lite-build.md`, `/tmp/transfer2.log` — the
  3%-vs-78% proof that the scorer is the edge.
