# Open question — 2026-05-29 — can the perf-chain headroom be spent OFF the chooser?

## Setup

Today's three subprocess-isolated A/Bs all spent (or attempted to
spend) the ~400 ms/turn nominal headroom from the perf chain
(WC=800ms + KT singleton + vectorized `predict_fleet_fate`):

- **Level 0** (do nothing differently): parity vs pre-perf
- **Level 1** (spend on JOINT breadth: TOP_K 3→15, MAX_PAIRS 20→200,
  AGGR=True): parity vs same-bundle no-JOINT
- **Level 2** (spend on proposer admissibility via H44 wait_N filter):
  REGRESSED 9.4pp

All three are "spend headroom inside the chooser/proposer stack."
Per Rule 37 the class is closed on this branch.

## The unanswered question

Where else COULD the headroom go that would actually help?

Candidates not yet explored:

1. **`BASELINE_OPP_TIER=1` ladder-realistic opp model** — `chooser.py:46-50`
   the per-call opp model is selectable: `lite_greedy_policy` (default,
   ~1-2ms/call) or `top_tier_mirror_policy` (~5-10ms/call, ladder-
   realistic v3.5.1 aggressive snipe pipeline). With the headroom
   the 5-10× cost is affordable per-rollout. Hypothesis: the chooser's
   rollouts are mis-pricing leaves because the OPP model in-rollout is
   too dumb. **Untested standalone.** Cost: 1 env flip + n=32 ~45 min.

2. **Wider `MAX_HORIZON`** (`proposer.py:30`, currently 40): linear
   cost in horizon, but EDA H41 (Mine 4) said 76% of top-10 winners
   expand ship-share in the last 100 turns. H41 floor=50 falsified
   yesterday — but that was floor magnitude, not horizon depth. Horizon
   bump to 60-80 is a different lever. **Risk:** H41-class falsification.

3. **Multi-source `wait_N` JOINT pairs** (not just fire-now pairs).
   `chooser_trajectory.py:931-969` currently pairs only `wait_N=0` candidates
   in JOINT enumeration. With the headroom, expanding to `wait_N>0`
   joint pairs is feasible. **Untested.**

4. **Spend headroom OUTSIDE the chooser entirely:** opening-phase deeper
   look-ahead, defensive in-flight ledger projection, or beam search
   over candidate portfolios (`lib/candidate_portfolios.py` is
   partially built per `state/hypothesis-board.md:374`).

## Falsification gate before pursuing

Rule 37 says one more chooser/proposer test closes the class. Any
candidate above that lives in `chooser_trajectory.py` /
`chooser.py` / `proposer.py` already counts toward Rule 37. The only
candidates that survive the gate are:

- Anything in `lib/candidate_portfolios.py` (different file = different
  axis under the looser taxonomy)
- Anything that runs BEFORE the chooser (proposer breadth, opening
  classifier, archetype meta-selector — but most of these are also
  Rule-37-saturated)
- Anything in the value head that's NOT leaf-scoring (e.g., explicit
  endgame-predicate gate, distillation-trained head from Phase A)

## Why this is a question, not a hypothesis

The headroom is a real budget. The question of where to spend it has
been re-asked four times this week (H41 floor, perf-chain, JOINT,
wait_N filter) and the answer has been "nowhere I've tried yet" four
times. Either:

- The answer is genuinely "the chooser was already at the local
  optimum and headroom is wasted" — in which case the headroom commits
  should be reverted (they cost zero μ either way, but the perf-chain
  audit complexity is real overhead).
- The answer is "somewhere we haven't tried" — in which case the next
  experiment should be one of the candidates above, framed clearly
  as "spend headroom on X, not as a chooser tweak."
- The answer is "the headroom is only valuable if combined with a
  bigger structural shift" — in which case headroom-spending tests
  are misframed entirely; the structural shift comes first.

PI input needed to choose direction; this branch is exhausted of
hypotheses I'd confidently test next.
