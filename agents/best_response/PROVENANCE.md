# Provenance — `best_response` agent

**What this is.** A search agent built on our own fast, parity-tested engine
(`lib.fast_sim`). Per turn it runs the vendored Producer **once per seat** (our
own plan + each opponent's predicted move), then uses the cheap engine to
forward-simulate a sparse set of candidate first-moves over the Producer's
18-step planning horizon and keeps the best reply. See the module docstring in
`main.py` for the full rationale.

**Built for.** The PI's 2026-06-16 request: "run the producer once, then apply
your simulation to the 18 steps the producer has converged to … backwards
search the best response along a sparse set of actions." It is a clean redo of
the search-wrapper idea that prior attempts (`scripts/search_wrapper.py`,
`scripts/producer_opp_wrapper.py`) tied on — the difference is the Producer runs
O(1) times per turn, freeing the budget for an 18-step horizon and a richer
candidate set.

**Dependencies.** `lib.fast_sim`, `lib.opp_model.lite_greedy_policy`, and the
vendored Producer at `agents/producer/` (used as opponent model + candidate
source). Needs `torch` (CPU).

**Submission caveat (READ before any `kaggle competitions submit`).** This agent
embeds / depends on the vendored third-party Producer, whose `PROVENANCE.md`
says it is for *local evaluation only* ("we do not submit it … or derive agents
from it"). This is a **local research/eval build**. Submitting it is a separate,
PI-gated decision that must first resolve the Producer's redistribution/
licensing question. (Note: our actual ladder line `producer_plus` is itself
built on the Producer's `orbit_lite` engine, so the team may have already
settled this — confirm with the PI.)
