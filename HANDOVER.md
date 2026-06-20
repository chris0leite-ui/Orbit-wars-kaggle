# HANDOVER.md — next-session brief

> **Refreshed 2026-06-20 (deep-search-refuted session).** Live agent is
> `least_resistance` (2-ply take-and-hold). This session tested four strength
> ideas, ALL null/negative at proper sample — read the two new KB thoughts:
> `2026-06-20-deep-search-refuted-strength-vs-timing-bind.md` and
> `2026-06-20-scorer-objective-not-myopic-loss-decided-step30-50.md`. The
> 06-18 dropout brief below is HISTORICAL (that line is superseded by
> least_resistance).

## ⚠️ 2026-06-20 — state of play (read this first)

- **Nothing shipped this session. Zero submits.** Rolling pair unchanged:
  `{53858931 lr_concentrate (selective-concentration PROBE, still warming),
  53842571 lr_2ply_nocomet (champion ~1008)}`. The broken depth-3 ERROR slot was
  already evicted by the probe last session.
- **FIRST THING NEXT SESSION:** `kaggle competitions submissions orbit-wars | head -5`
  — read the **concentration probe `53858931`** μ. It tests whether selective
  force-concentration pays on the WEAK-opponent ladder (the local strong panel
  can't see that regime). If μ clears the champion ~1100 → promote concentration
  (one-line default flip + Rule 46/42); if it settles even → close that line.
  This ladder read is the one real-world signal pending.

- **Four strength ideas REFUTED this session (all banked, do NOT re-walk):**
  1. **Deep search depth-3** — n=32 paired vs V2: PARITY with 2-ply even with the
     accurate producer rollout opponent (Δmargin −0.062, p=1.00); the old 17/28
     was n=28 noise. And it's caught in a **strength-vs-timing bind**: the accurate
     opponent is too slow for the wall (1581 ms spikes → the ladder ERROR), the
     wall-safe cheap (`LR_DEEP_OPP=1`) opponent REGRESSES (−0.688, p=0.01).
     Anytime-guard code (`rollout_value` deadline + candidate[0] guard) +
     `tests/test_deep_search_anytime.py` are committed, **default-OFF, safe if ever
     revisited; depth-3 is NOT baked ON.**
  2. **Over-commit / scatter** (prior session) — n=40 inert vs strong panel.
  3. **Scorer myopia** (it under-values territory) — REFUTED: ship-margin and
     planet-margin move together; ship advantage is a good LEADING indicator. The
     scorer's objective is fine.
  4. (implied) more search depth / longer horizon / value-territory-more — none move
     the needle.

- **Where the evidence triangulates:** losses are **close games that tip wrong and
  then snowball** (even at step 30; only marginally behind at step 50: −2.8 ship
  margin; blowout later via production compounding). NOT defense (we shed *fewer*
  of our own planets in losses), NOT raw neutral count (we grab slightly *more*
  neutrals in losses, 2.75 vs 2.50). The gap is **move quality at the margin in the
  early-midgame**, and it smells partly seat/map-bound (losses are the low-expansion
  maps). `scripts/diag_scorer.py` + the two KB thoughts have the numbers.

- **THE OPEN FORK (PI deferred with "wrap up" — decide next session):**
  (a) **Test a neutral-QUALITY fix** — measure whether we capture lower-PRODUCTION
      neutrals than V2 by step 50, and if so bias neutral ranking (currently
      `rank = prod/eta`, a cheap-fast bias) toward production. Modest confidence
      (4 tactical knobs already died at n=32).
  (b) **Wait for the probe `53858931`** before more compute (deadline 2026-06-23).
  (c) **Pivot to win-equity / variance-when-behind** — the standing KB recommendation
      (`2026-06-14-why-stuck-4p-value-head-wrong-objective.md`): value the gap to the
      single strongest opponent and accept variance to steal the close games, instead
      of playing safe 2nd. Bigger, architectural. `LR_LEADER_RELATIVE_4P` exists but
      regressed 4P at n=16 — would need a fresh, correct take + n≥32.

- **Postmortem:** `audit/2026-06-20-postmortem-kaggle-dropout-strategy-improve-g57iln.md`
  — key lesson + PENDING improvements.md promotion candidate (`revalidate-subbar-prior`:
  re-measure a sub-n-32 strength claim at n≥32 BEFORE planning to ship it; the
  deep-search line burned ~half a session hardening a premise that collapsed).
  Not yet ratified — PI to approve.

---

> **Refreshed 2026-06-18 (smart-dropout session).** Active line =
> **smart dropout on `agents/producer_plus`**. Read `state/DROPOUT_PLAN.md`
> first (the executable roadmap), then
> `knowledge-base/thoughts/2026-06-18-dropout-and-seat-eval-confound.md`.
> The take-and-hold / least_resistance brief below is retained for history.

## ⚠️ Dropout session — state of play (2026-06-18)
- **Smart dropout** (model-free robustness REPLACING opponent modelling) is
  built on `agents/producer_plus`, all default-OFF, OFF path byte-identical.
  Variants in `scripts/bundle_producer_plus.py`: `dropout`, `dropout_live`
  (add-on to the live stack), `dropout_repl` (replaces opp model).
- **Result:** dropout replaces the opponent model at PARITY for ~half the
  per-turn cost. Compute axes (deeper/more-sims/naive-gen) don't help — the
  DROP MEASURE is the lever. Phase 1a incentive-weighting
  (`PRODUCER_PLUS_DROPOUT_INCENTIVE`) committed default-OFF, A/B in progress.
- **Eval lesson (load-bearing):** outcome is MAP-determined and seat-invariant;
  NO first-mover effect, NO seat bias. Evaluate on many DIVERSE map-seeds, one
  game per seed; do NOT condition on seat (confounds with map); do NOT use
  `fast.py eval` (correlated map-pairs). Run variants SEQUENTIALLY (parallel
  torch OOM-kills heavy variants). Full detail in the thoughts entry.
- **Clean wide-map A/B (28 maps vs V2):** base 15/28, more-sims 14 (parity),
  incentive 13, winprob 12/11, deeper(H30) 6. **Nothing beats base** — the
  bolt-on is SATURATED (producer's one-ply static value function is the
  ceiling). Two eval bugs found+fixed this session: seat tied to seed parity
  (a confound, not a seat effect), and a single-process knob env-leak (run one
  bundle per fresh subprocess).

- **THE FORK (decide next session):**
  (a) **Ship the cheap replacement** — dropout replacing the opponent mirror is
      ~54% vs V2 at ~half cost; harden (more maps + opponents) and submit via
      the Rule 42 gate; OR
  (b) **Build the dropout-NATIVE agent** — see **`state/DROPOUT_NATIVE_DESIGN.md`**
      (full build plan): value = expectation/CVaR over an ensemble of stochastic
      flip-HAZARD rollouts (mean-field v1, deterministic, no RNG), reusing the
      batched `_run_exact_recurrence` + `_reactive_reinforcement_margin` +
      producer's shortlist. Phase A is a hard KILL-GATE: a distribution-aware
      forward model scoring the same candidates must beat base 15/28, else stop.
  Do NOT keep refining the producer bolt-on — the data says it's done.

- **FIRST THING NEXT SESSION:** read `state/DROPOUT_PLAN.md` then
  `state/DROPOUT_NATIVE_DESIGN.md`; re-pull Producer V2
  (`kaggle kernels pull slawekbiel/the-producer-v2`); decide the fork above.

---

## (historical) take-and-hold brief — 2026-06-17

> Refreshed **2026-06-17** (take-and-hold session). Supersedes the 2026-06-15
> producer_plus brief. ~6 days to the 06-23 deadline. Full session record:
> `knowledge-base/thoughts/2026-06-17-take-and-hold-and-threat-aware-margin.md`.

## State of play

- **Live agent = `least_resistance`** (`agents/least_resistance/main.py`), NOT
  producer_plus (that line is superseded). It is the producer's `orbit_lite`
  garrison-flow scorer + a 2-ply lookahead + (NEW this session) **take-and-hold**.
- **Shipped this session:** sub **53768768** "take-and-hold" (tarball sha
  `dc3a4f17`), two levers now **default-ON** in code:
  - `LR_HOLD_MARGIN=0.5` — size enemy captures to take AND hold (⇒ concentration).
  - `LR_DEFEND=1` — reinforce own planets about to be flipped (regroup/defense).
- **Confirmed (n=32 independent seeds):** 2P vs Producer V2 **14/32 → 21/32 (+7)**;
  4P vs {V2, Roman-1224, konbu17} **18/32 → 19/32 (parity, no regression)**.
- Rolling pair: `{53768768 take-and-hold, 53741746 lr-fixed @μ1115}`. Broken μ172
  agent evicted.

## ⚠️ FIRST THING TO DO

1. **Re-check sub 53768768.** At session end it was **PENDING ~2 h** (anomalous).
   `kaggle competitions submissions orbit-wars -v | head -3 | awk -F',' '{print $1,$(NF-2),$(NF-1)}'`
   - **COMPLETE + μ** → it passed; compare μ to the 1115 backstop. If μ > 1115 the
     2P lift converted to the ladder → great. If ≤ 1115, the ladder field differs
     from our local panel (V2 isn't the ladder) — note it, don't panic.
   - **Error** → real validation failure. But the EXACT bundle passed a local
     real-loader self-play (2P + 4P-to-250, max 469 ms, 0 timeouts, entry=`agent`,
     `_ORBIT_OK`=True), so re-extract `/tmp/lr_submission.tar.gz` and diagnose what
     Kaggle does differently. (Bundle was almost certainly fine → queue delay.)
2. **Then build the threat-aware dynamic margin** (the agreed next lever, full spec
   in the thoughts entry §"NEXT TASK"). Replaces the placeholder flat `0.5`.

## The method that produced the win (use it)

**PI replay observation → diagnose the mechanism IN CODE → smallest native fix →
A/B at n≥32 INDEPENDENT seeds vs the REAL peer (Producer V2) with a regression
check → ship.** The producer/V2 is the discriminating opponent; **v7_0 is a
ceiling (we crush it) — do not tune against it.**

### CRITICAL methodology rule (learned the hard way, in improvements.md)
**A/B independence: one game per fresh distinct seed; rotate the SEAT across
DIFFERENT seeds, never within a seed.** Seat-reusing one seed = correlated games;
it masked the real 2P lift under a false "parity". `scripts/verify_confirm.py`
already does this (64 distinct seeds, 32/block, OFF/ON paired). Reuse it.

## Real public-agent benchmark panel (this session's biggest unlock)

External kernels live in `audit/external/` (gitignored — re-pull on a fresh
container). **Reproduce:** `kaggle kernels pull <ref> -p audit/external/kernels-pulled/<name>`,
then extract the `%%writefile *.py` cell containing `def agent` to a runnable
`main.py`. Confirmed-playing opponents: **konbu17** (~85% panel ML-hybrid),
**Roman-1224**, **ykhnkf-1100**, **vickimar-1110**, **Producer V2**.
- **Arity gotcha:** `def agent(obs)` vs `agent(obs,cfg)` — call via `env.run([...])`
  (handles arity) or an arity-aware wrapper; a 2-arg call to a 1-arg agent looks
  like "idle".
- **Producer V2 runs on OUR `orbit_lite`** (verified) — no module conflict.
- We **crush the field** (2P 6/6, 4P 8/8 vs konbu17/Roman/ykhnkf); **V2 is the
  one peer** (we're built on its engine). Beating V2 is the climb.

## DO NOT re-walk (refuted at proper sample vs the producer/V2)
leader-relative 4P objective (regressed 4P 12→6) · value-commit (washed out) ·
enemy-boost (regressed 2P) · anytime (null; never spent the bank) · deep rollout
search (parity, fixed-2/broke-2). All still gated default-OFF in `main.py` —
**strip them in a cleanup.** Lesson: depth/objective/value alone fail; the moves
must be *generated* (breadth), which is what take-and-hold fixed.

## Bundle / smoke / submit (the least_resistance path)
- Build: `bash scripts/build_least_resistance.sh` → `/tmp/lr_submission.tar.gz`
  (root: `lib main.py orbit_lite producer_main.py`). Code defaults bake the ship
  config (LR_HOLD_MARGIN=0.5, LR_DEFEND on).
- Rule 46 smoke: `python -m pytest tests/test_least_resistance_entry.py -q`
  (agent is last callable) **+** real-loader self-play (extract tar, strip repo
  from `sys.path`, `env.run([m.agent]*N)`, 2P AND 4P, confirm active turns, max
  turn < 1000 ms, `_ORBIT_OK`=True).
- Rule 42 gate: `kaggle competitions submissions orbit-wars | head -5`, append a
  claim row to `state/MULTI_BRANCH.md` (evicted sub_id + μ vs predicted μ), submit
  single-shot with PI sign-off. Budget 5/day, rolling-last-2.

## Open questions / flags
- **Q:** did 53768768 pass, and does the 2P +7 vs V2 convert to ladder μ > 1115?
- **Q:** 4P is only at *parity* with the strong field — what's the 4P-specific
  lever (FFA targeting / kingmaker)? It's ~60% of the ladder = the bigger prize.
- **FLAG (timing):** a 1441 ms 4P-vs-field midgame turn was seen (self-play clean
  ≤570 ms). Bound the total turn so we never eat overage on the ladder.
- **FLAG (cleanup):** strip the 5 refuted default-OFF levers from `main.py`.

## Pointers
- `knowledge-base/thoughts/2026-06-17-take-and-hold-and-threat-aware-margin.md` — this session + the dynamic-margin spec.
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42); top row = the take-and-hold submit.
- `.claude/skills/kaggle-comp/improvements.md` — the A/B-independence rule.
- `CLAUDE.md` — process rules.
</content>
