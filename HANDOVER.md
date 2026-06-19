# HANDOVER.md — next-session brief

## 🛑 FRESH 2026-06-19 (late) — deep-search line REFUTED; back to the DEFAULT agent
- **Panel A/B verdict (n=27 paired, stratified `SEED_PANEL_128[::4]`, 1v1 vs V2, P0):**
  **DEFAULT least_resistance (2-ply take-and-hold, `LR_DEEP_OPP=0`) = 18/27 (67%),
  margin +275 — it BEATS V2.** Every deep-search variant this session loses:
  contagion-d6 9/27 (33%, −1068); wide+calib 3/13; wide 2/12. The contagion / depth /
  wide-candidate / calibration line is a ~34-pp REGRESSION over the default. Refuted.
  Full write-up: `knowledge-base/thoughts/2026-06-19-contagion-deepsearch-refuted-vs-v2.md`.
- **KEEP:** the sun-clearance fix (base agent correctness, commit `4f9cf4e`). All
  deep-search code is default-OFF / gated (shipped agent = the default = unaffected).
- **DO NOT** keep pushing contagion/deep-search vs V2 — the panel says it loses.
- **NEXT (the path to crush V2): improve the DEFAULT agent's LOSSES.** It already wins
  ~67%; the leverage is the ~1/3 it loses. On the panel it lost seeds 32 (−1922),
  78 (−1410), 647 (−836). Pull those replays, see what V2 does to the take-and-hold
  agent, fix THAT observation-driven. Anchor EVERY future A/B to the default baseline
  (the methodological miss this session: comparing regressions to each other).
- **Ladder hygiene still open:** the dead depth-3 ERROR still occupies a final-eval
  slot; the DEFAULT agent is timing-safe and beats V2 — a clean resubmit candidate
  (Rule 42 claim + PI sign-off).

---

## ⚡ 2026-06-19 — Phase 1 cheap-opponent landed; run the kill-gate next (SUPERSEDED by the refutation above)
- **Why we're here:** sub **53836276** (`lr_depth3`) ERRORed on Kaggle —
  *"Validation Episode failed."* = a per-turn **timeout** in the self-vs-self
  validation game. Depth-3 re-runs the torch producer mirror at every node and
  has near-zero headroom under the 1 s wall; the slower validation slot tips it
  over. This is the exact wall `state/DROPOUT_NATIVE_DESIGN.md` predicted.
- **Done this session (implement + smoke only, NO submit):** the strategy's
  **Phase 1** — knob `LR_DEEP_OPP` in `agents/least_resistance/main.py` makes the
  deep-search opponent swappable; `1` plugs in the cheap `lite_greedy_policy`
  (~1-2 ms/node vs the mirror's ~10-50 ms). Default `0` = mirror = byte-identical.
  Unit-tested (`tests/test_deep_opp_dispatch.py`) + timing-smoked. Also installed
  CPU torch and folded it into `bootstrap.sh` (was missing → degraded prior repro).
- **Phase 2 landed (`LR_DEEP_OPP=2`):** model-free **contagion** opponent — each
  rollout step flips neutrals + my under-defended planets to the strongest single
  reachable rival (replaces opponent launches; `_apply_contagion`). Reuses the
  `claude/dropout-plan-review-rb5817` principles (the branch's `native_forward.py`
  was refuted as a leaf scorer; here the ideas are an *opponent model*). Tests:
  `tests/test_contagion_opponent.py` (7 pass). Default-OFF byte-identical.
- **n=8 triage vs Producer V2 (1v1, P0, seeds 5000-5007) — margin MONOTONE in depth:**
  mirror_d3 4/8 (−538, maxms **10067**=timeout); lite_d3 2/8 (−3168, 284 ms);
  contagion d3 2/8 (−2594, 330) → d5 2/8 (−2017, 277) → **d6 3/8 (−504, 311 ms)**.
  Contagion at **depth-6 reaches mirror-depth-3 strength (−504 ≈ −538) at ~30× less
  time** — the "cheaper-deeper ≥ mirror" thesis. No config wins outright vs a strong
  1v1 V2 on these hard seeds (the original "17/28" was a different/4P set), but the
  depth trend is real.
- **FIRST THING NEXT SESSION:** escalate the contagion line — n≥32 (Rule 45) at
  depth 6 (and try d7+, where the margin trend may cross zero), plus **4P + the
  original 28-map set** (1v1-vs-V2 seeds look unrepresentative). Tune
  `LR_CONTAGION_REACH_TICKS` / the flip threshold if contagion is too aggressive
  (every-candidate-flips → inert ranking, the branch's warning). Submit only on an
  n≥32 PASS + Rule 42 claim + PI sign-off.
- Details: `knowledge-base/thoughts/2026-06-19-validation-timeout-and-phase1-cheap-opponent.md`.

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
