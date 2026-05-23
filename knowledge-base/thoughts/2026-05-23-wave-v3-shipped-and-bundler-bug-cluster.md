# 2026-05-23 — wave V3 shipped + bundler-bug cluster

## What happened
- Submitted V3 wave (convergence wave + leaf-Δ value-head gate +
  planet_positions cache) as sub 52966655. PENDING at wrap.
- Local n=16 vs baseline_full: 12/16 = 75%, Wilson [0.505, 0.898].
  Clears Rule 45 minimum 0.50 by 0.005.
- Latency profile (n=16 eval): p50=664ms, p95=739ms, max=1199ms
  (down from V2's 2473ms max — cache delivered -45% per-turn p50).
- Three latent bundler bugs surfaced sequentially: in-function
  import indent-strip, multi-line paren import continuation-leak,
  and unlink-on-failure blocking inspection. All three fixed
  this session in source or via workaround; promotion candidates
  drafted for `scripts/bundle_agent.py` improvements.

## What I'm sitting with
- **Local 75% A/B → ??? μ is the unknown that matters.** The
  sibling branch (`claude/extract-physics-trajectory-Vjaz9`) had
  an identical 12/16 = 75% local result on a different value-head
  axis and settled at μ ≈ 963-985. If V3 lands in that band, the
  whole wave family is locally-strong-but-LB-noisy and the
  mechanism is exploiting baseline_full specifically. If V3 lands
  ≥ 1050, the wave is genuinely transitive and V4 (positional
  value, attack_pull, etc.) becomes the natural next axis.
- **Wilson-lo 0.505 is uncomfortably tight.** We submitted on
  n=16; n=32 would have either tightened to 0.55+ (panel-clear)
  or collapsed to 0.45 (parity not lift). The choice was PI-led
  ("submit"+"go") but the resulting calibration point will be
  noisier than it could have been.
- **Latency-tail 1199ms still over 1000ms cap.** Improved from
  V2's 2473ms but not eliminated. Could timeout on Kaggle hardware
  on adversarial seeds. The sibling sub 52963659 ERROR'd with
  similar tail profile.

## What I'd do next session (in priority order)
1. **Read V3 μ from the LB and append to `state/calibration-ladder.md`.**
   This is THE most informative event between sessions; everything
   else is conditional on it.
2. **If V3 μ ≥ 1050:** sketch V4 = positional value head (attack_pull
   from the orbitfix sibling's diagnosis "agent is too passive").
   The wave is transitive; the next axis is leaf scoring.
3. **If V3 μ ∈ [990, 1050]:** the wave is real but small. Don't
   double-down on the same axis — run a Rule 43 panel on V3 to
   see if it beats opponents other than baseline_full, then
   decide.
4. **If V3 μ < 990 or ERROR:** the wave family is not LB-transitive
   OR the latency tail caused timeout. Pivot. The 1199ms max-turn
   becomes the priority-1 fix; the wave mechanism gets shelved
   until latency is < 1000ms.
5. **Regardless of outcome:** promote the three bundler-bug fixes
   to `scripts/bundle_agent.py` itself (preserve indent in
   `_clean_lib_source`, AST scanner for multi-line paren imports,
   `--keep-on-failure` flag). The bugs are latent and will
   recur each time a long-paused branch re-bundles.

## Patterns I want to remember
- **Latent bundler bugs surface only at submit time.** Three weeks
  of code changes can accumulate in-function imports and
  multi-line paren imports without anyone noticing until the next
  bundle attempt. WRAPUP step should consider a periodic dry-bundle
  smoke even on non-submit days, OR the bundler should add
  AST-level pre-checks that catch these without needing the
  ~minute-long full bundle.
- **PI "go" after PI "submit" is permission to compress the gate
  chain, not skip it.** The bundle + parity gates are mandatory
  (Rule 46) and were run. The Rule 43 panel was skipped — that's
  the gate that most often gets dropped under time pressure and
  the one that most often catches single-opponent A/B noise.
