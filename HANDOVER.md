# HANDOVER.md — next-session brief

> Last written: 2026-06-01 PM by `claude/champion-strategy-rules-00JzI`.
> Stale dated sections (2026-05-20 … 2026-05-31) were removed this session
> per PI cleanup request — they live in git history. Durable knowledge moved
> to `audit/` docs and `state/MULTI_BRANCH.md`.

## Read order (Rule 44 — mandatory)

1. **`state/MULTI_BRANCH.md`** — live Kaggle rolling pair, track registry,
   closed tracks (falsified knowledge), push claim board.
2. **`state/TOOLS.md`** — A/B harnesses, diagnostics, validation suite.
3. **`CLAUDE.md`** — rules 1-48.
4. **This file.**
5. `audit/friction.md` before touching a fragile path.

## Live ladder — read it directly, never transcribed

`kaggle competitions submissions orbit-wars`. Rolling pair = the two
**most-recent** submissions (Kaggle auto-keeps exactly these for final
eval; Rule 12). Deadline **2026-06-23 23:59 UTC**. Budget 5/day.

As of 2026-06-01 PM (will go stale — re-read): champion
`baseline_launch_rules_universal` (sub 53182323) **μ=1183.7**, our best ever,
**evicted**. Sync coalition (53223160) μ=1150.2, evicted. **Rolling pair is
soft:** expand_credit (53259633, **1059.4**, newest) + size_balance
(53248277, 1142.0). Recovering it needs *two* champion-class pushes (the
newest weak one survives the first push). 4/5 submits used 2026-06-01.

## 🟢 ACTIVE — 2026-06-01 PM: loss mode re-diagnosed (NOT hoarding); adaptive-K built (v1 in A/B); redesign = state-driven horizon

**One-line state:** the "ship-hoarding / under-expansion" loss-mode framing
is **refuted** — we launch plenty (more in losses). Built a v1 step-schedule
adaptive horizon K (default OFF, committed `9985e98`); its A/B is in flight
(interim 67%). PI redirected K to be **state-driven, not a fixed schedule**.
Top build candidate going forward: **contest-urgency (win the race / arrival
timing)**.

### What this session established (with data + caveats)

- **Loss mode is NOT hoarding** (`audit/2026-06-01-loss-mode-diagnosis.md`).
  Champion's own 121 live games: opening planet-gap ≈ 0 (we're not behind
  early); the split is **midgame capture *rate*** (2P: wins capture 25 vs
  losses 12) while defense is ~equal. In losses we carry a *higher* in-flight
  ship fraction — we out-*launch* but under-*capture*. So it's a **conversion
  gap, not a volume gap.** This explains the flat-expand-credit regression
  (−124μ: added volume to a non-volume problem).
- **PI corrections (load-bearing):** (1) **selection bias** — win/loss is
  confounded by opponent strength, so the conversion gap is *not* proven as a
  fixable mechanism; (2) **fleets do NOT die in flight** (no air collisions)
  → the H44 "destroyed in-flight" lever is **dropped**; conversion =
  sizing + timing + winning the race, never survival; (3) opening *tempo* is
  real ("we open too slowly") and chains into the midgame.
- **Paired positioning check is a NULL** — within-game (selection-bias-free),
  our ships are no more "rear" than the opponent's (US 39.2 vs OPP 39.2 mean
  dist to enemy; 56% vs 55% rear). The "ships stuck in the rear" hypothesis
  doesn't hold on the field average; if real it's opponent-specific or a
  tempo/rate effect a snapshot can't see.
- **Adaptive horizon K built** (`audit/2026-06-01-adaptive-horizon-k-
  investigation.md`). Single lever: `launch_rules.capture_horizon_k(step)` is
  read by the launch gate + proposer prune + sync cap, so phase-awareness
  propagates consistently; value head already sees horizon ~40. v1 =
  step-decay K_OPEN=20→floor 10 by step 30 (default OFF → byte-identical
  champion, 19 tests green, wallclock smoke clean p95≈283ms).
- **A/B contamination caught:** ON-vs-OFF of the *same* bundle is invalid —
  `capture_horizon_k` reads env live, `env.run` shares one process, so the ON
  bundle's baked `ADAPTIVE_K=1` leaks and turns OFF adaptive too. Valid A/Bs
  use the pre-edit `submissions/baseline_champion_nokt.py` as the **immune**
  opponent (0 refs to the new env).

### Next-session first actions (ranked)

1. **Read the v1 adaptive-K A/B verdict** (`/tmp/adaptiveK_vs_champ_ab.txt`
   or re-run `clean_ab champ_adaptiveK_on.py baseline_champion_nokt.py
   --seeds 16`). Interim was 12W/6L. If non-negative → the horizon lever has
   signal; build the **state-driven** redesign (§8 of the K doc): per-target
   `K ≈ time-until-enemy-interference`, clamped — raises horizon in midgame
   lulls, which the step-schedule misses. If clearly negative → the lever is
   dead, pivot to contest-urgency.
2. **Contest-urgency (top build candidate).** Conversion = sizing+timing+race.
   Sizing alone failed (size-balance regressed); the untested half is
   **timing/race** — prioritise captures we *narrowly win* the race for,
   defer bankable ones. Value the opponent's reachability of each target.
   Evaluate vs *aggressive* opponents, not the champion mirror.
3. **Recover the rolling pair.** It's soft (1059+1142). Push champion KT-OFF
   (`submissions/baseline_champion_nokt.py`, ready, config baked) — needs two
   pushes to fully clear; surface the Rule-42 claim each time.

### Submission discipline reminders for this branch

- **Bundle-baking gotcha (load-bearing):** Kaggle runs with **no env vars**,
  so every `BASELINE_*` toggle falls to its code default (sync, pv_eta,
  launch_rules all default OFF). Bake the full tested config as an
  `os.environ.setdefault(...)` header **above the first inlined module**
  (constants are read at import). Verify with a clean-env smoke.
- **Rebuild recipe:** `python scripts/bundle_agent.py agents/baseline
  --out-dir submissions --force --skip-parity-gate` (internal parity gate
  breaks in-container — `agents` collides with `kaggle_environments.lux_ai_s3`;
  verify via `test_bundle.py` + clean-env smoke), then splice the baked
  header after the `from __future__` line.
- `submissions/*` is git-ignored → bundles do NOT survive a fresh clone.
  Rebuild before trusting any A/B.

## Pointers (durable)

- `audit/2026-06-01-loss-mode-diagnosis.md` — the not-hoarding diagnosis +
  PI corrections + positioning null.
- `audit/2026-06-01-adaptive-horizon-k-investigation.md` — K lever map,
  reachability data, v1 build, state-driven redesign (§8).
- `audit/2026-06-01-live-replay-diagnosis.md` — the *prior* hoarding
  diagnosis (now superseded; kept for the fleet-outcome distribution).
- `state/mechanism-ledger.md` — every agent family tried.
- `state/hypothesis-board.md` — open ideas, killed list.
- `knowledge-base/concepts/evaluation-metrics.md` — Rule 48 eval protocol.
- `audit/2026-05-18-seed-panel.md` — 128-seed geometry panel for A/B.

## Rule reminders (most relevant)

- **1 / 12 / 42:** submissions PI-approved single-shot; rolling-last-2; fill
  the push claim board before any submit.
- **40:** prefer modeling-correctness over restriction-tuning.
- **41:** confound-sweep before correlational conclusion (the selection-bias
  point above is exactly this).
- **43 / 45:** multi-opponent panel + champion h2h, n≥32, before submit.
- **47:** physics-primitive verification before agent design.

> Prior writers (superseded, per-branch): `kaggle-baseline-strategy-lO4mm`,
> `audit-workflow-performance-btjeK`, `strategy-framework-design-OyoYR(-rebased)`,
> `ml-competition-strategy-PFhzM`, `analyze-game-strategy-EpMVP`,
> `review-skills-improvements-moKOR`.
