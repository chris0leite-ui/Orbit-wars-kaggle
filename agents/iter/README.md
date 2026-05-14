# `agents/iter` — fast-iteration scaffold

Forkable copy of **v7_pv** (= `v7_0_drop_one` chooser + `PV_GAMMA=0.99`),
the ladder-best v7-family agent (mu=1064.4). Day-zero behaviour is
functionally equivalent. Iterate by editing knobs or hooks; eval against
a multi-opponent panel; bundle when a variant passes.

## Knobs (top of `main.py`)

| Knob | Default | Sweep range | Notes |
|------|---------|-------------|-------|
| `K` | 10 | 8 / 10 / 12 / 15 | Lookahead horizon. Watch wallclock. |
| `WALLCLOCK_MS` | 700.0 | 500-900 | Per-turn budget; Kaggle eval cap is 1000 ms. |
| `ENUMERATOR_MODE` | `"drop_one"` | see `lib.v7_search` | Candidate proposer. |
| `OPP_TIERS` | `(1,)` | `(1,)` / `(1,2)` / `(0,1,2)` | >1 entry triggers MAXIMIN robustness. |
| `PV_GAMMA` | 0.99 | 0.95-1.0 | 1.0 reverts to plain v7_0_drop_one. |
| `VALUE_FN` | `"default"` | `"default"` / `"composite"` | Leaf-state head. |

## Patch surfaces (four places to hook)

1. **Top-of-file knob** — change one constant, run eval. 1-line edit.
2. **`PRE_FILTER` hook** inside `agent()` — short-circuit or mutate `obs`
   before `choose()`. E.g. *"if `step > 480` and lead is locked, return `[]`."*
3. **`POST_PROCESS` hook** — sanitise the returned action. E.g.
   *"drop launches whose trajectory leaves the play area"* via
   `lib.trajectory`.
4. **New value head** — add a function to `lib/value_heads.py`, extend
   `_resolve_value_fn` here, set `VALUE_FN = "your_head"`.

## Eval (Wilson-gated, ~3-8 min wallclock)

```bash
# 2P, 3-opponent calibration panel (v7_0, v4_planner, v3.5.1).
python fast.py eval iter --vs-panel default --max-seeds 32

# Broader 2P panel.
python fast.py eval iter --vs-panel v7_0,v4_planner,v3.5.1,v3_snipe,roi_baseline --max-seeds 32

# 4P first-place rate, fixed background.
python -m scripts.ffa_panel --focals agents/iter/main.py --background weakest enemy_first baseline --seeds 32
```

## Loss-mode inspection (after a submission settles)

```bash
bash scripts/iter_losses.sh <submission_id>
# Pulls replays, classifies (opening_lost / mid_economy_lost / ...),
# prints Counter and CSV path for drill-down.
```

## Bundle + parity gate

```bash
python scripts/bundle_agent.py agents/iter
pytest tests/test_iter_agent.py tests/test_bundle.py -q
```

## Submission gate (Rule 12 reminder)

Kaggle keeps your **rolling last 2 submissions** for final evaluation.
A push of `submissions/iter.py` evicts the previous oldest — currently
that's **v7_pv at mu=1064.4** (our team-best). Only push iter if its
local-panel Wilson lower bound + a margin for the
`local-overpredict-2x` calibration warning clears 1064.4.

`fast.py eval --vs-panel` over-predicts the ladder by 80-150 mu in the
last two submissions. Until we add more opponent classes, treat a
panel-PASS variant as ~70% likely to hold ladder rank, not 100%.
