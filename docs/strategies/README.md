# Strategy notes — Orbit Wars

One markdown per agent we've built or considered. The point: a future
agent (or a returning human) can read **what** the strategy does and
**why** in plain English, separate from the code's how.

## Files

- `shipped-baseline.md` — comp-shipped Nearest Planet Sniper. Floor anchor.
- `v1_orbitfix.md` — first submitted variant (v1, v1.1). Orbit-aware aim
  + tie-break randomisation + production-aware sizing.
- `roadmap.md` — planned v2 (arrival ledger), v3 (mission classes),
  v4 (opponent modeling or MCTS), with the public-research touchstones.
- `heuristics-research.md` — universal strategy primitives
  (deterministic-win predicates, ROI scoring, N-step lookahead, ship
  bundling, compete-relative play, 15-item brainstorm). §K cross-
  references panel + Phase 1 verdicts and recommends path A.
  Companion plan:
  `audit/2026-05-10-research-driven-next-experiments.md`.

### Simple-strategy panel (target-selection ablations)

Five agents under `agents/simple/` share v1.1's mechanism stack and vary
only the target-selection score function. Run via
`python -m scripts.strategy_panel`. Plan:
`/root/.claude/plans/read-the-handover-next-imperative-whisper.md`.

- `simple-nearest.md` — distance-greedy (control; reproduces v1).
- `simple-production.md` — argmax target.production. **75.0% mean panel
  winrate at 8 seeds; 69% vs v1_orbitfix.**
- `simple-roi.md` — argmax production / distance. **96.9% mean panel
  winrate at 8 seeds; 100% (16/16) vs v1_orbitfix — strongest signal.**
- `simple-weakest.md` — argmin target.ships. Falsified at 8 seeds
  (15.6%); kept for opponent-panel diversity.
- `simple-enemy_first.md` — enemy planets first, then nearest.
  Falsified at 8 seeds (32.3%); kept for opponent-panel diversity.

## Naming convention

`<vN>_<one-word-handle>` — handle describes the **load-bearing change**
vs the previous version. The shipped baseline has no `vN` prefix because
it isn't ours.

## Per-agent file template

1. **One-liner** — what's the agent's elevator pitch?
2. **Mechanism** — bullets covering perception → planning → action.
3. **Why it works (or doesn't)** — the strategic argument.
4. **Gotchas** — failure modes the next version should fix.
5. **Evidence** — local gates cleared + live μ if shipped.
6. **What it does NOT do** — the explicit gap that motivates the next version.
