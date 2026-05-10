# Strategy notes — Orbit Wars

One markdown per agent we've built or considered. The point: a future
agent (or a returning human) can read **what** the strategy does and
**why** in plain English, separate from the code's how.

## Files

- `shipped-baseline.md` — comp-shipped Nearest Planet Sniper. Floor anchor.
- `v1_orbitfix.md` — first submitted variant. Orbit-aware aim + tie-break
  randomisation. Beats baseline 40/40 locally; live μ TBD.
- `roadmap.md` — planned v2 (arrival ledger), v3 (mission classes),
  v4 (opponent modeling or MCTS), with the public-research touchstones.

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
