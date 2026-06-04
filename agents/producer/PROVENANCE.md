# Provenance — "Producer" agent

**What this is.** A vendored copy of a third-party *public* Orbit Wars
agent published as a Kaggle notebook ("The Producer", a self-contained
torch + stdlib single-game agent). The author's `USAGE.md` is preserved
alongside the code.

**Why it is in our repo.** It is used **only as an internal evaluation
opponent** — a strong, architecturally-distinct calibration target for
our local A/B panel (registered as the `producer` short-name in
`fast.py`, and a member of `DEFAULT_PANEL`). We do **not** submit it,
copy its strategy, or derive agents from it. It was vendored (rather than
left in `/tmp`) because the remote execution container is ephemeral and a
panel opponent must survive container recycling.

**Layout.**
- `producer_agent.py` — our 2-arg harness adapter (loads `main.py` under a
  unique module name to avoid `sys.modules["main"]` collisions).
- `main.py` — the upstream agent entry point (exports `agent(obs)`).
- `orbit_lite/` — the upstream support package (geometry, planner, etc.).
- `USAGE.md` — the upstream author's usage notes.

**Vendored.** 2026-06-04, from `/tmp/producer` (notebook + utils zip
extracted). Bytecode caches and the source `.zip`/`.ipynb` were not
vendored.

**Attribution / licensing caveat.** This is someone else's published
work. If we ever do anything with it beyond local evaluation — or if the
repo's public-leaderboard adjacency raises a redistribution concern —
revisit the original notebook's license first.
