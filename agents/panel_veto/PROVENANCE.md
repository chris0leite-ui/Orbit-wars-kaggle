# Provenance — "panel_veto" agent (eval-only)

**What this is.** A vendored copy of a third-party *public* Orbit Wars
agent (Kaggle notebook `anthonytherrien/floor-matched-fleets-target-veto-evacuation`, "Orbit Wars
| I'M SMARTER"). Like every strong agent in the public field, it is a
**ProducerLite variant** — its `main.py` imports the shared `orbit_lite`
package (`ProducerLiteConfig` / `plan_lite_waves` / `ProducerLiteRuntime`),
which we already vendor at `agents/producer/orbit_lite`. The adapter
(`agent_entry.py`) points its imports there.

**Why it is here.** Used **only as a local A/B panel opponent** — a
strong, differently-tuned ProducerLite reference. We do **not** submit it,
copy its strategy, or derive agents from it.

**Vendored.** 2026-06-13, from the public notebook's `%%writefile main.py`
cell. Only `main.py` is the agent; `orbit_lite` is shared from the
Producer vendor. Notebook scaffolding/training cells were not vendored.

**Caveat (referee-blindness finding).** This agent does NOT, on its own,
reduce local referee blindness for the shot-MLP class of mechanism: as a
strong ProducerLite opponent our response-veto mirror models it well, so
it elicits ~0% low-probability attacks from us (vs ~33% on the live
ladder). Its value is as one member of a HETEROGENEOUS 4P panel. See
`audit/2026-06-13-referee-blindness-diagnosis.md`.

**Licensing.** Someone else's published work; eval-only use. Revisit the
original notebook's license before any use beyond local evaluation.
