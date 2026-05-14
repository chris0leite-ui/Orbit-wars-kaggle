"""lib/geo — geometric sense + posture + joint allocator for the `geo` agent.

Three modules:
- sense:     clustering, Voronoi cells, front planets, threat budgets, comet claims
- posture:   one-of-four decision rule (OPENING / EXPAND / DEFEND / BREAK)
- allocator: LP-based joint settlement; falls back to settle_plan on failure

All state lives in dataclasses returned per call. No module-level mutables.
"""
