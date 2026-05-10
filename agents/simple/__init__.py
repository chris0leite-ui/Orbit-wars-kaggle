"""Simple-strategy panel — single-axis target-selection ablations of v1.

Each module under this package implements one `propose_intents(obs)` that
varies only the *target-selection score function*. All five share v1's
tie-break RNG, the `DEFAULT_MECHANISMS` pipeline, and the
"one launch per owned planet per turn" structure — so any winrate gap
across the panel is attributable to the targeting axis alone.

See plan: /root/.claude/plans/read-the-handover-next-imperative-whisper.md
Runner:    scripts/strategy_panel.py
"""
