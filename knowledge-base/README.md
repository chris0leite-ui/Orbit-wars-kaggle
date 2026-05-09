# knowledge-base/ — PI second-brain

> Permanent. Append-only. Per CLAUDE.md Rules 35–36 — never overwrite,
> delete, or archive on cleanup.

## Subdirs

- `thoughts/` — PI voice-dumps, dated, one file per session or topic.
  Format: `YYYY-MM-DD-<slug>.md`. The agent transcribes lightly and
  links related entries.
- `concepts/` — durable abstractions PI wants the future-agent to
  internalise (operational environment, decision-quality framing,
  BOTE recipes, etc.). Cross-comp; a new entry here means a lesson
  the PI wants to outlast this competition.
- `friction/` — cross-session friction observations beyond what
  `audit/friction.md` captures. Use this when a friction is slow-burn
  (felt over many sessions) rather than per-session.
- `flags/` — standing concerns / things-to-watch-for that the agent
  should re-read at session start (e.g., "this comp's eval is
  high-variance; treat single-LB outcomes with caution").
- `questions/` — open questions PI raised that have not been answered
  yet. Closed entries move to `concepts/` with the answer.

## Discipline

- Append, never overwrite.
- One topic per file. If a topic grows, split into dated successors.
- File naming: lowercase-kebab, ISO date prefix where applicable.
- Cross-link liberally. Markdown links are fine.
