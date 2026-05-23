# Flag: random-elim gate is a candidate Rule, not yet a Rule

**Filed**: 2026-05-23 (claude/session-EqJuT)
**Severity**: medium (PI-ratified; needs CLAUDE.md promotion next audit)

PI ratified `random-elim-gate-mandatory` as a candidate Rule-48 this
session, but it is currently filed only in
`.claude/skills/kaggle-comp/improvements.md` as a pending entry. It
is NOT yet in `CLAUDE.md ## Operating rules`, so it does not bind
agent behaviour automatically — every submission decision needs the
PI or a vigilant agent to apply it manually until the next audit
pass promotes it.

**Action item for the next audit-workflow session:** move the entry
from improvements.md to CLAUDE.md as Rule 48, sub-clause of Rule 12
(submission discipline) and Rule 43 (multi-opponent panel). Move-
out marker should reference this flag for traceability.

**Why this flag exists**: the gate caught 2 latent bugs in
lagrange_simple that single-game smoke missed. Until the rule binds,
the next agent that builds a new candidate may skip the gate and
ship those same bug classes again. Risk: rolling-pair slot burn on
a regression that 3 minutes of n=16 testing would have caught.
