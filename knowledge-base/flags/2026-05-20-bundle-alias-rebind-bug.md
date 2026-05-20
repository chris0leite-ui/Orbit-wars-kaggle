# 2026-05-20 — `scripts/bundle_agent.py` drops indent on alias rebinds

**Standing flag.** Any function-local `from X import Y as Z`
import in `agents/baseline/main.py` (or any agent module that's
bundled) produces a broken bundle: the rebind line `Z = Y` is
emitted at column 0 instead of matching the original line's indent,
causing `IndentationError` at module import.

Pre-existing on `origin/claude/audit-workflow-performance-btjeK`
since at least 2026-05-19. Hit fresh this session.

**Workaround used this session:** hoist the offending import to
module level. See `agents/baseline/main.py:96-98`.

**Real fix:** `scripts/bundle_agent.py::_strip_intra_package_imports`
should preserve the leading whitespace of the original import line
when emitting the rebind. ~1 line change.

**Why this is a flag, not just a friction:** the workaround is silent
and easy to forget. The next agent that adds a function-local
aliased import will hit the same bug and may not know the workaround
exists. Until the bundler is fixed, **all bundled agent code must
keep aliased imports at module level.**
