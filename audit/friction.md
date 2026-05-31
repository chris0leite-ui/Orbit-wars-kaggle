# friction.md — current friction summary (concise)

> Rolling log of operational frictions. Newest at top. Full detail in
> `audit/<date>-*.md`. Promotion candidates flow to
> `.claude/skills/kaggle-comp/improvements.md` → CLAUDE.md rules.

> See also `audit/INDEX.md` for the audit-doc map.

> NOTE (2026-05-31): this file was found CORRUPTED at HEAD — lines 11–46
> were a prior session's stray thinking-text (no real entries survived).
> Rewritten clean below. Pre-corruption entries, if needed, live deeper in
> `git log -p audit/friction.md`. Cleanup of that history is a separate task.

## 2026-05-31 — sync-coalition session (champion-strategy-rules-00JzI)

### F1 — Kaggle-has-no-env-vars bundle-baking gotcha (SYSTEMIC; promotion candidate)
Our agents read all config from `os.environ` (`BASELINE_JOINT_SYNC`,
`BASELINE_PV_ETA`, `BASELINE_ORBITAL_SAFETY`, `BASELINE_LAUNCH_RULES`, …),
each defaulting OFF; local A/B drivers set them in the shell. **Kaggle sets
none** → a vanilla bundle runs everything OFF, i.e. a near-baseline agent,
NOT the one we measured. The local "focal" bundles
(`baseline_joint_sync_focal.py`, `baseline_joint_sync_hold_focal.py`) do not
bake config and were never valid submissions. Nearly submitted an inert
agent. **Fix:** prepend an `os.environ.setdefault(...)` header with the full
tested env block ABOVE the first inlined module (modules read constants at
import), then verify with a **clean-env smoke** (scrub all `BASELINE_*`/`PV_*`,
import the bundle — register in `sys.modules` first or `@dataclass` resolution
fails — assert baked values, run one full game). The champion bundle already
does this (`_lr_os.environ.setdefault` header). → **proposed Rule 49** (see
postmortem).

### F2 — bundler internal parity gate broken in this container (RECURRING)
`scripts/bundle_agent.py`'s parity gate loads source `main.py`, whose import
chain pulls `kaggle_environments`, which shadows our top-level `agents`
package with `kaggle_environments.envs.lux_ai_s3.agents` → `ImportError:
attempted relative import with no known parent package`. Already noted in the
`baseline_redeploy_gangup` submission. **Workaround:** bundle with
`--skip-parity-gate`, verify via structural `tests/test_bundle.py` (compiles /
callable / one-future / alias-rebind) + the clean-env play smoke. The bundle
artifact itself is fine (imports are stripped); only the gate's source-load
breaks.

### F3 — foreground timeouts kill heavy jobs on the slow/contended box
Full games run ~70–730s each here; `timeout 300–600` foreground runs got
SIGTERM'd (exit 143) repeatedly with no output (clean_ab buffers per-arm).
**Fix:** run heavy compute in the background with generous timeouts (≥1700s
for an A/B arm); add per-game progress + flush so partial runs survive.

### F4 — friction.md was corrupted at HEAD
A prior session committed stray thinking-text over the real entries. Found
during this wrap-up; rewritten clean. Watch for the same failure mode in
other rolling docs.
