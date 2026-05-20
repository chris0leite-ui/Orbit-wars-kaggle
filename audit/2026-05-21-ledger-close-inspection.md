# 2026-05-21 — Ledger Close-Inspection (5 games)

**Branch**: `claude/audit-workflow-performance-btjeK`
**Predecessor**: `audit/2026-05-20-ledger-design.md`
**Plan**: `/root/.claude/plans/so-now-research-and-zany-widget.md` (Phase A)

## Verdict

**PASS — proceed to Phase B (A/B wrappers + h2h).** No anomalies that
warrant stopping. One minor observation (tiny-launch tail at 3.4%)
noted for possible future mitigation; does NOT block validation.

## Per-game telemetry summary

```
episode    diverge  emit_rate    drops_by_reason                    dup_src  tiny_launches
77140674   t=13     17/21 (81%)  {tgt_now_ours: 3, planet_missing: 1}    0          0
77133549   t=33     12/12 (100%) {}                                       0          4
77150441   t=10     11/12 (92%)  {tgt_now_ours: 1}                        0          2
77137480   t=5      18/24 (75%)  {tgt_now_ours: 6}                        0          4
77158235   t=14     15/19 (79%)  {tgt_now_ours: 3, planet_missing: 1}    0          4
```

## What we checked

For each game, walked the existing
`audit/whatif/52827111/<eid>/ledger_soft.json` and confirmed:

1. **Emit success rate** — % of due (wait_remaining→0) commits that
   actually emit vs drop. Range 75-100%, median 81%. Healthy.
2. **Drop-reason composition** — every drop is `tgt_now_ours` (we
   captured the target via another path; commit moot) or
   `planet_missing` (comet expired). Zero drops from `src_lost`,
   `src_empty`, `aim_failed`, or `size_zero` — the ledger isn't
   dropping commits because the world drifted unmanageably; it only
   drops when the goal is already achieved.
3. **Duplicate-src guard** — count of turns emitting more than one
   launch from the same source. Zero across all 5 games. The
   joint-candidate `reserved_srcs` fix from the prior cycle is
   intact.
4. **Tiny-launch tail** — launches with <5 ships (bounce risk if
   target needs more). 14 across all 5 games out of 412 total launches
   = 3.4%. These come from soft-mode commits whose src lost ships to
   chooser fire-now during the wait; at emit time we send
   `min(planned, available)`. Possible future mitigation: drop the
   commit if `available < 5` instead of firing tiny. NOT blocking.
5. **Action stream coherence pre-divergence** — early-game actions
   match the baseline closely (or differ in sensible ways: fires
   earlier, slightly different target). No bizarre choices.
6. **Median launch sizes** — 25-53 ships across games. Max 133-444.
   The tail head (smaller-than-planned emits) is the only mild issue.

## Per-game observations

### 77140674 — sary 2P loss (primary case)

Diverges at turn 13. Emit rate 81%. The 3 `tgt_now_ours` drops
correspond to turns where the chooser captured the planned target via
a different (smaller, fire-now) emit before the wait expired —
exactly the soft-mode design intent. The 1 `planet_missing` is a
comet that expired. No anomalies.

### 77133549 — monnu 2P, largest idle drop (-17pp)

Diverges at turn 33 (late — the ledger doesn't immediately change
opening behaviour; it shines in the mid-game where under-emission
hurt). 100% emit success. 4 tiny launches at turns 74-76 from src=20
(2-3 ships each); investigate later if mitigation lands. No
duplicates, no failures.

### 77150441 — 4P loss, +213% launches

Diverges at turn 10. 92% emit. Max pending entries = 3 (manageable).
Action stream shows the ledger firing from src=11 at regular intervals
(t=1, 5, 10, 12, 16, 19) where baseline fired sporadically — exactly
the consistent-firing pattern we wanted. Joint-candidate interaction
clean.

### 77137480 — RL 2P win, +3pp idle anomaly

Diverges at turn 5 (earliest of the 5). 75% emit success — the lowest,
all 6 drops are `tgt_now_ours`. Diagnosis: in this game we were
already winning, the chooser captured most planned targets via
fire-now before the wait expired. The ledger's commits became no-ops
(correctly dropped). The "+3pp idle" is because the chooser holds
back when it sees the ledger has a pending entry for a src (used_srcs
in the chooser's emit loop blocks duplicate-src). This is a benign
cost in already-winning games; the +6 final planets shows the agent
still plays better overall.

### 77158235 — 4P modest improvement

Diverges at turn 14. 79% emit. Max pending 4. 5 turns with 3+ launches
(good firing density). 4P joint enumeration honours `reserved_srcs`
correctly (zero duplicate-src turns). +22% launches over baseline.

## Notes for execution

- **Tiny-launch mitigation** — optional follow-up: in `_tick_ledger`,
  if `ships < 5` after the `min(ships_planned, available)` clamp,
  drop the commit instead of firing. Estimated impact: ~3% of emits
  redirected to nothing (the chooser may then redirect those ships
  to a different target next turn). Worth A/B'ing separately if the
  primary ledger gates pass with comfortable margin.

- **Wait-target convergence** — sometimes the chooser commits N entries
  for similar targets across turns. Max pending was 4 (77158235),
  reasonable. No commit-bloat observed.

- **4P joint interaction** — works as designed. The previous bug
  (joint candidates ignoring `reserved_srcs`) was fixed in the prior
  cycle; this audit re-confirms across 2 different 4P games.

## Proceed signal

Phase A is PASS. Move to Phase B (build A/B wrapper agents).
