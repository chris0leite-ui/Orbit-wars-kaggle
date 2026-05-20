# Postmortem — 2026-05-20 phase7-btjek-chain-bonus

## What went wrong

### Bad decisions (given priors at decision-time)

1. **Aggregate-only iteration through Phase 7 → 8 without close-read.**
   After Phase 7 came back 7/16 INCONCLUSIVE-negative I went straight
   to Phase 8 (full port + chooser bypass). Both decisions were sized
   on aggregate winrate alone. PI had to prompt "Have you also looked
   at one game closely?" before I instrumented `scripts/inspect_chain_game.py`
   and discovered the load-bearing finding: 31 chain launches fired,
   0 relay completions — the bonus credits leg-1 with leg-2 value
   that never materialises.

   Given the priors at decision-time (Phase 7 INCONCLUSIVE-negative,
   inspect not yet built), what I SHOULD have done was build the
   inspect FIRST. The post-Phase-7 question "why didn't the bonus
   lift winrate" has two plausible answers (mechanism not firing
   vs mechanism firing but underweighted by chooser), and the close-
   read distinguishes them in 1 game vs 16 game-equivalents of A/B
   noise.

   Decision-quality category: **default-workflow blind-spot**. The
   "A/B → iterate" loop is in my muscle memory; "watch a game" is
   not. This is a generalisable failure mode worth promoting (see
   step 3).

2. **Phase 9 design assumed the bonus was directionally correct.**
   Inspect (Phase 8) showed 0 relays. I interpreted this as
   "mechanism not delivering its promise, force it via commitment"
   rather than "the bonus credits a future the agent doesn't
   actually want to commit to." The two read very differently:
   (i) implies α (force the relay), (ii) implies β (drop the bonus).
   Phase 9 going to 1/16 strongly suggests (ii) was the right read.

   Mitigating: I DID present α/β/γ to PI, and PI picked α. So this
   was a PI-ratified decision. The lesson is calibration-only:
   "inspect-reveals-mechanism-doesn't-fire" should bias toward
   "the bonus is wrong" over "we need to add more enforcement."

### PI overrides (calibration points)

- 1× this session — PI prompted "Have you also looked at one game
  closely?" mid-iteration. This is the only PI override. Calibration
  signal: my default workflow understates close-read by ≥1 per
  mechanism family.

### Rule-bypass failures

- **None this session.** Rule 1 (PI sign-off on submissions) held —
  I did not submit any of Phases 7-9 despite proximity to a 5/day
  budget. Rule 37 (consecutive-falsification cap) fired correctly
  after Phase 9; the recommendation to STOP came from me before
  PI confirmation.

### Rule-gap failures

- **No standing rule about close-reading before scaling A/B.**
  The "watch a game" step would have flagged "0 relays" after
  Phase 7 in <2 min wallclock and changed the entire Phase 8/9
  trajectory. Promotion candidate (see step 3).

## Frictions logged this session

Cross-linked to `audit/friction.md` 2026-05-20 (claude/phase7-btjek-chain-bonus):

- `tag: aggregate-without-close-read` — drove Phase 7 → 8 → 9 on
  winrate aggregates only. PI prompted close-read; that single game
  revealed 0 relay completions.
- `tag: rule37-axis-exhaustion-chain-bonus` — three same-axis
  variants failed (7/16 → 0 relays → 1/16). Rule fired correctly.
- `tag: bundler-alias-rebind-drops-indent` — pre-existing on btjeK;
  `scripts/bundle_agent.py` drops the indent of alias rebinds for
  function-local imports. Hoisted the import to module level as
  workaround. Real fix is in the bundler.
- `tag: pre-existing-wallclock-test-fails-on-btjeK` — recurrence;
  `test_baseline_wallclock_under_budget_favor` fails 539ms vs 300ms
  on clean btjeK checkout. Not a Phase 7-9 regression.

## Promotion candidates (PI ratified: PENDING)

### [ ] CLAUDE.md `## Operating rules — concise` — add Rule 42

**Tag:** `aggregate-without-close-read` (this session, 1 PI override,
cost: ~4h compute on Phases 8 & 9 that close-read after Phase 7
would have killed).

**Where to insert:** new rule below Rule 41 (or wherever the next
ordinal is open).

**What to add:**
```
42. **Close-read before scaling.** Before running any A/B at n > 16
    on a new mechanism, run a single-game close-read of the focal
    agent (1 game, focal vs control on a panel seed, dumped
    per-turn: mechanism-firing count, mechanism-emission count,
    downstream outcome). The close-read distinguishes "mechanism
    not firing" from "mechanism firing but underweighted by
    chooser" in <2 min wallclock; pure A/B at n=16 cannot.
    Origin: 2026-05-20 chain-bonus session — drove Phase 7 → 8 → 9
    on aggregate winrate only; PI prompted close-read mid-iteration,
    which immediately revealed 0/31 relay completions and would
    have killed Phases 8 + 9 had it run after Phase 7.
```

**Why:** matches the cost gate (≥1h compute waste this session AND
≥1 PI override). Pattern is generalisable beyond chain-bonus — any
new mechanism family on the proposer/chooser stack benefits.

### [ ] `scripts/bundle_agent.py` — preserve indent on alias rebinds

**Tag:** `bundler-alias-rebind-drops-indent` (pre-existing on btjeK
since at least 2026-05-19; first observed this session).

**Where to insert:** `_strip_intra_package_imports` (around line 219+
based on the file structure I read).

**What to add:** when emitting `{asname} = {original}` after
commenting out a `from X import Y as Z` line, prepend the original
line's leading whitespace. Single-line fix on the existing rebind
emission.

**Why:** any function-local import with an alias produces broken
bundles. Real cost: this session lost ~10 min to the diagnosis +
workaround (hoist to module level). Other agents may hit this and
not know about the workaround.

## PI additions (from step 4)

(awaiting PI input — see step 4 question below)

## Framework version at session-end

- Commit SHA: 4f6cd5a (will be a wrap-up commit on top by step 6)
- Active rules: 1..41 per CLAUDE.md (Rule 41 pending PI ratification
  in prior session — kept as candidate)
- Loaded skills this session: postmortem (this), kaggle-comp
  (implicit via project context).

## Calibration snapshot

No submission this session → no row added to
`state/calibration-ladder.md`.

## Key artifacts produced

- `agents/baseline/proposer.py` — chain-bonus helper + 9-tuple
  (Phases 7/8/9 all touched this file).
- `agents/baseline/chooser_trajectory.py` — bypass logic (Phase 8) +
  leg-2 relay commit (Phase 9).
- `agents/baseline/chooser.py`, `chooser_roi.py` — unpacking
  updates for 9-tuple compatibility.
- `agents/baseline/main.py` — auto-engage ledger path when
  BASELINE_CHAIN_BONUS=1 (Phase 9).
- `agents/_chain_on/`, `agents/_chain_off/` — A/B wrappers.
- `tests/fixtures/replays/claws_77164175_step223.json` — replay
  fixture (port from EpMVP branch).
- `tests/test_baseline_replay_regression.py` — new file with the
  Claws step-223 chain-bonus regression test.
- `tests/test_baseline_proposer.py`, `test_chooser_trajectory.py`,
  `test_baseline_chooser.py` — updated for new tuple shape + bypass.
- `scripts/inspect_chain_game.py` — one-game close-read (load-
  bearing artifact; **keep this for future mechanism work**).

## What the branch ships

`claude/phase7-btjek-chain-bonus` HEAD = `4f6cd5a Phase 9: force the
chain relay via leg-2 commit`. Per Rule 37 axis exhaustion, **do not
ship anything from this branch.** btjeK at chain-off (commit
0b83734, the branch base) is strictly better than any of the three
Phase variants.

Next session should pivot the merge-target to a different mechanism
family. The close-read script + the regression fixture are the
useful permanent artifacts.
