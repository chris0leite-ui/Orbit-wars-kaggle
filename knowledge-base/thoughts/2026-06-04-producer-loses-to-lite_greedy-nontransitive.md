# Producer loses to lite_greedy — the opponent ranking is non-transitive

_2026-06-04, branch JzIAr. Surfaced while building the Producer-lite opponent
model (the cheap primary fidelity gate). Plain-English record for the PI._

## The finding

The vendored public **Producer** torch agent — the one that beats our champion
line ~60% — **loses 0 of 16 games to `lite_greedy`**, the cheap, weak opponent
model our lookahead currently rolls opponents forward with.

So the skill ranking is not a clean ladder. At least on the seeds measured:

- Producer **beats** our hoarding champion (~60%, the original observation).
- `lite_greedy` **beats** Producer (0/16 for Producer).
- (Our champion vs `lite_greedy` is the third leg — not re-measured this
  session, but the champion is a strong agent that should be competitive.)

This is rock-paper-scissors, not A > B > C.

## Why (traced, not guessed)

Producer's signature is **all-in aggression**. Its `safe_drain` ships out the
maximum a planet can spare while still being held in a do-nothing projection —
which, for a healthy growing planet, is essentially the whole garrison. On turn
zero it empties its home planet to grab a distant target.

- Against our **champion** (hoards ships, waits, K=10 launch discipline), that
  early tempo and territory grab wins.
- Against **`lite_greedy`** (keeps every planet affordable to defend, only
  commits to captures it can actually complete), Producer's emptied home planets
  get picked off and it bleeds out. Discipline beats over-extension.

## Why it matters

1. **The cheap primary fidelity gate's premise is false.** The Producer-lite
   plan made "producer_lite must beat `lite_greedy` (Wilson-lo ≥ 0.65)" the
   first, fastest faithfulness check, reasoning "full Producer trounces
   `lite_greedy`, so a faithful port must too." The oracle itself fails that
   bar. Beating `lite_greedy` therefore measures *over-aggression vs that one
   opponent*, NOT faithfulness to Producer. The gate that actually measures
   fidelity is move-agreement + winrate-**transfer** vs our champion.

2. **`lite_greedy` may be underrated as a rollout opponent — and as a strategy
   cue.** It beats the strong public agent. Its disciplined "only launch
   affordable captures, keep planets defensible" rule is exactly the behaviour
   that punishes over-extension. Worth remembering when we reason about why our
   own over-expansion experiments regressed.

3. **Single observation, big caveat: n=16.** The band on 0/16 is roughly
   0–19%. A larger run + the champion-leg is needed before treating the ranking
   as settled. But 0/16 is already strong directional evidence.

## Open question for the PI

If the threat that beats us (Producer) is itself beaten by a simple disciplined
expander, is the right opponent model to anticipate (a) Producer-like
aggression, (b) `lite_greedy`-like discipline, or (c) a mix? The whole point of
the opponent model is to calibrate our defense to the threat that actually beats
us — and the live field is neither pure Producer nor pure `lite_greedy`.

---

## ⛔ CORRECTION (same day) — the above is WRONG. It was a harness bug.

The PI flagged "this sounds like a bug." It was. The vendored
`agents/producer/producer_agent.py` shim referenced `__file__` at module level,
which is **undefined when kaggle_environments execs an agent as raw source**
(`exec(code, {})`, no `__file__`). So full Producer **failed to load and idled
every turn** under `env.run` / `scripts/clean_ab` — it never played. The 0/16
was Producer doing nothing, not Producer losing.

The earlier "full Producer step-0 move" spot-check loaded the agent via
`importlib` (which DOES define `__file__`), so it looked fine — masking the bug.
Classic Rule 38: I verified producer_lite's own wrapper ran in the harness but
**assumed the oracle ran too**. It didn't.

After fixing the shim (recover the dir from `sys.path`, since the loader appends
the agent's directory before exec), full Producer plays correctly and **beats
`lite_greedy` decisively** (eliminates it by ~step 80 on seed 1). So:

- The opponent ranking is **NOT** the non-transitive cycle claimed above.
- The cheap primary fidelity gate's premise (**full Producer trounces
  `lite_greedy`**) is **restored and valid**.
- All earlier Producer-vs-X numbers from clean_ab are void; re-run with the
  fixed shim.

**Lesson:** when an oracle's result is shocking, verify the oracle is actually
*running* in the exact harness before interpreting — load-by-path (`env.run`)
and load-by-importlib are not the same; only the latter defines `__file__`.
