"""v7.7 — enemy-target multiplier in the snipe scorer (H10).

The diagnostic + top-10 replay analysis
(`knowledge-base/concepts/top-performer-strategies.md` §H10) shows
top-10 picks enemy targets at 32 % vs midpack 14 % — a ×2.3 gap.
The current v7_0 snipe scorer treats enemy and neutral targets
equivalently in the priority term (both get a 1.0 multiplier).

v7_7 bakes `lib.missions.snipe.ENEMY_MULTIPLIER = 1.3` into the bundle
by setting that constant in `lib/missions/snipe.py` at bundle time and
bundling v7_7 against the modified lib. v7_0_drop_one's bundle was
built when the constant was 1.0, so the A/B isolates the multiplier
change.

Same drop-one chooser, same K=10 rollout, same `composite_capture_value`
head as v7_4. Module-mutation at agent-call time was tried first
(commit aborted on parity-gate fail) — the bundler's text inlining
gives the bundle its own copy of every lib constant, so the runtime
patch only affects the real lib namespace, not the inlined copy.
"""

from __future__ import annotations

from lib.v7_search import choose
from lib.value_heads import composite_capture_value


def agent(obs, configuration=None):
    return choose(
        obs, configuration,
        enumerator_mode="drop_one",
        K=10,
        wallclock_ms=700.0,
        value_fn=composite_capture_value,
    )
