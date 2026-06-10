# 2026-06-10 — PI instinct: the opening is solvable

PI (verbatim, transcribed): "the opening game which is about capturing
planets before any opponent contact should be solvable in an optimal way
to compound production, be ready to counter the next opponent neutral
captures to our advantage and then use the slightest production advantage
to over roll the opponent."

Three-part thesis: (1) pre-contact neutral expansion is a deterministic,
fully-observable scheduling problem — solve it near-optimally for
compounding production; (2) contested neutrals are counter-play, not
races — predict the opponent's captures (symmetric maps make the mirror
near-exact pre-contact) and snipe the freshly-flipped planet one tick
later at survivor cost ("let them pay the toll"); (3) a small production
lead, played for tempo, compounds into a roll-over.

Supporting evidence already banked: material decision step p50 30-54 in
every corpus; our live losses are out-expansion losses (6 vs 7 planets at
step 40); the terminal-value mechanism failed for lack of exactly this
structure (no safe/contested distinction, no counter-readiness).

Structural gap discovered while assessing: the planner is now-or-never —
it cannot WAIT to time an arrival (e.g., launch in 3 turns to land one
tick after the opponent's capture). Delayed-launch candidates are the
smallest piece of the thesis. Top agents' low launch rate (0.26) reads
as timing, not abstaining.

Agreed order: (a) offline beam-search optimum on the seed panel, score
our agent's openings against it (size the prize first); (b) delayed-
launch snipe candidates; (c) full opening scheduler behind a default-OFF
gate with a worst-case-rush safety envelope.
