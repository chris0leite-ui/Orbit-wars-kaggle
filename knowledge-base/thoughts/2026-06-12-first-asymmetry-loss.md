# PI observation — first-asymmetry loss vs producer-type (2026-06-12)

PI, from the replay viewer (seed 1493019744, 207 steps, opponent
RYOTAaaaaa rated 1160, we 1214, rating hit -69):

"This is the first non-symmetry in a game against a producer type
strategy which we lose."

I.e. the game stays near-symmetric until the first structural divergence
(~step 45 in the screenshot: both sides hold mirrored +4/+3 planets;
fleets 38+101 orange vs 135 blue in flight), and the divergence resolves
against us. Producer-type opponent = the mirror-adjacent strategy class
our own engine descends from — these should be our most winnable games.

## Diagnosis (same morning, decision-trace + geometry)

The 135-ship wave (their planet 18 -> our planet 19) was visible from
t=39; planet 19 fell t=48 at garrison 96. Geometry check: from t=39
onward NO feasible reinforcement could arrive in time (the one saving
combination, 35 ships from planet 10, exceeded planet 10's garrison of
21). The planner's silence at t=44 was locally rational — the planet was
already doomed. The actual failures:

1. PRE-POSITIONING BLINDNESS (the decisive one): the opponent visibly
   massed 135+ ships on planet 18 over ~30 ticks. Our garrison-value
   term (live in this sub!) never registered a deficit at planet 19
   because the threat is weighted by SOURCE_SAFETY=0.5 — half of 135 is
   ~67, below 19's garrison-plus-growth. The half-weight was a
   multi-counting guard, but the rival-cap (one credited target per
   rival) now handles that structurally; the magnitude should be FULL.
   In 2P especially: one opponent, their reserve goes one place.
2. DONATED GARRISON: 96 ships sat on the doomed planet and transferred
   to the opponent. No evacuation concept in the engine (ledger branch
   had one). 24-strength swing became ~120 effective.
3. The counter-grab (101 ships -> their 28-garrison planet 16) was the
   only positive-scoring outlet; their massing source 18 itself was
   floor-blocked (reactive floor on their massive routable support).

Proposed minimal fix (one mechanism): garrison-value deficit uses FULL
routable threat (weight 1.0 internally, R-cap unchanged, source-safety
cap stays at its own 0.5). Rule 38 verification: replay THIS game's
t=25-35 positions through the planner and confirm reinforcement toward
planet 19 fires while parcels can still arrive; then panel + mirror.
Evacuation = separate future mechanism, logged.
