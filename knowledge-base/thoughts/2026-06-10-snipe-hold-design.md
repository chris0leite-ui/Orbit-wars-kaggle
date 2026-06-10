# Snipe-hold v1 design (toll-sniping, PI opening thesis part 2)

Gap: the planner is now-or-never. When the projection shows an opponent
capturing a planet at tick k_f (owner trajectory of a non-owned planet
transitions to an opponent id within H, or a background-predicted opp
launch cracks a neutral), the cheap play is arriving at k_f+1 for
survivor+1 ships (status.ships[p, k_f] is the survivor; capture_floor at
that arrival tick already reflects it). If our eta now < k_f+1 we cannot
"wait" — and worse, the regroup lane drains the idle ships away so
nothing is in range when the flip lands.

v1 (small, safe): snipe-hold = reserve idle ships with a dated snipe
appointment. Per turn, with the gate PRODUCER_PLUS_SNIPE_HOLD=1:
1. Detect flip events (k_f, survivor) for non-owned planets within H —
   trajectory scan + background captures (mirror predictions).
2. For each, find owned sources that can afford survivor+1+overhead from
   ships REMAINING after this turn's waves, and whose travel time allows
   launching at t = k_f+1−eta (eta computed at that future launch time;
   intercept_angle fixed-point).
3. Mark those sources reserved → filter THEIR regroup entries this turn
   (same post-planning entry-filter pattern as the response veto / convoy
   filter). Attack waves untouched in v1.
4. Re-planning each turn fires the actual snipe naturally when eta lines
   up (the normal candidate's floor at arrival is then survivor+1).

v2 (later): full deferred candidates competing in greedy (eta override
to k_f+1, strip-if-selected), which can also outbid attack waves.

Score gating: only reserve if the snipe's flow score (launches=[snipe at
eta k_f+1] vs background) clears the roi margin — reuse the veto's
scoring helper pattern. Unit tests: synthetic trajectory with a flip at
k=3, survivor 4; assert regroup from the reserved source is filtered and
others pass; assert no reservation when the flip is unsnipeable (too far
/ too poor) or scores below margin.
