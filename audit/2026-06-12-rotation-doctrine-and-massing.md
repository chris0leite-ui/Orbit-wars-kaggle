# The mathematician + battlefield read (PI directive, 2026-06-12)

## Q: is asymmetric counterattack + counterclockwise commitment dominant?

### The physics
Planets co-rotate (angular velocity ~0.035-0.046, radius ~36 => tangential
speed ~1.25/tick) while fleets fly in inertial space at speed 2-4.2
(grows with fleet size). Intercept solving for a 90-degree separation on
the RYOTA map: striking the CLOCKWISE-BEHIND neighbor (rotating toward
you, under counterclockwise system rotation) lands in ~10 ticks; the
same-distance strike at the counterclockwise-ahead neighbor (running
away) takes ~15. A ~50% tempo edge, worth ~20 garrison-growth ticks
denied to the defender.

### The empirics (8,459 opponent + 4,004 our attack waves, all live games)
- Everyone already attacks ~2:1 in the favored direction (us 2,720 vs
  1,284; field 5,768 vs 2,691). The advantage is implicitly priced via
  intercept etas in candidate scoring — ours and apparently the field's.
- Capture rates are FLAT across direction (74% both ways for us; 57%
  both ways for the field): direction governs which targets are
  feasible, not the success of chosen strikes.
- Verdict on the attack side: NOT a new dominant doctrine — the engine
  already exploits it wherever it pays. Forcing more would be
  restriction, not modeling (Rule 40).

### Where direction WAS a blindspot: defense
All balance-of-force margins (threat/help/floors-margin in the new
mechanisms) used static distances — they cannot see that enemy mass
arrives sooner from the favored direction, or that OUR planets rotate
into/out of enemy reach. Fixed tonight (gated
PRODUCER_PLUS_ROTATION_AWARE_MARGINS): reach tests use the per-k
intercept slices cross_dist[k].

### The deeper loss mechanism: relay concentration beats snapshots
RYOTA's kill: backline 84-stack -> staging planet -> merged 135-stack ->
strike. Because fleet speed GROWS with mass, relayed concentration flies
faster than its components — and a threat model reading CURRENT garrisons
goes positive exactly when defense becomes infeasible. Fixed
(concentration_speed in the garrison-value deficit): reach is priced at
the speed of the enemy's combined strength = what they CAN concentrate.
Rule 38 chain on the RYOTA replay: old stack idle through the whole
pre-positioning window -> new stack fires 64 ships to the frontier at
t=32, 16 ticks before the historical strike (allocation between the two
tied frontier planets remains enemy-choice-dependent; games judge that).

### Strategic synthesis for the PI
The dominant strategy is not "always attack counterclockwise". It is:
attack wherever intercept math says (already done); DEFEND as if the
enemy strikes in HIS favored direction at HIS best concentration speed
(built tonight). Asymmetric counterattack remains situational — we
counter-attack 65% of inbound threats already, and the RYOTA game shows
the missing move was pre-positioned defense, not a sharper counter.

## Battery + bisect outcome (same day, late) — the calibration frontier

The full fix chain ROUTED the local battery (4P final share 14.0% vs the
night-1 build's 36.4%; 2P mirror 2/12, -54%@250): full-weight
max-concentration threat + rotation-aware floors = over-insurance, the
turtle again. Bisect isolated the components:

- KEPT (mirror 7/12, +20.2%@250 — identical to night-1 strength):
  deficit-shortlist appender + attractiveness tie-break + concentration-
  speed threat at the HALF weight (expected-value massing estimate).
  This is the current vetorf4p_sync_garval variant config.
- REFUTED locally: GARRISON_VALUE_THREAT_W=1.0 and
  ROTATION_AWARE_MARGINS=1 (both stay in the engine, gated default-off).

But the half-weight config does NOT re-fire the RYOTA pre-positioning
(deficit never crosses). The frontier, stated plainly:
  half-weight threat  -> mirror-safe, blind to the RYOTA relay-massing kill;
  full-weight threat  -> RYOTA fixed, mirror-routed.
The mirror is not the field (doctrine); only a live slot can price the
middle. Candidate resolutions for a future session: per-player-count
threat weight (2P opponent concentrates freely -> higher weight);
adaptive weight from observed enemy massing; or a massing DETECTOR
(garrison growth rate at enemy forward planets) gating the full weight.

Live state unchanged: 53588922 (night-1 garval) healthy at 61.5%
overall / 85.7% 2P (n=13, warming). No new submit from this thread.
