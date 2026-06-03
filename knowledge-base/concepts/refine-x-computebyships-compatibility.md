# refine × compute-by-ships: compatibility analysis + next-session plan

**PI ask (2026-06-03):** our latest submission (`champ_computeByShips_on`, sub
53332500) scales the compute available to each planet with the number of ships
on that planet. Is it compatible with the validated teamwork refiner, and can
we get the best of both worlds next session?

## The two mechanisms

**compute-by-ships** — `BASELINE_COMPUTE_BY_SHIPS=1`
(`chooser_trajectory.py` / proposer). Per-source SOLO search effort scales with
the planet's garrison: *enumeration breadth 4→16 log-scaled by ship surplus*,
and *the horizon-K cap raised up to +50% for high-ship planets*. A big planet
considers more target/size candidates and looks further ahead. **Operates in
the champion's solo candidate enumeration.** Its own solo A/B was **7/16**
(weak/negative standalone) — submitted as a calibration probe.

**refine** — `BASELINE_CHOOSER=refine` (`chooser_refine.py`). Runs the champion
verbatim, then ADDS only oracle-positive two-source coalition atoms that don't
conflict with the champion's locks. **Operates as a post-champion augment.**
Validated 2026-06-03: **78.1% vs the adaptive-K champion** (50% parity), net +9
paired.

## Compatibility: likely YES, with two measurable interactions

**They touch different layers** — compute-by-ships shapes *solo* enumeration
breadth/depth per source; refine *adds teamwork* on top of whatever the champion
emits. refine wraps `choose_trajectory`, so a broader/deeper champion bundle is
simply the new base it augments. No structural conflict.

**Interaction 1 — possible cannibalization (the interesting one).**
compute-by-ships raises K (+50%) for high-ship planets ⇒ those planets can
solo-capture farther/more targets ⇒ fewer "neither source can solo-take this"
situations ⇒ potentially *fewer* coalition opportunities for refine. BUT the
coalition seam is driven by *contested / out-resourced* (low-ship) planets,
where compute-by-ships gives little extra breadth (low surplus → breadth stays
~4). So the natural hypothesis is **complementarity, not competition**:
compute-by-ships helps *high-ship* planets solo-expand; refine helps
*low-ship / contested* planets coordinate — different regimes of the same game.
This is testable (see plan step 5).

**Interaction 2 — wallclock (the real risk).** Both ADD compute.
compute-by-ships broadens enumeration AND deepens K for big planets (more
candidates × deeper rollouts); refine adds oracle rollouts for coalitions.
Stacked on adaptive-K (already deeper early horizon), the per-turn cost rises.
refine alone benched p95 633 / max 777 ms (table ON). Combined **must be
re-benched** — this is the gate that could force a budget-split or a breadth cap.

**Interaction 3 — shared K knob (code check needed).** compute-by-ships's
"+50% K cap", adaptive-K's step-decay, and the generator's `MAX_HORIZON` all
touch the horizon lever. Confirm they compose (no double-application; the
generator sees a sane K) before trusting any combined A/B.

## Next-session build sequence

1. **Code read (Rule 44):** trace where compute-by-ships and adaptive-K both
   touch `capture_horizon_k(step)` / enumeration breadth; confirm the +50% cap
   applies on top of the decayed K without double-counting, and that
   `generate_sync_coalitions` sees a coherent horizon.
2. **Combined bundle:** refine + adaptive-K + compute-by-ships + table
   (extend `scripts/_build_refine_adaptivek_bundle.sh` with
   `BASELINE_COMPUTE_BY_SHIPS=1`).
3. **Combined wallclock bench FIRST (Rule 2/30):** `fast.py bench`, must pass
   p95<800 / 0 over 1000ms. If it blows, tune `BASELINE_REFINE_CHAMP_PCT` /
   enumeration cap before any A/B.
4. **Same-seed paired A/B (clean_ab, n≥32):** combined vs (a) refine-alone and
   (b) the champion, on the adaptive-K config. Question: does compute-by-ships
   ADD to refine, or is it neutral/negative (its 7/16 solo result says it may
   not help on its own)?
5. **Cannibalization check:** run `scripts/refine_seam_contested.py` with
   `BASELINE_COMPUTE_BY_SHIPS=1` vs off; compare midgame raw coalition counts.
   If compute-by-ships materially lowers the coalition count, that quantifies
   the trade (more solo reach ↔ fewer teamwork needs).
6. **Decide:** ship combined only if combined > refine-alone at Wilson-lo;
   otherwise ship refine-alone (compute-by-ships is weak standalone and may not
   add once teamwork is present).

## Caveat

compute-by-ships was **7/16 standalone** — it is not an established win on its
own. The expected-value case for combining is the complementarity hypothesis
(Interaction 1), not a presumption that two levers add. Test it; don't assume.
