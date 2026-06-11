# Open questions — 2026-06-12

- Which of the 14 concentration mechanisms (c42c9fc) are individually
  positive on a clean 3-opponent panel? (Bisection never ran; the
  stick-rate fixes halved waste vs the Producer in traces and might
  help vs mid-ladder snowballers even if the matchup stays lost.)
- Why does the full-compute Producer never freeze: does its value
  function have ANY board state it declines to attack, or does it
  always find a target? (If the latter, deterrence is structurally
  impossible and only out-producing it works.)
- Is the live ladder's Producer-style population also torch-based —
  i.e., do LIVE opponents degrade under kaggle's compute limits in
  ways our local Producer does not? (Live episode replays could show
  no-op-turn signatures.)
