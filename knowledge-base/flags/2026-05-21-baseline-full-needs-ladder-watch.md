# 2026-05-21 — flag: baseline_full needs 12-24h ladder watch

Sub **52893236** `baseline_full` shipped on n=4 evidence (2/4 = 50%,
Wilson [0.150, 0.850]). Replaced `baseline_joint_aggr` (sub 52874528,
μ ≈ 1135) in the rolling pair. Risk: if true μ < 1000, we lost
~135 μ floor in the rolling pair until next submit.

**Action:** monitor sub 52893236 μ over next 12–24h. If below 1000
after ~30 ladder games, propose a rollback strategy (re-submit
consolidated to evict baseline_full, or push a refined variant).
