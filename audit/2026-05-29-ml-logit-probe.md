# Reframe A — Step-0 diagnostic probe report

Trace file: `/tmp/probe_combined.jsonl`
Candidates analysed (with valid P): **28661**
Turns analysed: **499**, median candidates/turn: 58.0

## Per-turn statistics (medians across turns)

- σ(delta): **70.4668**
- σ(P_success): **0.2628**  (p25 0.2364, p75 0.2923)
- σ(logit P): **1.5578**
- median(P_success): **0.7908**
- |Spearman ρ(delta, logit P)|: **0.1315**  (p25 0.0658, p75 0.2249)

## Gates

- `sigma_p_>=_0.05`: **PASS**
- `abs_rho_<_0.85`: **PASS**
- `median_p_in_[0.2, 0.8]`: **PASS**

**Verdict: PASS — proceed to Reframe A**

## P_success distribution

| bin | count | frac |
|---|---:|---:|
| [0.0,0.1) | 1201 | 0.0419 |
| [0.1,0.2) | 1461 | 0.0510 |
| [0.2,0.3) | 1324 | 0.0462 |
| [0.3,0.4) | 1609 | 0.0561 |
| [0.4,0.5) | 1592 | 0.0555 |
| [0.5,0.6) | 2157 | 0.0753 |
| [0.6,0.7) | 2574 | 0.0898 |
| [0.7,0.8) | 3092 | 0.1079 |
| [0.8,0.9) | 5236 | 0.1827 |
| [0.9,1.0) | 8415 | 0.2936 |

Tail (P<0.1 or P>0.9) fraction: 0.3355

## Suggested λ sweep

Targeting ML-logit magnitudes {0.1, 0.3, 1.0} × σ(delta):
- λ candidates: **[4.523, 13.57, 45.234]**

Sweep these via `BASELINE_ML_LAMBDA=<λ>` in Step 7. Centered-logit form: `λ * (logit(P) - logit(0.5))`.
