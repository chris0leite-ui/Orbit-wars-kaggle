# idle-trajectory audit — 2026-05-17

Idle ship-turns: my-planet ships attributed by min-distance to nearest non-our planet.
Launch ETA: ceil(launch flight distance / fleet speed).
Staging opportunity: long launch had own/neutral planet in ±15° corridor at <60% target distance.

## 52754310 (8 eps)

**Idle ship-turns by distance bucket:**

| bucket | range | ship-turns | % |
|---|---|---:|---:|
| frontier | 0-20 | 1190740 | 22.9% |
| mid | 20-35 | 1176635 | 22.6% |
| rear | 35-50 | 556799 | 10.7% |
| isolated | 50-999 | 2282129 | 43.8% |
| **TOTAL** | | **5206303** | 2127 steps |

**Launch ETA distribution** (980 launches, 45004 ships):

| bucket | range | launches | % | ships% |
|---|---|---:|---:|---:|
| short | 0-10 | 534 | 54.5% | 62.8% |
| medium | 11-20 | 247 | 25.2% | 21.6% |
| long | 21-30 | 68 | 6.9% | 6.3% |
| very_long | 31-999 | 46 | 4.7% | 2.2% |

**Staging opportunity (ETA > 20)**: 54 / 117 = 46.2%

## 52744856 (13 eps)

**Idle ship-turns by distance bucket:**

| bucket | range | ship-turns | % |
|---|---|---:|---:|
| frontier | 0-20 | 5232377 | 23.9% |
| mid | 20-35 | 5421076 | 24.8% |
| rear | 35-50 | 3585218 | 16.4% |
| isolated | 50-999 | 7646127 | 34.9% |
| **TOTAL** | | **21884798** | 3897 steps |

**Launch ETA distribution** (1608 launches, 153032 ships):

| bucket | range | launches | % | ships% |
|---|---|---:|---:|---:|
| short | 0-10 | 1108 | 68.9% | 85.2% |
| medium | 11-20 | 264 | 16.4% | 9.6% |
| long | 21-30 | 76 | 4.7% | 1.7% |
| very_long | 31-999 | 60 | 3.7% | 0.8% |

**Staging opportunity (ETA > 20)**: 58 / 142 = 40.8%

## 52710995 (92 eps)

**Idle ship-turns by distance bucket:**

| bucket | range | ship-turns | % |
|---|---|---:|---:|
| frontier | 0-20 | 3588395 | 20.0% |
| mid | 20-35 | 4203512 | 23.5% |
| rear | 35-50 | 2934601 | 16.4% |
| isolated | 50-999 | 7195314 | 40.1% |
| **TOTAL** | | **17921822** | 20579 steps |

**Launch ETA distribution** (8227 launches, 401957 ships):

| bucket | range | launches | % | ships% |
|---|---|---:|---:|---:|
| short | 0-10 | 5092 | 61.9% | 70.3% |
| medium | 11-20 | 1890 | 23.0% | 18.8% |
| long | 21-30 | 402 | 4.9% | 3.5% |
| very_long | 31-999 | 277 | 3.4% | 1.2% |

**Staging opportunity (ETA > 20)**: 290 / 702 = 41.3%

---

**Decision gates:**
- isolated+rear > 30% of ship-turns → spatial leaf fix high-leverage
- long+very_long > 25% of launches AND staging-rate > 50% → staging proposer high-leverage
