# R3 technical report — protected (KrunTechBench v1 with restored labels + R3 extension)

## raw: KrunTechBench v1 part (choice items; macro over the 10 tracks)

| model | macro | api | backend | cloud | database | devops | distributed | mlinfra | security | software_repair | state_action | hard neg. | flips | unseen comb. | seen comb. | unseen tech. | large k | unseen family | gold>hardest | margin | OOD FA | involved (noul) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 30.9 | 26.3 | 26.4 | 25.2 | 28.3 | 4.3 | 41.3 | 40.7 | 47.9 | 59.6 | 9.0 | 39.2 | 18.2 | 52.1 | 39.5 | 39.0 | 24.2 | 12.7 | 66.9 | +1.12 | 0.4 | 60.8 |
| r2l4na_s1 | 61.7 | 55.3 | 73.0 | 46.5 | 69.7 | 87.0 | 94.8 | 47.4 | 50.3 | 74.9 | 18.1 | 65.7 | 63.0 | 59.9 | 99.8 | 76.0 | 51.1 | 9.8 | 79.4 | +4.21 | 0.2 | 49.2 |
| r3r_s1 | 62.9 | 60.4 | 54.4 | 49.0 | 68.2 | 90.7 | 94.7 | 57.1 | 60.0 | 74.1 | 20.1 | 65.8 | 67.4 | 58.3 | 100.0 | 75.5 | 51.4 | 8.8 | 79.5 | +4.08 | 0.2 | 83.3 |
| r3r_s2 | 61.6 | 55.4 | 63.9 | 45.2 | 66.7 | 89.4 | 94.3 | 54.7 | 51.5 | 75.6 | 19.3 | 65.5 | 63.3 | 59.8 | 100.0 | 74.5 | 50.4 | 8.1 | 79.5 | +4.19 | 0.0 | 80.8 |

## raw: unseen requirement combinations and counterfactual families

| model | unseen 1 req | unseen 2 req | unseen 3 req | catalog (unseen sets) | table families (unseen sets) | v1 flips: both / stuck | table families: both / stuck | capability families: both / stuck |
|---|---:|---:|---:|---:|---:|---|---|---|
| v1 | 46.6 | 55.1 | 55.8 | 31.7 | 40.0 | 2.8 / 65.1 (n=286) | 9.5 / 51.0 (n=472) | 1.3 / 95.0 (n=1171) |
| r2l4na_s1 | 37.1 | 70.5 | 97.7 | 27.4 | 41.4 | 46.5 / 20.8 (n=286) | 10.6 / 33.3 (n=472) | 1.7 / 91.9 (n=1171) |
| r3r_s1 | 40.3 | 65.8 | 100.0 | 55.6 | 37.4 | 49.7 / 20.1 (n=286) | 16.5 / 32.7 (n=472) | 3.8 / 85.4 (n=1171) |
| r3r_s2 | 38.9 | 69.7 | 93.0 | 54.7 | 38.9 | 44.4 / 21.7 (n=286) | 15.7 / 32.9 (n=472) | 3.8 / 85.0 (n=1171) |

## raw: capability tracks (R3 extension; all question types)

| model | macro | numeric | temporal | multi_hop | long_policy | tradeoff | mixed | seen scenario | unseen scenario | choice | true/false | ordinal | 2 hops | 3 hops | 4 hops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 37.2 | 40.1 | 34.8 | 33.4 | 38.3 | 33.5 | 43.1 | 37.3 | 36.4 | 27.5 | 54.5 | 42.0 | 33.3 | 34.4 | 28.3 |
| r2l4na_s1 | 37.0 | 39.2 | 34.1 | 33.1 | 39.3 | 32.7 | 43.8 | 37.2 | 36.2 | 27.7 | 53.5 | 42.0 | 29.8 | 36.5 | 26.7 |
| r3r_s1 | 41.2 | 43.2 | 42.0 | 32.8 | 45.9 | 37.7 | 45.7 | 42.5 | 39.2 | 32.0 | 56.2 | 55.5 | 30.7 | 33.8 | 35.0 |
| r3r_s2 | 41.0 | 44.3 | 41.2 | 32.8 | 45.5 | 34.6 | 47.8 | 42.6 | 38.8 | 31.6 | 56.5 | 54.6 | 31.6 | 32.6 | 38.3 |

## feat: KrunTechBench v1 part (choice items; macro over the 10 tracks)

| model | macro | api | backend | cloud | database | devops | distributed | mlinfra | security | software_repair | state_action | hard neg. | flips | unseen comb. | seen comb. | unseen tech. | large k | unseen family | gold>hardest | margin | OOD FA | involved (noul) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 45.3 | 45.0 | 59.1 | 35.2 | 47.8 | 13.7 | 58.7 | 60.9 | 75.8 | — | 11.1 | 32.5 | 28.0 | 75.3 | 67.8 | 74.5 | 30.1 | 24.0 | 72.5 | +2.02 | 0.0 | — |
| r2l4na_s1 | 83.9 | 85.8 | 77.7 | 63.6 | 88.9 | 100.0 | 99.2 | 89.2 | 84.8 | — | 65.5 | 77.2 | 97.3 | 74.3 | 99.8 | 90.5 | 81.0 | 14.8 | 90.5 | +7.88 | 0.5 | — |
| r3r_s1 | 84.6 | 91.8 | 70.6 | 69.9 | 88.9 | 100.0 | 98.8 | 95.1 | 80.6 | — | 65.2 | 78.3 | 97.8 | 79.3 | 100.0 | 89.5 | 82.1 | 21.7 | 90.8 | +7.72 | 0.0 | — |
| r3r_s2 | 86.5 | 87.6 | 76.7 | 77.4 | 88.4 | 100.0 | 99.0 | 96.8 | 85.5 | — | 67.3 | 81.8 | 98.0 | 79.7 | 100.0 | 91.5 | 83.4 | 32.7 | 93.1 | +7.92 | 0.0 | — |

## feat: unseen requirement combinations and counterfactual families

| model | unseen 1 req | unseen 2 req | unseen 3 req | catalog (unseen sets) | table families (unseen sets) | v1 flips: both / stuck | table families: both / stuck | capability families: both / stuck |
|---|---:|---:|---:|---:|---:|---|---|---|
| v1 | 68.6 | 78.4 | 86.0 | 32.1 | 73.5 | 8.7 / 39.4 (n=286) | 31.6 / 19.8 (n=472) | 10.2 / 58.5 (n=754) |
| r2l4na_s1 | 65.7 | 77.6 | 100.0 | 36.9 | 64.2 | 94.4 / 0.0 (n=286) | 24.8 / 22.0 (n=472) | 19.1 / 35.8 (n=754) |
| r3r_s1 | 71.4 | 82.4 | 100.0 | 72.6 | 62.4 | 95.8 / 0.0 (n=286) | 44.7 / 17.6 (n=472) | 71.0 / 7.3 (n=754) |
| r3r_s2 | 68.3 | 85.3 | 95.3 | 70.9 | 66.6 | 95.8 / 0.0 (n=286) | 41.5 / 20.4 (n=472) | 72.1 / 5.2 (n=754) |

## feat: capability tracks (R3 extension; all question types)

| model | macro | numeric | temporal | multi_hop | long_policy | tradeoff | mixed | seen scenario | unseen scenario | choice | true/false | ordinal | 2 hops | 3 hops | 4 hops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 35.6 | 33.9 | 31.6 | 34.0 | 44.9 | 30.5 | 38.8 | 34.5 | 36.5 | 35.5 | — | — | 27.3 | 37.6 | 39.6 |
| r2l4na_s1 | 45.2 | 48.0 | 44.1 | 39.2 | 57.1 | 35.3 | 47.5 | 43.3 | 45.9 | 44.7 | — | — | 33.7 | 42.6 | 41.7 |
| r3r_s1 | 81.7 | 68.8 | 85.2 | 86.8 | 88.0 | 79.7 | 81.9 | 80.8 | 84.7 | 82.9 | — | — | 84.9 | 86.8 | 93.8 |
| r3r_s2 | 81.4 | 67.0 | 87.2 | 89.4 | 87.4 | 76.6 | 80.6 | 81.2 | 84.5 | 83.0 | — | — | 86.6 | 90.9 | 91.7 |

## Paired top-1 difference vs v1 (raw, bootstrap 95% CI)

| model | KrunTechBench v1 choice items | capability items |
|---|---|---|
| r2l4na_s1 | +26.5 pp [+24.9, +28.1] (n=5302) | -0.2 pp [-1.5, +1.1] (n=2821) |
| r3r_s1 | +27.7 pp [+26.1, +29.2] (n=5302) | +3.9 pp [+1.8, +6.0] (n=2821) |
| r3r_s2 | +26.5 pp [+24.9, +28.1] (n=5302) | +3.8 pp [+1.6, +5.8] (n=2821) |
