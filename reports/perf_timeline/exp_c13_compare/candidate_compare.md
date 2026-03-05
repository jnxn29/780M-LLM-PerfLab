# C11/C12/C13 Candidate Compare

generated_at_utc: `2026-03-03T21:15:01Z`

| candidate | all_hard | phi3_passed | r1_r4_avg_ttft_ms | r1_r4_ttft_win_count | r5_ttft_ms | avg_ttft_ms | avg_tps_mean | r5_tps_mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C11 | True | True | 13.488 | 3 | 385.366 | 87.864 | 135.782 | 5.057 |
| C13B | True | True | 13.977 | 1 | 430.817 | 97.345 | 137.350 | 5.500 |
| C12B_rep | True | True | 14.175 | 0 | 365.930 | 84.526 | 136.824 | 5.397 |
| C12A_rep | True | True | 14.206 | 0 | 368.519 | 85.069 | 135.045 | 5.277 |
| C13A | True | True | 14.865 | 0 | 374.621 | 86.816 | 136.522 | 5.237 |

Decision: keep `C11` as resume optimized point because it remains the best balanced profile on `r1-r4` (wins 3/4 TTFT dimensions), while C12/C13 gains mainly come from r5-only shifts.
