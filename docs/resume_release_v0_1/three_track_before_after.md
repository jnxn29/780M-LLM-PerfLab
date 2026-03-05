# Four-track Before/After (Stage-A best TPS profile)

generated_at_utc: `2026-03-05T00:42:37Z`
csv: `three_track_before_after.csv`

| backend | before_snapshot | after_snapshot | before_profile | after_profile | before_ttft_ms | after_ttft_ms | ttft_improve_pct | before_tps | after_tps | tps_ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama | N_B0 | r_non_ort_target_c_torch_lite_sdpa_r2_20260305_080003 | L0 | L0 | 1255.587 | 443.011 | 64.717 | 6.453 | 12.228 | 1.895 |
| mlc | N_B0 | r_non_ort_target_c_torch_lite_sdpa_r2_20260305_080003 | M0 | M0 | 4629.093 | 1574.600 | 65.985 | 11.210 | 16.967 | 1.514 |
| torch_rocm | N_B0 | r_non_ort_target_c_torch_lite_sdpa_r2_20260305_080003 | T0 | T0 | 1793.380 | 462.173 | 74.229 | 0.869 | 6.101 | 7.021 |


