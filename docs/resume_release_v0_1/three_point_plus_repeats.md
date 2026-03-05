# Resume Three-point + Repeat Metrics

generated_at_utc: `2026-03-03T20:20:59Z`
csv: `three_point_plus_repeats.csv`

| run_name | B0_ttft | C06_ttft | C11_ttft | C12A_ttft | C12B_ttft | B0_tps | C06_tps | C11_tps | C12A_tps | C12B_tps | B0_to_C11_ttft_improve_pct | B0_to_C12A_ttft_improve_pct | B0_to_C12B_ttft_improve_pct | B0_to_C11_tps_ratio | B0_to_C12A_tps_ratio | B0_to_C12B_tps_ratio | hard_gate_C11 | hard_gate_C12A | hard_gate_C12B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| r1_baseline | 20.063 | 15.820 | 13.419 | 13.903 | 14.357 | 116.909 | 146.755 | 168.248 | 171.469 | 167.001 | 33.116 | 30.703 | 28.440 | 1.439 | 1.467 | 1.428 | True | True | True |
| r2_flash_only | 19.540 | 15.213 | 13.530 | 14.572 | 13.707 | 117.471 | 141.537 | 168.709 | 161.446 | 171.180 | 30.757 | 25.425 | 29.852 | 1.436 | 1.374 | 1.457 | True | True | True |
| r3_compile_only | 18.241 | 15.440 | 13.586 | 14.051 | 14.041 | 117.355 | 147.353 | 170.075 | 168.963 | 170.501 | 25.519 | 22.970 | 23.025 | 1.449 | 1.440 | 1.453 | True | True | True |
| r4_flash_compile | 18.159 | 15.211 | 13.418 | 14.298 | 14.594 | 117.405 | 146.701 | 166.821 | 168.070 | 170.042 | 26.108 | 21.262 | 19.632 | 1.421 | 1.432 | 1.448 | True | True | True |
| r5_phi3_confirm | 1481.101 | 388.969 | 385.366 | 368.519 | 365.930 | 1.267 | 5.067 | 5.057 | 5.277 | 5.397 | 73.981 | 75.119 | 75.293 | 3.991 | 4.165 | 4.260 | True | True | True |

## Repeat Stability

- all_hard: `True`
- all_tps_95: `True`
- repeat_stability_overall: `True`

| run_name | ttft_delta_ratio | tps_delta_ratio | repeat_stable_ttft | repeat_stable_tps | repeat_stable |
| --- | ---: | ---: | --- | --- | --- |
| r1_baseline | 0.032655 | 0.026754 | True | True | True |
| r2_flash_only | 0.063106 | 0.060293 | True | True | True |
| r3_compile_only | 0.000712 | 0.009103 | True | True | True |
| r4_flash_compile | 0.020702 | 0.011733 | True | True | True |
| r5_phi3_confirm | 0.007075 | 0.02274 | True | True | True |


