# Resume Three-point Metrics (B0/C06/C11)

generated_at_utc: `2026-03-03T20:21:10Z`
csv: `<repo-root>\reports\perf_timeline\exp_c12_repeat_compare\three_point_metrics.csv`

| run_name | B0_ttft | C06_ttft | C11_ttft | B0_tps | C06_tps | C11_tps | B0_to_C11_ttft_improve_pct | C06_to_C11_ttft_improve_pct | B0_to_C11_tps_ratio | C11_hard_gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r1_baseline | 20.063 | 15.820 | 13.419 | 116.909 | 146.755 | 168.248 | 33.116 | 15.177 | 1.439 | True |
| r2_flash_only | 19.540 | 15.213 | 13.530 | 117.471 | 141.537 | 168.709 | 30.757 | 11.063 | 1.436 | True |
| r3_compile_only | 18.241 | 15.440 | 13.586 | 117.355 | 147.353 | 170.075 | 25.519 | 12.008 | 1.449 | True |
| r4_flash_compile | 18.159 | 15.211 | 13.418 | 117.405 | 146.701 | 166.821 | 26.108 | 11.788 | 1.421 | True |
| r5_phi3_confirm | 1481.101 | 388.969 | 385.366 | 1.267 | 5.067 | 5.057 | 73.981 | 0.926 | 3.991 | True |

## KPI Summary

- r1_r5_avg_ttft_improve_pct: `37.896`
- r1_r4_min_ttft_improve_pct: `25.519`
- r5_ttft_improve_pct: `73.981`
- r5_tps_ratio: `3.991`
- r5_tps_guardrail_pass: `True`
- all_hard: `True`
- all_tps_95: `True`
- all_ttft_targets: `True`


