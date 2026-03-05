# 优化前后指标对比

更新时间：`2026-03-05`

## 1. 三点对比（B0/C06/C11）

数据源：
- `docs/resume_release_v0_1/three_point_metrics.csv`
- `docs/resume_release_v0_1/resume_kpi_summary.json`

| run_name | B0_ttft | C06_ttft | C11_ttft | B0_tps | C06_tps | C11_tps | B0_to_C11_ttft_improve_pct | C06_to_C11_ttft_improve_pct | B0_to_C11_tps_ratio | hard_gate_C11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r1_baseline | 20.063 | 15.820 | 13.419 | 116.909 | 146.755 | 168.248 | 33.116% | 15.177% | 1.439 | true |
| r2_flash_only | 19.540 | 15.213 | 13.530 | 117.471 | 141.537 | 168.709 | 30.757% | 11.063% | 1.436 | true |
| r3_compile_only | 18.241 | 15.440 | 13.586 | 117.355 | 147.353 | 170.075 | 25.519% | 12.008% | 1.449 | true |
| r4_flash_compile | 18.159 | 15.211 | 13.418 | 117.405 | 146.701 | 166.821 | 26.108% | 11.788% | 1.421 | true |
| r5_phi3_confirm | 1481.101 | 388.969 | 385.366 | 1.267 | 5.067 | 5.057 | 73.981% | 0.926% | 3.991 | true |

## 2. 同参复验（C12A/C12B）

数据源：
- `docs/resume_release_v0_1/three_point_plus_repeats.csv`
- `docs/resume_release_v0_1/repeat_stability.json`

| run_name | C12A_ttft | C12B_ttft | C12A_tps | C12B_tps | B0_to_C12A_ttft_improve_pct | B0_to_C12B_ttft_improve_pct | B0_to_C12A_tps_ratio | B0_to_C12B_tps_ratio | hard_gate_C12A | hard_gate_C12B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| r1_baseline | 13.903 | 14.357 | 171.469 | 167.001 | 30.703% | 28.440% | 1.467 | 1.428 | true | true |
| r2_flash_only | 14.572 | 13.707 | 161.446 | 171.180 | 25.425% | 29.852% | 1.374 | 1.457 | true | true |
| r3_compile_only | 14.051 | 14.041 | 168.963 | 170.501 | 22.970% | 23.025% | 1.440 | 1.453 | true | true |
| r4_flash_compile | 14.298 | 14.594 | 168.070 | 170.042 | 21.262% | 19.632% | 1.432 | 1.448 | true | true |
| r5_phi3_confirm | 368.519 | 365.930 | 5.277 | 5.397 | 75.119% | 75.293% | 4.165 | 4.260 | true | true |

复验判定：
- `all_hard=true`
- `all_tps_95=true`
- `repeat_stability_overall=true`

## 3. 三轨发布对比（发布口径）

数据源：`docs/resume_release_v0_1/three_track_before_after.csv`

| backend | before_snapshot | after_snapshot | before_profile | after_profile | before_ttft_ms | after_ttft_ms | ttft_improve_pct | before_tps | after_tps | tps_ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama | N_B0 | Top candidate | L0 | L0 | 1255.587 | 443.011 | 64.717% | 6.453 | 12.228 | 1.895 |
| mlc | N_B0 | Top candidate | M0 | M0 | 4629.093 | 1574.600 | 65.985% | 11.210 | 16.967 | 1.514 |
| torch_rocm | N_B0 | Top candidate | T0 | T0 | 1793.380 | 462.173 | 74.229% | 0.869 | 6.101 | 7.021 |

## 4. Sidecar 结论

- 证据文件：
  - `reports/perf_timeline/operator_sidecar_summary.csv`
  - `reports/perf_timeline/promotion_recommendation.json`
- 推荐候选：`O4 (flash_only + compile=reduce-overhead)`

## 5. 发布范围说明

- 本次发布严格采用 no-ORT 口径：`llama + mlc + torch_rocm`。
- 显存带宽证据采用 RGP CSV 聚合。
- ORT/GPU-Z 路径不进入发布包。
