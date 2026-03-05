# Non-ORT Tuning Report

generated_at_utc: `2026-03-05T00:40:26Z`
required_backends: `llama,mlc,torch_rocm`
strict_thresholds: `ttft_improve_pct>=15.0, tps_ratio>=0.95`

- summary_csv: `<repo-root>\reports\perf_timeline\non_ort_tuning_summary.csv`
- gate_json: `<repo-root>\reports\perf_timeline\non_ort_release_gate.json`

## Candidate Summary

| candidate_run_name | hard_gate | strict_gate_pass | avg_ttft_ms | avg_tps |
| --- | --- | --- | ---: | ---: |
| r_non_ort_target_c_torch_lite_sdpa_r2_20260305_080003 | True | True | 716.041 | 11.737 |
| r_non_ort_target_c_low2_ondemand_eager_20260305_1 | True | True | 721.107 | 11.553 |
| r_non_ort_target_c_low2_ondemand_20260305_1 | True | True | 784.660 | 11.734 |
| r_non_ort_target_c_torch_lite_sdpa_compile_20260305_082736 | True | True | 786.906 | 11.897 |
| r_non_ort_target_c_torch_lite_sdpa_20260305_074749 | True | True | 810.636 | 11.778 |
| r_non_ort_target_c_torch_lite3_sdpa_20260305_081253 | True | True | 823.652 | 11.792 |
| r_non_ort_target_c_low2_ondemand_flash_20260305_1 | True | True | 837.825 | 11.559 |
| r_non_ort_target_c_low2_20260304_1 | True | True | 1285.832 | 10.720 |

## Backend Detail

| candidate_run_name | backend | baseline_ttft_ms | candidate_ttft_ms | ttft_improve_pct | baseline_tps | candidate_tps | tps_ratio | hard_gate | strict_gate_pass | rgp_status | rgp_dram_bytes_per_token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| r_non_ort_target_c_low2_20260304_1 | llama | 1473.694 | 629.457 | 57.287 | 4.650 | 10.224 | 2.199 | True | True |  | nan |
| r_non_ort_target_c_low2_20260304_1 | mlc | 4499.779 | 2855.394 | 36.544 | 11.029 | 16.512 | 1.497 | True | True |  | nan |
| r_non_ort_target_c_low2_20260304_1 | torch_rocm | 1532.064 | 372.644 | 75.677 | 0.841 | 5.423 | 6.448 | True | True |  | nan |
| r_non_ort_target_c_low2_ondemand_20260305_1 | llama | 1473.694 | 421.329 | 71.410 | 4.650 | 12.451 | 2.678 | True | True |  | nan |
| r_non_ort_target_c_low2_ondemand_20260305_1 | mlc | 4499.779 | 1444.961 | 67.888 | 11.029 | 17.047 | 1.546 | True | True |  | nan |
| r_non_ort_target_c_low2_ondemand_20260305_1 | torch_rocm | 1532.064 | 487.690 | 68.168 | 0.841 | 5.704 | 6.782 | True | True |  | nan |
| r_non_ort_target_c_low2_ondemand_flash_20260305_1 | llama | 1473.694 | 441.724 | 70.026 | 4.650 | 12.338 | 2.653 | True | True |  | nan |
| r_non_ort_target_c_low2_ondemand_flash_20260305_1 | mlc | 4499.779 | 1564.540 | 65.231 | 11.029 | 16.922 | 1.534 | True | True |  | nan |
| r_non_ort_target_c_low2_ondemand_flash_20260305_1 | torch_rocm | 1532.064 | 507.212 | 66.894 | 0.841 | 5.417 | 6.441 | True | True |  | nan |
| r_non_ort_target_c_low2_ondemand_eager_20260305_1 | llama | 1473.694 | 435.779 | 70.429 | 4.650 | 12.252 | 2.635 | True | True |  | nan |
| r_non_ort_target_c_low2_ondemand_eager_20260305_1 | mlc | 4499.779 | 1225.805 | 72.759 | 11.029 | 16.987 | 1.540 | True | True |  | nan |
| r_non_ort_target_c_low2_ondemand_eager_20260305_1 | torch_rocm | 1532.064 | 501.738 | 67.251 | 0.841 | 5.420 | 6.445 | True | True |  | nan |
| r_non_ort_target_c_torch_lite_sdpa_20260305_074749 | llama | 1473.694 | 434.934 | 70.487 | 4.650 | 12.297 | 2.645 | True | True |  | nan |
| r_non_ort_target_c_torch_lite_sdpa_20260305_074749 | mlc | 4499.779 | 1534.886 | 65.890 | 11.029 | 17.029 | 1.544 | True | True |  | nan |
| r_non_ort_target_c_torch_lite_sdpa_20260305_074749 | torch_rocm | 1532.064 | 462.088 | 69.839 | 0.841 | 6.009 | 7.145 | True | True |  | nan |
| r_non_ort_target_c_torch_lite_sdpa_r2_20260305_080003 | llama | 1473.694 | 439.466 | 70.179 | 4.650 | 12.274 | 2.640 | True | True |  | nan |
| r_non_ort_target_c_torch_lite_sdpa_r2_20260305_080003 | mlc | 4499.779 | 1228.665 | 72.695 | 11.029 | 17.012 | 1.542 | True | True |  | nan |
| r_non_ort_target_c_torch_lite_sdpa_r2_20260305_080003 | torch_rocm | 1532.064 | 479.992 | 68.670 | 0.841 | 5.925 | 7.045 | True | True |  | nan |
| r_non_ort_target_c_torch_lite3_sdpa_20260305_081253 | llama | 1473.694 | 442.411 | 69.979 | 4.650 | 12.226 | 2.629 | True | True |  | nan |
| r_non_ort_target_c_torch_lite3_sdpa_20260305_081253 | mlc | 4499.779 | 1554.126 | 65.462 | 11.029 | 17.003 | 1.542 | True | True |  | nan |
| r_non_ort_target_c_torch_lite3_sdpa_20260305_081253 | torch_rocm | 1532.064 | 474.420 | 69.034 | 0.841 | 6.148 | 7.310 | True | True |  | nan |
| r_non_ort_target_c_torch_lite_sdpa_compile_20260305_082736 | llama | 1473.694 | 428.382 | 70.931 | 4.650 | 12.321 | 2.650 | True | True |  | nan |
| r_non_ort_target_c_torch_lite_sdpa_compile_20260305_082736 | mlc | 4499.779 | 1474.626 | 67.229 | 11.029 | 17.061 | 1.547 | True | True |  | nan |
| r_non_ort_target_c_torch_lite_sdpa_compile_20260305_082736 | torch_rocm | 1532.064 | 457.711 | 70.125 | 0.841 | 6.309 | 7.502 | True | True |  | nan |

## Selection

- selected_candidate: `r_non_ort_target_c_torch_lite_sdpa_r2_20260305_080003`
- strict_gate_pass: `true`


