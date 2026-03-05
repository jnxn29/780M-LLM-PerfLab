# Sidecar Operator Summary

generated_at_utc: `2026-03-03T20:21:10Z`

## Recommendation

- selected_experiment_id: `O4`
- baseline_r5_tps: `5.067`
- tps_guardrail: `4.81365`
- selected_ttft_p50_ms: `377.929`
- selected_tps_mean: `5.098`
- selected_sdpa_profile_effective: `flash_only`
- selected_compile_enabled: `True`

## O1~O4 Snapshot

| experiment_id | sdpa_profile_effective | compile_enabled | compile_mode | ttft_p50_ms | tps_mean | eligible | ttft_signal_source | fallback_triggered | runtime_device_fallback |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| O1 | balanced | False | reduce-overhead | 390.227 | 4.942 | True | time_to_first_token | False | False |
| O2 | flash_only | False | reduce-overhead | 391.23 | 4.948 | True | time_to_first_token | False | False |
| O3 | balanced | True | reduce-overhead | 393.627 | 4.934 | True | time_to_first_token | False | False |
| O4 | flash_only | True | reduce-overhead | 377.929 | 5.098 | True | time_to_first_token | False | False |

