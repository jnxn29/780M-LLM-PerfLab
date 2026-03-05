# 简历版发布说明 v0.1

更新时间：`2026-03-05`

## 1. 问题定义

- 在 `Win11 + Radeon 780M` 上构建本地 LLM 推理性能优化流水线。
- 同时满足三类要求：可复现、可量化、可发布。

## 2. 系统设计

- 编排入口：
  - `scripts/run_gpu_util_uplift_windows.ps1`
  - `scripts/run_non_ort_release_tuning_windows.ps1`
  - `scripts/run_torch_rocm_push_windows.ps1`
- 指标口径：`TTFT p50` + `TPS mean`（stage_b_recheck）
- 硬门禁：
  - `eligible=true`
  - `torch_device=cuda`
  - `fallback_triggered=false`
  - `runtime_device_fallback=false`
- 证据链：`pipeline_meta.json`、汇总 CSV/MD、release readiness JSON、RGP/MLflow 证据

## 3. 优化路径

- 基线点：`N_B0`
- 收敛候选：`Top candidate`
- 复验：`C12A/C12B`

发布口径固定为三轨：`llama + mlc + torch_rocm`。

## 4. 量化结果

| backend | baseline TTFT (ms) | optimized TTFT (ms) | TTFT improve | baseline TPS | optimized TPS | TPS ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama | 1255.587 | 443.011 | 64.717% | 6.453 | 12.228 | 1.895x |
| mlc | 4629.093 | 1574.600 | 65.985% | 11.210 | 16.967 | 1.514x |
| torch_rocm | 1793.380 | 462.173 | 74.229% | 0.869 | 6.101 | 7.021x |

发布门禁文件：
- `reports/perf_timeline/non_ort_release_gate.json`
- `reports/perf_timeline/mlflow_evidence.json`
- `reports/perf_timeline/release_readiness.json`

## 5. 复现实验命令

```powershell
python scripts/build_perf_timeline_report.py --results-root results --out-root reports/perf_timeline
python scripts/collect_rgp_metrics.py --input-root reports/rgp_raw --out-root reports/perf_timeline
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/prepare_github_release_folder.ps1 -ReleaseProfile resume_clean_no_ort
python scripts/verify_release_bundle.py --release-root <release_root>
```

## 6. Resume-ready 英文描述

1. Built a reproducible local LLM performance pipeline on Win11 + Radeon 780M with strict reliability gates.
2. Delivered significant TTFT reductions across three backends while maintaining TPS guardrails.
3. Implemented evidence-driven release quality controls including lint/tests/readiness/bundle verification.
