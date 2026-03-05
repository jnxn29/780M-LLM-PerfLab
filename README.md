# 780M-LLM-PerfLab

## 项目定位 / Project Positioning

面向 `Win11 + Radeon 780M` 的本地 LLM 推理性能实验与发布工程项目，发布口径聚焦 `llama + mlc + torch_rocm`，强调可复现实验、量化优化、门禁收敛与发布验包。

A Windows-first local LLM inference performance lab for `Win11 + Radeon 780M`, with publish scope focused on `llama + mlc + torch_rocm`, emphasizing reproducible experiments, measurable optimization, strict gating, and release validation.

## 发布门禁 / Release Gates

| artifact | required value |
| --- | --- |
| `reports/perf_timeline/non_ort_release_gate.json` | `pass=true` |
| `reports/perf_timeline/mlflow_evidence.json` | `mlflow_used=true` |
| `reports/perf_timeline/release_readiness.json` | `ready=true` |

## 优化结果快照 / Optimization Snapshot

关键对比（`N_B0 -> Top candidate`, stage_b）如下。

Core comparison (`N_B0 -> Top candidate`, stage_b) is shown below.

| backend | baseline TTFT (ms) | optimized TTFT (ms) | TTFT improve | baseline TPS | optimized TPS | TPS ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama | 1255.587 | 443.011 | 64.717% | 6.453 | 12.228 | 1.895x |
| mlc | 4629.093 | 1574.600 | 65.985% | 11.210 | 16.967 | 1.514x |
| torch_rocm | 1793.380 | 462.173 | 74.229% | 0.869 | 6.101 | 7.021x |

## 证据链与工具 / Evidence Chain and Tools

主链路编排：`run_gpu_util_uplift_windows.ps1`、`run_non_ort_release_tuning_windows.ps1`、`run_torch_rocm_push_windows.ps1`。报告聚合：`build_perf_timeline_report.py`、`collect_non_ort_tuning_report.py`。显存带宽证据：RGP CSV -> `collect_rgp_metrics.py`。试验追踪证据：`collect_mlflow_evidence.py`。发布验包：`verify_release_bundle.py`。

Main orchestration: `run_gpu_util_uplift_windows.ps1`, `run_non_ort_release_tuning_windows.ps1`, `run_torch_rocm_push_windows.ps1`. Report aggregation: `build_perf_timeline_report.py`, `collect_non_ort_tuning_report.py`. Memory-bandwidth evidence: RGP CSV -> `collect_rgp_metrics.py`. Experiment tracking evidence: `collect_mlflow_evidence.py`. Bundle verification: `verify_release_bundle.py`.

## 快速开始 / Quick Start

```bash
conda env create -f environment.yml
conda activate 780m-perflab
python harness/benchctl.py doctor --out results/demo/
python harness/benchctl.py validate --input fixtures/sample_metrics.jsonl
python harness/benchctl.py report --input fixtures/sample_run/ --out reports/demo/
```

## 三轨调优执行 / Run Three-Track Tuning

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_non_ort_release_tuning_windows.ps1 `
  -OutRoot results/r_non_ort_tuning_publish/
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_torch_rocm_push_windows.ps1 `
  -OutRoot results/r_torch_rocm_push_publish/
```

## RGP CSV 聚合 / RGP CSV Aggregation

```powershell
python scripts/collect_rgp_metrics.py --input-root reports/rgp_raw --out-root reports/perf_timeline
```

Expected outputs:
1. `reports/perf_timeline/rgp_summary.csv`
2. `reports/perf_timeline/rgp_report.md`
3. `reports/perf_timeline/rgp_bytes_per_token.png`
4. `reports/perf_timeline/rgp_dram_read_write.png`

## 发布质量检查 / Release Quality Checks

```bash
python -m ruff check harness scripts tests
pytest -q
python scripts/verify_release_bundle.py --release-root <release_bundle_root>
```

发布包会强制检查必备文件、禁入路径、编码乱码、路径隐私泄露和 readiness 门禁。

The release bundle enforces required files, forbidden paths, encoding sanity, path privacy checks, and readiness gates.

## 发布范围说明 / Publish Scope Notes

ORT 与 GPU-Z 路径仅保留在主仓库历史中，不进入本次发布包。

ORT and GPU-Z paths are kept only as historical/internal tracks and are excluded from this release bundle.
