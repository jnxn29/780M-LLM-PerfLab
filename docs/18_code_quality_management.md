# Code Quality Management (Release v0.1.2)

Date: `2026-03-05`

## 1) Scope

This release-quality document records checks for the clean publish bundle:
- publish scope: `llama + mlc + torch_rocm`
- evidence scope: TTFT/TPS + RGP CSV + MLflow
- excluded from publish bundle: ORT and GPU-Z paths

## 2) Lint Gate

Command:

```bash
python -m ruff check harness scripts tests
```

Expected result:
- no lint errors

## 3) Test Gate (Minimal Release Set)

Command:

```bash
pytest -q tests/test_build_resume_release_metrics.py tests/test_build_four_track_before_after_metrics.py tests/test_collect_non_ort_tuning_report.py tests/test_collect_rgp_metrics.py tests/test_collect_mlflow_evidence.py tests/test_run_non_ort_release_tuning_windows_contract.py tests/test_run_torch_rocm_push_windows_contract.py tests/test_run_gpu_util_uplift_windows_contract.py tests/test_verify_release_bundle.py
```

Expected result:
- all selected tests pass

## 4) Readiness Gate

Command:

```bash
python scripts/check_release_readiness.py --c06-meta ... --fourtrack-before-meta ... --fourtrack-after-meta ... --three-point-csv ... --repeat-stability-json ... --quality-report-json ... --non-ort-release-gate-json reports/perf_timeline/non_ort_release_gate.json --mlflow-evidence-json reports/perf_timeline/mlflow_evidence.json --out-json reports/perf_timeline/release_readiness.json
```

Expected result:
- `reports/perf_timeline/release_readiness.json` has `ready=true`

## 5) Bundle Verification Gate

Command:

```bash
python scripts/verify_release_bundle.py --release-root <release_root>
```

Expected result:
- `release_bundle_check.json.ready=true`
- no forbidden publish paths
- no encoding issues in README and core docs

## 6) Required Evidence Files

- `reports/perf_timeline/non_ort_tuning_summary.csv`
- `reports/perf_timeline/non_ort_tuning_report.md`
- `reports/perf_timeline/non_ort_release_gate.json`
- `reports/perf_timeline/mlflow_evidence.json`
- `reports/perf_timeline/release_readiness.json`
- `reports/perf_timeline/rgp_summary.csv`
- `reports/perf_timeline/rgp_report.md`
