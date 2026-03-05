# GitHub 发布 Runbook（手动执行）

更新时间：`2026-03-05`

## 1) 发布范围

`v0.1.2` 发布范围：
- 主轨道：`llama + mlc + torch_rocm`
- 证据：TTFT/TPS + RGP CSV + MLflow
- 发布 profile：`resume_clean_no_ort`

发布包排除：
- ORT 发布路径
- GPU-Z 相关脚本/测试/文档
- 原始重资产目录（`results/`、`reports/rgp_raw/`）

## 2) 发布前检查

```powershell
python -m ruff check harness scripts tests
pytest -q tests/test_build_resume_release_metrics.py tests/test_build_four_track_before_after_metrics.py tests/test_collect_non_ort_tuning_report.py tests/test_collect_rgp_metrics.py tests/test_collect_mlflow_evidence.py tests/test_run_non_ort_release_tuning_windows_contract.py tests/test_run_torch_rocm_push_windows_contract.py tests/test_run_gpu_util_uplift_windows_contract.py tests/test_verify_release_bundle.py
```

Readiness（默认路径脱敏输出）：

```powershell
python scripts/check_release_readiness.py `
  --c06-meta <c06_meta_json> `
  --fourtrack-before-meta <before_meta_json> `
  --fourtrack-after-meta <after_meta_json> `
  --three-point-csv docs/resume_release_v0_1/three_point_metrics.csv `
  --repeat-stability-json docs/resume_release_v0_1/repeat_stability.json `
  --quality-report-json reports/perf_timeline/quality_report.json `
  --non-ort-release-gate-json reports/perf_timeline/non_ort_release_gate.json `
  --mlflow-evidence-json reports/perf_timeline/mlflow_evidence.json `
  --out-json reports/perf_timeline/release_readiness.json
```

目标：`reports/perf_timeline/release_readiness.json` 中 `ready=true`。

## 3) 生成隔离发布目录

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/prepare_github_release_folder.ps1 `
  -ReleaseProfile resume_clean_no_ort `
  -SanitizePaths $true
```

默认输出目录：
- `release/v0.1.2-clean-no-ort-<date>/780m-llm-perflab`

## 4) 验包

```powershell
python scripts/verify_release_bundle.py --release-root release/v0.1.2-clean-no-ort-<date>/780m-llm-perflab
```

目标：`release_bundle_check.json.ready == true`。

验证项：
- 必备文件/目录
- 禁入路径（`*ort*`、`*gpuz*`、`results/`、`reports/rgp_raw/`）
- 编码乱码检查
- 路径隐私泄露检查
- readiness 门禁

## 5) GitHub 手动发布命令

在发布目录根执行：

```powershell
git init
git checkout -b main
git add .
git commit -m "release: v0.1.2 clean no-ort resume edition"
git remote add origin https://github.com/<your-username>/780m-llm-perflab.git
git push -u origin main
git tag -a v0.1.2 -m "resume release v0.1.2 clean no-ort"
git push origin v0.1.2
```

## 6) 回滚策略

1. 在主仓库修复问题。
2. 重新执行 `prepare_github_release_folder.ps1`。
3. 重新执行 `verify_release_bundle.py`。
4. 按补丁版本重新发布（`v0.1.2.x`）。
