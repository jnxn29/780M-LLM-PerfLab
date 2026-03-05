# Resume Bullets (CN/EN)

Updated: `2026-03-05`

## 1) One-line Project Statement

在 Win11 + Radeon 780M 上构建本地 LLM 推理性能优化流水线，形成可复现实验、严格门禁和发布级证据闭环。

Built a local LLM inference optimization pipeline on Win11 + Radeon 780M with reproducible experiments, strict gates, and release-ready evidence closure.

## 2) 中文简历要点（可直接使用）

1. 设计并落地三轨主链路（`llama + mlc + torch_rocm`）性能优化流程，建立 `eligible/cuda/no-fallback` 硬门禁，保证结果可比较、可复验。  
2. 采用流式首 token 口径优化 TTFT，在发布候选中实现明显时延下降，同时保持 TPS 护栏。  
3. 建立发布级质量管理流程：`ruff + 合同/回归测试 + readiness gate + 验包`，支持隔离发布目录直接投递 GitHub。  
4. 建立 RGP CSV 证据链，补充显存带宽相关分析，形成“指标 + 证据 + 文档”闭环。  
5. ORT 与 GPU-Z 仅作为内部历史研究路径，发布包中严格剔除，确保公开版本简洁稳定。  

## 3) English Resume Bullets

1. Designed and implemented a three-track optimization pipeline (`llama + mlc + torch_rocm`) with strict hard gates (`eligible/cuda/no-fallback`) for reproducible benchmarking.  
2. Optimized TTFT using streaming first-token measurement while preserving TPS guardrails across release candidates.  
3. Built release-grade quality controls (`ruff`, contract/regression tests, readiness gate, bundle verification) for GitHub-ready delivery.  
4. Added an RGP CSV evidence path to support memory-bandwidth-oriented analysis and evidence closure.  
5. Kept ORT and GPU-Z paths as internal research history in the main repo, but excluded them from the public release bundle for cleanliness and stability.  

## 4) 30-second Interview Pitch

我做的是 AI Infra 工程化，不是一次性跑分。我把性能优化、可靠性门禁、复验稳定性、证据报告和发布验包串成了可持续迭代的流水线。  

I focused on AI infra engineering rather than one-off benchmarks. I built a sustainable pipeline that connects optimization, reliability gates, rerun stability, evidence reports, and release verification.
