# 本地大模型运行边界

VisionGuard AI 比赛版支持以本地OpenAI兼容服务启用真实大语言模型推理，并在不可用、超时、非法JSON或虚构证据引用时自动切换为确定性回退。

## 冻结组件

- 推理运行时：`ggml-org/llama.cpp` Windows x64 Vulkan，版本`b10581`；压缩包SHA-256：`03d22a6267330005a56de5841a39ee6efaf5524337b9c08030122ef17e189a80`。
- 模型：`Qwen/Qwen2.5-1.5B-Instruct-GGUF`，仓库提交`91cad51170dc346986eccefdc2dd33a9da36ead9`，量化`Q4_K_M`；文件SHA-256：`6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`。
- 运行时和权重保存在被Git忽略的`runtime/`、`models/`目录，不提交到代码仓库。

## 安全约束

本地大模型只负责解释已存在的视觉证据、知识依据和风险评估，不直接读取原图、不代替专业视觉模型、不生成新的检测类别。响应必须为结构化JSON，且引用的`evidence_id`和`rule_id`必须真实存在；任何越界引用都会被拒绝并触发确定性回退。

该1.5B量化模型用于离线可演示和隐私友好的任务解释，不将其表述为工业安全领域专用大模型，也不以模型参数规模代替实际效果验证。

## 已验证运行结果

在2026-08-22记录的比赛机上，20张固定全链路样本以4并发运行：请求成功率、Agent成功率、本地LLM实际采用率与证据约束推理率均为100%，确定性推理回退0次；平均响应7.29秒，P95响应11.17秒。原始记录为`evaluation/full_chain_results_local_llm_v4.json`。

该结果仅适用于冻结模型、运行时、硬件、样本与配置。确定性推理配置下的视觉链路时延应作为另一组独立基准，不与本地LLM端到端时延混写。

## 安装与启动

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_local_llm.ps1
powershell -ExecutionPolicy Bypass -File scripts\start_competition.ps1
```
