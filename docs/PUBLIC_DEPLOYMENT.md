# VisionGuard AI公开评委演示部署

## 定位

公开部署用于让评委直接访问产品界面、运行三类可复现案例并查看Agent、知识检索、风险分析和报告导出闭环。默认云端配置明确使用Demo视觉fallback，不把它表述为专业模型效果；完整YOLO、三路火情确认与本地大模型仍以比赛机配置运行。

## Render Blueprint

1. 将本分支推送至GitHub。
2. 在Render创建Blueprint并选择本仓库，平台会读取根目录的`render.yaml`。
3. 部署完成后访问`/api/v1/health`，确认`status=ok`与`deployment_profile=public-demo`。
4. 进入首页运行“赛前环境自检”。公开配置显示视觉、分类和本地大模型降级属于预期状态；知识、Agent工具、案例与报告导出必须为就绪。

Blueprint设置为仅在GitHub检查通过后自动部署，避免未通过测试的提交直接替换评委演示版本。

## 完整模型部署

如云端环境具备合法模型文件、足够内存/GPU和安全密钥管理，可覆盖下列环境变量启用完整链路：

- `VISION_BACKEND=yolo`
- `YOLO_MODEL_PATH=/secure/models/model.pt`
- `YOLO_EXPECTED_SHA256=<固定摘要>`
- `FIRE_CLASSIFIER_ENABLED=true`
- `FIRE_CLASSIFIER_ROOT=/secure/models/fire-classifier`
- `REASONING_BACKEND=llm`
- `LLM_BASE_URL`、`LLM_MODEL`及安全保存的`LLM_API_KEY`

模型权重、密钥、上传图片和本地虚拟环境不得提交至GitHub。

## 公网运行保护

- 上传文件按分块读取，并在超过限制时立即终止。
- 同时分析任务数量受`MAX_CONCURRENT_ANALYSES`限制；排队超时返回503并提示重试。
- API响应禁止缓存，页面附加内容安全、点击劫持和权限策略响应头。
- 上传图片仅在内存处理，分析记录不保存原图和原始文件名。
- 0.13.0起，分析详情、历史列表、指标、追问和报告按浏览器Cookie会话隔离；无Cookie或其他会话访问已知报告编号统一返回404。报告地址不能作为跨浏览器分享链接。
- Cookie有效期8小时，HttpOnly、SameSite=Strict，HTTPS下为Secure。生产代理应正确传递HTTPS信息。此机制仅做匿名会话隔离，不代替账号鉴权、企业权限、持久审计、限流或防滥用网关；不得把本演示直接宣传为企业生产部署。
- 内存最多保存100条记录，淘汰或重启后不可恢复；不支持跨进程共享记录，请保持单应用进程。API调用方需保存Cookie，并在首次并发调用前建立会话。
- 图片推理前执行实际格式、静态帧、像素上限及完整性检查，统一方向与透明背景；最高2400万像素、单边12000像素。应依据部署内存进一步限制上传和并发，而不是仅依据压缩文件大小估计内存。
