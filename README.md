# VisionGuard AI

面向工业安全场景的多模态视觉风险智能体。

当前产品版本：`0.13.0`（2026-09-23）。在0.12.0模型链路基础上增加现场风险核查界面、异步结果隔离、图片方向校正、可点击证据、浏览器会话隔离和报告打印修复。下文原有模型/并发数字仍属于0.12.0历史验证，本轮未重跑该基准。

参赛定位：**2026年iCAN大学生创新创业大赛AI应用创新挑战赛——软件赛道**。项目以可在线演示或可运行的工业安全视觉风险智能体为作品形态，围绕创新性、技术实现、实用价值、用户体验和展示效果组织参赛证据；不再按浙江省国际大学生创新大赛人工智能命题项目表述。

本仓库是独立比赛项目，不是 FactorySafe 的分支、子模块或后续版本，也不依赖 FactorySafe 代码。

## MVP 闭环

```text
图片 + 自然语言任务
        ↓
VisionGuard Agent
        ↓
专业目标检测 → 检测框与定位证据
        +
三路火情分类 → 整图高精度确认
        ↓
双模型证据融合 → 一致、冲突与未定位火情
        ↓
安全知识检索 → 依据与处置要点
        ↓
风险分析 → 风险等级与人工复核边界
        ↓
结构化安全报告
```

## 当前实现状态

| 能力 | 状态 | 说明 |
|---|---|---|
| 图片上传与校验 | 已完成 | 支持 PNG/JPEG/WebP，默认限制 10 MB |
| 视觉检测接口 | 已完成 | 统一协议、演示适配器、YOLO适配器和显式回退 |
| YOLO定位模型权重 | 已接入 | 本地保存、固定来源、SHA-256校验，不进入Git |
| 高精度火情确认 | 已接入 | 调用队友冻结三路模型，严格OOF F1为0.943481；权重本地配置，不进入Git |
| 双模型证据融合 | 已完成 | YOLO负责定位，三路分类模型负责整图确认，显式处理一致与冲突 |
| Agent 工具调用 | 已完成 | 任务解释、工具注册、顺序编排、调用轨迹 |
| 可追溯知识增强 | 已完成 | 权威来源、条款、版本、链接、相关度和适用边界 |
| 可解释混合 RAG | 已完成 | 风险类别约束、中文TF-IDF向量召回、领域关键词与权威等级重排，返回检索方式和命中词 |
| 本地大语言模型推理 | 已完成 | Qwen2.5-1.5B经llama.cpp Vulkan离线运行，兼容外部接口，严格校验证据引用并保留确定性回退 |
| 基于证据的智能追问 | 已完成 | 复用本次视觉与知识证据回答，不重复检测、不脱离证据聊天 |
| 风险分析与报告 | 已完成 | 输出证据、依据、等级、建议与复核提示 |
| 检测框与证据定位 | 已完成 | 原图叠加类别、置信度和检测框 |
| 推理可追溯信息 | 已完成 | 记录模型版本、摘要、阈值、设备、耗时和回退原因 |
| 比赛演示模式 | 已完成 | 三类明确标注的合成标准案例 |
| 脱敏分析记录 | 已完成 | 内存保存最近100条，不保存上传图片和密钥 |
| 运行指标 | 已完成 | 次数、平均耗时、风险分布、后端与回退统计 |
| 报告与证据导出 | 已完成 | 可打印HTML报告、结构化JSON和浏览器端标注证据PNG下载 |
| 固定跨来源验证 | 已完成 | 80张能力集与20张全链路集，记录来源、许可、分组和哈希；80张集已用于阈值选择，不是最终未见测试集 |
| 并发稳定性 | 已完成 | DirectML单常驻模型，20张/4并发接口与Agent成功率均为100%，无回退 |
| 视觉链路2秒时延验证 | 已完成（当前硬件） | 确定性推理配置下20张/4并发平均1.02秒、P95 1.43秒 |
| 本地LLM全链路验证 | 已完成（当前硬件） | 0.12.0在20张/4并发下请求、Agent、LLM采用及证据约束率均为100%，推理回退0次 |

> `demo` 视觉后端仍作为显式回退保留，采用颜色启发式，不是经过训练或认证的工业安全模型，输出不可用于真实生产决策。

Agent当前采用固定顺序的工具编排，大模型负责证据约束的解释与追问，不宣称自主选择任意工具或自动执行现场处置。表中模型与性能数字是既有记录，不是2026-09-22重跑得到的新结果；历史“证据约束率”只校验引用，不代表解释正确率或专家认可率。

## 现场核查体验

- 首页以图片、核查要求、结果和下一步核查建议为主，模型参数与调用轨迹可展开查看。
- 切换图片、修改任务或停止等待会清除旧报告；迟到的分析和追问不会覆盖新任务。
- 上传前检查格式、文件大小和图片完整性；服务繁忙或超时时提供明确反馈和重试入口。
- 证据PNG按原图像素尺寸导出，桌面与手机端使用同一套证据定位。
- 演示视觉、规则解释及模型回退会明确显示，不以合成案例证明专业识别能力。
- 视觉引用可定位原图检测框，点击框可返回证据卡；知识引用会展开对应条款及适用条件，支持键盘操作。定位高亮不会改变导出的完整证据图。
- 手机照片统一按EXIF拍摄方向校正，透明区域合成白底；各视觉分支共享同一份RGB像素。报告记录输入图片SHA-256及处理版本，不保留原图或拍摄元数据。
- HTML报告支持证据跳转；打印按钮使用同源脚本，兼容当前内容安全策略。

## 会话与图片边界

历史记录、运行统计、报告及追问仅能由创建记录的浏览器会话访问；报告URL本身不授予访问权限。使用8小时有效的HttpOnly、SameSite=Strict Cookie，HTTPS下附加Secure标志。其他浏览器或清除Cookie后访问旧报告会返回404，请及时导出。记录仍是单进程内存中最近100条，不是持久审计；服务重启或记录淘汰后不可恢复。同一浏览器配置共享会话，这不是账号登录、企业多租户或正式权限管理。

API客户端需要保留响应Cookie；并发调用前先请求一次 `/api/v1/health` 建立会话。系统评测脚本已自动完成该步骤。专业推理前会校验静态JPEG/PNG/WebP的实际内容，单图不超过2400万像素、单边不超过12000像素；损坏文件、动图和超限图片不会进入视觉模型。像素标准化有额外开销，旧性能结果不能视为本版本的重新验证。

队友从 `feat/product-review-20260921` 分支接力，任务与验收方法见 [队友交接](docs/TEAM_HANDOFF.md)。`submission/ican2026/` 的PDF、DOCX和ZIP属于旧快照，本轮未重新生成，不应当作本轮界面更新后的提交包。

## 比赛版一键启动

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_competition.ps1
```

脚本会检查并按固定发布地址下载YOLO权重、校验SHA-256，启用真实火焰/烟雾定位；本地队友分类模型及其隔离环境存在时，同时启用双模型融合；本地LLM运行时与权重存在时，启用离线证据约束推理。首次安装本地LLM可运行 `scripts\setup_local_llm.ps1`。来源、许可、公开指标与本机验证边界见 [模型卡](docs/YOLO_MODEL_CARD.md) 与 [本地大模型说明](docs/LOCAL_LLM.md)。

经过验证的指标、硬件、阈值与可申报边界见 [iCAN参赛定位](docs/ICAN_2026_POSITIONING.md)、[比赛就绪报告](docs/COMPETITION_READINESS.md) 和 [机器可读结果](evaluation/VERIFIED_RESULTS.json)。

## 本地运行

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

浏览器访问 `http://127.0.0.1:8000`。

## 公开评委演示部署

仓库根目录提供 `Dockerfile` 与 `render.yaml`，可在Render通过Blueprint创建公开评委演示站点。该云端档位明确标记为 `public-demo`，默认使用Demo视觉回退，只用于体验图片上传、Agent编排、知识检索、风险解释和报告导出闭环，不作为YOLO、三路分类模型或本地大模型的效果证明。

完整模型在比赛机通过 `scripts/start_competition.ps1` 启动；公开演示部署步骤、资源边界和安全配置见 [公开部署说明](docs/PUBLIC_DEPLOYMENT.md)。Render免费实例可能休眠或重启，正式提交前应预热并复查访问状态。

## 接入 YOLO

安装可选依赖并配置模型路径：

```powershell
pip install -e ".[yolo]"
$env:VISION_BACKEND="yolo"
$env:YOLO_MODEL_PATH="C:\path\to\model.pt"
$env:YOLO_MODEL_VERSION="fire-smoke-v1"
$env:YOLO_CONFIDENCE_THRESHOLD="0.35"
$env:YOLO_IOU_THRESHOLD="0.45"
$env:YOLO_DEVICE="auto"
uvicorn app.main:app
```

模型类别名称中包含 `fire`、`flame`、`smoke` 时会映射为两类标准风险证据。其他类别不会进入当前比赛版风险链路。发布方原始口径采用置信度 `0.25`、NMS IoU `0.30`；VisionGuard比赛版在80张外部固定验证集上选择高召回阈值 `0.10`，并保留人工复核。该调参结果不能替代新的未见测试集。

当YOLO后端未配置或运行失败且 `VISION_FALLBACK_ENABLED=true` 时，系统会回退到Demo检测器，并在结构化结果和页面中明确记录回退原因。权重不进入Git；发布方指标与VisionGuard系统指标严格分开记录。

每次视觉调用返回：

- 模型名称、模型版本和模型文件摘要；
- 置信度阈值、IoU阈值和推理设备；
- 模型推理耗时；
- 检测类别、置信度、检测框、区域占比和证据编号；
- 是否发生Demo回退及其原因。

## 接入队友三路火情分类模型

队友模型用于图像级“存在可见火焰”判断，不输出检测框，也不把烟雾单独判为火焰。
VisionGuard将其作为高精度确认分支，与YOLO定位证据协同，而不是用整图分类替代目标检测。

模型发布包约460MB，必须保存在本地 `models/` 或独立模型目录中，禁止提交到GitHub。
本机已校验并解压到 `models/fire_highscore_submission`。首次运行需要建立独立推理环境：

```powershell
python -m venv .venv-firebench
.venv-firebench\Scripts\python.exe -m pip install -r models\fire_highscore_submission\requirements-lock.txt
```

Windows比赛机建议另建DirectML隔离环境，现有CPU环境不会被覆盖：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_directml.ps1
```

启动前配置：

```powershell
$env:FIRE_CLASSIFIER_ENABLED="true"
$env:FIRE_CLASSIFIER_ROOT="$PWD\models\fire_highscore_submission"
$env:FIRE_CLASSIFIER_PYTHON="$PWD\.venv-firebench-dml\Scripts\python.exe"
$env:FIRE_CLASSIFIER_DEVICE="dml"
$env:FIRE_CLASSIFIER_MODE="persistent"
$env:FIRE_CLASSIFIER_WORKER_COUNT="1"
$env:FIRE_CLASSIFIER_EAGER_START="true"
$env:FIRE_CLASSIFIER_TIMEOUT_SECONDS="180"
uvicorn app.main:app
```

`persistent` 模式会启动可配置的隔离常驻进程池，后续请求复用内存中的模型并由安全队列分配，避免并发响应串线；设置为 `cli` 可继续逐次调用队友的冻结命令行入口。比赛脚本会优先选择CUDA，其次选择DirectML GPU，两种GPU模式均使用1个常驻工作进程以避免重复占用显存；仅CPU模式使用4个工作进程。常驻进程仍使用队友锁定的独立Python环境，主应用不直接加载深度学习依赖。

比赛展示建议同时启用 `FIRE_CLASSIFIER_EAGER_START=true`，把模型加载、DirectML算子编译及YOLO首轮推理前移到服务启动阶段。`/api/v1/health` 中的 `fire_classifier_runtime.state` 为 `ready` 后再开始正式展示；若预热失败，服务仍可启动并在健康接口中记录错误，不会伪装为模型已就绪。

本机RTX 5060 Laptop GPU的DirectML比赛配置中，确定性推理基准在启动预热后的20张固定全链路样本、4并发下全部成功：平均1.02秒、P95 1.43秒、最大1.46秒，未发生视觉回退或分类不可用。启用本地Qwen2.5-1.5B后，同一20张/4并发验证中请求、Agent、LLM采用和证据约束率均为100%，推理回退0次，平均7.29秒、P95 11.17秒。两组结果只适用于记录的硬件、模型、样本与配置，不能外推为所有部署环境的性能。

当分类器不可用或超时时，系统不会把失败解释成“无火”，而是记录为分类分支不可用并继续保留YOLO证据。融合状态包括：

- `confirmed`：YOLO定位和整图分类共同确认火焰；
- `classifier_only`：分类判断有火但没有检测框，按疑似极小或远距离火焰复核；
- `detector_only`：YOLO有候选框但分类未确认，标记模型证据冲突；
- `no_fire_evidence`：两个分支均未形成火焰证据，但不表述为绝对安全；
- `classifier_unavailable`：分类分支不可用，结论仅依据检测证据。

## 可信推理配置

普通开发启动默认使用离线确定性推理器；比赛启动脚本在本地运行时与权重就绪时，默认启用Qwen2.5-1.5B离线推理。Agent调用大模型生成风险解释时，只允许引用输入中存在的视觉证据编号和知识规则编号：

```powershell
$env:REASONING_BACKEND="llm"
$env:LLM_BASE_URL="https://your-compatible-endpoint.example/v1"
$env:LLM_API_KEY="your-key"
$env:LLM_MODEL="your-model"
$env:REASONING_FALLBACK_ENABLED="true"
uvicorn app.main:app
```

本地比赛模型可按固定版本与SHA-256安装：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_local_llm.ps1
powershell -ExecutionPolicy Bypass -File scripts\start_competition.ps1
```

## 系统级实验

队友模型的OOF指标不作为整套系统指标。准备独立测试清单后，可使用 `scripts/evaluate_system.py` 对运行中的接口自动统计分类Precision、Recall、F1、混淆计数、请求与Agent成功率、P95延迟、融合状态和回退次数。完整数据要求及命令见 [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md)。

当接口不可用、配置缺失或大模型引用不存在的证据时，系统会切换为确定性推理，并在响应和页面中明确记录回退原因。密钥不会写入报告或调用轨迹。

当前知识库已接入国家法律法规数据库和应急管理部公开依据，同时保留明确标注的系统方法边界。每条结果记录引用编号、条款、版本、原文链接、权威级别、检索相关度和适用条件；视觉结果不构成火灾、事故隐患等级或执法认定。来源清单见 [docs/KNOWLEDGE_BASE.md](docs/KNOWLEDGE_BASE.md)。

## 比赛演示与报告

首页提供三种合成案例：疑似明火、疑似烟雾和无明确证据。案例由程序生成，只用于稳定展示系统闭环，不属于模型训练集或准确率证明。

每次分析完成后可以：

- 打开适合打印或另存为PDF的HTML安全报告；
- 下载包含完整证据链的JSON报告；
- 下载叠加检测框、类别与置信度的标注证据PNG；
- 在演示看板查看累计分析次数、平均总耗时、平均视觉耗时和回退次数。

系统仅在进程内保存最近100条脱敏记录，服务重启后自动清空。上传图片、原始文件名和大模型密钥不会进入记录。该实现用于比赛演示，不等同于生产级审计存储。

## 测试

```powershell
pytest
```

浏览器交互回归使用独立本地测试服务与演示视觉后端，不需要模型权重或大模型密钥：

```powershell
pip install -e ".[dev,browser]"
python -m playwright install chromium
pytest -q browser_tests
```

2026-09-22本机验证：后端及评测契约测试50项通过，Chromium交互测试11项通过。浏览器首轮有一次页面加载网络暂停，完整复跑11项全部通过。截图与服务日志写入被Git忽略的 `browser-artifacts/`，GitHub CI亦会保存为构建附件。以上结果证明所列回归场景可运行，不证明模型准确率、真实用户满意度或公网部署可用性。

2026-09-23本机复核：后端测试74项通过，Chromium交互测试19项通过；新增覆盖8种EXIF方向、透明像素、无效图片拦截、跨会话读取隔离、移动端知识定位、键盘证据定位、原图导出及CSP下报告打印。已检查桌面、手机端与HTML报告截图；未开展真实模型准确率复测或用户试点。

系统评测采用 `trace-and-reference-v2`：要求六个工具步骤按顺序各完成一次，且解释、复核边界及必要引用完整合法。`grounded_reasoning_rate` 是结构与引用合规率，**不是语义正确率**；新报告会保存解释文本与引用，供人工复核，不追溯改写旧评测数字。

## 安全边界

- 系统输出是风险辅助分析，不替代现场人员判断和企业应急制度。
- 证据不足时返回“需人工复核”，不生成确定性事故结论。
- 上传图片当前仅在内存中处理，不默认落盘。
- FactorySafe 的数字孪生、路径规划、人员核查与事件管理不属于本 MVP。
