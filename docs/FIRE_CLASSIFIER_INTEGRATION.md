# 高精度火情分类模型接入说明

## 模型角色

队友冻结模型由 SigLIP2-Base/256、DINOv2-S/14-336 全局分支和
DINOv2-S/14-336 patch-MIL 小火焰分支组成，通过二层 Logistic Stacking
输出整图可见火焰概率。严格去近重复五折 OOF 指标为：F1 0.943481、
Precision 0.913326、Recall 0.975694，正式阈值为 0.506986200885。

该指标属于图像级二分类，不能与YOLO的mAP直接比较。模型不提供目标框，
因此在VisionGuard中承担“高精度整图确认”角色，YOLO继续承担定位与可视化取证。

## 代码与权重边界

- VisionGuard仓库只保存适配器、融合逻辑、配置和测试。
- 队友发布包保持独立，运行时通过无Shell命令调用冻结推理入口。
- `models/`、`.venv-firebench/` 与 `.venv-firebench-dml/` 已加入Git忽略，权重和隔离环境不会上传GitHub。
- 适配器读取发布包中的模型清单、阈值和bundle摘要，并记录到结构化证据。
- 队友共享包、官方训练数据及外部数据均不复制到仓库。

## 运行配置

```powershell
python -m venv .venv-firebench
.venv-firebench\Scripts\python.exe -m pip install -r models\fire_highscore_submission\requirements-lock.txt

$env:FIRE_CLASSIFIER_ENABLED="true"
$env:FIRE_CLASSIFIER_ROOT="$PWD\models\fire_highscore_submission"
$env:FIRE_CLASSIFIER_PYTHON="$PWD\.venv-firebench-dml\Scripts\python.exe"
$env:FIRE_CLASSIFIER_DEVICE="dml"
$env:FIRE_CLASSIFIER_MODE="persistent"
$env:FIRE_CLASSIFIER_WORKER_COUNT="1"
$env:FIRE_CLASSIFIER_EAGER_START="true"
$env:FIRE_CLASSIFIER_TIMEOUT_SECONDS="180"
```

如发布包位于其他目录，可直接把 `FIRE_CLASSIFIER_ROOT` 指向其中的
`fire_highscore_submission`。如模型bundle单独存放，可设置
`FIRE_CLASSIFIER_BUNDLE` 覆盖默认的 `artifacts/final_ensemble`。

## 常驻推理模式

比赛展示建议使用 `FIRE_CLASSIFIER_MODE=persistent`。系统首次调用时启动隔离的JSON-lines工作进程并校验、加载两套冻结骨干模型，后续请求复用已加载模型。该模式不修改队友冻结包，也不改变融合权重、阈值或图像预处理。

启用 `FIRE_CLASSIFIER_EAGER_START=true` 后，模型加载在Web服务接收请求前完成。健康接口会报告 `ready`、`lazy`、`warming`、`error`、`disabled` 或 `cli`，常驻模式现场演示应仅在状态为 `ready` 时开始。

设置 `FIRE_CLASSIFIER_MODE=cli` 可使用原始冻结命令行入口进行逐次推理，便于结果复核。两种模式必须得到相同的融合分数和判断；本机真实权重对照样例均得到 `0.964269`。

当前比赛机采用DirectML GPU单常驻进程，并在服务接收请求前完成三轮算子预热；YOLO与分类分支同时启动推理。确定性证据推理基准在20张固定全链路样本、4并发下平均1.02秒、P95 1.43秒、最大1.46秒。CPU环境继续作为独立回退保留。该记录不是启用本地大模型后的端到端时延，也不是跨设备性能结论；正式申报必须同时披露运行配置、固定测试集、硬件、预热条件与并发数。

## 融合安全规则

1. 分类阳性且YOLO有框：形成双模型确认的火焰风险证据。
2. 分类阳性但YOLO无框：生成无定位分类证据，按极小或远距离火焰进入人工复核。
3. 分类阴性但YOLO有框：保留检测框并标记证据冲突，不自动删除YOLO证据。
4. 两个分支均阴性：只能表述为未形成可见火焰证据，不能表述为现场安全。
5. 分类进程失败或超时：记录不可用原因，不把异常降级为阴性预测。
6. 烟雾证据独立处理，smoke-only不能被分类分支映射成火焰阳性。
