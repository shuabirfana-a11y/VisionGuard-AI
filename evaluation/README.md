# 独立跨来源评测集

该目录保存 VisionGuard AI 比赛版的固定评测清单与真实运行结果。图片位于被 Git 忽略的 `evaluation/data/`，通过脚本按固定版本、固定哈希和固定选样盐重新取得。

## 数据来源

1. `shahriar-5/IFireSmoke@22b3c06db783d3e75c74234faa511020912eac64`
   - 固定测试集抽取30张火焰、30张烟雾；
   - Hugging Face卡片声明CC BY 4.0，压缩包内README声明MIT；按更严格的CC BY 4.0履行署名；
   - 用于室内火焰/烟雾跨来源测试。
2. `fireviewer/fire-smoke-detection-corpus-v1@85ad763e6275537386f7eefdae5e3a18a55f1c71` 的 Pyro-SDIS 子集
   - 抽取10张烟雾、10张无目标场景，每张来自不同序列；
   - Apache-2.0；
   - 用于远距离烟雾、小目标与误报测试。

当前YOLO模型卡声明训练源为D-Fire，上述来源用于跨数据源验证。但队友分类模型未提供完整训练图片来源，因此分类结果应表述为“外部固定集验证”，不能绝对声称与其训练数据零重叠。

## 固定选样规则

- 使用 `visionguard-independent-evaluation-v1` 作为固定哈希盐；
- IFireSmoke按原始图片组去重后哈希排序选取；
- Pyro-SDIS跨测试集等距取候选，并按序列去重后哈希排序选取；
- 不依据模型预测结果增删样本。
- Pyro-SDIS清单同时记录语料中的原始图像哈希与数据预览接口重新编码后的实际JPEG哈希，不混用两种口径。

## 重建与运行

```powershell
.venv\Scripts\python.exe scripts\prepare_independent_evaluation.py
powershell -ExecutionPolicy Bypass -File scripts\start_competition.ps1
.venv\Scripts\python.exe scripts\evaluate_components.py `
  --manifest evaluation\independent_manifest.csv `
  --output evaluation\component_results.json
.venv\Scripts\python.exe scripts\evaluate_system.py `
  --manifest evaluation\full_chain_manifest.csv `
  --concurrency 4 `
  --output evaluation\independent_results.json
```

输出分别报告目标框级、图片级、分类分支、接口成功率、Agent成功率、回退情况和延迟，不合并为含义模糊的单一“准确率”。
