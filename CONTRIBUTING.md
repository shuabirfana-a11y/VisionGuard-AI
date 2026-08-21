# VisionGuard AI 协作规范

## 项目边界

VisionGuard AI 是独立项目。任何提交不得复制、修改或依赖 FactorySafe 的源码、配置、模型资产和仓库历史。

## 分支与提交

- 从 `main` 创建短生命周期分支：`feat/*`、`fix/*`、`docs/*`、`test/*`。
- 禁止直接向 `main` 推送功能代码。
- 每个Pull Request只解决一个明确问题，并关联对应Issue。
- 提交信息建议使用 Conventional Commits，例如 `feat: add validated yolo model adapter`。

## Pull Request 验收

- `pytest`全部通过；
- 不包含API密钥、个人信息、上传图片和未授权数据；
- 模型结果附带版本、阈值、数据范围和可复现实验记录；
- 新增能力区分“已完成”“实验性”“计划”；
- 不把合成案例当作真实指标证据；
- 不把内部草案知识当作正式法规。

## 数据与模型

数据集和 `.pt` 权重不得进入普通Git历史。仓库只保存：

- 数据来源、许可和划分说明；
- 模型名称、版本、摘要和获取方式；
- 训练配置与评测脚本；
- 可复现的汇总指标和错误案例清单。

大文件应使用经团队确认的Git LFS或对象存储方案。

