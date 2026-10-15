# 近期代码导航

2026-10-16整理。实验按用户要求暂停；阅读和CPU验证不会启动实验。主设计仅维护 [model_and_experiment_plan.md](../../design/model_and_experiment_plan.md)，结果及限制见[近期实验报告](../../output/experiment_record/improvement_to_50000/近期实验报告.md)。

## 当前最佳方案

用户反馈公榜47291.34608，PCA候选034010，开发验证52201.21；原50000目标达到，40000未达到。`pe_whole_pca_bandwidth_kernel.py`在训练折拟合PCA并按训练距离中位数校准带宽；`pca_kernel_component.py`显式处理验证回放、8000标签全量重拟合与测试变换。

| 职责 | 入口 |
| --- | --- |
| 冻结特征原价分组核回归 | `frozen_kernel_search.py` |
| 图像特征读入、冻结融合全量准备 | `visual_kernel_submission.py` |
| 普通核/PLS/PCA显式适配 | `visual_kernel_components.py`、`pca_kernel_component.py` |
| 训练OOF凸融合与增量门槛 | `encoder_oof_fusion.py`、`pe_giant_incremental_fusion.py`、`pe_pca_incremental_fusion.py` |
| 已核验测试特征复用 | `infer_mixed_verified_features.py` |
| 全量核身份检查、独立3000条价格回放 | `audit_visual_kernel_refit.py`、`audit_visual_submission.py` |

## 表示与对照

| 方向 | 文件 |
| --- | --- |
| PE-G加载、权重/资源预检、原视图 | `pe_core_giant_preflight.py`、`pe_core_giant_encoder.py`、`pe_core_giant_reference.py` |
| 原视图＋整图 | `pe_giant_whole_view.py`、`pe_core_giant_whole_encoder.py`、`audit_pe_giant_whole_feature_adapter.py` |
| AIMv2/EVA独立视觉参照 | `aimv2_*`、`eva_clip_*` |
| PLS价格度量、价位加权、核网格 | `visual_price_metric_kernel.py`、`weighted_visual_kernel.py`、`refined_whole_kernel.py` |
| 视图权重、PCA与半径对照 | `pe_whole_view_weight_kernel.py`、`pe_whole_pca*.py` |
| LoRA预检、查询适配与同空间读出 | `pe_giant_lora_preflight.py`、`pe_lora_*` |
| 第40块预检、编码器、全量提取/新进程审计 | `pe_giant_mid40_preflight.py`、`pe_giant_mid40_encoder.py`、`pe_giant_mid40_reference.py` |
| 数据质量与场景误差 | `current_fusion_scene_audit.py`、`whole_fusion_scene_audit.py`、`pe_giant_frame_geometry_audit.py` |
| 只读汇总结果 | `improvement_summary.py` |

文件名通配用于导航，不是批量执行指令。只登记或未通过门槛的代码不代表有效模型；MLLM本地参照独立排除在课程主线之外。

## 环境与回放

基础依赖见 `requirements.txt`，近期视觉扩展环境见 `requirements-visual.txt`；Python3.11。数据/划分/缓存经 `local_paths.json` 或 `CV_*`环境变量配置，示例在仓库根目录。数据及权重不随Git上传，普通新checkout不能直接复现全部结果，需要对应哈希的本地产物。

23份关键编排/审计源码原样保存在 [archive快照](../../archive/experiment_orchestration/20261016_snapshot/README.md)，`manifest.json`列出原`.local`路径和SHA256。它们绑定历史ID，不能从归档目录直接执行；也不要为整理路径而修改已登记源码并冒充原运行。

第40块暂停在600/8000，现有入口不支持续跑，不能直接重启；恢复须用户指令并先实现保存块核验/接续。此分支只整理与推送，不恢复GPU任务。

只重新生成文本/JSON汇总可运行：

```powershell
python scripts/experiments/improvement_summary.py
```

该脚本读取公开记录，不读取图片、不加载模型、不启动训练；公榜来源明确标记为用户反馈。
