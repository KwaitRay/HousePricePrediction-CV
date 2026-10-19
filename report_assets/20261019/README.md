# Report evidence pack — 2026-10-19

供两位组员写报告使用，不是最终 code submission 文件夹。下载本目录后打开 [index.html](index.html)，即可浏览七张可复制表格、推荐图、图注和完整证据索引，无需安装依赖。

- [中文草稿（共享文档，正文主版本）](https://docs.google.com/document/d/1osyQDBjpkzbW7s0LzdxijcCU7Aq3U85L345mT6FeMSk/edit)
- [本次中文正文快照](draft_zh.html)；修改请优先在云端协作，快照不自动双向同步。
- [全部表格显示值](tables/report_tables.json)；完整精度见 `data/`、`photometric/`、`ablation/`。
- [逐文件来源与 SHA256](manifest.json)；[中文图注](CAPTIONS_ZH.txt)。

## 写报告从哪里取材料

| 报告部分 | 首选材料 | 注意事项 |
| --- | --- | --- |
| 方法与实验总览 | `tables/overall.html`、`tables/OVERVIEW.html` | 区分三种子指标均值、单次已选配置、冻结融合 |
| 预处理 | `tables/preprocessing_summary.csv`、`data/preprocessing_decision.json` | 最终保留拉伸；频域单次最好不代表复核胜出 |
| 训练调优 | `data/training_aggregates.json`、`figures/training_comparison.png`、`data/stages/04_training/` | 所有阶段决定和失败实验均保留；不要挑最好曲线代替正式汇总 |
| 增强对照 | `tables/AUGMENT.html`、`tables/PHOTO.html`、`photometric/` | 新光度实验使用新对照，不能覆盖旧实验 |
| 结构筛选 | `tables/architecture.html`、`data/architecture.json` | 单种子AlexNet改动，不是完整经典网络排名 |
| 最终模型消融 | `ablation/`、`data/final_ablation.json` | 删整个分支，不是纯PCA/PLS操作消融 |
| 最终模型误差 | `error_analysis/`、`data/final_error_analysis.json` | 不要把旧AlexNet诊断图当最终模型结果 |
| 后续未晋级尝试 | `data/continuation_screen.json`、`data/mid40_fusion.json` | 保留负结果，没有替换原提交 |
| 各组训练过程 | `archive_curves/` | 包含历史运行，按阶段决定筛选正文所用曲线 |

## 必须一致的口径

- 价格和MAE单位：千美元；MSE单位：(千美元)²。不是澳洲房产数据。
- 固定开发划分：6,399训练 / 1,601验证；尾部 >1,300，共159张。
- 最终模型版本：`20261015T034010Z_0c94a543`。
- 固定验证 MSE **52,201.21**；训练侧 meta-crossfit **52,210.22**；历史已提交公开榜 **47,291.34608**。三者不能混用，公开榜不是当前排名或私榜。
- 最终权重在训练OOF上学习，不使用验证标签拟合。meta-crossfit并非完整嵌套无偏评估；开发验证已被反复使用。
- 水平翻转三种子均值改善0.75%，只有2/3配对改善，不能写成每次稳定提升。
- 光度补充单种子：对照131,389.94，亮度131,433.91，对比度128,860.53。对比度总体−1.93%，尾部+17.14%。未做多种子复核，未应用于最终融合。
- 最终PCA配置256维、白化指数0，不能写成采用了白化。
- 删除PE-L PLS后验证略好，但meta略差，不支持宣称所有分支均有稳定独立贡献。
- 缓存特征CPU回放通过，不等于从原图重新跑过编码器的端到端复现。

## 范围和版本

本分支基于 `664393e`（亮度/对比度实验，PR #5）新建；没有合并PR或修改模型代码/最终预测。最终模型证据来自 `feature/best_model` 的已核验包，来源记录在manifest。

不包含原始图片、带原图的案例拼图/遮挡图、逐图标签或预测、模型权重、特征缓存、账号信息及design目录。案例原图仍从课程数据和受控本地产物取用，不在Git中重复发布。

正文案例与图选择属于描述性分析，不是随机样本或因果验证。探索性视觉覆盖图来自每价位100张、共400张训练侧抽样，不能写成全量人工普查。

云端当前是中文工作稿。最终需两人审核并压成IEEE格式：A4正文最多4页，参考文献可延至第5页。

`build_pack.py` 为原工作站的整理脚本，依赖未公开的 `.local` 聚合源文件；队友直接使用已提交材料即可。更新素材时保留旧数值及来源，不要无标注覆盖已发表的实验记录。
