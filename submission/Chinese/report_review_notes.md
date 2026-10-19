# 当前提交版与历史核对记录

最新状态：署名已填写，英文为四页正文加一页参考文献。报告已按问题、方法与对照、结果及下一问题重组；表格补齐三个编码器分支和对照，PLS 未改善自身验证 MSE 的结果如实保留。图 3 补充按第 25 绝对误差百分位选取的成功案例 518，与失败案例 2138 对照。

版式更新：Table 1 改为单栏三列表格，仅显示 MSE、MAE 点估计；表注区分三种子均值与冻结拟合，标准差仍保留在原 CSV。房价散点和分价位误差改为单栏三面板；PCA 诊断及多模型热图保留跨双栏。未缩小正文或修改 IEEE 页边距。

新增 `../English/code/reproduce_final.py`，提供 check、extract、fit、predict 四阶段入口，绕过历史运行编号。已通过元数据核验，并用原有头和特征核验全部 3000 条 CPU 预测，最大绝对差约 2.31e-11。本轮未重新执行 GPU 提取和全量回归拟合。代码 ZIP、两份 PDF 和英文整体包均已更新，不包含模型权重、特征数组和课程原图。

以下保留早期核对记录，其中“作者待填”“代码包未核验”“融合消融未独立复核”等状态已由上述更新替代，不能视为当前未完成项。

日期：2026-10-19。正文为英文，主文件 `report_draft.tex`；`report_draft.pdf` 为实际编译的 IEEE 双栏预览，`report_draft.md` 为方便讨论的同内容阅读版。当前英文版为 **4 页 A4 正文，第 5 页仅参考文献**，符合 Project 上限；中文版共 4 页。作者姓名、学号仍为占位内容，不能直接当作提交终稿。

## 1. 采用的要求来源

- 原始要求：`../../COMP90086_2026_Project.pdf`，完整读取 5 页。任务与允许方法见第 1–2 页；提交、格式、方法与分析要求见第 3 页；评分见第 4 页。
- 老师补充：`../../discussion/discussion_content.md`。#156 建议 Introduction、Methods、Results、Discussion，不需要 Literature Review，并强烈建议不写摘要。#162 要求报告讨论到的工作都包含代码。#144 强调理解、应用、错误分析与结果解释，不以预测完全准确为目标。
- 评分细则：`../../report_rubric.txt`，写作 5 分、方法及理由 10 分、实验及评价 10 分。
- 上述均为工作区提供的课程材料副本；没有登录 Canvas 核对后续公告。它们的日期不能替代最新课程通知。

## 2. 要求如何落实

| Project / Ed 要求 | 初稿中的对应内容 | 状态与边界 |
|---|---|---|
| IEEE 会议短论文格式、A4、正文不超过 4 页；参考文献可延伸到第 5 页 | `IEEEtran` 的 `conference,a4paper` 选项；默认 10 pt、双栏及模板页边距 | 已编译，英文正文 4 页，第 5 页仅参考文献；没有缩小正文或改页边距 |
| 说明方法、设计选择及 CV 理论理由 | Methods 的参照模型、受控 CNN 实验、冻结表示、视图与核、融合及最终重拟合 | 同时写出关键训练设置、特征维度、核公式、带宽校准及融合权重 |
| 解释从基本方案到改进方案的实验步骤 | Results 按从零与传统参照 → 冻结视觉表示 → 视野与距离尺度 → 融合展开 | 训练调优、增强与架构实验保留；没有只写最终模型 |
| 比较评价、最终结果 | Table I；开发验证 MSE 52,201.21、MAE 159.58；公榜 47,291.35 | 公榜来自人工提交反馈；没有冒充独立读取的榜单或私榜分数 |
| 错误分析：何时成功、何时失败、解释与改进方向 | 价位偏差、平方误差集中、最后一步改善的分段贡献、具体图片案例、遮挡敏感性 | 图 2 展示分段误差；案例不是随机样本，机制解释仅为假设 |
| 表图说明充分 | 一张精简结果表；三组图依次为 PCA/带宽诊断、预测散点与分价位误差、三模型区域遮挡热图 | 散点包含全部 1,601 个验证样本，热图统一色标；有样本数、单位、对照及分析边界 |
| 自己设计方法，允许通用预训练特征 | 冻结视觉编码器 + 自定义视图、价格监督 PLS、PCA 带宽校准、OOF 融合 | 最终方案不使用 MLLM、额外 SoCal 数据或已有房价任务权重 |
| 禁止测试标签、测试训练 | Methods 明确折内拟合、训练 OOF 选择、固定后 8,000 标签重拟合、测试仅推理 | 开发验证反复使用的局限如实说明；没有称为全嵌套无偏评价 |
| 不写摘要、不需要单列综述 | 直接从 Introduction 开始，采用四大模块 | 已遵守老师 #156 的建议；引用在相关方法处给出 |
| 代码与报告分开提交；报告讨论的实验均提供代码 | 下节列出已有实现与记录 | 本轮撰写没有新建或验证最终 Canvas 代码 ZIP，提交前须核查最终包，不能把本轮当作打包验收 |

## 3. 各模块内部的逻辑

**Introduction**：估价需要房屋属性、地理与交易环境 → 图像能够提供外观与场景线索，但信息不完整 → 手工描述子、CNN 与大规模预训练表示的发展 → SIFT 参照、AlexNet 受控实验、冻结编码器三阶段研究路线 → 初步结果及主要剩余误差。CLIP、SigLIP2、PE 的研究背景依据原始论文，不将通用迁移研究写成未经验证的房价预测行业排名。

**Methods**：先说数据和评价边界，再描述传统与从零参照及受控实验；随后说明为什么采用冻结特征，如何保留视野、改变距离和监督方向，最后交代 OOF 融合与最终重拟合。关键公式写进正文，读者无需查看代码才能理解基本算法。

**Results**：先证明图像包含预测信号，再解释哪些早期收益跨种子保持、哪些没有；冻结特征是完整路线对比，不是纯单因素。PE-G 整图对照支持覆盖价值，PCA 对照重点说明距离尺度混杂。最后用融合消融与分价位错误限制总体分数的解释。

**Discussion**：回扣引言的三条路线：手工局部特征的证据组织及空间局限 → 从零 CNN 的监督规模与容量匹配 → 冻结预训练表示的迁移、视图与任务适配 → 综合为信息量、覆盖和估价映射的共同作用，进一步讨论视觉信息的边界与后续研究。移除重复指标和硬件耗时，保留评价局限；不将跨路线比较解释为纯编码器的因果消融。

## 4. 数值与来源

相对路径以 `Final_Assignment` 为根：

| 内容 | 优先依据 |
|---|---|
| 早期三种子均值、标准差、配对改善 | `submission/report/figures/analysis_sources/early_experiment_paired_deltas.csv`、`early_experiment_results_summary.csv` |
| 原始 padded AlexNet、均价参照 | `cv-house-price/output/experiment_record/02_baselines/实验报告.md` |
| SIFT 三种子 | `cv-house-price/output/experiment_record/03_traditional/实验报告.md`，正式结果部分 |
| PE-G 单/双视图与 PCA | `submission/report/figures/analysis_sources/pe_g_view_pca_results_summary.csv`；`cv-house-price/output/experiment_record/19_image_embedding_reference/pe_giant_whole_pca_bandwidth_kernel/20261015T033416Z_4b848f75/registration.json` 与 `results.json` |
| 最终指标、meta-CV | `cv-house-price/output/experiment_record/15_fixed_ensembles/pe_whole_pca_incremental_fusion/20261015T033810Z_8aa6ec62/results.json` |
| 最终权重、特征维度与全量重拟合 | `cv-house-price/output/experiment_record/08_final/visual_kernel_ensemble/20261015T034010Z_0c94a543/registration.json` |
| 公榜反馈 | 同目录 `kaggle_feedback.json`，标注未由工具独立访问 Kaggle |
| 最终模型分段与案例 | `submission/report/figures/6.txt`、`7.txt` 和 `1.png`–`5.png` |
| 融合移除消融 | 组员提供的 `submission/report/figures/8.png`；本轮逐项读取核对，没有重新拟合消融 |
| 遮挡协议、数值及替换一致性 | `submission/report/figures/analysis_sources/occlusion/` 和 `analysis_figures_readme.md` |

**权重映射特别核对**：以正式导出登记为准，顺序为 NaFlex **0.1080651221**、PE-L PLS **0.0808055778**、PE-G PCA **0.8111293001**。旧主设计摘要曾交换前两项，初稿没有沿用该错误，也没有改写历史记录。

**评价特别核对**：去掉 PE-L 后验证 MSE 52,194.81 略优于完整融合 52,201.21，但 meta-CV 52,244.24 略差于 52,210.22。正文没有把完整融合描述为所有评价下都最优，或声称所有分支都不可替代。

**单位与统计**：价格、MAE、RMSE、偏差为千美元，MSE 为（千美元）²；三种子平均指标不等于三模型预测平均后的集成指标；SD 不是置信区间；单种子未晋级不等于普遍无效。PCA 固定带宽的 77,771.43 是 OOF，不是验证 MSE。

## 5. 讨论内容的代码对应与提交前核查

已有实现位于 `cv-house-price/scripts/experiments/`：

- 数据划分、传统参照与 CNN：`preparation/`、`traditional/`、`models/`、`training/`、`run.py`、`train_cached.py`、`train_augmented.py`、`architecture_stage.py`、`encoder_stage.py` 及对应 `configs/`。
- 通用冻结表示与视图：`pretrained_image_reference.py`、`siglip_naflex_reference.py`、`pe_core_multiview.py`、`pe_core_giant_reference.py`、`pe_giant_whole_view.py`。
- PLS、PCA 与融合：`visual_price_metric_kernel.py`、`pe_whole_pca_bandwidth_kernel.py`、`pe_pca_incremental_fusion.py`、`fixed_kernel_ensemble.py`、`visual_kernel_submission.py`、`pca_kernel_component.py`。
- 本轮补充图表及区域敏感性：本目录 `build_analysis_figures.py`、`build_results_tables.py`、`build_compact_results_table.py`、`build_occlusion_figures.py`。正文紧凑面板由 `build_report_analysis_panels.py` 从归档 CSV 重绘，输出 PNG、SVG 和正文引用的 PDF；来源哈希见 `figures/analysis_sources/report_analysis_panels_provenance.json`。本次没有训练或 GPU 推理。

以上是代码位置索引，不保证所有旧队列都能从主入口一键恢复。最终 ZIP 必须含已讨论实验的实际源码、依赖与运行说明、必要的划分及结果文件，并排除提供的训练/测试原图。特别需要组员补齐 **8.png 的融合移除分析脚本/原始数值文件，以及 1–5.png 的原始生成脚本**；当前目录有图与文字，尚未定位这些组员图的生成源码。若不能补齐，终稿应移除对应未附代码的实验内容。图像文件不能替代实验代码。

## 6. 尚待完成，不能默认已解决

1. 填写两位作者的真实姓名、学号，核对课程组名；当前 PDF 顶部明确显示占位内容。
2. 亮度和对比度已纳入中英文稿的 Methods 与 Results，使用 PR #5 中新跑的 seed 2026 无增强对照（MSE 131,389.94），不与旧三种子对照直接配对。两项均以概率 1、倍率 U(0.9,1.1) 独立应用。对比度总体 MSE 改善 1.93%，但高价尾部 MSE 恶化 17.14%；亮度总体 MSE 基本不变。保留原九行总览表，通过正文补充这组独立比较。饱和度仍待提供。
3. 与组员核对融合移除消融的重拟合协议、原始数值与代码；目前结论依据已提供的 8.png，未独立重跑。
4. 最终代码 ZIP 需要单独验收，尤其是组员已有图的生成代码及所有讨论实验。原图不可装入 ZIP，README 说明预期数据路径。
5. 作者信息、光度实验或图表变化后必须重新编译检查四页上限。不得默认本次四页保证后续版本仍为四页。
6. 如引用课程作业原始数据来源，只引用项目提供的信息，不下载完整公开 SoCal 数据补充样本或价格。

## 7. 编译、阅读版与检查

原始文献补充：中英文正文现有 25 篇编号引用，按首次出现顺序排列，覆盖 SIFT、视觉词袋、Ridge/KRR、Adam/AdamW、Dropout、余弦衰减、初始化、卷积结构、Transformer、PCA/PLS、预训练编码器、融合和遮挡分析。PDF 中的文献标题可点击访问原始论文（PLS 1966 历史书章未获得公开全文，不附错误链接）。全部 39 条正文与备查来源见 `references_experiment_sources.md`，可导入的文献库为 `references_all_experiments.bib`，机器可读元数据为 `references_catalog.json`。`build_references.py` 统一生成编号、文献与清单，两个阅读版不再手工写死四篇文献的编号。未在正文展开的 DINOv2、BLIP、StreetCLIP、AIMv2、EVA-CLIP、LoRA 等保留在备查清单，避免引用与正文内容脱节。

光度实验来源为 PR #5、提交 `664393e2342fe5b838db6649554b31bc64a1c763`，本轮仅整合已 review 的结果，没有重新训练。数据副本与来源哈希见 `figures/analysis_sources/photometric_screen_20261019/`。对应执行与导出代码仍在 PR 分支，最终打包时需包含 `photometric_screen.py`、`export_photometric_screen.py` 及其共享依赖；本轮不自动合并 PR。Review 指出的导出器固定 40 轮问题不影响已保存的三组 40 轮结果，但应在复用前修正。

2026-10-19 语言修订：正文删除了重复的合规清单、核验流程和防御性声明，改用“设计动机 → 方法 → 结果 → 分析”的连贯叙述。实验设置集中于 Methods，评价局限集中于 Discussion。作者信息、待补实验和代码打包检查继续保留在本说明，不混入论文正文。英文 Markdown 阅读版也移除了编辑提示。

工作目录为本文件所在目录。标准 LaTeX 环境可运行 `pdflatex report_draft.tex` 两次；也可上传 `.tex`、`figures/report_compact_results_table.tex` 及以下三张被引用 PDF 到 Overleaf，选用标准 IEEEtran：`report_pca_distance_bandwidth_oof.pdf`、`report_prediction_scatter_price_band_errors.pdf`、`report_pe_g_occlusion_model_comparison.pdf`。正文与坐标使用矢量输出，照片和散点保留嵌入图像。

本机已将便携 Tectonic 下载到 `.tools/`，没有改动实验 Python 环境：

```powershell
.\.tools\tectonic.exe report_draft.tex --keep-logs
..\..\cv-house-price\.venv\Scripts\python.exe -X utf8 prepare_draft_review.py
```

`prepare_draft_review.py` 生成 Markdown 阅读版和 `draft_validation.json`，检查 A4、总页数、主结构、图表依赖、正式权重映射、最终验证指标及文件哈希。PDF 字体与排版另行逐页目视核查，不用文本提取代替视觉检查。渲染预览在 `tmp/pdfs/`，工具和中间文件不应加入课程提交包。
