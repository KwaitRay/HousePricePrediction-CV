# 按实验顺序运行房价预测

> 2026-10-03迁移说明：当前入口、路径配置和暂停状态以[仓库首页](../../README.md)为准。以下保留原实现的接口说明，旧输出路径已由 `local_paths.json` 映射到本地目录，虚拟环境需自行创建。历史队列不随迁移自动运行；阶段阅读报告位于仓库 `output/experiment_record`，新运行原始产物位于本地 `records_root`。03为传统参照，04为训练调优，后续目录顺延。

本目录实现[设计文档](../../design/model_and_experiment_plan.md)中的实验。共用脚本负责配置、训练和评价；不同路线的专用实现分开保存。所有网络从零训练，不下载预训练权重。候选运行不自动代表获胜，下一阶段读取人工登记的选定配置。

2026-10-01 准备验收更新：新运行统一写入 [output/experiment_record](../../output/experiment_record/README.md)，完成或失败后自动追加对应阶段 Markdown。下文旧 `outputs/` 路径仅代表历史产物或示例；当前基础配置使用 `../../output/experiment_record`，实际结果位于 `阶段/runs/实验名/运行时间/`，阶段文档位于 `阶段/README.md`，专属日志归入 `阶段/logs/`，通用环境与测试位于 `shared/`。代码 review、GPU 安装与实际验收结果见该索引。

新增准备命令（工作目录仍为本目录）：

```powershell
.\.venv\Scripts\python.exe -X utf8 preflight.py audit --device cpu
.\.venv\Scripts\python.exe -X utf8 preflight.py learn --device cuda --samples 32 --epochs 60 --workers 0
.\.venv\Scripts\python.exe -X utf8 preflight.py learn --device cuda --samples 32 --epochs 2 --workers 2
```

`audit` 复核冻结划分并检查全部11000张图片的解码和基础变换；不覆盖划分。`learn` 仅从训练侧固定抽取图片，不使用验证或测试标签，保留完整AlexNet、学习率、随机失活、正则化和有效批量。`workers=0` 减少Windows小样本反复启动进程的成本，`workers=2` 短检查核对正式加载设置；所有覆盖保存到配置。诊断结果不会进入正式性能比较，也不能选为阶段模型。默认60轮诊断的预登记验收线为训练MSE至少下降50%，短检查仅验收加载链路。

## 一 目录与入口

以下为脚本文件夹的结构。先在阶段配置目录中找到中文实验名称，再按后面的对应表查找实现文件；可复用的逻辑只保留一份。

```text
Final_Assignment/
└── scripts/
    ├── analyze_dataset.py              原有数据探索与完整性分析
    ├── README.md                       原有分析脚本说明
    ├── requirements.txt                原有分析依赖
    ├── tests/                          原有分析测试
    ├── .venv/                          原有分析环境
    └── experiments/                    实验代码的统一位置
        ├── run.py                      所有阶段共用的命令入口
        ├── generate_configs.py         配置目录与索引生成器
        ├── README.md                   本说明
        ├── VERIFICATION.md             已完成的检查及其限制
        ├── requirements.txt            实验依赖
        ├── .venv/                      独立实验环境
        ├── configs/
        │   ├── base.json               基础结构、输入及训练设置
        │   ├── README.md               全部实验配置的中文索引
        │   ├── 00_preparation/         划分与六项预处理备选
        │   ├── 01_baselines/           平均房价、传统特征、AlexNet
        │   ├── 02_training/            学习率、优化器、正则化等
        │   ├── 03_augmentation/        单项增强与组合
        │   ├── 04_architecture/        小卷积核、多尺度、残差
        │   ├── 05_encoder/             汇总头、单层与双层编码器
        │   └── selected/               前一阶段已确认保留的配置
        ├── preparation/
        │   ├── split.py                数据检查、重复分组、冻结划分
        │   └── images.py               图像处理、在线增强、图片预览
        ├── traditional/
        │   └── pipeline.py             SIFT、词袋、附加特征与回归
        ├── models/
        │   ├── alexnet.py              AlexNet 及三类卷积结构改动
        │   └── encoder.py              小型聚合头及 Transformer 编码器
        ├── training/
        │   └── engine.py               共用训练、优化、早停与权重保存
        ├── evaluation/
        │   └── report.py               指标汇总、成对比较与重采样
        ├── common/
        │   ├── config.py               配置继承、校验、运行与环境记录
        │   └── metrics.py              三个指标、价格变换、预测保存
        ├── tests/
        │   └── test_experiments.py     模块及完整运行流程测试
        ├── artifacts/                  划分、词典缓存、预览等产物
        └── outputs/                    每次实验的配置、权重、指标和图表
```

| 要找的实验 | 配置目录 | 对应实现 |
| --- | --- | --- |
| 划分与预处理 | `configs/00_preparation/` | [数据划分](preparation/split.py)、[图像处理](preparation/images.py)；预处理效果比较复用神经网络训练入口 |
| 平均房价基线 | `configs/01_baselines/mean_price.json` | [统一入口](run.py)中的 `experiment` 分支 |
| SIFT 与手工特征回归 | `configs/01_baselines/` 中传统路线配置 | [传统方法实现](traditional/pipeline.py) |
| AlexNet 基线 | `configs/01_baselines/alexnet.json` | [网络结构](models/alexnet.py)、[共用训练](training/engine.py) |
| 训练参数比较 | `configs/02_training/` | [共用训练](training/engine.py)；随机失活和初始化定义在[模型文件](models/alexnet.py) |
| 数据增强比较 | `configs/03_augmentation/` | [增强操作](preparation/images.py)，训练仍使用同一脚本 |
| 小卷积核、多尺度与残差 | `configs/04_architecture/` | [卷积模型与模块](models/alexnet.py) |
| 聚合头及编码器 | `configs/05_encoder/` | [特征处理模块](models/encoder.py)，由 AlexNet 模型接入 |
| 所有方法的统一评价 | 各运行已保存的完整配置 | [指标定义](common/metrics.py)、[结果比较](evaluation/report.py) |
| 全量重训与预测导出 | `configs/selected/final.json`，选定后生成 | [统一入口](run.py)中的 `retrain` 与 `predict` 命令 |

代码、配置和本说明已从原 `Final_Assignment/experiments` 整体迁入此处。历史运行记录中的原始绝对路径按当时状态保留，不改写历史证据；新运行使用当前位置和已更新的数据相对路径。当前历史产物只有平均价基线与小样本检查，没有需要迁移加载的正式全量模型。

| 位置 | 用途 |
| --- | --- |
| `run.py` | 准备、预览、训练、阶段选择、结果比较、全量重训与导出的统一入口 |
| [配置索引](configs/README.md) | 全部五十六份配置与中文实验编号；同一个实验可有多个候选参数 |
| `configs/base.json` | AlexNet 基础结构、输入、训练设置与数据路径 |
| `configs/00_preparation/` | 冻结划分与六项预处理备选 |
| `configs/01_baselines/` | 平均房价、传统特征和 AlexNet 基线 |
| `configs/02_training/` | 学习率、优化器、调度、批量、随机失活、正则化与停止规则 |
| `configs/03_augmentation/` | 十项单独增强和组合登记模板 |
| `configs/04_architecture/` | 小卷积核、多尺度与残差 |
| `configs/05_encoder/` | 平均聚合头、单层与双层编码器 |
| `configs/selected/` | 使用 `select` 生成的阶段选择记录，初始为空，不预设实验结论 |
| `preparation/` | 分组划分、图像预处理、在线增强和预览 |
| `traditional/` | SIFT、视觉词袋、空间金字塔、颜色或明暗或梯度统计、岭回归和支持向量回归 |
| `models/` | AlexNet 与可选卷积模块；编码器单独实现在 `encoder.py` |
| `training/` | 共用神经网络训练、梯度累积、正则化、提前停止和记录 |
| `evaluation/` | 跨模型汇总、成对误差和按图片组重采样 |
| `common/` | 配置继承、文件记录、目标变换和三个统一指标 |
| `tests/` | 合成数据与模型行为测试，不使用测试集价格 |
| `artifacts/`、`outputs/` | 冻结划分及每次运行结果，不作为源代码打包 |

配置中的 `experiment.name` 是文档中的完整中文实验编号，`id` 只是安全的目录名。`question` 说明研究问题，`comparison` 说明对照，`changed_fields` 声明允许改动的字段。配置继承时会检查额外改动，避免把优化器或增强变化隐藏在结构实验中。

## 二 环境

使用 Python 3.11 或兼容版本，在本目录建立独立环境，不修改原有 `scripts/.venv`。下面命令以 `Final_Assignment/scripts/experiments` 为工作目录。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

本次检查环境复用了本机已安装的 CPU 版 PyTorch，并在独立实验环境中安装其他依赖；它不是便携环境。正式 GPU 训练前，请按 PyTorch 官方安装说明安装相互匹配的 GPU 版 PyTorch 与 torchvision，再确认能识别显卡。运行会保存实际版本，CPU 检查不能证明 GPU 耗时或显存占用。

`device` 为 `auto` 时优先使用可用 GPU；设为 `cpu` 可强制 CPU，设为 `cuda` 时无可用 GPU 会报错。首次实现验证使用单精度，暂未引入混合精度。原有分析环境保持不变。

## 三 实验准备与初始基线

```powershell
.\.venv\Scripts\python.exe -X utf8 run.py prepare configs/00_preparation/prepare_split.json
.\.venv\Scripts\python.exe -X utf8 run.py train configs/01_baselines/mean_price.json
.\.venv\Scripts\python.exe -X utf8 run.py train configs/01_baselines/alexnet.json --smoke
.\.venv\Scripts\python.exe -X utf8 run.py train configs/01_baselines/sift_ridge.json --smoke
```

划分文件已存在时，准备命令会拒绝覆盖。需要修改分组时，在配置中改用新的划分路径；正式运行都会核对训练标签、分组和图片内容。训练和测试清单分别要求八千和三千条，只有合成测试才修改这个数量。

`--smoke` 使用八张训练图片、最多四张验证图片、一个训练轮次；传统词典临时缩小到八个词，实际覆盖设置保存在运行配置。它只检查数据、梯度、保存和评价，不代表泛化性能，不允许登记为阶段优胜者或进入正式比较表。使用正式 AlexNet 的完整回归头，CPU 上仍可能占用较多内存。

检查通过后，去掉 `--smoke` 才是正式实验。传统路线先运行局部特征基线，再按问题选择空间金字塔或颜色补充。预处理备选按需要独立运行，不自动遍历全部候选。

```powershell
.\.venv\Scripts\python.exe -X utf8 run.py train configs/01_baselines/alexnet.json
.\.venv\Scripts\python.exe -X utf8 run.py train configs/00_preparation/preprocess_01_stretch.json
```

预处理保持原方案时，登记正式 AlexNet 结果；替代输入有效时，登记对应预处理实验结果。不要把平均房价或传统回归结果登记到神经网络训练阶段。

## 四 按阶段登记选择并继续实验

下面的运行目录是示意，替换成命令实际输出的目录。`select` 只复制配置及选择理由，不复制训练权重；新实验仍重新初始化。

```powershell
.\.venv\Scripts\python.exe -X utf8 run.py select outputs/alexnet/实际运行目录 --slot input --reason "输入对照复核后保留基础预处理"
.\.venv\Scripts\python.exe -X utf8 run.py train configs/02_training/tune_01_lr_3e-05.json
.\.venv\Scripts\python.exe -X utf8 run.py train configs/02_training/tune_02_sgd_0.01.json
```

| 选定记录名称 | 何时生成 | 供哪些后续配置使用 |
| --- | --- | --- |
| `input` | 基础输入或预处理对照完成后，选择无增强原始 AlexNet | 固定学习率与优化器候选；AdamW 与随机梯度下降各三个学习率 |
| `optimizer` | 学习率与优化器选择后 | 余弦调度 |
| `schedule` | 调度实验后，或决定沿用固定学习率时 | 有效批量比较 |
| `batch` | 批量实验后，或决定沿用原批量时 | 随机失活比较 |
| `dropout` | 随机失活比较后 | 无惩罚、绝对值惩罚、平方惩罚及解耦衰减 |
| `regularization` | 权重正则化选择后 | 提前停止 |
| `training` | 完成无增强 AlexNet 调优和复核后 | 十项独立增强、组合模板和条件初始化检查 |
| `augmentation` | 增强选择后，可仍为无增强模型 | 小卷积核实验 |
| `small` | 小卷积核实验后选择保留模型，可仍是其对照 | 多尺度实验 |
| `multi` | 多尺度实验后选择保留模型，可仍是其对照 | 残差实验 |
| `convolution` | 卷积结构阶段保留模型 | 小型平均聚合头 |
| `head` | 保存平均聚合头实验结果，即使它弱于原卷积模型 | 单层编码器的直接对照 |
| `encoder` | 单层编码器稳定有效后 | 双层编码器 |
| `final` | 最终比较、逐项移除和复核完成后 | 全量重训与最终推理 |

按需实验被跳过时，把此前保留的正式运行登记到对应记录名即可；不需要为了建立文件再训练同样配置。现有记录默认不覆盖，确认更改选择时加 `--replace`，之前所有运行保持原样。选择记录会提醒单次成绩不等于已完成多种子复核。

初始化检查和增强组合是条件模板，默认 `status` 为 `design_required`，不会误执行。根据诊断或有效单项填写真实候选、问题和对照后，再将状态改为 `ready`。组合模板中的翻转与亮度只是示例，不表示已验证有效。解耦权重衰减候选只允许在 AdamW 下运行。

```powershell
.\.venv\Scripts\python.exe -X utf8 run.py catalog
.\.venv\Scripts\python.exe -X utf8 run.py check configs/03_augmentation/augment_01_flip.json
.\.venv\Scripts\python.exe -X utf8 run.py preview configs/03_augmentation/augment_01_flip.json --output artifacts/flip_preview.png
.\.venv\Scripts\python.exe -X utf8 run.py train configs/03_augmentation/augment_01_flip.json
```

`catalog` 会区分可执行配置、尚未选定前序方案和需要补充设计的模板。`check` 展示继承后的完整设置，并检查未知字段、重复正则化和未登记变化。所有相对数据路径以本实验目录为基准，`extends` 则以当前配置文件所在目录为基准。

## 五 重复实验与公平比较

```powershell
.\.venv\Scripts\python.exe -X utf8 run.py train configs/04_architecture/structure_01_small_kernels.json --seed 2027
.\.venv\Scripts\python.exe -X utf8 run.py train configs/04_architecture/structure_01_small_kernels.json --seed 2028
.\.venv\Scripts\python.exe -X utf8 run.py compare 对照运行目录 候选运行目录 --paired --output outputs/某组成对比较
.\.venv\Scripts\python.exe -X utf8 run.py compare 运行目录一 运行目录二 运行目录三 --output outputs/最终比较
```

成对比较要求两个运行种子相同，配置差异只能是候选登记的因素；输出逐样本误差变化和按图片组重采样的区间。汇总拒绝混用划分、标签或同一配置的重复种子。MSE 为核心，另外报告 MAE 和高价尾部 MSE；按完整设置指纹分组，避免同名不同参数被错误平均。只有一个种子时标准差为空，不能称稳定改善。

架构独立调优时，复制已有配置为新文件，写清训练参数变化；需解释模块作用时，再给对照结构使用同一训练设置。逐项移除也通过独立配置实现，例如在最终结构上只把 `model.residual` 改为 `false`，并将 `changed_fields` 设为该字段。数组整体替换，对象逐项合并；要关闭某项继承增强，将它的 `probability` 设为零，不要用空对象假定清除所有继承项。

目前入口支持显式逐项执行，不提供自动大规模搜索，防止尚未确认的候选或未复核的选择被自动串成结论。

## 六 输出与最终提交

每次输出到独立目录，主要文件包括完整配置、配置差异、数据及环境指纹、运行状态、模型结构、初始化指纹、训练历史、增强实际触发统计、学习曲线、固定样本卷积特征图数组、最佳权重、逐图验证预测、三个指标和成本。传统方法保存词典、标准化器及回归器；只加载本项目可信运行产物。

```powershell
.\.venv\Scripts\python.exe -X utf8 run.py retrain configs/selected/final.json --epochs 事先确定的轮次
.\.venv\Scripts\python.exe -X utf8 run.py predict 全量重训运行目录 --output outputs/submission.csv
```

全量轮次从最终开发配置的最佳轮次事先确定，例如三个种子的中位数；传统方法也使用同一命令，轮次可填一。全量重训重新初始化网络，或重拟合词典与标准化器。预测仅接受完整全量重训产物，关闭随机增强和失活，按测试清单顺序输出三千行，保留千美元单位。

解释性热图、复杂稳健性研究、模型集成和测试时增强在设计中属于后置选项，本入口没有默认启用；当前已提供固定特征数组与学习曲线用于检查。也未实现中断恢复，失败运行保留状态与错误记录，重试创建新目录，不能将已训练权重作为新实验初值。

## 七 测试

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

测试覆盖分组泄漏、指标单位、增强随机流隔离、所有模块的尺寸及初始化、编码器梯度、梯度累积、停止逻辑、配置登记，以及三条路线从合成数据训练到保存、全量重训和预测导出。合成数据结果不进入正式报告。
