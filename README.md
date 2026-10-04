# 图像房价预测实验

本仓库集中维护实验代码、设计和各阶段报告。数据集、模型权重、预处理缓存和逐图预测保留在本地，不上传 GitHub。训练调优实验六已结束，保留原配置；实验七复用既有早停回放与实现验收，实验八按用户授权开展卷积初始化比较。

## 目录

```text
project.py                      统一命令入口
scripts/                        数据分析与实验代码
  experiments/configs/          配置与阶段选择
  experiments/tests/            自动检查
design/                         模型设计、实验计划与改进方案
output/experiment_record/       阶段报告、汇总指标、训练历史与曲线
  00_preparation/                实验前准备
  01_preprocessing/              预处理
  02_baselines/                  基线
  04_training/                   训练调优；03留给传统参照
archive/orchestration/          历史队列源码，仅供审计，不直接执行
local_paths.example.json        本地路径配置示例
local_paths.json                本机配置，不提交
.local/                        新实验输出和缓存，不提交
```

从[实验记录导航](output/experiment_record/README.md)阅读结果，从[模型与实验计划](design/model_and_experiment_plan.md)阅读设计。每个阶段只维护一篇实验报告，运行目录保存证据。

协作时使用[固定划分索引](data_split/README.md)：包含全部8000张有标签图片的训练／验证归属，不含价格或图片。所有同学沿用该索引，避免重新划分导致结果无法直接比较。

## 本地配置与运行

使用 Python 3.11，在本地虚拟环境安装 `scripts/experiments/requirements.txt`。历史实验使用 PyTorch 2.11.0、torchvision 0.26.0；Muon 需要包含 `torch.optim.Muon` 的 PyTorch。GPU版本应与本机驱动匹配。

复制 `local_paths.example.json` 为 `local_paths.json`，填写数据目录、固定划分文件和本地输出位置。相对路径均相对仓库根目录，也支持 `CV_DATA_ROOT`、`CV_SPLIT_PATH`、`CV_RECORDS_ROOT`、`CV_CACHE_ROOT`、`CV_EDA_OUTPUT_ROOT` 环境变量覆盖。本机已配置复用原数据和划分。

```powershell
python project.py paths
python project.py check configs/optimized_v2/alexnet.json
python project.py catalog
# 仅在明确需要开始新实验时执行：
python project.py train configs/optimized_v2/alexnet.json --seed 2026
```

`prepare` 会创建或检查划分；复现历史比较必须使用原有固定划分，不能重新随机划分。训练结果写入本地 `records_root`，经分析后才将报告、汇总指标和曲线整理进仓库的对应阶段。最佳权重按验证集指标选取，不能用训练集误差代替验证结论。

```powershell
# CPU合成样本检查，不运行正式GPU实验
$env:CUDA_VISIBLE_DEVICES='-1'
python -m unittest discover -s scripts/experiments/tests -v
```

## 当前结论

预处理选用直接拉伸。训练调优保留AdamW 0.0001、固定学习率、有效批量32、随机失活0.5，三种子平均验证均方误差127774.42。Adam复核、余弦衰减、批量16/64及随机失活0/0.2均未满足替换条件。原因、控制变量和局限统一见[训练调优报告](output/experiment_record/04_training/实验报告.md)。Muon剩余候选不继续。

本次迁移保留历史指标，不把新代码当作旧版本的逐位复现。历史队列归档为文本。2026-10-03新增 `scripts/experiments/train_cached.py`，通过环境变量 `CV_STRETCH_CACHE_DIRECTORY` 显式连接本地缓存；启动前检查文件、原图和包版本。新顺序队列 `continue_training.py --legacy-root <原Final_Assignment目录>` 先核验核心代码结构和32张独立重算，再复核Adam并进入实验三。普通 `project.py train` 仍使用原图加载，缓存仅对显式入口生效。详见[整理与路径核验](design/整理与路径核验.md)。
