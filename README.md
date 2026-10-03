# 图像房价预测实验

这是**开发仓库，不是最终提交包**。先按下面三个入口阅读，不需要从头浏览全部文件。

| 你想做什么 | 从哪里开始 |
| --- | --- |
| 配置路径、训练或生成预测 | `project.py`、`local_paths.example.json`、`scripts/experiments/` |
| 查进度与内部实验依据 | [阶段导航](output/experiment_record/README.md)；这些材料不进入最终提交 |
| 导出干净的代码提交包 | `scripts/package_submission.py`；只导出指定配置及其依赖 |

## 最终提交边界

- 保留：模型与运行代码、依赖、必要 JSON 配置、简短 `README.txt`、正式测试预测和经双方确认的固定划分。
- 排除：`design/`、`archive/`、`output/experiment_record/`、所有内部叙述性 Markdown、本机路径、虚拟环境、缓存和提供的图片。
- 正式报告单独提交 PDF，不将内部阶段报告 Markdown 当成正式报告。
- 原始开发记录继续保留；不移动运行中的目录、不删除证据、不改变实验方案。

目前仅导出传统参照阶段预览；冻结划分已核对一致，可通过 `--split` 打入ZIP，但尚无最终测试预测，**不能直接作为最终提交**。

```powershell
python scripts/package_submission.py --config configs/01_baselines/sift_ridge.json --output .local/submission_preview/traditional_code.zip
# 只看将导出的文件，不创建 ZIP：在同一命令后加 --dry-run
```

交付前再明确加入报告中使用的各个配置（重复 `--config`）、确认的划分 `--split` 和最终预测 `--predictions`。
使用 `--final` 时缺少后两者会拒绝导出；已有 ZIP 也不会被覆盖。该检查不替代双方对最终模型和复现结果的验收。

<details>
<summary>开发环境、完整目录与历史运行说明（需要时展开）</summary>

本仓库集中维护实验代码、设计和各阶段报告。数据集、模型权重、预处理缓存和逐图预测保留在本地，不上传 GitHub。main的训练调优实验一至五记录已完成；实验六的更新由队友分支另行维护，本PR不启动或修改该队列。

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
  03_traditional/                传统参照
  04_training/                   训练调优
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

### 03 传统参照（Haoxuan Ying）

传统路线可使用独立CPU环境，无需为SIFT和岭回归安装PyTorch。
复现2026-10-03运行的依赖见 `scripts/experiments/requirements-traditional-lock.txt`；
同样必须通过 `local_paths.json` 对齐数据与冻结划分。

```powershell
uv pip install --python .venv/Scripts/python.exe -r scripts/experiments/requirements-traditional-lock.txt
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s scripts/experiments/tests -p test_traditional.py -v
.\.venv\Scripts\python.exe -X utf8 project.py train configs/01_baselines/sift_ridge.json --seed 2026
```

完整三种子结果、误差分析和已核对的划分指纹见
[传统参照报告](output/experiment_record/03_traditional/实验报告.md)。
分析入口 `scripts/analyze_traditional.py` 将本地运行整理为精简证据，
不导出模型、原始图片或逐图预测。本PR不改变队友的训练调优队列。

### 神经网络主线

预处理选用直接拉伸。训练调优保留AdamW 0.0001、固定学习率、有效批量32、随机失活0.5，三种子平均验证均方误差127774.42。Adam复核、余弦衰减、批量16/64及随机失活0/0.2均未满足替换条件。原因、控制变量和局限统一见[训练调优报告](output/experiment_record/04_training/实验报告.md)。Muon剩余候选不继续。

本次迁移保留历史指标，不把新代码当作旧版本的逐位复现。历史队列归档为文本。2026-10-03新增 `scripts/experiments/train_cached.py`，通过环境变量 `CV_STRETCH_CACHE_DIRECTORY` 显式连接本地缓存；启动前检查文件、原图和包版本。新顺序队列 `continue_training.py --legacy-root <原Final_Assignment目录>` 先核验核心代码结构和32张独立重算，再复核Adam并进入实验三。普通 `project.py train` 仍使用原图加载，缓存仅对显式入口生效。详见[整理与路径核验](design/整理与路径核验.md)。

</details>
