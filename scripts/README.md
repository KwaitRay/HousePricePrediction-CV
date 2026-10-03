# 本地数据分析运行说明

模型训练、阶段配置与统一评测入口见[实验代码说明](experiments/README.md)。本目录继续保留原有数据探索程序和独立环境。

分析方案见 [统一设计文档](../design/model_and_experiment_plan.md)中的实验准备部分。代码仅在本地读取现有数据，无需 GPU、Kaggle 账号、API key 或网络访问；首次安装 Python 依赖可能需要网络。

## 1. 安装

需要 Python 3.10 或更高版本。建议使用独立环境，避免影响课程其他项目。以下命令在 PowerShell 中执行：

```powershell
Set-Location -LiteralPath '${LOCAL_ASSIGNMENT}'
python -m venv .\scripts\.venv
.\scripts\.venv\Scripts\python.exe -m pip install -r .\scripts\requirements.txt
```

代码、测试、依赖清单及虚拟环境统一放在 `Final_Assignment/scripts/`，环境路径为 `scripts/.venv/`。环境已存在时可跳过创建步骤。无需激活环境，后续直接使用环境中的 Python，因此不依赖 PowerShell 的脚本激活策略。如果已有合适的环境，可以将下列 Python 路径替换为现有环境的解释器。

## 2. 建议先验证环境

```powershell
.\scripts\.venv\Scripts\python.exe -X utf8 .\scripts\analyze_dataset.py --max-images 100 --workers 4
```

每个 split 最多扫描 100 张随机图片，仍检查全部 CSV 和全部训练售价。抽样图片结果不能视为全量结论；此模式通常不会生成划分或基线。看到 `Status: completed` 后，再执行全量流程。

## 3. 完整分析

```powershell
.\scripts\.venv\Scripts\python.exe -X utf8 .\scripts\analyze_dataset.py --workers 4
```

默认数据位置根据脚本目录解析，不依赖启动目录：`Final_Assignment/comp-90086-2026/house_dataset/`。默认扫描全部 8,000 张训练图和 3,000 张测试图。图片读取和近重复检索期间会输出进度；耗时取决于本地磁盘、CPU、杀毒扫描和候选数量，图片 I/O 可能占主要时间。

脚本默认只使用 CPU。若磁盘较慢，可以尝试 `--workers 2`；不建议盲目设置很大的线程数。每次都生成新的输出目录，不会覆盖上次结果；本版本不提供断点续跑或缓存。

## 4. 输出位置与阅读顺序

启动后终端会打印实际输出路径，例如：

```text
Final_Assignment/analysis_outputs/20260921T050000Z_a1b2c3d4/
    report.md
    summary.json
    run_manifest.json
    tables/
    figures/
```

建议先看 `report.md`，再核对 `summary.json` 的 `status`、`full_scan` 和 `warnings`。接着检查重复/标签冲突表及图片，最后看划分草案和基线。若要继续讨论结果，可以提供该目录路径，优先读取报告、摘要和关键 CSV。

重点输出：

- `image_inventory.csv`、`image_errors.csv`：读取完整性。
- `price_statistics.csv`、`price_distribution.png`：训练售价分布，单位千美元。
- `train_feature_spearman.csv`、`quality_review_candidates.csv`：特征关系与质量复核。
- `exact_duplicate_members.csv`、`train_exact_label_conflicts.csv`：完全重复和训练标签冲突。
- `train_near_duplicate_candidates.csv`、`train_near_duplicate_pairs.png`：待确认的近重复候选；无候选时不生成配对图。
- `split_candidate.csv`、`near_pairs_crossing_split.csv`：验证划分草案及残留候选风险。
- `baseline_metrics.csv`、`baseline_error_by_price_band.csv`：本地常数基线，不是 Kaggle 成绩。

CSV 在 `tables/` 下，PNG 在 `figures/` 下。只有全量图片扫描且训练图片全部成功解码、至少有两个独立像素组时，才生成划分及基线文件。

`completed` 表示程序完成，不代表候选异常已经人工复核。`split_candidate.csv` 始终是草案；有跨集合近重复候选时应先复核。近重复搜索可能标记 `complete`、`sample_only`、`truncated` 或 `skipped`；`complete` 仅代表按当前 dHash 阈值完成检索，不保证发现所有同房屋图片。

不要将 `analysis_outputs/`、虚拟环境或课程图片整包作为最终代码提交。分析图表可能嵌入训练照片；应有选择地将必要图表放入报告，并遵守课程提交要求。

## 5. 可配置参数

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data-root` | 脚本相对的课程数据目录 | 输入目录，包含三个 CSV 和 train/test |
| `--output-root` | `Final_Assignment/analysis_outputs` | 新运行目录的父目录，必须在数据目录外 |
| `--workers` | 4 | 图片扫描线程数 |
| `--seed` | 2026 | 抽样、画图和划分随机种子 |
| `--max-images` | 0 | 每个 split 最多扫描数量；0 为全量 |
| `--validation-fraction` | 0.2 | 草案期望验证组比例，实际行比例会单独报告 |
| `--near-threshold` | 4 | 64 位 dHash Hamming 半径，允许 0–8 |
| `--max-near-pairs` | 50000 | 近重复候选上限；超出后明确标记检索截断 |
| `--skip-near-duplicates` | 不开启 | 跳过近重复检索，其余阶段仍运行；结果不能声称已完成该检查 |
| `--expected-train` / `--expected-test` | 8000 / 3000 | Schema 数量检查；仅为有意更换数据或合成测试调整，不用于掩盖缺失 |

自定义路径示例：

```powershell
.\scripts\.venv\Scripts\python.exe -X utf8 .\scripts\analyze_dataset.py --data-root 'D:\datasets\house_dataset' --output-root 'D:\results\house_eda' --workers 4
```

## 6. 失败处理与测试

退出码 0 表示流程完成；退出码 2 表示参数错误、schema 失败、图片解码错误或执行异常。图片错误可能保留其余分析结果；请检查 `image_errors.csv`。执行异常时 `run_manifest.json` 记录 `failed_runtime` 和错误信息；中断可能留下 `running`，应重新运行并查看新的目录。

常见问题：

- `ModuleNotFoundError`：确认安装依赖与执行脚本使用的是同一个 Python。
- 数量或文件对应错误：核对 `--data-root` 和解压是否完整，不要直接降低预期数量绕过检查。
- 找不到划分文件：检查是否处于抽样模式、训练图片是否全部解码、独立图片组是否至少有两个。
- 候选截断：先看低纹理图片是否导致大量误报，再决定是否提高上限；上限内结果不是完整列表。

可以运行合成数据测试。测试不读取课程数据，不访问网络；使用临时文件验证哈希检索、分组划分、训练侧基线、非法输入、坏图记录及全量/抽样端到端流程：

```powershell
.\scripts\.venv\Scripts\python.exe -X utf8 -m unittest discover -s .\scripts\tests -v
```

最低版本范围便于安装，不是锁定环境。每次运行都会记录实际依赖版本；确认环境后可自行锁定版本用于后续复现。
