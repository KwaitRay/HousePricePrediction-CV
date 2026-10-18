# Best model: 47291.34608

实验版本：`20261015T034010Z_0c94a543`。Kaggle 公榜 MSE：**47291.34608**（用户反馈）；固定验证 MSE：52201.21126766699。

归档：[`20261015T034010Z_0c94a543_features_predictions.zip`](20261015T034010Z_0c94a543_features_predictions.zip)，1,046,039,525 字节，使用 Git LFS 保存。

SHA256：`d164cfc3ca835e759d235c4957656029244f515c13e5f60fb3d054fc091db557`。

## 内容

- SigLIP2 NaFlex、PE-Core-L 多视角、PE-Core-G whole-frame + PCA 的训练/测试特征。
- 三个全训练模型头、PLS metric、PCA 变换及冻结融合权重。
- 原始 3000 行 `predictions.csv`，以及各分支预测、加权贡献、融合预测与逐图误差表。
- 相关实验设计、代码、验证/OOF 预测、审计记录、输入 CSV 和固定数据划分。
- 文件 SHA256 清单、完整性校验脚本和不加载编码器的 CPU 回放脚本。
- 三个预训练编码器的下载链接、精确 revision、文件 SHA256 和可选下载脚本。

不含预训练编码器权重或原始图片。使用归档特征回放预测无需下载权重；从图片重新提取特征时需另行准备图片和权重。

## 下载与核验

在仓库根目录执行：

```powershell
git lfs install
git lfs pull --include="best_model/20261015T034010Z_0c94a543_features_predictions.zip"
Get-FileHash best_model/20261015T034010Z_0c94a543_features_predictions.zip -Algorithm SHA256
```

解压到独立目录，在该目录使用兼容的 Python 环境执行：

```powershell
python verify_bundle.py
python replay_predictions.py
```

依赖版本参考包内 `scripts/experiments/requirements-visual.txt`。CPU 回放所需包为 NumPy、SciPy、pandas、joblib、threadpoolctl 和 scikit-learn。

本地已验证全部 ZIP 条目的 CRC、所有保留文件与原归档的 SHA256 一致性，以及原提交 CSV 的 SHA256。3000 行 CPU 预测回放最大绝对误差为 `1.3642420526593924e-12`。

原始 `predictions.csv` 的 SHA256：`c39009ec65fd89a0f531991d1cfc26a2ecf4095e512ce9905b89c38da06b8da3`。

## 补充：最佳方案的交叉拟合预测

[`meta_crossfit_predictions.csv`](meta_crossfit_predictions.csv) 单独保存在此目录，未放入上述 ZIP。它来自 `15_fixed_ensembles/pe_whole_pca_incremental_fusion/20261015T033810Z_8aa6ec62`，是最终最佳提交所沿用的 PCA 融合方案的训练侧评估记录。

文件共 6399 行，包含 `imageid`、`baseline_meta_prediction`、`augmented_meta_prediction`。在基础模型的 OOF 预测上执行三折 GroupKFold：每折使用另外两折学习融合权重，生成留出折预测。两列分别对应基线方案和新增 PCA 分支后的方案，价格单位与课程标签一致（千美元）。

从逐图预测重新计算，基线交叉拟合 MSE 为 **53052.56579**，新增 PCA 后为 **52210.22283**，三折均改善；固定验证集 MSE 为 **52201.21127**。最终冻结方案在全部 8000 条有标签数据上重新拟合后，测试提交公榜 MSE 为 **47291.34608**。这些分数对应不同样本和评估流程。

[`crossfit_evaluation/`](crossfit_evaluation/) 提供注册信息、三折指标及权重、融合配置、原执行源码、诊断和文件 SHA256。已验证 CSV 覆盖固定训练划分的全部 6399 条样本、顺序一致，重新计算的 MSE 与原记录相符。

基础模型超参数先前已选定，因此该交叉拟合不是完全嵌套的无偏验证；CSV 中的预测也不是最终全训练模型对测试集的预测。
