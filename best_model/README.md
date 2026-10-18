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
