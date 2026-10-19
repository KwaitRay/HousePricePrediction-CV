# Haoxuan Ying — 合并中文论文稿

- `report.pdf`：可直接阅读的 PDF，正文及图表 4 页，参考文献第 5 页。
- `main.tex`：LaTeX 源文件，参考文献已包含在文件末尾。
- `figures/`：编译所需的三张 PDF 插图，需与源码一起下载。

本稿结合 Haoxuan Ying 的中文稿与队友 `feat/submission` 中的论文和实验分析（合并内容参考提交 `108a4ac7541d353a8b566792f92d6535575409a4`）。本目录于 2026-10-19 添加到分支最新提交 `8efa670` 之上，不覆盖 `Chinese/`、`English/` 及其提交压缩包，也不宣称已吸收其后所有文字修改。

作者仍为 Shifeng Chen 与 Haoxuan Ying；文件夹名仅标识稿件维护人。该版本是供团队审阅的中文稿，并非替代英文最终提交材料。

## 编译

使用 XeLaTeX（不要用 pdfLaTeX），在本目录运行两次：

```sh
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

也可运行 `tectonic main.tex`。编译输出为 `main.pdf`；仓库中的 `report.pdf` 是当前已编译预览。Overleaf 需上传 `main.tex` 和完整 `figures/`，选择 XeLaTeX。源码有 Windows 字体与 TeX 字体回退，换环境后请重新检查分页。

## 当前内容与边界

保留增强对照、融合消融与价位误差分析，补入整图视野、PCA 带宽诊断和区域遮挡案例。无单独 Abstract，表格为英文。叙述区分实测结果、可能原因与尚未验证的下一步。

当前三张跨栏图集中在正文第 4 页，尚未实施就近穿插排版。本次上传不重新训练模型，不修改实验结果或队友原稿。
