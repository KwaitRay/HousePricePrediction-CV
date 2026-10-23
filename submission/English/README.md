# Final hand-in

Authors: Shifeng Chen (1686631), Haoxuan Ying (1255329).

- Upload **report.pdf** to Canvas **Final Project: Report**. It is the selected main_v1 revision: four A4 IEEE content pages, followed by references on page five.
- Upload **code.zip** to Canvas **Final Project: Code**. It contains model/experiment source, dependencies, fixed split, numeric evidence, the 3000-row Kaggle predictions, and README.txt with run commands.
- `code/` is the expanded copy of the same ZIP; `report_source/main.tex` is the matching XeLaTeX source. `report.md` is a reading copy of that source.
- Recipe and predictions correspond to public MSE **47291.34608** (user-reported). Later experiments are not silently substituted.

No supplied train/test images, fitted weights, encoder weights or feature arrays are included. The README explains the dataset layout and pinned weight downloads. Reported experiment code, including photometric, fusion and occlusion analysis, is included.

Validation: portable metadata check, recomputed validation/OOF/ablation metrics, CPU figure rebuilding, Python syntax, prediction IDs/hash, ZIP CRC and exact manifest hashes. All 3000 report-recipe predictions were also replayed locally using retained heads/features; fresh CUDA extraction/full refitting was not rerun.

The combined archive outside this directory is for transfer only. Group registration, the Kaggle submission and individual peer assessment are separate course activities; they are not performed by this package.
