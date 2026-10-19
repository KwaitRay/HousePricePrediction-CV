COMP90086 HOUSE PRICE PREDICTION - CODE SUBMISSION
Authors: Shifeng Chen (1686631), Haoxuan Ying (1255329).

1. CONTENTS
Model/experiment code, configurations, fixed 6399/1601 split, dependencies,
3000-row Kaggle predictions.csv, numerical results and report-analysis code.
No supplied train/test images, pretrained weights, fitted heads or feature arrays.
The public MSE 47291.34608 was reported after manual Kaggle submission.

2. ENVIRONMENT AND DATA
Python 3.11. Install a PyTorch/CUDA build appropriate to your machine, then:
  python -m pip install -r scripts/experiments/requirements-visual.txt
  python -m pip install -r requirements-analysis.txt
The CLIP reference additionally needs open_clip_torch (see original code).
Copy local_paths.example.json to local_paths.json. Set data_root to the supplied
house_dataset folder containing train/, test/, train.csv and test.csv. The default
is code/dataset/. Keep split_path=inputs/split_v1.csv; do not regenerate the split.
The supplied label/ID CSV copies in inputs/ enable result checks without images.
  python project.py paths

3. EXPERIMENTS DISCUSSED IN THE REPORT
SIFT/BoVW/Ridge: scripts/experiments/traditional/ and configs/01_baselines/.
AlexNet: models/, training/, run.py and configs/optimized_v2/.
Training tuning: continue_training.py, batch_dropout.py, regularization.py,
initialization.py, and configs/02_training/ and 04_training_recorded/.
Augmentation: augmentation_stage.py, train_augmented.py, augmentation_robustness.py.
Architecture: architecture_stage.py, encoder_stage.py and configs/04_architecture/.
Photometric: photometric_screen.py and export_photometric_screen.py (PR #5).
CLIP: clip_reference.py. Frozen encoders: siglip_naflex_reference.py,
pe_core_reference.py, pe_core_multiview.py, pe_core_giant_reference.py,
pe_giant_whole_view.py, visual_kernel_inference.py and corresponding adapters.
Regressors: frozen_kernel_search.py, visual_price_metric_kernel.py,
pe_whole_pca_bandwidth_kernel.py, encoder_oof_fusion.py, pe_pca_incremental_fusion.py.
Final full-data refit/test inference: visual_kernel_submission.py and adapters.
All paths above are relative to scripts/experiments/ unless otherwise specified.

Run an early experiment from this directory, for example:
  python project.py train configs/01_baselines/sift_ridge.json --seed 2026
Read script --help / argparse definitions before explicitly starting GPU training.
Historical frozen-feature scripts contain registered run IDs, source hashes and
dependency guards. They require regenerated predecessor runs, not merely copying
the archived JSON registrations. They are preserved as executed research code;
the archive does not claim one-command retraining from an empty cache.
final_recipe.json fixes the three selected encoders, revisions, preprocessing,
hyperparameters and weights. pretrained_downloads.json and the download script
provide pinned encoder sources; downloading is needed only to rerun image encoding.
No download or GPU work is triggered by unpacking or checking this package.

4. REPORT RESULTS WITHOUT RETRAINING
  python evaluate_report.py
  python verify_submission.py
These check the fixed split, original test-CSV hash, validation/OOF metrics and
selected-branch ablations from archived predictions; they do not infer from images.
To redraw the three report panels, supply images at data_root and run:
  python report_analysis/build_panels.py
For a non-default location set CV_DATA_ROOT to the dataset folder first.
The PCA/occlusion CSVs are the archived measurements; plotting does not repeat GPU
occlusion inference. occlusion_experiment_original.py preserves the original study
source and its historical path assumptions. The photometric exporter is specific
to the archived 40-epoch runs; it must be adapted before auditing longer reruns.

5. HAND-IN
Upload code.zip to Canvas Final Project: Code, and English/report.pdf separately
to Final Project: Report. English_all_materials.zip is a combined handover archive.
Project-plan registration/submission and individual peer assessment are separate
course activities; blank templates are not included as completed deliverables.
