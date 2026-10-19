"""Assemble the current bilingual hand-in snapshot. No training or inference.

Run from this repository: python scripts/prepare_submission_release.py
Requires the sibling submission/report working directory. Refuses to overwrite.
"""
from pathlib import Path
import ast
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import zipfile

REPO=Path(__file__).resolve().parents[1]
ASSIGNMENT=REPO.parent
REPORT=ASSIGNMENT/'submission/report'
DEST=REPO/'submission'
LOCAL=REPO/'.local/experiment_record'
FINAL='08_final/visual_kernel_ensemble/20261015T034010Z_0c94a543'
FUSION='15_fixed_ensembles/pe_whole_pca_incremental_fusion/20261015T033810Z_8aa6ec62'
PREVIOUS='15_fixed_ensembles/pe_giant_whole_incremental_fusion/20261014T082727Z_c5152cef'
PANELS=['report_pca_distance_bandwidth_oof','report_prediction_scatter_price_band_errors','report_pe_g_occlusion_model_comparison']


def put(path,data):
    if os.name=='nt' and len(str(path))>235 and not str(path).startswith('\\\\?\\'):
        path=Path('\\\\?\\'+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data if isinstance(data,bytes) else data.encode('utf-8'))


def copy(src,dst):put(dst,src.read_bytes())
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,data):put(path,json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(folder,path,prefix=''):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,prefix+p.relative_to(folder).as_posix())
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))


def main():
    if DEST.exists() and not (DEST/'English/code/submission_provenance.json').exists():
        print('Resuming an incomplete assembly; existing source originals are untouched.',flush=True)
    elif DEST.exists():raise FileExistsError(f'Refusing to overwrite completed assembly {DEST}')
    en=DEST/'English';zh=DEST/'Chinese';code=en/'code'
    code.mkdir(parents=True,exist_ok=True)
    copy(REPO/'project.py',code/'project.py')
    # All source/configuration files, including the code behind screened candidates.
    for src in (REPO/'scripts').rglob('*'):
        rel=src.relative_to(REPO)
        if not src.is_file() or '__pycache__' in rel.parts or src.suffix not in {'.py','.json','.txt'}:continue
        if 'mllm' in rel.as_posix().lower() or src.name=='prepare_submission_release.py':continue
        copy(src,code/rel)
    # PR #5 only adds these modules/test; the shared implementation is unchanged.
    pr=REPO/'.local/pr_review_5/scripts/experiments'
    for name in ['photometric_screen.py','export_photometric_screen.py','tests/test_photometric_export.py']:
        copy(pr/name,code/'scripts/experiments'/name)
    settings={'data_root':'dataset','split_path':'inputs/split_v1.csv','records_root':'.local/experiment_record',
              'cache_root':'.local/cache','eda_output_root':'.local/analysis_outputs'}
    write(code/'local_paths.example.json',settings)
    local=read(REPO/'local_paths.json')
    split=Path(local['split_path'])
    copy(split,code/'inputs/split_v1.csv');copy(split.with_suffix('.json'),code/'inputs/split_v1.json')
    data=Path(local['data_root'])
    copy(data/'train.csv',code/'inputs/train.csv');copy(data/'test.csv',code/'inputs/test.csv')
    copy(LOCAL/FINAL/'predictions.csv',code/'predictions.csv')
    assert sha(code/'predictions.csv')=='c39009ec65fd89a0f531991d1cfc26a2ecf4095e512ce9905b89c38da06b8da3'
    rows=list(csv.DictReader(io.StringIO((code/'predictions.csv').read_text(encoding='utf-8-sig'))))
    assert len(rows)==3000 and len({r['imageid'] for r in rows})==3000
    # Numeric evidence and execution registrations; no prose development diaries.
    for src in (REPO/'output/experiment_record').rglob('*'):
        rel=src.relative_to(REPO)
        relevant=any(part.startswith(('00_','01_','02_','03_','04_','05_','06_','08_','15_')) for part in rel.parts)
        if src.is_file() and src.suffix in {'.json','.csv'} and relevant:
            copy(src,code/rel)
    # Predictions needed to evaluate the final recipe and its selected branches.
    fusion=read(LOCAL/FUSION/'ensemble.json')
    for model in fusion['models']:
        model=model.replace('\\','/')
        for name in ['predictions.csv','selected_training_oof.csv']:
            copy(LOCAL/model/name,code/'evidence/components'/model/name)
    for name in ['ensemble.json','meta_crossfit_predictions.csv','predictions.csv','metrics.json']:
        copy(LOCAL/FUSION/name,code/'evidence/final_fusion'/name)
    copy(LOCAL/FUSION/'predictions.csv',code/'evidence/final_validation_predictions.csv')
    copy(LOCAL/PREVIOUS/'predictions.csv',code/'evidence/previous_validation_predictions.csv')
    copy(REPO/'best_model/meta_crossfit_predictions.csv',code/'evidence/meta_crossfit_predictions.csv')
    reg=read(REPO/'output/experiment_record'/FINAL/'registration.json')
    with zipfile.ZipFile(REPO/'best_model/20261015T034010Z_0c94a543_features_predictions.zip') as z:
        bundle=json.loads(z.read('bundle.json'))
        for name in ['pretrained_downloads.json','download_pretrained_weights.py']:
            put(code/name,z.read(name))
    recipe={'name':'Selected visual-only three-encoder fusion','run':FINAL,'authors':[
        {'name':'Shifeng Chen','student_id':'1686631'},{'name':'Haoxuan Ying','student_id':'1255329'}],
        'models':bundle['components'],'weights':reg['weights'],'components':reg['components'],
        'validation_mse':reg['reference_validation_mse'],'public_mse_user_reported':47291.34608,
        'target_scale':1000,'price_unit':'thousand USD','pretrained_weights_included':False,
        'fitted_heads_included':False,'feature_arrays_included':False,
        'prediction_sha256':sha(code/'predictions.csv')}
    write(code/'final_recipe.json',recipe)
    # Exact feature/preprocessing manifests and download provenance, without arrays/weights.
    for component in reg['components']:
        family,key=component['cache_family'],component['cache_key']
        manifest=REPO/'.local/cache'/family/key/'manifest.json'
        copy(manifest,code/'.local/cache'/family/key/'manifest.json')
        m=read(manifest)
        if 'original_cache_key' in m:
            origin='pe_core_features' if family=='pe_core_multiview' else 'pe_core_giant_features'
            src=REPO/'.local/cache'/origin/m['original_cache_key']/'manifest.json'
            if src.exists():copy(src,code/'.local/cache'/origin/m['original_cache_key']/'manifest.json')
    downloads=read(code/'pretrained_downloads.json')
    write(code/'submission_provenance.json',{'base_commit':subprocess.check_output(['git','-c',f'safe.directory={REPO.as_posix()}','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'photometric_pr_commit':'664393e2342fe5b838db6649554b31bc64a1c763','report_source':'Sibling submission/report',
        'weights_excluded_by_request':True,'provided_images_excluded':True,'test_labels_used':False})
    analysis=code/'report_analysis'
    for src in (REPORT/'figures/analysis_sources').rglob('*'):
        if src.is_file() and src.suffix in {'.json','.csv'}:
            copy(src,analysis/'figures/analysis_sources'/src.relative_to(REPORT/'figures/analysis_sources'))
    source=(REPORT/'build_report_analysis_panels.py').read_text(encoding='utf-8')
    source=source.replace('import hashlib','import os\nimport hashlib',1)
    source=source.replace('ASSIGNMENT = ROOT.parents[1]','ASSIGNMENT = ROOT.parent').replace("REPO = ASSIGNMENT / 'cv-house-price'",'REPO = ROOT.parent')
    source=re.sub(r"PRED = REPO / .*", "PRED = REPO / 'evidence/final_validation_predictions.csv'",source,count=1)
    source=source.replace("DATA = ASSIGNMENT / 'comp-90086-2026/house_dataset/train'","DATA = Path(os.environ.get('CV_DATA_ROOT', str(REPO/'dataset')))/'train'")
    source=source.replace('str(p.relative_to(ASSIGNMENT))','str(p)')
    put(analysis/'build_panels.py',source)
    copy(REPORT/'build_occlusion_figures.py',analysis/'occlusion_experiment_original.py')
    for stem in PANELS:
        for ext in ['png','svg','pdf']:copy(REPORT/f'figures/{stem}.{ext}',analysis/f'figures/{stem}.{ext}')
    copy(REPORT/'figures/analysis_sources/report_compact_results_table.csv',analysis/'representative_results.csv')
    for language,root in [('English',en),('Chinese',zh)]:
        name='report_draft' if language=='English' else 'report_draft_zh'
        copy(REPORT/f'{name}.pdf',root/'report.pdf')
        copy(REPORT/f'{name}.md',root/'report.md')
        tex=(REPORT/f'{name}.tex').read_text(encoding='utf-8')
        if language=='Chinese':
            tex=re.sub(r'\\setmainfont[^\n]*',r'\\setmainfont{TeX Gyre Termes}',tex)
            tex=re.sub(r'\\setCJKmainfont[^\n]*',r'\\setCJKmainfont{FandolSong-Regular}',tex)
            copy(REPORT/'references_draft_shared.tex',root/'report_source/references_draft_shared.tex')
        put(root/f'report_source/{name}.tex',tex)
        copy(REPORT/'references_all_experiments.bib',root/'report_source/references_all_experiments.bib')
        table='report_compact_results_table' + ('_zh' if language=='Chinese' else '')
        copy(REPORT/f'figures/{table}.tex',root/f'report_source/figures/{table}.tex')
        for stem in PANELS:
            for ext in ['pdf','png']:copy(REPORT/f'figures/{stem}.{ext}',root/f'report_source/figures/{stem}.{ext}')
    put(code/'requirements-analysis.txt','numpy\npandas\nscipy\nscikit-learn\nmatplotlib\nPillow\npypdf\nthreadpoolctl\n')
    put(code/'README.txt',CODE_README)
    put(en/'README.md',EN_README)
    put(zh/'README.md',ZH_README)
    copy(REPORT/'report_requirements_and_revision_notes.md',zh/'report_review_notes.md')
    copy(REPORT/'references_experiment_sources.md',zh/'experiment_references.md')
    put(DEST/'README.md',ROOT_README)
    # Exact-byte manifest is portable and excludes itself.
    manifest={p.relative_to(code).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(code.rglob('*')) if p.is_file()}
    write(code/'MANIFEST.json',manifest)
    for p in code.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    forbidden={'.pt','.pth','.ckpt','.npz','.npy','.joblib','.pkl','.safetensors','.jpg','.jpeg'}
    assert not [p for p in code.rglob('*') if p.is_file() and p.suffix.lower() in forbidden]
    print(f'Prepared {len(manifest)} code/evidence files; ready for independent checks.',flush=True)


CODE_README='''COMP90086 HOUSE PRICE PREDICTION - CODE SUBMISSION
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
'''

EN_README='''# COMP90086 submission materials

Shifeng Chen — 1686631; Haoxuan Ying — 1255329.

- **report.pdf**: English IEEE report, four A4 content pages plus references on page five.
- **code.zip**: separate Canvas code upload (also expanded under `code/` for review).
- **code/predictions.csv**: unchanged 3000-row Kaggle submission, public MSE 47291.34608 (manual feedback).
- **report_source/**: LaTeX, references and figures. Compile `report_draft.tex` with IEEEtran/pdflatex twice or Tectonic.
- **code/README.txt**: dependencies, expected dataset layout, experiment entry points and reproduction scope.

Canvas requires the report and code as separate uploads. The combined English archive
is for transfer/review. Trained model heads, pretrained weights, feature caches and
supplied images are omitted. All experiment source/configuration code discussed in
the report is included; historical scripts retain run-dependency guards.

The report and prediction files are the current snapshot. Confirm the Kaggle team
name contains both student IDs and complete the course's separate peer assessment.
'''

ZH_README='''# 中文阅读材料

署名顺序：Shifeng Chen（1686631）、Haoxuan Ying（1255329）。

- `report.pdf` / `report.md`：中文报告阅读版。
- `report_source/`：中文 LaTeX 与图表。可在 Overleaf 选择 XeLaTeX；源文件使用通用 Fandol 中文字体，不打包本机字体。
- `experiment_references.md`：实验与原始文献的对应清单。
- `report_review_notes.md`：报告要求核对和历史修改记录；其中早期“待填写作者/待打包”条目是历史记录，以本目录 README 和最终打包清单为准。

正式英文提交材料与共用代码位于 `../English/`；中文目录不重复保存代码。
Canvas 分别上传英文 `report.pdf` 和 `code.zip`。Project 要求的是模型代码，
本次不打包训练权重、预训练权重、特征缓存、课程原图、虚拟环境或下载工具。
英文所有材料压缩包为上一级 `English_all_materials.zip`。
'''

ROOT_README='''# Submission snapshot

Authors: Shifeng Chen (1686631), Haoxuan Ying (1255329).

| Directory / file | Purpose |
|---|---|
| English/ | English report, source, expanded code and separate code.zip |
| Chinese/ | Chinese reading version and review notes; shares English code |
| English_all_materials.zip | All English hand-in materials for transfer |
| release_manifest.json | SHA256 and verification summary |

For Canvas upload English/report.pdf and English/code.zip separately.
Only this submission/ directory defines the hand-in snapshot. Historical best_model/
archives elsewhere in the development repository are not part of the submission.
No fitted model weights, encoder checkpoints, feature arrays or supplied images are
included. Source for all reported experiments, frozen split and test predictions are included.
'''

if __name__=='__main__':main()
