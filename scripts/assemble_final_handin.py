"""Stage the selected main_v1 report and the existing audited report recipe.

Does not train, download, commit, push, or overwrite the working report.
The publish phase is separate and requires a verified staged PDF/code package.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import zipfile

REPO=Path(__file__).resolve().parents[1]
ASSIGNMENT=REPO.parent
STAGE=ASSIGNMENT.parent/'.handin_20261023'
SOURCE=ASSIGNMENT/'submission/report_1021'
DEST=REPO/'submission/English'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def filesystem_path(p):
    import os
    value=str(p.resolve())
    return '\\\\?\\'+value if os.name=='nt' and not value.startswith('\\\\?\\') else value


def copy_tree(source,target,ignore):
    return shutil.copytree(filesystem_path(source),filesystem_path(target),ignore=ignore)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def prepare():
    if STAGE.exists():raise FileExistsError(STAGE)
    STAGE.mkdir(parents=True)
    copy_tree(DEST/'code',STAGE/'code',shutil.ignore_patterns('__pycache__','*.pyc'))
    original=(SOURCE/'main_v1.tex').read_text(encoding='utf-8-sig')
    (STAGE/'source_main_v1_original.tex').write_text(original,encoding='utf-8')
    tex=original
    changes=[
        (r'\renewcommand{\refname}{参考文献}',r'\renewcommand{\refname}{References}'),
        (r'\title{基于冻结视觉特征与核回归融合的房价预测}',r'\title{House Price Prediction with Frozen Visual Features and Kernel Fusion}'),
        ('墨尔本大学，COMP90086 计算机视觉','COMP90086 Computer Vision, The University of Melbourne'),
        ('145,336.73pm1,091.58',r'145,336.73\pm1,091.58'),
        ('$4times4$',r'$4\times4$'),
        ('（table~', '(Table~'),('）。At', '). At'),('where $\\alpha=1$。','where $\\alpha=1$.'),
        (r'\section{Result}',r'\section{Results}'),
    ]
    applied=[]
    for old,new in changes:
        if old in tex:tex=tex.replace(old,new);applied.append({'before':old,'after':new})
    assert not re.search('[\u4e00-\u9fff]',tex),'Unexpected untranslated Chinese in English report'
    src=STAGE/'report_source';src.mkdir()
    (src/'main.tex').write_text(tex,encoding='utf-8')
    deps=[]
    for kind,name in re.findall(r'\\(input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}',tex):
        if kind=='input' and not Path(name).suffix:name+='.tex'
        p=SOURCE/name;assert p.is_file(),p
        target=src/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        deps.append(name)
    shutil.copyfile(SOURCE/'latexmkrc',src/'latexmkrc')
    (src/'README.txt').write_text('Main document: main.tex. Compiler: XeLaTeX.\nAll referenced figures, tables and bibliography are included.\nPrepared from main_v1.tex; original working source is unchanged.\n',encoding='utf-8')
    # Local compile only: explicit paths to the same TeX Gyre/Fandol fonts.
    # This preview source is never published or included in hand-in archives.
    shutil.copytree(SOURCE/'tmp/layout_v1/fonts',src/'_preview_fonts')
    fonts='_preview_fonts/'
    preview=tex.replace(r'\setCJKmainfont{FandolSong-Regular}',r'\setCJKmainfont[Path='+fonts+r']{FandolSong-Regular.otf}')
    preview=preview.replace(r'\setmainfont{TeX Gyre Termes}',r'\setmainfont[Path='+fonts+r',BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf,BoldItalicFont=texgyretermes-bolditalic.otf]{texgyretermes-regular.otf}')
    (src/'_preview.tex').write_text(preview,encoding='utf-8')
    provenance=json.loads((STAGE/'code/submission_provenance.json').read_text(encoding='utf-8'))
    provenance.update(report_source='English/report_source/main.tex',
        selected_report_original='Final_Assignment/submission/report_1021/main_v1.tex',
        selected_report_original_sha256=sha(SOURCE/'main_v1.tex'),
        report_changes=applied,selected_prediction_recipe='20261015T034010Z_0c94a543',
        subsequent_47196_exploration_included=False)
    dump(STAGE/'code/submission_provenance.json',provenance)
    dump(STAGE/'preparation.json',{'source_sha256':sha(SOURCE/'main_v1.tex'),
        'source_changes':applied,'dependencies':deps,'selected_public_mse':47291.34608})
    print('Prepared',STAGE)


def make_zip(folder,path):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
                z.write(p,p.relative_to(folder).as_posix())
    with zipfile.ZipFile(path) as z:assert z.testzip() is None


def publish():
    audit=json.loads((STAGE/'release_validation.json').read_text(encoding='utf-8'))
    assert audit['body_pages']==4 and audit['reference_pages']==1 and audit['test_predictions']==3000
    assert audit['code_verified'] and audit['pdf_visually_reviewed']
    assert sha(STAGE/'report.pdf')==audit['report_sha256']
    assert sha(STAGE/'code.zip')==audit['code_zip_sha256']
    # Preserve the superseded submission snapshot locally before replacing it.
    backup=REPO/'.local/submission_before_20261023'
    if backup.exists():raise FileExistsError(backup)
    copy_tree(REPO/'submission',backup,shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['code','report_source']:
        target=(DEST/name).resolve()
        assert target.is_relative_to((REPO/'submission').resolve())
        shutil.rmtree(filesystem_path(target))
        copy_tree(STAGE/name,target,shutil.ignore_patterns('_preview.tex','_preview_fonts','__pycache__','*.pyc','*.log','*.aux','*.out'))
    for name in ['report.pdf','report.md','code.zip']:
        shutil.copyfile(STAGE/name,DEST/name)
    shutil.copyfile(STAGE/'release_validation.json',DEST/'release_validation.json')
    shutil.copyfile(STAGE/'English_README.md',DEST/'README.md')
    shutil.copyfile(STAGE/'submission_README.md',REPO/'submission/README.md')
    make_zip(DEST,REPO/'submission/English_all_materials.zip')
    old=json.loads((REPO/'submission/release_manifest.json').read_text(encoding='utf-8'))
    old.update(selected_report='main_v1.tex',selected_prediction_public_mse_user_reported=47291.34608,
        selected_prediction_sha256=sha(DEST/'code/predictions.csv'),
        audit='English/release_validation.json')
    old['validation'].update(english_content_pages=4,test_predictions=3000,
        manifest_files=audit['manifest_files'],zip_crc_checked=True,
        encoder_extraction_rerun=False,full_refit_rerun=False)
    for name in list(old['files']):
        p=REPO/'submission'/name
        old['files'][name]={'bytes':p.stat().st_size,'sha256':sha(p)}
    dump(REPO/'submission/release_manifest.json',old)
    handin=ASSIGNMENT/'submission/final_20261023';handin.mkdir(exist_ok=True)
    for name in ['report.pdf','code.zip']:
        shutil.copyfile(DEST/name,handin/name)
    (handin/'README.txt').write_text('Canvas Final Project: Report -> report.pdf\nCanvas Final Project: Code -> code.zip\nSelected recipe: 47291.34608 (report-consistent).\nDo not upload the combined transfer archive to either slot.\nGroup registration, Kaggle submission and individual peer assessment remain separate course activities.\n',encoding='utf-8')
    print('Published verified hand-in files:',handin)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','publish']);a=p.parse_args()
    prepare() if a.phase=='prepare' else publish()
