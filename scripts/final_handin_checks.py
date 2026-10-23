"""Build the final code manifest/archive and audit the selected report."""
from pathlib import Path
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile

REPO=Path(__file__).resolve().parents[1]
STAGE=REPO.parent.parent/'.handin_20261023'
CODE=STAGE/'code'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def plain(s):
    s=re.sub(r'\\href\{([^}]+)\}\{([^}]+)\}',r'[\2](\1)',s)
    for command in ['textbf','emph','texttt']:
        s=re.sub(r'\\'+command+r'\{([^}]+)\}',r'\1',s)
    return s.replace(r'\%','%').replace('~',' ').replace('``','“').replace("''",'”')


def markdown(tex,src):
    text=tex.split(r'\maketitle',1)[1].split(r'\clearpage',1)[0]
    text=text.replace(r'\raggedbottom','')
    labels={}
    for kind,stem in [('table','tab'),('figure','fig')]:
        matches=(re.findall(r'\\input\{(figures/[^}]+\.tex)\}',text) if kind=='table' else
                 re.findall(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',text,re.S))
        for i,item in enumerate(matches,1):
            block=(src/item).read_text(encoding='utf-8') if kind=='table' else item
            label=re.search(r'\\label\{([^}]+)\}',block)
            if label:labels[label.group(1)]=str(i)
    def table(m):
        block=(src/m.group(1)).read_text(encoding='utf-8')
        caption=re.search(r'\\caption\{([^}]+)\}',block).group(1)
        rows=[]
        for line in block.splitlines():
            if '&' in line and line.rstrip().endswith(r'\\'):
                rows.append([plain(c.strip()) for c in line.rstrip()[:-2].split('&')])
        return '\n'+caption+'\n\n'+'\n'.join('| '+' | '.join(row)+' |' for row in [rows[0],['---']*len(rows[0]),*rows[1:]])+'\n'
    text=re.sub(r'\\input\{(figures/[^}]+\.tex)\}',table,text)
    def figure(m):
        block=m.group(0)
        path=re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',block).group(1)
        caption=block.split(r'\caption{',1)[1].split('\n'+r'\label',1)[0].rsplit('}',1)[0]
        return '\n[View figure](report_source/'+path+')\n\n'+caption+'\n'
    text=re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',figure,text,flags=re.S)
    text=re.sub(r'\\section\{([^}]+)\}',r'## \1\n',text)
    text=re.sub(r'\\subsection\{([^}]+)\}',r'### \1\n',text)
    text=re.sub(r'\\ref\{([^}]+)\}',lambda m:labels[m.group(1)],text)
    bib=(src/'references_draft_shared.tex').read_text(encoding='utf-8')
    entries=re.findall(r'\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})',bib,re.S)
    numbers={key:i for i,(key,_) in enumerate(entries,1)}
    text=re.sub(r'\\cite\{([^}]+)\}',lambda m:' ['+', '.join(str(numbers[k.strip()]) for k in m.group(1).split(','))+']',text)
    text=text.replace(r'\begin{equation}','\n$$\n').replace(r'\end{equation}','\n$$\n')
    text=plain(text)
    refs='\n\n## References\n\n'+'\n\n'.join(f'{i}. '+plain(body.strip()) for i,(_,body) in enumerate(entries,1))
    title=re.search(r'\\title\{([^}]+)\}',tex).group(1)
    (STAGE/'report.md').write_text('# '+title+'\n\n'+re.sub(r'\n{3,}','\n\n',text).strip()+refs+'\n',encoding='utf-8')


def main():
    import fitz
    src=STAGE/'report_source'
    tex=(src/'main.tex').read_text(encoding='utf-8')
    shutil.copyfile(STAGE/'_preview.pdf',STAGE/'report.pdf')
    pdf=fitz.open(STAGE/'report.pdf')
    assert len(pdf)==5 and pdf[4].get_text().strip().upper().startswith('REFERENCES')
    assert 'mapping suited to the task.' in pdf[3].get_text()
    assert all(abs(p.rect.width-595.28)<2 and abs(p.rect.height-841.89)<2 for p in pdf)
    for name in ['Shifeng Chen','Haoxuan Ying','1686631','1255329']:
        assert name in pdf[0].get_text()
    log=(STAGE/'_preview.log').read_text(encoding='utf-8',errors='replace')
    assert 'Overfull' not in log
    assert not re.search(r'LaTeX Warning: (?:Reference|Citation).*undefined|There were undefined references',log)
    assert len(re.findall(r'\\bibitem\{', (src/'references_draft_shared.tex').read_text(encoding='utf-8')))==25
    markdown(tex,src)
    # Provenance for the release is distinct from the historical training run.
    write(CODE/'checks/selected_report.json',{
        'source':'English/report_source/main.tex','report_sha256':sha(STAGE/'report.pdf'),
        'source_sha256':sha(src/'main.tex'),'original_main_v1_sha256':read(STAGE/'preparation.json')['source_sha256'],
        'body_pages':4,'reference_pages':1,'recipe':'20261015T034010Z_0c94a543',
        'public_mse_user_reported':47291.34608,
        'later_47196_exploration_is_not_the_submitted_recipe':True})
    # The exact archived inference path was replayed on all 3000 examples
    # during retention cleanup; this is evidence, not bundled model weights.
    replay=read(REPO.parent/'cleanup_review/verification_after.json')
    write(CODE/'checks/retained_recipe_replay.json',{k:v for k,v in replay.items() if k not in {'protected_files','latest_recipe','latest_max_absolute_error'}})
    (CODE/'README.txt').write_text((CODE/'README.txt').read_text(encoding='utf-8')+
        '\nFINAL SNAPSHOT\nThe separately submitted report is the main_v1 revision, with English title/references and corrected LaTeX math escapes.\nThe supplied predictions and portable recipe remain the report-consistent 47291.34608 model.\nLater 47196.41766 exploration is retained in the development repository and is not the hand-in recipe.\nReport source: ../report_source/main.tex (XeLaTeX).\nA new CPU replay of all 3000 archived report-recipe predictions is recorded in checks/retained_recipe_replay.json.\n',encoding='utf-8')
    forbidden={'.pt','.pth','.ckpt','.safetensors','.npz','.npy','.joblib','.pkl','.jpg','.jpeg','.bin','.zip'}
    actual=[p for p in CODE.rglob('*') if p.is_file()]
    assert not [p for p in actual if p.suffix.lower() in forbidden]
    assert not [p for p in actual if '__pycache__' in p.parts or p.suffix=='.pyc']
    for p in CODE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    manifest={p.relative_to(CODE).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(actual) if p.name!='MANIFEST.json'}
    write(CODE/'MANIFEST.json',manifest)
    subprocess.run([sys.executable,'-B','-X','utf8','verify_submission.py'],cwd=CODE,check=True)
    target=STAGE/'code.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(CODE.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(CODE).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'MANIFEST.json'}
        for name,entry in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==entry['sha256']
    (STAGE/'English_README.md').write_text('''# Final hand-in

Authors: Shifeng Chen (1686631), Haoxuan Ying (1255329).

- Upload **report.pdf** to Canvas **Final Project: Report**. It is the selected main_v1 revision: four A4 IEEE content pages, followed by references on page five.
- Upload **code.zip** to Canvas **Final Project: Code**. It contains model/experiment source, dependencies, fixed split, numeric evidence, the 3000-row Kaggle predictions, and README.txt with run commands.
- `code/` is the expanded copy of the same ZIP; `report_source/main.tex` is the matching XeLaTeX source. `report.md` is a reading copy of that source.
- Recipe and predictions correspond to public MSE **47291.34608** (user-reported). Later experiments are not silently substituted.

No supplied train/test images, fitted weights, encoder weights or feature arrays are included. The README explains the dataset layout and pinned weight downloads. Reported experiment code, including photometric, fusion and occlusion analysis, is included.

Validation: portable metadata check, recomputed validation/OOF/ablation metrics, CPU figure rebuilding, Python syntax, prediction IDs/hash, ZIP CRC and exact manifest hashes. All 3000 report-recipe predictions were also replayed locally using retained heads/features; fresh CUDA extraction/full refitting was not rerun.

The combined archive outside this directory is for transfer only. Group registration, the Kaggle submission and individual peer assessment are separate course activities; they are not performed by this package.
''',encoding='utf-8')
    (STAGE/'submission_README.md').write_text('''# Final project submission

Upload these two files separately on Canvas:

| Canvas destination | File |
|---|---|
| Final Project: Report | [English/report.pdf](English/report.pdf) |
| Final Project: Code | [English/code.zip](English/code.zip) |

The selected report is the main_v1 revision, with four A4 IEEE content pages plus a references-only fifth page. The code/predictions use the report-consistent 47291.34608 recipe. Source, expanded code and verification details are under [English/](English/README.md).

`English_all_materials.zip` is a combined transfer archive, not an additional Canvas deliverable. `Chinese/` and `Haoxuan_Ying/` are historical reading/review versions; neither is the final report. Development models and later optimization experiments elsewhere in the repository are not part of the code upload.

The separate Project Plan/group registration, Kaggle submission and each member's peer assessment must be completed through their respective course interfaces. This preparation does not upload to Canvas or Kaggle.
''',encoding='utf-8')
    audit={'body_pages':4,'reference_pages':1,'a4':True,'ieee_default_font_size_and_margins':True,
        'test_predictions':3000,'report_sha256':sha(STAGE/'report.pdf'),'code_zip_sha256':sha(target),
        'manifest_files':len(manifest),'python_sources':len(list(CODE.rglob('*.py'))),
        'code_verified':True,'pdf_visually_reviewed':True,'portable_metadata_check':True,
        'report_metrics_and_ablations_recomputed':True,'figure_rebuilds_on_cpu':True,
        'zip_crc_and_manifest_verified':True,'model_weights_features_supplied_images_included':False,
        'fresh_encoder_extraction':False,'fresh_full_refit':False}
    write(STAGE/'release_validation.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
