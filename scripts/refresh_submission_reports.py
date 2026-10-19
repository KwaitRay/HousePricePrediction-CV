"""Refresh both reports and the combined English archive; preserve code.zip."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import zipfile
from pypdf import PdfReader

REPO=Path(__file__).resolve().parents[1]
SOURCE=REPO.parent/'submission/report'
DEST=REPO/'submission'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path=DEST/'release_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    code_sha=sha(DEST/'English/code.zip')
    for lang,stem in [('English','report_draft'),('Chinese','report_draft_zh')]:
        pdf=SOURCE/f'{stem}.pdf';reader=PdfReader(pdf)
        assert len(reader.pages)<=5
        if lang=='English':
            assert len(reader.pages)==5
            assert reader.pages[4].extract_text().strip().startswith('REFERENCES')
        shutil.copyfile(pdf,DEST/lang/'report.pdf')
        md=(SOURCE/f'{stem}.md').read_text(encoding='utf-8').replace('(figures/','(report_source/figures/')
        (DEST/lang/'report.md').write_text(md,encoding='utf-8')
        tex=(SOURCE/f'{stem}.tex').read_text(encoding='utf-8')
        if lang=='Chinese':
            tex=re.sub(r'\\setmainfont[^\n]*',r'\\setmainfont{TeX Gyre Termes}',tex)
            tex=re.sub(r'\\setCJKmainfont[^\n]*',r'\\setCJKmainfont{FandolSong-Regular}',tex)
        (DEST/lang/f'report_source/{stem}.tex').write_text(tex,encoding='utf-8')
        manifest['validation']['pdf_pages'][lang]=len(reader.pages)
    target=DEST/'English_all_materials.zip';temp=target.with_suffix('.refresh.zip')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted((DEST/'English').rglob('*')):
            if p.is_file():z.write(p,p.relative_to(DEST/'English').as_posix())
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        assert hashlib.sha256(z.read('code.zip')).hexdigest()==code_sha
        assert z.read('report.pdf')==(DEST/'English/report.pdf').read_bytes()
    temp.replace(target)
    for name in manifest['files']:
        p=DEST/name
        manifest['files'][name]={'bytes':p.stat().st_size,'sha256':sha(p)}
    assert manifest['files']['English/code.zip']['sha256']==code_sha
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Updated bilingual PDFs/source/Markdown and verified combined archive; code.zip preserved.')

if __name__=='__main__':main()
