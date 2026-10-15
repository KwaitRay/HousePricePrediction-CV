from pathlib import Path
import hashlib
import json
import re

root = Path(__file__).resolve().parents[1]
design = root / 'design'
files = list(design.iterdir())
assert len(files) == 1 and files[0].name == 'model_and_experiment_plan.md'
text = files[0].read_text(encoding='utf-8')
assert '\ufffd' not in text
anchors = set(re.findall(r'<a id="([^"]+)"', text))
missing = []
for label, target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text):
    if target.startswith(('http:', 'https:')):
        continue
    path, _, fragment = target.partition('#')
    destination = (design / path).resolve() if path else files[0]
    if not destination.exists():
        missing.append(target)
    elif fragment and destination == files[0] and fragment not in anchors:
        missing.append(target)
assert not missing, missing
backup = root / 'archive/design_history/20261014_consolidation'
assert len(list(backup.glob('*.md'))) == 5
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in backup.glob('*.md')}
(backup / 'sha256.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
for expected in ['49543.82739', '52555.25635', '52130.67815', '40000', '0.7165%', '54105.31', '53462.04']:
    assert expected in text, expected
print(json.dumps({'design_files': [p.name for p in files], 'links_checked': True,
    'archive_documents': len(manifest), 'utf8_checked': True}, ensure_ascii=False))
