"""Check the exact-byte package manifest and 3000-row Kaggle prediction file."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parent


def main():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    for name,entry in manifest.items():
        p=ROOT/name
        assert p.is_file(),name
        assert p.stat().st_size==entry['bytes'],name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],name
    with (ROOT/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==['imageid','price'];rows=list(reader)
    assert len(rows)==3000 and len({r['imageid'] for r in rows})==3000
    assert all(math.isfinite(float(r['price'])) and float(r['price'])>=0 for r in rows)
    with (ROOT/'inputs/test.csv').open(encoding='utf-8-sig',newline='') as f:test=list(csv.DictReader(f))
    assert [r['imageid'] for r in rows]==[r['imageid'] for r in test]
    expected='c39009ec65fd89a0f531991d1cfc26a2ecf4095e512ce9905b89c38da06b8da3'
    assert hashlib.sha256((ROOT/'predictions.csv').read_bytes()).hexdigest()==expected
    forbidden={'.pt','.pth','.ckpt','.safetensors','.npz','.npy','.joblib','.pkl','.jpg','.jpeg'}
    assert not [p for p in ROOT.rglob('*') if p.is_file() and p.suffix.lower() in forbidden]
    sources=list(ROOT.rglob('*.py'))
    for p in sources:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    print(json.dumps({'verified_files':len(manifest),'python_sources_parsed':len(sources),'test_rows':len(rows),'prediction_sha256':expected,'forbidden_artifacts':0},indent=2))

if __name__=='__main__':main()
