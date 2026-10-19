"""Equal-block fusion of frozen CLIP multi-view and StreetCLIP image features."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from common.config import digest, write_json
from common.local_paths import local_path, REPOSITORY


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--street-reference', type=Path, required=True)
    parser.add_argument('--original-manifest', type=Path, required=True)
    parser.add_argument('--multiview-reference', type=Path, required=True)
    args = parser.parse_args()
    base = local_path('cache_root')
    split = local_path('split_path')
    frame = pd.read_csv(split, dtype={'imageid': str})
    original_manifest = json.loads(args.original_manifest.read_text(encoding='utf-8'))
    original_key = hashlib.sha256(json.dumps(original_manifest, sort_keys=True).encode()).hexdigest()
    multiview = json.loads((args.multiview_reference / 'registration.json').read_text(encoding='utf-8'))
    multiview_cost = json.loads((args.multiview_reference / 'cost.json').read_text(encoding='utf-8'))
    street = json.loads((args.street_reference / 'registration.json').read_text(encoding='utf-8'))
    if multiview['original_manifest_sha256'] != digest(args.original_manifest):
        raise ValueError('Multi-view and original feature provenance differ')
    sources = [base / 'clip_features' / original_key,
               base / 'clip_multiview' / multiview_cost['feature_cache_key'],
               base / 'streetclip_features' / street['cache_key']]
    arrays, provenance = [], []
    for folder in sources:
        manifest_path = folder / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['imageids'] != frame.imageid.tolist() or manifest['split_sha256'] != digest(split):
            raise ValueError('Source image order or split differs')
        chunks = sorted(folder.glob('*.npy'))
        values = np.concatenate([np.load(p, allow_pickle=False) for p in chunks]).astype(np.float64)
        if len(values) != 8000 or not np.isfinite(values).all():
            raise ValueError('Invalid source features')
        arrays.append(values)
        provenance.append({'cache': str(folder.relative_to(base)), 'manifest_sha256': digest(manifest_path),
                           'chunks_sha256': [digest(p) for p in chunks]})
    original, views, street_features = arrays
    if original.shape != (8000, 768) or views.shape != (8000, 2, 768) or street_features.shape != (8000, 768):
        raise ValueError('Unexpected model feature shape')
    multi_features = np.concatenate([original, views.reshape(8000, 1536)], axis=1)
    blocks = [multi_features, street_features]
    blocks = [x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12) for x in blocks]
    joined = np.concatenate(blocks, axis=1)
    joined /= np.maximum(np.linalg.norm(joined, axis=1, keepdims=True), 1e-12)
    if not np.allclose(np.linalg.norm(joined, axis=1), 1, atol=1e-8):
        raise ValueError('Feature normalization failed')
    manifest = {'imageids': frame.imageid.tolist(), 'split_sha256': digest(split), 'sources': provenance,
                'feature_dim': 3072, 'source_sha256': digest(__file__), 'test_images_used': False,
                'fusion': 'Equal unit-norm CLIP L14 three-view block and StreetCLIP block, concatenate then global L2 normalize',
                'course_scope': 'Frozen visual CLIP/StreetCLIP exploratory features; eligibility pending confirmation'}
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    folder = base / 'fused_features' / key
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'manifest.json', manifest)
    np.save(folder / '00000_8000.npy', joined.astype(np.float32))
    public = REPOSITORY / 'output/experiment_record/17_geo_visual_reference'
    write_json(public / 'feature_fusion_registration.json', {**{k: v for k, v in manifest.items() if k != 'imageids'},
                'cache_key': key, 'head_protocol': 'Existing 12-candidate training-only 3-fold GroupKFold kernel search'})
    print(key, flush=True)


if __name__ == '__main__':
    main()
