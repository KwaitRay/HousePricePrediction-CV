"""Train-only native-image audit, with grouped OOF errors and reviewed scenes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from common.config import digest, write_json
from common.local_paths import local_path, REPOSITORY


def metrics(frame):
    error = frame.prediction.to_numpy() - frame.price.to_numpy()
    return {'n': len(frame), 'mse': float(np.mean(error**2)), 'mae': float(np.mean(abs(error))),
            'bias': float(np.mean(error)), 'mean_native_width': float(frame.width.mean()),
            'mean_native_height': float(frame.height.mean())}


def main():
    split = local_path('split_path')
    train = pd.read_csv(split, dtype={'imageid': str, 'group_id': str})
    train = train[train.partition.eq('train')].copy()
    reference = local_path('records_root') / '13_clip_optimization/clip_l14_multiview_kernel/20261013T120554Z_e2a41622'
    oof = pd.read_csv(reference / 'selected_training_oof.csv', dtype={'imageid': str})
    if len(train) != 6399 or not oof.imageid.is_unique or set(oof.imageid) != set(train.imageid):
        raise ValueError('Unexpected training OOF coverage')
    train['prediction'] = oof.set_index('imageid').loc[train.imageid, 'oof_prediction'].to_numpy()
    train['price_tier'] = np.where(train.price <= 300, 'low', np.where(train.price <= 1300, 'middle', 'high'))
    native = []
    for name in train.imageid:
        with Image.open(local_path('data_root') / 'train' / name) as source:
            exif = source.getexif()
            corrected = ImageOps.exif_transpose(source)
            width, height = corrected.size
            native.append({'imageid': name, 'width': width, 'height': height, 'aspect_ratio': width / height,
                           'exif_tag_n': len(exif), 'gps_tag_present': 34853 in exif, 'format': source.format})
    train = train.merge(pd.DataFrame(native), on='imageid', validate='one_to_one')
    annotations = pd.read_csv(REPOSITORY / '.local/artifacts/training_scene_index/training_scene_index.csv', dtype={'imageid': str})
    train = train.merge(annotations[['imageid', 'scene_primary', 'annotation_status', 'visible_pool', 'partial_subject', 'large_building']],
                        on='imageid', validate='one_to_one')
    private = local_path('records_root') / '09_data_quality/training_geometry_l14_multiview'
    public = REPOSITORY / 'output/experiment_record/09_data_quality/training_geometry_l14_multiview'
    private.mkdir(parents=True, exist_ok=True)
    train.to_csv(private / 'training_geometry.csv', index=False, encoding='utf-8-sig')
    sizes = train.groupby(['width', 'height']).size().sort_values(ascending=False)
    summary = {'scope': 'Only 6399 training images; no validation or test pixels/labels used',
               'split_sha256': digest(split), 'oof_sha256': digest(reference / 'selected_training_oof.csv'),
               'source_sha256': digest(__file__), 'all_training': metrics(train),
               'native_size_counts': [{'width': int(w), 'height': int(h), 'n': int(n)} for (w, h), n in sizes.items()],
               'min_side_at_least': {str(edge): int((train[['width', 'height']].min(axis=1) >= edge).sum()) for edge in [224, 336, 384]},
               'exif_nonempty_n': int((train.exif_tag_n > 0).sum()), 'gps_tag_present_n': int(train.gps_tag_present.sum()),
               'metadata_used_as_model_features': False,
               'caveats': ['Native dimensions do not measure property size or subject pixel occupancy.',
                           'OOF hyperparameters previously selected in these folds; association is descriptive, not causal or nested-unbiased.',
                           'Only high-price scene annotations cover their entire price tier; other tiers have unreviewed entries.']}
    strata = []
    for tier, subset in train.groupby('price_tier'):
        strata.append({'price_tier': tier, **metrics(subset)})
    summary['price_strata'] = strata
    high = train[train.price_tier.eq('high')]
    scenes = [{'scene': str(scene), **metrics(subset)} for scene, subset in high.groupby('scene_primary')]
    summary['high_price_scenes'] = scenes
    summary['high_price_flags'] = [{'feature': flag, 'state': str(state), **metrics(subset)}
        for flag in ['visible_pool', 'partial_subject', 'large_building'] for state, subset in high.groupby(flag)]
    # Observed variance for exactly matching canonical pixels and dimensions; not a test/Bayes bound.
    mean_by_image = train.groupby(['pixel_hash', 'width', 'height']).price.transform('mean')
    summary['training_identical_pixel_price_variance_mse'] = float(np.mean((train.price - mean_by_image)**2))
    summary['identical_pixel_variance_scope'] = 'Empirical training-set disagreement only; cannot establish an unseen-data error floor or label corruption'
    write_json(public / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ['all_training', 'min_side_at_least', 'exif_nonempty_n', 'gps_tag_present_n', 'high_price_scenes', 'training_identical_pixel_price_variance_mse']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
