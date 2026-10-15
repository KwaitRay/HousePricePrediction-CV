"""Train-only description of native aspect ratios and grouped OOF errors."""
from pathlib import Path
import time
import uuid
import numpy as np
import pandas as pd
from PIL import Image
from common.config import digest, write_json
from common.local_paths import local_path, REPOSITORY

REFERENCE = '15_fixed_ensembles/pe_giant_incremental_fusion/20261014T065752Z_88efe38c'


def main():
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    train = frame[frame.partition.eq('train')].reset_index(drop=True)
    source = local_path('records_root') / REFERENCE / 'meta_crossfit_predictions.csv'
    predictions = pd.read_csv(source, dtype={'imageid': str})
    assert train.imageid.tolist() == predictions.imageid.tolist() and len(train) == 6399
    sizes = []
    for name in train.imageid:
        with Image.open(local_path('data_root') / 'train' / name) as image:
            sizes.append(image.size)
    dimensions = np.asarray(sizes)
    retained = dimensions.min(axis=1) / dimensions.max(axis=1)
    train['width'], train['height'] = dimensions[:, 0], dimensions[:, 1]
    train['center_square_retained_area_fraction'] = retained
    train['prediction'] = predictions.augmented_meta_prediction.to_numpy()
    train['error'] = train.prediction - train.price
    rows = []
    for label, mask in [('all', np.ones(len(train), dtype=bool)),
        ('retained_le_half', retained <= .5), ('retained_half_to_two_thirds', (retained > .5) & (retained <= 2/3)),
        ('retained_two_thirds_to_90pct', (retained > 2/3) & (retained <= .9)), ('retained_over_90pct', retained > .9)]:
        for tier, prices in [('all', np.ones(len(train), dtype=bool)), ('high_gt1300', train.price.to_numpy() > 1300)]:
            subset = train[mask & prices]
            if len(subset):
                rows.append({'geometry': label, 'price_tier': tier, 'n': len(subset),
                    'mean_price': float(subset.price.mean()), 'median_price': float(subset.price.median()),
                    'mse': float(np.mean(subset.error ** 2)), 'mae': float(np.mean(abs(subset.error))),
                    'bias': float(subset.error.mean()), 'mean_retained_fraction': float(subset.center_square_retained_area_fraction.mean())})
    relative = Path('09_data_quality/pe_giant_frame_geometry') / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out, public = local_path('records_root') / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    write_json(public / 'registration.json', {'source_sha256': digest(__file__),
        'split_sha256': digest(local_path('split_path')), 'prediction_sha256': digest(source),
        'scope': '6399 training rows only; native dimensions and current meta crossfit errors',
        'geometry': 'Ideal center-square retained area after shortest-side resize with crop_pct=1, min(width,height)/max(width,height)',
        'caveat': 'Descriptive association only; price and scene distributions confound comparisons. Does not establish discarded pixels contain useful price information.',
        'test_images_used': False})
    train.to_csv(out / 'geometry_and_training_errors.csv', index=False)
    public.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(public / 'geometry_metrics.csv', index=False)
    summary = {'n': len(train), 'median_retained_area_fraction': float(np.median(retained)),
        'native_area_outside_center_square_mean': float(np.mean(1 - retained)),
        'n_more_than_one_third_native_area_outside_crop': int((retained < 2/3).sum()),
        'source': REFERENCE, 'price_model_selection_claim': None}
    write_json(public / 'summary.json', summary)
    write_json(public / 'status.json', {'status': 'completed'})
    print(public, flush=True)
    print(summary, flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
