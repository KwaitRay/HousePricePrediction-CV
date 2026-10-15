"""Verify real exported heads using the kernel-ridge linear-system identity."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from visual_kernel_components import feature_registration, predict_component
from visual_kernel_submission import features_for


def main(out):
    out = out.resolve()
    public = REPOSITORY / 'output/experiment_record' / out.relative_to(local_path('records_root'))
    registration = read_json(out / 'registration.json')
    prepared = read_json(out / 'prepared.json')
    if not prepared['reference_replay_passed'] or prepared['full_train_n'] != 8000:
        raise ValueError('Full refit has not completed')
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    indices = np.random.default_rng(2026).choice(len(frame), 12, replace=False)
    results = []
    for index, component in enumerate(registration['components']):
        source = read_json(REPOSITORY / 'output/experiment_record' / component['source'] / 'registration.json')
        values = features_for(feature_registration(source), frame)
        prediction = predict_component(out, index, component, values[indices])
        with np.load(out / f'component_{index}_full_head.npz', allow_pickle=False) as head:
            # (K + alpha I)c = y - mean implies Kc + mean = y - alpha c.
            expected = np.maximum(frame.price.to_numpy()[indices] - float(head['alpha']) * head['coefficients'][indices] * 1000, 0)
        np.testing.assert_allclose(prediction, expected, rtol=1e-8, atol=1e-5)
        results.append({'component': index, 'kernel_type': component['kernel_type'],
                        'max_abs_prediction_identity_error': float(np.max(np.abs(prediction - expected))),
                        'full_head_sha256': component['full_head_sha256']})
        print(f'Full-refit identity verified for component {index + 1}', flush=True)
    write_json(public / 'kernel_adapter_audit.json', {'passed': True,
        'scope': '12 supplied labeled images per full-refit component; KRR linear-system identity independent of prediction adapter',
        'all_validation_predictions_replayed_before_refit': prepared['reference_replay_passed'],
        'imageids': frame.imageid.iloc[indices].tolist(), 'components': results,
        'source_sha256': digest(__file__), 'test_images_used': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-run', type=Path, required=True)
    main(parser.parse_args().prepared_run)
