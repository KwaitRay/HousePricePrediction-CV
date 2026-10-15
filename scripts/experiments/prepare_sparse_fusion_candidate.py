"""Compile only exact zero coefficients, verify compositions, then full-label refit."""
from pathlib import Path
import argparse
import shutil
import time
import uuid
import numpy as np
import pandas as pd
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from visual_kernel_submission import prepare
from audit_visual_kernel_refit import main as audit_refit


def main(reference):
    root = local_path('records_root')
    reference = reference.resolve()
    relative_reference = reference.relative_to(root)
    pubroot = REPOSITORY / 'output/experiment_record'
    assert read_json(pubroot / relative_reference / 'status.json')['status'] == 'completed'
    results = read_json(pubroot / relative_reference / 'results.json')
    assert results['export_eligible']
    ensemble = read_json(reference / 'ensemble.json')
    weights = np.asarray(ensemble['weights'], dtype=np.float64)
    assert np.isfinite(weights).all() and (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(), 1., rtol=0, atol=1e-8)
    keep = np.flatnonzero(weights != 0).tolist()
    assert keep
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str})
    assert ensemble['split_sha256'] == digest(local_path('split_path'))
    for i, model in enumerate(ensemble['models']):
        assert digest(root / model / 'head.npz') == ensemble['model_sha256'][i]
        assert digest(root / model / 'predictions.csv') == ensemble['prediction_sha256'][i]
    errors = {}
    for filename, column, partition in [('selected_training_oof.csv', 'oof_prediction', 'train'), ('predictions.csv', 'prediction', 'validation')]:
        tables = [pd.read_csv(root / m / filename, dtype={'imageid': str}) for m in ensemble['models']]
        ids = frame.loc[frame.partition.eq(partition), 'imageid'].tolist()
        assert all(t.imageid.tolist() == ids for t in tables)
        matrix = np.column_stack([t[column] for t in tables])
        delta = float(np.max(abs(matrix @ weights - matrix[:, keep] @ weights[keep])))
        assert delta < 1e-7
        errors[column] = delta
    relative = Path('15_fixed_ensembles/fixed_sparse_export_reference') / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out, public = root / relative, pubroot / relative
    out.mkdir(parents=True, exist_ok=False)
    sparse = dict(ensemble)
    for field in ('models', 'weights', 'model_sha256', 'prediction_sha256'):
        sparse[field] = [ensemble[field][i] for i in keep]
    write_json(out / 'ensemble.json', sparse)
    for filename in ('predictions.csv', 'metrics.json', 'prediction_checks.json'):
        shutil.copyfile(reference / filename, out / filename)
    write_json(public / 'registration.json', {'method': 'Exact-zero compilation before full refit',
        'reference': str(relative_reference), 'reference_ensemble_sha256': digest(reference / 'ensemble.json'),
        'kept_indices': keep, 'max_abs_composition_errors': errors, 'no_new_selection': True,
        'test_images_used': False, 'source_sha256': digest(__file__)})
    write_json(public / 'results.json', results)
    write_json(public / 'status.json', {'status': 'completed'})
    prepared = prepare(out)
    audit_refit(prepared)
    write_json(pubroot / relative_reference / 'prepared_export.json', {'prepared_run': str(prepared),
        'full_refit_audit_passed': True, 'test_inference_completed': False})
    print('AUDITED_PREPARED_RUN', prepared, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-dir', type=Path, required=True)
    main(parser.parse_args().reference_dir)
