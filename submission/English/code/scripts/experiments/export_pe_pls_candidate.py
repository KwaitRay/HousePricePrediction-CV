"""Replay old/new PLS adapters, compile exact zeros, refit and audit candidate."""
from pathlib import Path
import shutil
import time
import uuid
import joblib
import numpy as np
import pandas as pd
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from visual_kernel_components import feature_registration, kernel_for
from visual_kernel_submission import features_for, prepare, infer
from audit_visual_kernel_refit import main as audit_refit
from audit_visual_submission import main as audit_submission
from threadpoolctl import threadpool_limits

REFERENCE = '15_fixed_ensembles/pe_pls_incremental_fusion/20261014T063915Z_573c1b3a'


def main():
    root = local_path('records_root')
    public_root = REPOSITORY / 'output/experiment_record'
    reference = root / REFERENCE
    assert read_json(public_root / REFERENCE / 'results.json')['export_eligible']
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    valid = frame.partition.eq('validation').to_numpy()
    # Exercise unchanged legacy SigLIP and the new PE branch on all held-out rows.
    audits = []
    for name in ['19_image_embedding_reference/siglip2_price_metric_kernel/20261013T155603Z_4129a7fd',
                 '19_image_embedding_reference/pe_core_multiview_price_metric_kernel/20261014T051116Z_13d3ea16']:
        registration = read_json(public_root / name / 'registration.json')
        values = features_for(feature_registration(registration), frame)
        source = root / name
        metric = joblib.load(source / 'feature_metric.joblib')
        with np.load(source / 'head.npz', allow_pickle=False) as head, threadpool_limits(limits=2):
            raw = (kernel_for(head, values[valid], metric) @ head['coefficients'] + float(head['target_mean'])) * 1000
        expected = pd.read_csv(source / 'predictions.csv').raw_prediction.to_numpy()
        np.testing.assert_allclose(raw, expected, rtol=1e-7, atol=1e-4)
        audits.append({'source': name, 'n': len(raw), 'max_abs_replay_error': float(np.max(abs(raw - expected)))})
        del values
    write_json(public_root / REFERENCE / 'pls_adapter_audit.json', {'passed': True, 'models': audits,
        'adapter_sha256': digest(Path(__file__).with_name('visual_kernel_components.py'))})
    ensemble = read_json(reference / 'ensemble.json')
    weights = np.asarray(ensemble['weights'])
    keep = np.flatnonzero(weights != 0).tolist()
    assert len(keep) == 3 and all(weights[i] == 0 for i in range(len(weights)) if i not in keep)
    # Removing exactly zero terms changes neither OOF nor validation prices.
    errors = {}
    for filename, column in [('selected_training_oof.csv', 'oof_prediction'), ('predictions.csv', 'prediction')]:
        arrays = [pd.read_csv(root / m / filename, dtype={'imageid': str}) for m in ensemble['models']]
        ids = frame.loc[~valid if column == 'oof_prediction' else valid, 'imageid'].tolist()
        assert all(a.imageid.tolist() == ids for a in arrays)
        matrix = np.column_stack([a[column] for a in arrays])
        delta = np.max(abs(matrix @ weights - matrix[:, keep] @ weights[keep]))
        assert delta < 1e-7
        errors[column] = float(delta)
    relative = Path('15_fixed_ensembles/pe_pls_sparse_export_reference') / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out, public = root / relative, public_root / relative
    out.mkdir(parents=True, exist_ok=False)
    sparse = dict(ensemble)
    for field in ('models', 'weights', 'model_sha256', 'prediction_sha256'):
        sparse[field] = [ensemble[field][i] for i in keep]
    write_json(out / 'ensemble.json', sparse)
    for filename in ('predictions.csv', 'metrics.json', 'prediction_checks.json'):
        shutil.copyfile(reference / filename, out / filename)
    write_json(public / 'registration.json', {'method': 'Exact-zero compilation before refit',
        'reference': REFERENCE, 'reference_ensemble_sha256': digest(reference / 'ensemble.json'),
        'kept_indices': keep, 'max_abs_composition_errors': errors, 'no_new_selection': True,
        'test_images_used': False, 'mllm_components': False, 'source_sha256': digest(__file__)})
    write_json(public / 'results.json', read_json(public_root / REFERENCE / 'results.json'))
    write_json(public / 'status.json', {'status': 'completed'})
    prepared = prepare(out)
    write_json(public_root / REFERENCE / 'export_queue.json', {'status': 'prepared', 'prepared_run': str(prepared)})
    audit_refit(prepared)
    infer(prepared)
    audit_submission(prepared)
    write_json(public_root / REFERENCE / 'export_queue.json', {'status': 'completed', 'prepared_run': str(prepared), 'independent_audit_passed': True})
    print('AUDITED_CSV', prepared / 'predictions.csv', flush=True)


if __name__ == '__main__':
    main()
