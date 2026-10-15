"""Copy a frozen full refit, removing only exactly zero composition terms."""
import argparse
import copy
from pathlib import Path
import shutil
import time
import uuid
import numpy as np
import pandas as pd
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY


def main(source):
    source = source.resolve()
    root = local_path('records_root')
    source_relative = source.relative_to(root)
    registration = read_json(source / 'registration.json')
    prepared = read_json(source / 'prepared.json')
    if not prepared['reference_replay_passed'] or not prepared['weights_frozen'] or prepared['full_train_n'] != 8000:
        raise ValueError('Frozen full refit is not verified')
    weights = np.asarray(registration['weights'], dtype=np.float64)
    if len(weights) != len(registration['components']) or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Invalid composition')
    keep = np.flatnonzero(weights != 0).tolist()
    removed = np.flatnonzero(weights == 0).tolist()
    if not removed:
        raise ValueError('No exact zero-weight terms to remove')
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str})
    valid_ids = frame.loc[frame.partition.eq('validation'), 'imageid'].tolist()
    train_ids = frame.loc[frame.partition.eq('train'), 'imageid'].tolist()
    valid, training = [], []
    for component in registration['components']:
        model = root / component['source']
        pv = pd.read_csv(model / 'predictions.csv', dtype={'imageid': str})
        pt = pd.read_csv(model / 'selected_training_oof.csv', dtype={'imageid': str})
        if pv.imageid.tolist() != valid_ids or pt.imageid.tolist() != train_ids:
            raise ValueError('Frozen component IDs differ')
        valid.append(pv.prediction.to_numpy())
        training.append(pt.oof_prediction.to_numpy())
    errors = {}
    for name, values in [('validation', valid), ('training_oof_inputs', training)]:
        values = np.column_stack(values)
        original = values @ weights
        compiled = values[:, keep] @ weights[keep]
        np.testing.assert_allclose(compiled, original, rtol=0, atol=1e-9)
        errors[name] = float(np.max(np.abs(compiled - original)))
    run = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    out = root / '08_final/visual_kernel_ensemble' / run
    public = REPOSITORY / 'output/experiment_record/08_final/visual_kernel_ensemble' / run
    out.mkdir(parents=True, exist_ok=False)
    compiled_registration = copy.deepcopy(registration)
    compiled_registration.update(weights=weights[keep].tolist(),
        models=[registration['models'][i] for i in keep], components=[],
        compiled_from=str(source_relative), original_weights=weights.tolist(),
        zero_weight_components_removed=removed, compilation_source_sha256=digest(__file__),
        compilation='Remove exactly zero terms only; nonzero weights, model parameters and price formula preserved')
    for new_index, old_index in enumerate(keep):
        component = copy.deepcopy(registration['components'][old_index])
        head = source / f'component_{old_index}_full_head.npz'
        if digest(head) != component['full_head_sha256']:
            raise ValueError('Frozen full-training head changed')
        copied = out / f'component_{new_index}_full_head.npz'
        shutil.copyfile(head, copied)
        if digest(copied) != component['full_head_sha256']:
            raise ValueError('Copied head differs')
        if 'full_metric_file' in component:
            metric = source / component['full_metric_file']
            if digest(metric) != component['full_metric_sha256']:
                raise ValueError('Frozen metric changed')
            name = f'component_{new_index}_full_metric.joblib'
            shutil.copyfile(metric, out / name)
            if digest(out / name) != component['full_metric_sha256']:
                raise ValueError('Copied supervised metric differs')
            component['full_metric_file'] = name
        component['original_component_index'] = old_index
        compiled_registration['components'].append(component)
    for name in ('preparation_source.py', 'component_adapter_source.py'):
        shutil.copyfile(source / name, out / name)
    (out / 'compilation_source.py').write_bytes(Path(__file__).read_bytes())
    # This is a fresh inference attempt; retain the failed source run and its snapshot.
    for name in ('inference_source_sha256', 'test_images_role'):
        compiled_registration.pop(name, None)
    compiled_registration['test_images_used'] = False
    write_json(out / 'registration.json', compiled_registration)
    write_json(public / 'registration.json', compiled_registration)
    write_json(out / 'prepared.json', prepared)
    write_json(public / 'compilation_audit.json', {'passed': True, 'removed_exact_zero_indices': removed,
        'kept_indices': keep, 'nonzero_weights_unchanged': True, 'model_files_byte_identical': True,
        'all_1601_validation_and_6399_training_inputs_replayed': True, 'max_abs_composition_errors': errors,
        'test_labels_used': False, 'no_new_training_or_parameter_selection': True})
    write_json(public / 'status.json', {'status': 'prepared', 'test_inference_completed': False})
    print(out, flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    main(parser.parse_args().source_run)
