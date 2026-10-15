"""Run a preregistered supervised-distance readout after PE-Core-G extraction."""
from pathlib import Path
import argparse
import os
import time
import uuid
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from encoder_oof_fusion import resolve_encoder
from pe_giant_incremental_fusion import SPEC


def preregister():
    relative = Path('19_image_embedding_reference/pe_giant_pls_planned') / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out = local_path('records_root') / relative
    public = REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    write_json(public / 'registration.json', {'encoder': SPEC, 'source_sha256': digest(__file__),
        'metric_source_sha256': digest(Path(__file__).with_name('visual_price_metric_kernel.py')),
        'feature_loader_source_sha256': digest(Path(__file__).with_name('visual_kernel_submission.py')),
        'split_sha256': digest(local_path('split_path')),
        'comparison': 'Fixed PE-Core-G ordinary image RBF versus same image RBF multiplied by fold-fitted supervised PLS RBF',
        'image_gamma': 'Freeze ordinary image gamma selected by the existing training GroupKFold search',
        'pls_fit_components': 16, 'score_components': [4, 16], 'pls_gammas': [.01, .1], 'alphas': [.1, 1., 10.],
        'null_control': 'Ordinary image RBF with its training-selected alpha, reuse existing OOF',
        'selection': 'Original-price unweighted training GroupKFold3 MSE; fixed validation once after selection',
        'caveat': 'Source image gamma previously selected on the same training folds; not nested unbiased CV',
        'test_images_used': False, 'mllm_components': False, 'automatic_test_export': False})
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    write_json(public / 'status.json', {'status': 'preregistered'})
    print(out, flush=True)


def run(out):
    out = out.resolve()
    public = REPOSITORY / 'output/experiment_record' / out.relative_to(local_path('records_root'))
    registration = read_json(public / 'registration.json')
    assert read_json(public / 'status.json')['status'] == 'preregistered'
    assert digest(__file__) == registration['source_sha256']
    assert digest(local_path('split_path')) == registration['split_sha256']
    assert digest(Path(__file__).with_name('visual_kernel_submission.py')) == registration['feature_loader_source_sha256']
    metric_source = Path(__file__).with_name('visual_price_metric_kernel.py')
    assert digest(metric_source) == registration['metric_source_sha256']
    try:
        write_json(public / 'status.json', {'status': 'running', 'pid': os.getpid()})
        source = resolve_encoder(registration['encoder'])
        head_sha = digest(local_path('records_root') / source / 'head.npz')
        write_json(public / 'resolved_source.json', {'source': source, 'source_head_sha256': head_sha})
        from visual_price_metric_kernel import main
        main(source, 'pe_core_giant_price_metric_kernel', registration['metric_source_sha256'], head_sha)
        write_json(public / 'status.json', {'status': 'completed', 'pid': os.getpid()})
    except BaseException as exc:
        write_json(public / 'status.json', {'status': 'failed', 'error': repr(exc), 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['preregister', 'run'], required=True)
    parser.add_argument('--protocol-dir', type=Path)
    args = parser.parse_args()
    if args.phase == 'preregister':
        preregister()
    elif args.protocol_dir:
        run(args.protocol_dir)
    else:
        parser.error('run requires protocol-dir')
