"""After PE extraction exits, register its exact view adapter and one fixed fusion."""
import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY

BRANCH = '''    if family == 'pe_core_multiview':
        from pe_core_multiview import views
        if digest(Path(__file__).with_name('pe_core_multiview.py')) != manifest['source_sha256']:
            raise ValueError('Registered PE multiview transformation source changed')
        original_folder = local_path('cache_root') / 'pe_core_features' / manifest['original_cache_key']
        original_path = original_folder / 'manifest.json'
        original = read_json(original_path)
        if digest(original_path) != manifest['original_manifest_sha256'] or original['model'] != manifest['model'] or original['revision'] != manifest['revision']:
            raise ValueError('Registered PE full-image source changed')
        if manifest['feature_dim'] != 3072 or manifest['batch_size'] != 4 or original.get('mllm_components') is not False:
            raise ValueError('Unsupported PE multiview feature contract')
        base_encode = encoder('pe_core_features', original)
        def encode(images):
            whole, detail = views(images)
            return normalize(np.concatenate([base_encode(images), base_encode(whole), base_encode(detail)], axis=1))
        return encode
'''


def main(reference, expected_source):
    if digest(__file__) != expected_source:
        raise ValueError('Queued fusion pipeline source changed')
    reference = reference.resolve()
    public = REPOSITORY / 'output/experiment_record' / reference.relative_to(local_path('records_root'))
    state = read_json(public / 'status.json')
    if state['status'] != 'completed' or not state.get('feature_extraction_completed'):
        raise ValueError('PE feature extraction must finish before editing shared encoder routes')
    registration = read_json(public / 'registration.json')
    if registration['cache_family'] != 'pe_core_multiview':
        raise ValueError('Only the pinned PE multiview extraction is supported')
    def status(value, **extra):
        write_json(public / 'fusion_pipeline_status.json', {'status': value, 'pid': os.getpid(), **extra})
    try:
        status('registering_adapter')
        scripts = Path(__file__).parent
        inference = scripts / 'visual_kernel_inference.py'
        submission = scripts / 'visual_kernel_submission.py'
        old_inference, old_submission = inference.read_text(encoding='utf-8'), submission.read_text(encoding='utf-8')
        for path in (inference, submission):
            if digest(path) != registration['dependency_sha256'][path.name] or digest(reference / path.name) != digest(path):
                raise ValueError('Shared source changed since PE preregistration; do not overwrite')
        anchor = "    if family == 'siglip2_giant_features':\n"
        family_anchor = "'pe_core_features', 'siglip2_giant_features'"
        if old_inference.count(anchor) != 1 or old_submission.count(family_anchor) != 1:
            raise ValueError('Expected one unmodified encoder dispatch and family whitelist')
        new_inference = old_inference.replace(anchor, BRANCH + anchor, 1)
        new_submission = old_submission.replace(family_anchor,
            "'pe_core_features', 'siglip2_giant_features', 'pe_core_multiview'", 1)
        ast.parse(new_inference)
        ast.parse(new_submission)
        if new_inference.replace(BRANCH, '', 1) != old_inference:
            raise ValueError('An existing encoder route changed')
        if new_submission.replace("'pe_core_features', 'siglip2_giant_features', 'pe_core_multiview'", family_anchor, 1) != old_submission:
            raise ValueError('An existing feature/refit formula changed')
        # Original source snapshots remain immutable in the extraction record.
        inference.write_text(new_inference, encoding='utf-8')
        submission.write_text(new_submission, encoding='utf-8')
        (reference / 'multiview_compatible_inference.py').write_bytes(inference.read_bytes())
        (reference / 'multiview_compatible_submission.py').write_bytes(submission.read_bytes())
        adapter_audit = {'passed': True, 'only_new_encoder_branch_and_whitelist_added': True,
            'all_existing_encoder_and_kernel_formulas_unchanged': True,
            'prior_source_sha256': registration['dependency_sha256'],
            'compatible_source_sha256': {p.name: digest(p) for p in (inference, submission)},
            'view_source_sha256': digest(scripts / 'pe_core_multiview.py'),
            'test_images_used': False, 'mllm_components': False}
        write_json(public / 'adapter_compatibility_audit.json', adapter_audit)
        status('probing_registered_views')
        from PIL import Image, ImageOps
        import torch
        from visual_kernel_inference import encoder
        from visual_kernel_submission import cache_array
        frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
        cost = read_json(public / 'cost.json')
        values, manifest = cache_array('pe_core_multiview', cost['feature_cache_key'], frame)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        positions = np.random.default_rng(2026).choice(8000, 8, replace=False)
        images = []
        for name in frame.imageid.iloc[positions]:
            with Image.open(local_path('data_root') / 'train' / name) as im:
                images.append(ImageOps.exif_transpose(im).convert('RGB'))
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        encode = encoder('pe_core_multiview', manifest)
        actual = np.concatenate([encode(images[offset:offset + 4]) for offset in range(0, 8, 4)])
        np.testing.assert_allclose(actual, values[positions], rtol=1e-4, atol=3e-6)
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, rtol=0, atol=1e-8)
        del encode, actual, values
        torch.cuda.empty_cache()
        adapter_audit.update(eight_actual_training_feature_probes_passed=True,
            probe_imageids=frame.imageid.iloc[positions].tolist())
        write_json(public / 'adapter_compatibility_audit.json', adapter_audit)
        # Freeze the joint model before the new PE kernel has been evaluated.
        from encoder_oof_fusion import preregister
        from siglip_giant_encoder import REPO, REVISION
        specs = [
            {'name': 'siglip2_giant_384_kernel', 'family': 'siglip2_giant_features', 'model': REPO, 'revision': REVISION},
            {'name': 'pe_core_l14_336_multiview_kernel', 'family': 'pe_core_multiview',
                'model': manifest['model'], 'revision': manifest['revision']}]
        baseline = '15_fixed_ensembles/encoder_family_oof_fusion/20261014T012302Z_b883fad1'
        out = preregister(baseline, specs, 'pe_multiview_giant_oof_fusion', 1e-12)
        write_json(public / 'joint_fusion_protocol.json', {'protocol_dir': str(out),
            'baseline_public_mse': 53754.87960, 'baseline_validation_mse': 57982.72725781495,
            'new_pe_prices_not_read_for_protocol': True, 'no_extra_model_subset_search': True})
        queue = public / 'pe_multiview_kernel_after_features_command_queue.json'
        kernel_controller = read_json(queue)['controller_pid']
        command_file = out / 'joint_fusion_after_pe_kernel_command.json'
        write_json(command_file, ['scripts/experiments/encoder_oof_fusion.py', '--phase', 'run',
            '--protocol-dir', str(out), '--gpu-predecessor-status', str(queue),
            '--gpu-predecessor-pid', str(kernel_controller)])
        command = [sys.executable, '-X', 'utf8', str(scripts / 'after_process_command.py'),
            '--status', str(queue), '--pid', str(kernel_controller), '--command', str(command_file)]
        status('waiting_for_kernel', protocol_dir=str(out), command=command)
        result = subprocess.run(command, check=False)
        if result.returncode:
            raise RuntimeError('PE/Giant joint fusion pipeline failed; preserve all records')
        status('completed', protocol_dir=str(out))
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    main(args.reference, args.source_sha256)
