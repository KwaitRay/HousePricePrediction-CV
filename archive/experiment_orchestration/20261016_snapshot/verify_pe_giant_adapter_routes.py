from pathlib import Path
import ast
import hashlib
import json

root = Path(__file__).resolve().parent.parent
code = root / 'scripts/experiments'
prior = root / '.local/experiment_record/08_final/visual_kernel_ensemble'
inference = code / 'visual_kernel_inference.py'
submission = code / 'visual_kernel_submission.py'
old_inference = (prior / '20261014T044646Z_7d1af3eb/inference_source_at_launch.py').read_text(encoding='utf-8')
old_submission = (prior / '20261014T064058Z_16e1be61/preparation_source.py').read_text(encoding='utf-8')
branch = "    if family == 'pe_core_giant_features':\n        from pe_core_giant_encoder import make_encoder\n        return make_encoder(manifest)\n"
current_inference = inference.read_text(encoding='utf-8')
current_submission = submission.read_text(encoding='utf-8')
assert current_inference.count(branch) == 1
assert current_inference.replace(branch, '', 1) == old_inference
assert current_submission.replace(", 'pe_core_giant_features'", '', 1) == old_submission
for path in (inference, submission, code / 'pe_core_giant_encoder.py'):
    ast.parse(path.read_text(encoding='utf-8'))
extraction = root / 'output/experiment_record/19_image_embedding_reference/pe_core_g14_448/20261014T065137Z_b822c3d4'
registration = json.loads((extraction / 'registration.json').read_text(encoding='utf-8'))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(code / 'pe_core_giant_preflight.py') == registration['loader_source_sha256']
assert sha(code / 'frozen_kernel_search.py') == registration['kernel_search_source_sha256']
record = {'new_encoder_branch_and_family_only': True, 'all_existing_encoder_and_refit_formulas_byte_equivalent_after_route_removal': True,
    'active_extraction_dependencies_unchanged': True, 'gpu_feature_replay_pending': True,
    'compatible_source_sha256': {path.name: sha(path) for path in (inference, submission, code / 'pe_core_giant_encoder.py')},
    'scope': 'Static dispatch preservation and active dependency identity; numerical GPU replay remains required', 'test_images_used': False}
(extraction / 'export_adapter_preparation.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
