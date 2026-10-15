"""Preregister a conditional readout comparison, preserving the upstream run."""
from pathlib import Path
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from common.local_paths import local_path
from common.config import read_json,write_json,digest
UPSTREAM=Path('20_price_lora/pe_giant_query_lora/20261014T112652Z_0fdd4154')
up=ROOT/'output/experiment_record'/UPSTREAM
reg=read_json(up/'registration.json')
relative=Path('20_price_lora/pe_giant_lora_coherent_readout')/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
local=local_path('records_root')/relative
public=ROOT/'output/experiment_record'/relative
local.mkdir(parents=True,exist_ok=False)
protocol={'method':'Fixed final price-supervised PE-G LoRA encoder followed by coherent same-encoder RBF bank/head refit and unchanged three-branch blend',
    'upstream':str(UPSTREAM),'upstream_registration_sha256':digest(up/'registration.json'),
    'upstream_source_sha256':reg['source_sha256'],'upstream_dependency_sha256':reg['dependency_sha256'],
    'source_checkpoint_epoch':2,'checkpoint_choice':'Fixed final epoch2 from the original two-epoch budget regardless of validation score; no checkpoint search',
    'upstream_status_observed':read_json(up/'status.json'),
    'upstream_epoch2_metrics_generated_at_registration':any(r['epoch']==2 for r in read_json(up/'history.json')),
    'start_condition':'Upstream run and its queued artifact audit must complete successfully; execute only if the upstream asymmetric-query screen fails. If it passes, prefer the already planned three-seed workflow.',
    'hypothesis':'Student query features changed while original teacher anchors/coefficients stayed fixed. Refitting anchors and coefficients in the same adapted feature space may preserve learned relative price cues and reduce coordinate/readout mismatch.',
    'direct_comparison':'Same trained student epoch2, images, gamma4 alpha1 and fixed baseline blend weights; change the complete RBF bank/head from original teacher space to the adapted student space',
    'baseline_fusion':reg['baseline_fusion'],'baseline_ensemble_sha256':reg['baseline_ensemble_sha256'],
    'original_teacher_kernel':reg['source_kernel'],'original_teacher_head_sha256':reg['source_head_sha256'],
    'split_sha256':reg['split_sha256'],'gamma':4.,'alpha':1.,'price_scale':1000,
    'fit_partition':'6399 train images only; fold/gamma/alpha selection is not repeated',
    'validation_partition':'All original 1601 validation images, original-price unweighted MSE',
    'fixed_blend_weights':True,'new_model_oof_generated':False,'meta_weights_relearned':False,
    'control':'Reproduce original teacher head and full fixed blend on all 1601 validation images before evaluating the adapted-bank alternative',
    'baseline_validation_mse':52524.78420579877,
    'screen_gate':{'minimum_relative_validation_gain':.01},
    'later_adoption':'Only after the readout screen succeeds: same complete training/readout pipeline for seeds2026/2027/2028, mean validation improvement and at least2/3 paired wins. No fresh model OOF or unbiased meta CV claimed.',
    'execution_source_path':'scripts/experiments/pe_lora_coherent_readout.py',
    'execution_binding':'Implementation and the completed upstream epoch2 adapter hash will be frozen in a separate execution_binding.json before extraction; this preregistration is preserved unchanged',
    'feature_family':'pe_giant_lora_price_features','supervised_encoder':True,
    'caveat':'Development validation has been reused. The encoder was supervised on all6399 training labels, so ordinary post-encoder CV would leak fold-heldout labels and is not used.',
    'test_images_used':False,'mllm_components':False,'automatic_test_export':False,
    'full_refit_adapter_supported':False,'source_sha256':digest(__file__)}
write_json(public/'registration.json',protocol)
write_json(public/'status.json',{'status':'preregistered_conditional','upstream':str(UPSTREAM)})
(local/'preregistration_source.py').write_bytes(Path(__file__).read_bytes())
print(local)
