"""Freeze and run existing fold-only PLS readout on the verified whole-view PE-G."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from common.config import digest,write_json
from common.local_paths import local_path
from visual_price_metric_kernel import main
source='19_image_embedding_reference/pe_core_g14_448_whole_kernel/20261014T090655Z_2d55c743'
code=ROOT/'scripts/experiments/visual_price_metric_kernel.py'
registration={'source_kernel':source,'source_sha256':digest(code),
    'source_head_sha256':digest(local_path('records_root')/source/'head.npz'),
    'recipe':'Unchanged existing fold-fitted PLS-score product kernel, 16 fit components, score dimensions4/16, gamma0.01/0.1, alpha0.1/1/10, plus null ordinary kernel',
    'new_factor':'Apply the existing PLS readout to the whole-view PE-G source instead of original crop-only PE-G',
    'fold_fitting_only':True,'test_images_used':False,'automatic_export':False,
    'fusion_baseline':'15_fixed_ensembles/pe_giant_whole_incremental_fusion/20261014T082727Z_c5152cef',
    'fusion_gate':'metaCV >=1% gain, >=2/3 wins and improved validation'}
target=ROOT/'output/experiment_record/improvement_to_50000/whole_pls_launch_registration.json'
assert not target.exists()
write_json(target,registration)
main(source,'pe_core_giant_whole_price_metric_kernel',registration['source_sha256'],registration['source_head_sha256'])
