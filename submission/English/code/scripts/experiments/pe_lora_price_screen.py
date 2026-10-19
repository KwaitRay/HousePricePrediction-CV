"""Price-supervised LoRA query adaptation against a frozen teacher RBF bank."""
import argparse
import os
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from safetensors.torch import save_file
from timm.data import create_transform
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from common.metrics import save_predictions
from visual_kernel_submission import features_for
from pe_core_giant_preflight import REVISION, build_meta, stream_load
from pe_giant_lora_preflight import LowRankLinear, unit
from pe_giant_whole_view import whole_view

SOURCE = '19_image_embedding_reference/pe_core_g14_448_whole_kernel/20261014T090655Z_2d55c743'
BASELINE = '15_fixed_ensembles/pe_giant_whole_incremental_fusion/20261014T082727Z_c5152cef'
PREFLIGHT = '19_image_embedding_reference/pe_lora_prototype_preflight/20261014T111614Z_95bf28f8'
DEPENDENCIES = ['pe_core_giant_preflight.py','pe_giant_lora_preflight.py','pe_giant_whole_view.py','visual_kernel_submission.py']


class QueryPrice(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        folder = local_path('cache_root')/'pe_core_giant_weights'/REVISION
        config = read_json(folder/'config.json')
        self.visual = stream_load(build_meta(config),folder/'model.safetensors').eval().requires_grad_(False)
        cfg = config['pretrained_cfg']
        self.processor = create_transform(input_size=cfg['input_size'],interpolation=cfg['interpolation'],
            mean=cfg['mean'],std=cfg['std'],crop_pct=cfg['crop_pct'],crop_mode=cfg['crop_mode'],is_training=False)
        for i in range(len(self.visual.blocks)-2,len(self.visual.blocks)):
            for leaf in ('qkv','proj'):
                parent = self.visual.get_submodule(f'blocks.{i}.attn')
                setattr(parent,leaf,LowRankLinear(getattr(parent,leaf)))
        self.visual.eval()
        self.register_buffer('anchors',torch.as_tensor(head['train_features'].copy(),device='cuda',dtype=torch.float64))
        self.register_buffer('coefficients',torch.as_tensor(head['coefficients'].copy(),device='cuda',dtype=torch.float64))
        self.gamma, self.mean = float(head['gamma']),float(head['target_mean'])

    def query(self,image):
        values = [unit(self.visual(self.processor(im).unsqueeze(0).to('cuda',torch.float16)))
            for im in [image,whole_view(image)]]
        value = torch.cat(values,dim=1)
        value = (value/value.norm(dim=-1,keepdim=True)).float().double()
        return value/value.norm(dim=-1,keepdim=True)

    def price(self,value):
        return (torch.exp(-self.gamma*torch.clamp(2-2*value@self.anchors.T,min=0))@self.coefficients+self.mean)*1000


def preregister(seed):
    root = local_path('records_root')
    pre = REPOSITORY/'output/experiment_record'/PREFLIGHT
    assert read_json(pre/'status.json')['status']=='completed' and read_json(pre/'results.json')['passed']
    relative = Path('20_price_lora/pe_giant_query_lora')/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
    out,public = root/relative,REPOSITORY/'output/experiment_record'/relative
    out.mkdir(parents=True,exist_ok=False)
    registration = {'method':'Price-supervised LoRA query adaptation with frozen teacher prototypes and fixed fusion weights',
        'source_kernel':SOURCE,'baseline_fusion':BASELINE,'preflight_reference':PREFLIGHT,
        'preflight_results_sha256':digest(pre/'results.json'),'source_sha256':digest(__file__),
        'dependency_sha256':{name:digest(Path(__file__).with_name(name)) for name in DEPENDENCIES},
        'split_sha256':digest(local_path('split_path')),'source_head_sha256':digest(root/SOURCE/'head.npz'),
        'baseline_ensemble_sha256':digest(root/BASELINE/'ensemble.json'),
        'source_feature_registration_sha256':digest(REPOSITORY/'output/experiment_record'/SOURCE/'registration.json'),
        'seed':seed,'epochs':2,'microbatch':1,'effective_batch':16,'rank':4,'adapter_alpha':4,
        'trainable_modules':'Last two block qkv/proj only; FP32 adapters, frozen FP16 base and learned projection',
        'optimizer':{'name':'AdamW','lr':1e-3,'weight_decay':1e-4,'betas':[.9,.999],'eps':1e-8},
        'gradient_clip_norm':1.,'teacher_drift_coefficient':.1,
        'loss':'MSE(max(0,query-price)/1000,real-price/1000) + 0.1*sum((query-feature-teacher-feature)^2)',
        'readout':'Fixed original training teacher anchors, coefficients, gamma and prior mean; adapt query encoder only',
        'augmentation':False,'selection':'Complete fixed validation MSE of unchanged three-branch convex blend; cached epoch-zero reference, two complete epochs, retain best improved checkpoint',
        'control':'All 1601 cached teacher prices replayed; first three train images independently encoded at zero LoRA; no claim all validation images freshly reencoded at epoch zero',
        'screen_gate':{'minimum_relative_validation_gain':.01},
        'later_adoption':'Only after screen succeeds, seeds 2026/2027/2028 require mean validation improvement and at least 2/3 paired wins; no newly learned meta weights',
        'caveat':'Development validation repeatedly used; original base parameters and blend weights already selected. No unbiased nested CV or fresh model OOF claimed.',
        'test_images_used':False,'mllm_components':False,'automatic_test_export':False,'full_refit_adapter_supported':False}
    write_json(public/'registration.json',registration)
    for name in [Path(__file__).name,*DEPENDENCIES]:
        (out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    write_json(public/'status.json',{'status':'preregistered'})
    print(out,flush=True)


def run(out):
    root = local_path('records_root')
    out = out.resolve()
    public = REPOSITORY/'output/experiment_record'/out.relative_to(root)
    reg = read_json(public/'registration.json')
    assert read_json(public/'status.json')['status']=='preregistered'
    assert digest(__file__)==reg['source_sha256'] and digest(local_path('split_path'))==reg['split_sha256']
    for name,sha in reg['dependency_sha256'].items():
        assert digest(Path(__file__).with_name(name))==sha
    assert digest(root/SOURCE/'head.npz')==reg['source_head_sha256']
    assert digest(root/BASELINE/'ensemble.json')==reg['baseline_ensemble_sha256']
    frame = pd.read_csv(local_path('split_path'),dtype={'imageid':str,'group_id':str})
    train,valid = frame.partition.eq('train').to_numpy(),frame.partition.eq('validation').to_numpy()
    assert train.sum()==6399 and valid.sum()==1601
    assert not set(frame.loc[train,'group_id'])&set(frame.loc[valid,'group_id'])
    source_reg = read_json(REPOSITORY/'output/experiment_record'/SOURCE/'registration.json')
    assert digest(REPOSITORY/'output/experiment_record'/SOURCE/'registration.json')==reg['source_feature_registration_sha256']
    features = features_for(source_reg,frame)
    ensemble = read_json(root/BASELINE/'ensemble.json')
    own = [i for i,m in enumerate(ensemble['models']) if Path(m)==Path(SOURCE)]
    assert len(own)==1
    own_index,other = own[0],np.zeros(1601)
    for i,(reference,weight) in enumerate(zip(ensemble['models'],ensemble['weights'])):
        if not weight or i==own_index:
            continue
        assert digest(root/reference/'predictions.csv')==ensemble['prediction_sha256'][i]
        table = pd.read_csv(root/reference/'predictions.csv',dtype={'imageid':str})
        assert table.imageid.tolist()==frame.loc[valid,'imageid'].tolist()
        other += weight*table.prediction.to_numpy()
    reference = pd.read_csv(root/BASELINE/'predictions.csv',dtype={'imageid':str})
    assert reference.imageid.tolist()==frame.loc[valid,'imageid'].tolist()
    baseline_mse = float(np.mean((reference.prediction.to_numpy()-frame.loc[valid,'price'].to_numpy())**2))
    np.testing.assert_allclose(baseline_mse,
        read_json(REPOSITORY/'output/experiment_record'/BASELINE/'results.json')['mse'],rtol=1e-10,atol=1e-6)
    def status(state,**extra):
        write_json(public/'status.json',{'status':state,'pid':os.getpid(),**extra})
    def image_for(name):
        with Image.open(local_path('data_root')/'train'/name) as image:
            return ImageOps.exif_transpose(image).convert('RGB')
    try:
        torch.set_num_threads(2)
        torch.manual_seed(reg['seed'])
        torch.cuda.reset_peak_memory_stats()
        status('loading_gpu')
        with np.load(root/SOURCE/'head.npz',allow_pickle=False) as head:
            model = QueryPrice(head)
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert sum(p.numel() for p in trainable)==73728
        control = []
        with torch.no_grad():
            for start in range(0,1601,128):
                value = torch.as_tensor(features[valid][start:start+128],device='cuda',dtype=torch.float64)
                control.extend(model.price(value).cpu().numpy().tolist())
            indices = np.flatnonzero(train)[:3]
            for i in indices:
                value = model.query(image_for(frame.imageid.iloc[i]))
                np.testing.assert_allclose(value.cpu().numpy(),features[i:i+1],rtol=1e-4,atol=3e-6)
        np.testing.assert_allclose(other+ensemble['weights'][own_index]*np.maximum(control,0),reference.prediction,rtol=1e-7,atol=1e-4)
        write_json(public/'zero_lora_control.json',{'passed':True,'cached_validation_prices_replayed':1601,
            'online_training_image_probes':3,'baseline_validation_mse':baseline_mse})
        optimizer = torch.optim.AdamW(trainable,lr=1e-3,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
        training = frame.loc[train].reset_index(drop=True)
        global_indices = np.flatnonzero(train)
        rng = np.random.default_rng(reg['seed'])
        best,history = baseline_mse,[]
        began = time.perf_counter()
        for epoch in range(1,reg['epochs']+1):
            order = rng.permutation(len(training))
            totals,drifts = [],[]
            for start in range(0,len(order),reg['effective_batch']):
                batch = order[start:start+reg['effective_batch']]
                optimizer.zero_grad(set_to_none=True)
                for offset in batch:
                    row = training.iloc[offset]
                    value = model.query(image_for(row.imageid))
                    raw = model.price(value)
                    teacher = torch.as_tensor(features[global_indices[offset]:global_indices[offset]+1],device='cuda',dtype=torch.float64)
                    price_loss = (torch.clamp(raw,min=0)/1000-float(row.price)/1000).square().mean()
                    drift = (value-teacher).square().sum()
                    loss = price_loss+reg['teacher_drift_coefficient']*drift
                    assert torch.isfinite(loss)
                    (loss/len(batch)).backward()
                    totals.append(float(price_loss.detach()))
                    drifts.append(float(drift.detach()))
                torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
                optimizer.step()
                if start%256==0:
                    status('training',epoch=epoch,images=start+len(batch),total=6399)
                    print(f'LoRA epoch {epoch}: {start+len(batch)}/6399',flush=True)
            values = []
            with torch.no_grad():
                for j,name in enumerate(frame.loc[valid,'imageid']):
                    values.append(float(model.price(model.query(image_for(name))).item()))
                    if j%100==0:
                        status('validating',epoch=epoch,images=j+1,total=1601)
                        print(f'LoRA validation {epoch}: {j+1}/1601',flush=True)
            raw = np.asarray(values)
            prediction = other+ensemble['weights'][own_index]*np.maximum(raw,0)
            epoch_out = out/f'epoch_{epoch}'
            epoch_out.mkdir()
            metrics = save_predictions(epoch_out,frame.loc[valid],prediction,1300)
            save_file({name:p.detach().cpu().contiguous() for name,p in model.visual.named_parameters() if p.requires_grad},str(epoch_out/'adapter.safetensors'))
            row = {'epoch':epoch,**metrics,'mean_training_price_loss':float(np.mean(totals)),
                'mean_teacher_squared_drift':float(np.mean(drifts)),'adapter_sha256':digest(epoch_out/'adapter.safetensors')}
            history.append(row)
            write_json(public/'history.json',history)
            if metrics['mse']<best:
                best = metrics['mse']
                write_json(out/'selected.json',{'epoch':epoch,'adapter_sha256':row['adapter_sha256'],'validation_mse':best})
            print(row,flush=True)
        result = {'baseline_validation_mse':baseline_mse,'best_validation_mse':best,
            'relative_gain':1-best/baseline_mse,'screen_passed':best<baseline_mse*.99,
            'three_seed_adoption_completed':False,'price_model_uses_fixed_prototypes':True,
            'seconds':time.perf_counter()-began,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'test_images_used':False,'export_eligible':False}
        write_json(public/'results.json',result)
        status('completed',screen_passed=result['screen_passed'])
        print(result,flush=True)
    except BaseException as exc:
        status('failed',error=repr(exc))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['preregister','run'],required=True)
    parser.add_argument('--protocol-dir',type=Path)
    parser.add_argument('--seed',type=int,default=2026)
    args = parser.parse_args()
    if args.phase=='preregister':
        preregister(args.seed)
    elif args.protocol_dir:
        run(args.protocol_dir)
    else:
        parser.error('run requires protocol-dir')
