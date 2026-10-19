"""Conditional equivalence preflight for frozen PE-G prefix reuse, no full cache."""
import os
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
import timm.models.eva as eva_module
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from pe_lora_price_screen import QueryPrice
from pe_giant_lora_preflight import unit
from pe_core_giant_preflight import REVISION

SCREEN=Path('20_price_lora/pe_giant_query_lora/20261014T112652Z_0fdd4154')

def main():
    screen_public=REPOSITORY/'output/experiment_record'/SCREEN
    assert read_json(screen_public/'status.json')['status']=='completed'
    assert read_json(screen_public/'results.json')['screen_passed'], 'No prefix experiment for a failed screen'
    assert read_json(screen_public/'checkpoint_audit_command_queue.json')['status']=='completed'
    assert read_json(screen_public/'checkpoint_replay_audit.json')['status']=='passed'
    reg=read_json(screen_public/'registration.json')
    assert digest(local_path('split_path'))==reg['split_sha256']
    for name,sha in {'pe_lora_price_screen.py':reg['source_sha256'],**reg['dependency_sha256']}.items():
        assert digest(Path(__file__).with_name(name))==sha
    head_path=local_path('records_root')/reg['source_kernel']/'head.npz'
    assert digest(head_path)==reg['source_head_sha256']
    weights=local_path('cache_root')/'pe_core_giant_weights'/REVISION
    provenance=read_json(weights/'download.json')
    assert provenance['files']['model.safetensors']['publisher_lfs_verified']
    assert digest(weights/'model.safetensors')==provenance['files']['model.safetensors']['sha256']
    frame=pd.read_csv(local_path('split_path'),dtype={'imageid':str})
    rows=frame.loc[frame.partition.eq('train')].iloc[:3]
    relative=Path('19_image_embedding_reference/pe_lora_prefix_equivalence_preflight')/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
    public=REPOSITORY/'output/experiment_record'/relative
    out=local_path('records_root')/relative
    out.mkdir(parents=True,exist_ok=False)
    registration={'source_sha256':digest(__file__),'screen_registration_sha256':digest(screen_public/'registration.json'),
        'screen_results_sha256':digest(screen_public/'results.json'),'checkpoint_audit_sha256':digest(screen_public/'checkpoint_replay_audit.json'),
        'eva_implementation_sha256':digest(eva_module.__file__),'head_sha256':digest(head_path),
        'base_checkpoint_sha256':provenance['files']['model.safetensors']['sha256'],
        'config_sha256':digest(weights/'config.json'),'split_sha256':reg['split_sha256'],
        'seed':2026,'training_imageids':rows.imageid.tolist(),
        'provided_training_image_sha256':{name:digest(local_path('data_root')/'train'/name) for name in rows.imageid},
        'optimizer_steps_per_path':2,
        'mini_batch_size':3,'optimizer':reg['optimizer'],'teacher_drift_coefficient':.1,
        'feature_tolerance':{'rtol':1e-4,'atol':3e-6},'price_tolerance':{'rtol':1e-7,'atol':1e-4},
        'gradient_update_tolerance':{'rtol':1e-4,'atol':1e-6},
        'scope':'Three training images only: full two-view graph versus FP16 input/rope replay from block48, two AdamW updates from identical adapter initialization',
        'no_full_cache_created':True,'validation_images_used':False,'test_images_used':False,
        'price_generalization_evaluated':False,'automatic_training_queue':False}
    write_json(public/'registration.json',registration)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    def status(state,**extra):
        write_json(public/'status.json',{'status':state,'pid':os.getpid(),**extra})
    try:
        status('loading_gpu')
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        torch.cuda.reset_peak_memory_stats()
        with np.load(head_path,allow_pickle=False) as head:
            model=QueryPrice(head)
        assert len(model.visual.blocks)==50 and not getattr(model.visual,'rope_mixed',False)
        params={name:p for name,p in model.visual.named_parameters() if p.requires_grad}
        assert sum(p.numel() for p in params.values())==73728
        initial={name:p.detach().cpu().clone() for name,p in params.items()}
        captures=[]
        def hook(module,args,kwargs):
            x=args[0]
            rope=kwargs['rope']
            assert x.shape==(1,1024,1536) and x.dtype==torch.float16 and not x.requires_grad
            assert isinstance(rope,torch.Tensor) and not rope.requires_grad
            assert kwargs.get('attn_mask') is None and kwargs.get('is_causal') is False
            captures.append((x.detach().cpu().clone(),rope.detach().cpu().clone()))
        handle=model.visual.blocks[48].register_forward_pre_hook(hook,with_kwargs=True)
        images,teachers,prefixes=[],[],[]
        probe_hashes={}
        with torch.no_grad():
            for i,row in enumerate(rows.itertuples()):
                with Image.open(local_path('data_root')/'train'/row.imageid) as im:
                    image=ImageOps.exif_transpose(im).convert('RGB')
                captures.clear()
                zero_query=model.query(image).detach()
                teacher=model.anchors[i:i+1].detach().clone()
                np.testing.assert_allclose(zero_query.cpu().numpy(),teacher.cpu().numpy(),rtol=1e-4,atol=3e-6)
                teachers.append(teacher)
                assert len(captures)==2
                restored=[]
                for view,(x,rope) in enumerate(captures):
                    stem=f'probe_{i}_view_{view}'
                    xfile,rfile=out/(stem+'_tokens.npy'),out/(stem+'_rope.npy')
                    np.save(xfile,x.numpy(),allow_pickle=False)
                    np.save(rfile,rope.numpy(),allow_pickle=False)
                    probe_hashes[xfile.name]=digest(xfile)
                    probe_hashes[rfile.name]=digest(rfile)
                    xback=torch.from_numpy(np.load(xfile,allow_pickle=False))
                    rback=torch.from_numpy(np.load(rfile,allow_pickle=False))
                    assert torch.equal(x,xback) and torch.equal(rope,rback)
                    restored.append((xback.to('cuda'),rback.to('cuda')))
                prefixes.append(restored)
                images.append(image)
        handle.remove()
        def cached_query(i):
            values=[]
            for value,rope in prefixes[i]:
                for block in model.visual.blocks[48:]:
                    value=block(value,rope=rope,attn_mask=None,is_causal=False)
                value=model.visual.norm(value)
                values.append(unit(model.visual.forward_head(value)))
            value=torch.cat(values,dim=1)
            value=(value/value.norm(dim=-1,keepdim=True)).float().double()
            return value/value.norm(dim=-1,keepdim=True)
        trajectories={}
        times={}
        for path in ['full','cached']:
            with torch.no_grad():
                for name,p in params.items(): p.copy_(initial[name])
            optimizer=torch.optim.AdamW(list(params.values()),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
            records=[]
            torch.cuda.synchronize()
            began=time.perf_counter()
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                features,prices,losses=[],[],[]
                for i,row in enumerate(rows.itertuples()):
                    query=model.query(images[i]) if path=='full' else cached_query(i)
                    price=model.price(query)
                    loss=(price.clamp(min=0)/1000-float(row.price)/1000).square().mean()+.1*(query-teachers[i]).square().sum()
                    assert torch.isfinite(loss)
                    (loss/len(images)).backward()
                    features.append(query.detach().cpu().numpy())
                    prices.append(float(price.detach().item()))
                    losses.append(float(loss.detach().item()))
                gradients={name:p.grad.detach().cpu().clone() for name,p in params.items()}
                assert all(torch.isfinite(g).all() for g in gradients.values())
                assert sum(float(g.norm()) for g in gradients.values())>0
                assert all(p.grad is None for p in model.visual.parameters() if not p.requires_grad)
                torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True)
                optimizer.step()
                records.append({'features':np.concatenate(features),'prices':np.asarray(prices),'losses':np.asarray(losses),
                    'gradients':gradients,'parameters':{name:p.detach().cpu().clone() for name,p in params.items()}})
            torch.cuda.synchronize()
            times[path]=time.perf_counter()-began
            trajectories[path]=records
        comparisons=[]
        for step in range(2):
            full,cached=[trajectories[p][step] for p in ['full','cached']]
            np.testing.assert_allclose(full['features'],cached['features'],rtol=1e-4,atol=3e-6)
            np.testing.assert_allclose(full['prices'],cached['prices'],rtol=1e-7,atol=1e-4)
            for kind in ['gradients','parameters']:
                for name in params:
                    np.testing.assert_allclose(full[kind][name].numpy(),cached[kind][name].numpy(),rtol=1e-4,atol=1e-6)
            comparisons.append({'step':step+1,'feature_max_error':float(abs(full['features']-cached['features']).max()),
                'price_max_error':float(abs(full['prices']-cached['prices']).max()),
                'gradient_max_error':max(float((full['gradients'][n]-cached['gradients'][n]).abs().max()) for n in params),
                'parameter_update_max_error':max(float((full['parameters'][n]-cached['parameters'][n]).abs().max()) for n in params)})
        result={'passed':True,'comparisons':comparisons,'small_probe_seconds':times,'probe_file_sha256':probe_hashes,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'no_full_cache_created':True,
            'full_training_trajectory_verified':False,'price_generalization_evaluated':False,'automatic_training_queue':False}
        write_json(public/'results.json',result)
        status('completed')
        print(result,flush=True)
    except BaseException as exc:
        status('failed',error=repr(exc))
        raise

if __name__=='__main__':
    main()
