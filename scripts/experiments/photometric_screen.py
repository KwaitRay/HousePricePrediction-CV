"""Controlled scratch-AlexNet brightness/contrast screen; no automatic promotion.

Usage: python photometric_screen.py prepare|smoke|launch --workdir PATH
The launch command runs only seed2026 control, brightness and contrast, in order.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from common.config import ROOT, resolve, validate, read_json, write_json, digest, differences
from common.local_paths import local_path

NAMES = ('control', 'brightness', 'contrast')

def state(work, status, **extra):
    write_json(work/'status.json', dict(status=status,pid=os.getpid(),
               updated_utc=datetime.now(timezone.utc).isoformat(),**extra))

def prepare(work):
    import numpy as np
    import torch
    from preparation.split import load_split
    from preparation.images import HouseImages, preview
    torch.set_num_threads(4)
    work.mkdir(parents=True,exist_ok=True)
    if (work/'registration.json').exists(): raise FileExistsError('Use a new workdir for a new protocol')
    base=validate(resolve(ROOT/'configs/selected/training.json'))
    data,meta=load_split(base)
    train=data[data.partition.eq('train')]
    valid=data[data.partition.eq('validation')]
    assert len(train)==6399 and len(valid)==1601 and not set(train.group_id)&set(valid.group_id)
    configs={}
    for name in NAMES:
        c=deepcopy(base)
        for key in list(c):
            if key.startswith('_') or key=='selection': del c[key]
        c.update(seed=2026,device='cuda',output_root=str(work/'artifacts'))
        c['experiment'].update(id='photometric_'+name,name='Photometric screen: '+name,
            stage='02_training' if name=='control' else '03_augmentation',
            question='Does this mild photometric augmentation improve held-out price regression?',
            comparison='Fresh local unaugmented AlexNet, same seed and all non-augmentation settings',
            changed_fields=['augmentation'],enforce_changes=False,status='ready')
        c['augmentation']={} if name=='control' else {name:dict(probability=1.,range=[.9,1.1])}
        write_json(work/'configs'/f'{name}.json',c)
        configs[name]=validate(resolve(work/'configs'/f'{name}.json'))
    for name in NAMES[1:]:
        diff=differences(configs['control'],configs[name])
        assert all(key=='augmentation' or key.startswith('augmentation.') for key in diff),diff
    sources=list(ROOT.glob('*.py'))+[p for d in ('common','preparation','models','training') for p in (ROOT/d).glob('*.py')]
    registration=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=2026,groups=list(NAMES),
        split_sha256=digest(local_path('split_path')),data_manifest=meta,
        training=base['training'],preprocess=base['preprocess'],model=base['model'],target=base['target'],
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in sources},
        config_hashes={n:digest(work/'configs'/f'{n}.json') for n in NAMES},
        input_contract='Original EXIF-corrected RGB -> online training-only augmentation -> stretch224 -> fixed normalization; no stochastic cache',
        evaluation='Fresh unaugmented control; paired validation MSE/MAE and price-bin errors; historical scores contextual only',
        screening_only=True,automatic_multiseed=False,automatic_selection=False,automatic_export=False,
        test_images_used_for_training=False,original_data_checks='Test CSV and file existence checked by standard split loader; no test inference')
    write_json(work/'registration.json',registration)
    sample=train.sample(16,random_state=2026)
    plain=HouseImages(local_path('data_root'),sample,configs['control'])
    checks={}
    for name in NAMES[1:]:
        c=configs[name]
        ds=HouseImages(local_path('data_root'),sample,c,training=True)
        validation=HouseImages(local_path('data_root'),sample,c,training=False)
        changes=[]
        for i in range(len(sample)):
            before=torch.random.get_rng_state().clone()
            a=ds[i];b=ds[i]
            assert torch.equal(a[0],b[0]) and a[1:]==b[1:]
            assert torch.equal(before,torch.random.get_rng_state())
            event=json.loads(a[3])[name]
            assert event['triggered'] and .9<=event['factor']<=1.1
            assert a[0].shape==(3,224,224) and torch.isfinite(a[0]).all()
            assert torch.equal(validation[i][0],plain[i][0])
            ds.epoch=1
            changes.append(not torch.equal(a[0],ds[i][0]))
            ds.epoch=0
        assert any(changes)
        checks[name]=dict(repeated_input_equal=True,global_rng_unchanged=True,
                         validation_equal_to_control=True,epoch_changed=sum(changes),samples=len(sample))
        preview(local_path('data_root'),train,c,work/f'{name}_preview.png',count=4,views=3)
    write_json(work/'input_checks.json',checks)
    state(work,'prepared')
    print(json.dumps(checks,indent=2),flush=True)

def check(work):
    reg=read_json(work/'registration.json')
    for name,sha in reg['source_hashes'].items():
        if digest(ROOT/name)!=sha: raise ValueError(f'Source changed: {name}')
    for name,sha in reg['config_hashes'].items():
        if digest(work/'configs'/f'{name}.json')!=sha: raise ValueError(f'Config changed: {name}')
    assert digest(local_path('split_path'))==reg['split_sha256']
    return reg

def worker(work,name,smoke):
    check(work)
    import run
    out=run.experiment(work/'configs'/f'{name}.json',smoke=smoke)
    write_json(work/(f'smoke_{name}.json' if smoke else f'run_{name}.json'),dict(path=str(out)))

def child(work,name,smoke,deadline):
    command=[sys.executable,'-X','utf8','-u',str(Path(__file__).resolve()),'worker',
             '--workdir',str(work),'--name',name]
    if smoke:command.append('--smoke')
    with (work/(('smoke_' if smoke else '')+name+'.log')).open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        state(work,'smoke_running' if smoke else 'training',group=name,child_pid=proc.pid,
              deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat())
        try:code=proc.wait(timeout=max(1,deadline-time.time()))
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT,check=False)
            proc.wait(timeout=30)
            raise TimeoutError('Queue time budget reached; saved checkpoints retained, run is incomplete')
    if code:raise RuntimeError(f'{name} failed, exit{code}; see log')
    print('Completed',name,'smoke' if smoke else 'training',flush=True)

def summarize(work):
    import numpy as np
    import pandas as pd
    control=Path(read_json(work/'run_control.json')['path'])
    init=read_json(control/'model_summary.json')['initialization_sha256']
    reference=pd.read_csv(control/'predictions.csv')
    rows=[]
    for name in NAMES:
        path=Path(read_json(work/f'run_{name}.json')['path'])
        assert read_json(path/'status.json')=={'status':'completed','kind':'development'}
        assert read_json(path/'model_summary.json')['initialization_sha256']==init
        hist=pd.read_csv(path/'history.csv')
        assert hist.samples.eq(6399).all() and hist.updates.eq(200).all()
        pred=pd.read_csv(path/'predictions.csv')
        assert pred.imageid.tolist()==reference.imageid.tolist()
        np.testing.assert_array_equal(pred.price,reference.price)
        if name!='control':
            events=read_json(path/'augmentation_history.json')
            assert len(events)==len(hist)
            for e in events:
                assert set(e['operations'])=={name}
                stats=e['operations'][name]
                assert stats['samples']==stats['triggered']==6399 and stats['skipped']==0
                assert .9<=stats['observed_ranges']['factor'][0]<=stats['observed_ranges']['factor'][1]<=1.1
        metrics=read_json(path/'metrics.json')
        np.testing.assert_allclose(metrics['mse'],np.mean((pred.prediction-pred.price)**2),rtol=1e-10)
        row=dict(group=name,run=str(path),**metrics,epochs=len(hist),
                 best_epoch=read_json(path/'stopping.json')['best_epoch'],
                 delta_mse=metrics['mse']-read_json(control/'metrics.json')['mse'])
        rows.append(row)
    write_json(work/'results.json',dict(rows=rows,initializations_equal=True,
        all_epoch_sample_counts_verified=True,screening_only=True,automatic_promotion=False,
        limitation='One seed only; best checkpoints use the repeatedly reused validation split. No general or statistically significant improvement claim.'))
    pd.DataFrame(rows).to_csv(work/'summary.csv',index=False)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['prepare','smoke','launch','worker'])
    p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--name',choices=NAMES)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--hours',type=float,default=2.)
    a=p.parse_args();work=a.workdir.resolve()
    try:
        if a.phase=='prepare':prepare(work)
        elif a.phase=='worker':worker(work,a.name,a.smoke)
        else:
            check(work)
            deadline=time.time()+a.hours*3600
            if a.phase=='launch':
                for name in NAMES: assert (work/f'smoke_{name}.json').exists()
                assert not any((work/f'run_{n}.json').exists() for n in NAMES),'Do not overwrite an existing queue'
            for name in NAMES:child(work,name,a.phase=='smoke',deadline)
            if a.phase=='launch':summarize(work)
            state(work,'smoke_completed' if a.phase=='smoke' else 'completed')
    except BaseException as exc:
        state(work,'failed',error=repr(exc))
        raise

if __name__=='__main__':main()
