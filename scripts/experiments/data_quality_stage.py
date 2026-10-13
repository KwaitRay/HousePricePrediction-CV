"""Registered scratch-model spatial, quality and train-distribution probes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from copy import deepcopy
from pathlib import Path
import json, subprocess, sys, time, shutil
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as F
from common.config import ROOT, resolve, validate, read_json, write_json, digest
from common.local_paths import local_path, REPOSITORY
from common.metrics import save_predictions
from preparation.images import HouseImages, read_rgb, geometry
from preparation.split import load_split
from models.alexnet import AlexNetRegressor

SEEDS = (2026, 2027, 2028)


def fixed_views(x):
    h, w = x.shape[-2:]
    ch, cw = max(1, round(h * .6)), max(1, round(w * .6))
    return [x[..., y:y+ch, z:z+cw] for y,z in ((0,0),(0,w-cw),(h-ch,0),(h-ch,w-cw))]


def mild_weights(prices):
    bands = np.digitize(prices, [300,700,1300], right=True)
    counts = np.bincount(bands, minlength=4)
    if (counts == 0).any(): raise ValueError('Empty training price band')
    weights = np.clip(np.sqrt(len(prices) / (4 * counts)), .8, 1.25)
    weights /= np.average(weights, weights=counts)
    return {'counts': counts.tolist(), 'weights': weights.tolist(), 'scope':'training price bands only'}


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    public = REPOSITORY / 'output/experiment_record/09_data_quality'
    public.mkdir(parents=True, exist_ok=True)
    work = local_path('records_root') / '09_data_quality' / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    work.mkdir(parents=True, exist_ok=False)
    import msvcrt
    lock = (public/'queue.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    c = validate(resolve(ROOT/'configs/selected/head.json'))
    frame, meta = load_split(c)
    train = frame[frame.partition=='train'].reset_index(drop=True)
    val = frame[frame.partition=='validation'].reset_index(drop=True)
    old = read_json(local_path('records_root')/'07_encoder/20261013T043346Z/state.json')
    refs = {s:Path(old['runs'][f'mean_{s}']) for s in SEEDS}
    for reference in refs.values():
        if read_json(reference/'status.json')['status']!='completed':raise ValueError('Incomplete baseline')
        p=pd.read_csv(reference/'predictions.csv')
        if not p[['imageid','group_id','price']].equals(val[['imageid','group_id','price']]):raise ValueError('Baseline split differs')
    sources={str(p):digest(p) for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts}
    registration = {'split_sha256':meta['split_sha256'],'seed':2026,'view_crop_fraction':.6,'fusion':'0.5 full + 0.5 mean of four fixed corner predictions',
        'brightness':{'probability':.5,'range':[.9,1.1]},'brightness_gate':'within-price-band brightness extreme MSE > 1.2x central, n>=30 in both groups',
        'weighting':mild_weights(train.price.to_numpy()),'selection':'2026 screen then 2027/2028 if MSE improves; unweighted MSE remains primary',
        'case_selection':'20 high-tail joint worst, 20 model disagreements, 20 tail success controls; no deletion or relabeling',
        'limits':'Development feedback on reused validation; saliency/occlusion are sensitivities, not causal attention or proof of absent price information',
        'test_images_used':False,'max_new_training_runs':6,'source_sha256':digest(__file__),'adapter_sha256':digest(ROOT/'diagnostic_training.py')}
    write_json(work/'registration.json',registration);write_json(public/'registration.json',registration)
    state={'status':'quality audit','results':[],'decisions':[],'runs':{}}
    def publish():
        write_json(work/'state.json',state)
        write_json(public/'status.json',{k:v for k,v in state.items() if k!='runs'})
        lines=['# 数据质量诊断与单项干预实验',f'当前状态：{state["status"]}。',
            '主线使用从零训练的小回归头；冻结划分。CLIP仅用于困难样本分歧审计。所有评估使用原始未加权MSE，不删图、不改价，不读取测试图片。',
            '| 实验 | 种子 | MSE | MAE | 尾部MSE | 相对小头MSE |','| --- | ---: | ---: | ---: | ---: | ---: |']
        for r in state['results']:lines.append(f'| {r["name"]} | {r["seed"]} | {r["mse"]:.2f} | {r["mae"]:.2f} | {r["tail_mse"]:.2f} | {r["delta_mse"]:+.2f} |')
        lines+=['',*state['decisions'], '', '共同失败候选不能确认为信息不足或脏数据。固定局部视图不等于检测到主体；曝光/清晰度相关性不证明因果。视觉覆盖抽样仅400张，其余未知，价位加权不宣称解决了全部语义长尾。']
        (public/'实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    publish()
    try:
        # Thresholds come only from training pixels. Gray quality is a proxy.
        import cv2
        quality=[]
        for i,row in frame.iterrows():
            with Image.open(local_path('data_root')/'train'/row.imageid) as im:
                gray=np.asarray(ImageOps.exif_transpose(im).convert('L').resize((128,128)),dtype=np.float64)/255.
            quality.append({'imageid':row.imageid,'brightness':float(gray.mean()),'sharpness':float(cv2.Laplacian(gray,cv2.CV_64F).var())})
            if i%1000==0: print(f'quality pixels {i}/{len(frame)}',flush=True)
        q=pd.DataFrame(quality).merge(frame,on='imageid',validate='one_to_one')
        q.to_csv(work/'quality.csv',index=False)
        cuts={k:q.loc[q.partition=='train',k].quantile([.1,.9]).tolist() for k in ('brightness','sharpness')}
        pred=pd.read_csv(refs[2026]/'predictions.csv');z=q[q.partition=='validation'].merge(pred[['imageid','squared_error']],on='imageid',validate='one_to_one')
        slices=[]; brightness_gate=False
        for key in cuts:
            lo,hi=cuts[key]
            for a,b in ((0,300),(300,700),(700,1300),(1300,np.inf)):
                band=z[(z.price>a)&(z.price<=b)];mid=band[(band[key]>lo)&(band[key]<hi)]
                for name,part in [('low',band[band[key]<=lo]),('high',band[band[key]>=hi])]:
                    ratio=float(part.squared_error.mean()/mid.squared_error.mean()) if len(part) and len(mid) else None
                    slices.append({'feature':key,'price_band':[a,None if np.isinf(b) else b],'group':name,'n':len(part),'central_n':len(mid),'mse_ratio':ratio})
                    if key=='brightness' and len(part)>=30 and len(mid)>=30 and ratio>1.2:brightness_gate=True
        write_json(public/'quality_audit.json',{'train_thresholds':cuts,'slices':slices,'brightness_gate':brightness_gate})
        # Purposeful validation cases are diagnostic, not a representative census.
        clip_run=local_path('records_root')/'03_clip_reference/runs/20261013T043141Z_c9f899f2'
        clip=pd.read_csv(clip_run/'predictions.csv')
        joined=pred.merge(clip[['imageid','squared_error']],on='imageid',suffixes=('_scratch','_clip'),validate='one_to_one')
        tail=joined[joined.price>1300].copy()
        tail['joint_score']=tail.squared_error_scratch.rank(pct=True)+tail.squared_error_clip.rank(pct=True)
        tail['disagreement']=abs(tail.squared_error_scratch-tail.squared_error_clip)
        bad=tail.nlargest(20,'joint_score');bad['case_group']='joint_failure_candidate'
        disagreement=tail[~tail.imageid.isin(bad.imageid)].nlargest(20,'disagreement');disagreement['case_group']='model_disagreement'
        success=tail[~tail.imageid.isin(pd.concat([bad,disagreement]).imageid)].nsmallest(20,'joint_score');success['case_group']='success_control'
        cases=pd.concat([bad,disagreement,success]);cases.to_csv(work/'cases.csv',index=False)
        write_json(public/'case_audit.json',{'counts':cases.case_group.value_counts().to_dict(),'case_file_sha256':digest(work/'cases.csv'),
            'interpretation':'Shared failures only: information limitation remains a hypothesis; no claim of a measured irreducible error bound'})
        def model_for(seed):
            ck=torch.load(refs[seed]/'best.pt',map_location='cpu',weights_only=True)
            model=AlexNetRegressor(ck['config']['model']);model.load_state_dict(ck['model']);model=model.cuda().eval()
            return model,ck
        # Saliency and occlusion on 6 joint cases and 6 successful controls.
        model,ck=model_for(2026);diagnosis=[]
        selected=pd.concat([bad.head(6),success.head(6)])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for _,row in selected.iterrows():
            ds=HouseImages(local_path('data_root'),val[val.imageid==row.imageid],ck['config'],ck['stats'])
            x=ds[0][0].cuda();probe=x[None].detach().requires_grad_(True)
            price=model(probe)*1000.;grad=torch.autograd.grad(price.sum(),probe)[0][0]
            saliency=(grad*x).abs().sum(0).detach().cpu().numpy()
            if not np.isclose(float(price.detach()),row.prediction,rtol=1e-5,atol=.05):raise ValueError('Sensitivity checkpoint mismatch')
            tiles=[]
            for y in range(7):
                for xcell in range(7):
                    tile=x.clone();tile[:,y*32:(y+1)*32,xcell*32:(xcell+1)*32]=0.;tiles.append(tile)
            with torch.inference_mode():
                outputs=torch.cat([model(torch.stack(tiles[i:i+8])) for i in range(0,49,8)]).cpu().numpy()*1000.
            delta=outputs.reshape(7,7)-float(price.detach())
            fig,axes=plt.subplots(1,3,figsize=(10,3));axes[0].imshow((x.cpu()*.5+.5).permute(1,2,0).numpy().clip(0,1));axes[1].imshow(saliency);axes[2].imshow(delta,cmap='coolwarm')
            for ax in axes:ax.axis('off')
            fig.savefig(work/f'sensitivity_{row.imageid}.png',dpi=120);plt.close(fig)
            diagnosis.append({'case_group':row.case_group,'maximum_absolute_occlusion_delta':float(abs(delta).max()),'absolute_error':float(abs(row.error))})
        write_json(public/'sensitivity_summary.json',{'n':len(diagnosis),'records':diagnosis,'note':'Input-gradient and gray occlusion sensitivity, not attention truth; raw images private'})
        del model
        state['status']='fixed local-view inference';publish()
        def metrics(name,seed,out,raw):
            out.mkdir(parents=True,exist_ok=True)
            m=save_predictions(out,val,raw,meta['tail_threshold'])
            bands=[]
            for lo,hi in ((0,300),(300,700),(700,1300),(1300,np.inf)):
                mask=(val.price>lo)&(val.price<=hi);err=np.maximum(raw[mask],0)-val.price.to_numpy()[mask]
                bands.append({'low':lo,'high':None if np.isinf(hi) else hi,'n':int(mask.sum()),'mse':float(np.mean(err**2)),'bias':float(err.mean())})
            m.update(name=name,seed=seed,delta_mse=m['mse']-read_json(refs[seed]/'metrics.json')['mse'],price_bands=bands)
            state['results'].append(m);write_json(public/f'{name}_{seed}.json',m);publish();return m
        def views(seed):
            model,ck=model_for(seed);raw=[]
            mean=torch.tensor(ck['stats']['mean'])[:,None,None];std=torch.tensor(ck['stats']['std'])[:,None,None]
            reference=pd.read_csv(refs[seed]/'predictions.csv')
            if reference.imageid.tolist()!=val.imageid.tolist():raise ValueError('Reference order mismatch')
            with torch.inference_mode():
                for i,row in val.iterrows():
                    x=read_rgb(local_path('data_root')/'train'/row.imageid)
                    batch=torch.stack([(geometry(view,ck['config']['preprocess'])-mean)/std for view in [x,*fixed_views(x)]])
                    p=model(batch.cuda()).cpu().numpy()*1000.
                    if not np.isclose(p[0],reference.iloc[i].raw_prediction,rtol=1e-5,atol=.05):raise ValueError('Full-view inference mismatch')
                    raw.append(.5*p[0]+.5*p[1:].mean())
                    if i%400==0:print(f'local-view seed{seed}: {i}/1601',flush=True)
            del model
            return metrics('local_views',seed,work/f'local_views_{seed}',np.asarray(raw))
        first=views(2026)
        if first['delta_mse']<0:
            for seed in SEEDS[1:]:views(seed)
        else:state['decisions'].append('固定局部视图2026未改善，不追加种子。')
        # Independent scratch training; the interventions never accumulate.
        for name in (['brightness'] if brightness_gate else [])+['price_weighting']:
            config=deepcopy(c);config['experiment'].update(id=f'diagnostic_{name}',name=f'数据诊断：{name}',stage='05_encoder',enforce_changes=False)
            if name=='brightness':config['augmentation']['brightness']=registration['brightness']
            else:config['_train_price_weights']=registration['weighting']
            for seed in SEEDS:
                for filename,expected in sources.items():
                    if digest(filename)!=expected:raise RuntimeError('Source changed during registered queue')
                config['seed']=seed;config['_parent_source']=None;validate(config)
                path=work/f'{name}_{seed}.json';write_json(path,config)
                parent=local_path('records_root')/'07_encoder/runs'/config['experiment']['id'];before=set(parent.glob('*'))
                state['status']=f'training {name} {seed}';publish()
                with (work/f'{name}_{seed}.log').open('x',encoding='utf-8') as log:
                    child=subprocess.Popen([sys.executable,'-u','-X','utf8',str(ROOT/'diagnostic_training.py'),'train',str(path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    try:
                        while child.poll() is None:
                            if (work/'STOP').exists():raise RuntimeError('Stop requested')
                            time.sleep(5)
                        if child.returncode:raise RuntimeError(f'{name} {seed} training failed')
                    except BaseException:
                        if child.poll() is None:subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
                        raise
                created=list(set(parent.glob('*'))-before)
                if len(created)!=1:raise ValueError('Ambiguous output')
                run=created[0];p=pd.read_csv(run/'predictions.csv')
                if p.imageid.tolist()!=val.imageid.tolist() or not pd.read_csv(run/'history.csv').samples.eq(6399).all():raise ValueError('Incomplete sample coverage')
                r=metrics(name,seed,work/f'{name}_{seed}_evaluation',p.raw_prediction.to_numpy())
                state['runs'][f'{name}_{seed}']=str(run)
                dest=public/'runs'/name/str(seed);dest.mkdir(parents=True,exist_ok=True)
                for filename in ('metrics.json','cost.json','history.csv','stopping.json','diagnostic_training.json'):shutil.copy2(run/filename,dest/filename)
                if seed==2026 and r['delta_mse']>=0:
                    state['decisions'].append(f'{name}初筛未改善，不追加种子。');break
            results=[r for r in state['results'] if r['name']==name]
            if len(results)==3:
                adopted=np.mean([r['delta_mse'] for r in results])<0 and sum(r['delta_mse']<0 for r in results)>=2
                state['decisions'].append(f'{name}三种子复核：'+('满足采用条件。' if adopted else '未满足采用条件。'))
        if not brightness_gate:state['decisions'].append('曝光分组未满足预登记关联门槛，本轮不启动亮度训练；清晰度仅诊断，不叠加滤波。')
        local_results=[r for r in state['results'] if r['name']=='local_views']
        if len(local_results)==3:
            adopted=np.mean([r['delta_mse'] for r in local_results])<0 and sum(r['delta_mse']<0 for r in local_results)>=2
            state['decisions'].append('固定局部视图三种子复核：'+('满足采用条件。' if adopted else '未满足采用条件。'))
        state['status']='completed';publish()
    except BaseException as exc:
        state['status']='failed';state['error']=str(exc);publish();raise


if __name__=='__main__':main()
