"""Small, user-authorized GPU inference study; no optimizer or model selection.

Compare archived development PE-G single-view, two-view and PCA two-view KRR.
Only two fixed validation cases; 4x4 original-image masks, gray and blur controls.
"""
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from threadpoolctl import threadpool_limits

REPORT = Path(__file__).resolve().parent
ASSIGNMENT = REPORT.parents[1]
REPO = ASSIGNMENT/'cv-house-price'
sys.path.insert(0,str(REPO/'scripts/experiments'))
FIGURES = REPORT/'figures'
SOURCES = FIGURES/'analysis_sources'
OUT = SOURCES/'occlusion'
OUT.mkdir(exist_ok=True)
LOCAL=REPO/'.local/experiment_record'
CACHE=REPO/'.local/cache'
SETTINGS=json.loads((REPO/'local_paths.json').read_text(encoding='utf-8-sig'))
DATA=Path(SETTINGS['data_root'])
BASE='19_image_embedding_reference/pe_core_g14_448_kernel/20261014T073812Z_e0339201'
WHOLE='19_image_embedding_reference/pe_core_g14_448_whole_kernel/20261014T090655Z_2d55c743'
PCA='19_image_embedding_reference/pe_giant_whole_pca_bandwidth_kernel/20261015T033416Z_4b848f75'
ORIGINAL='19_image_embedding_reference/pe_core_g14_448/20261014T065137Z_b822c3d4'
LABELS=['Single-view KRR','Two-view KRR','Two-view PCA KRR']
GRID=4
BLUR_RADIUS=8
SOURCE_HASHES={}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold','svg.fonttype':'none'})


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def track(p):
    SOURCE_HASHES[p.relative_to(ASSIGNMENT).as_posix()]=digest(p)
    return p


def read(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def unit(z):return z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)


def load_head(rel):
    with np.load(track(LOCAL/rel/'head.npz'),allow_pickle=False) as saved:
        return {k:saved[k].copy() for k in saved.files}


def predict(z,head):
    k=np.exp(-float(head['gamma'])*np.maximum(2-2*(z@head['train_features'].T),0))
    return float(np.maximum((k@head['coefficients']+float(head['target_mean']))*1000,0).item())


def cached_row(manifest,folder,name):
    index=manifest['imageids'].index(name)
    batch=int(manifest.get('batch_size',1));offset=(index//batch)*batch
    count=min(batch,len(manifest['imageids'])-offset)
    path=folder/f'{offset:05d}_{count}.npy'
    rows=np.load(track(path),allow_pickle=False).astype(np.float64)
    return unit(rows[index-offset:index-offset+1])


def plot(table,selection,protocol):
    vmax=float(table.delta_price.abs().max())
    vmax=max(vmax,1e-6)
    norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
    original=table[table['mode'].eq('original')]
    outputs=[]
    for mode in ['gray','blur']:
        fig,axes=plt.subplots(2,4,figsize=(13.0,7.5),gridspec_kw={'width_ratios':[1.2,1,1,1]})
        for i,case in selection.iterrows():
            name=case.imageid
            with Image.open(DATA/'train'/name) as src:image=ImageOps.exif_transpose(src).convert('RGB')
            axes[i,0].imshow(image)
            axes[i,0].set_title(f'{name} | true {case.price:,.1f}\n{case.selection}',fontsize=10)
            for j,label in enumerate(LABELS,1):
                rows=table[table.imageid.eq(name)&table['mode'].eq(mode)&table.model.eq(label)].sort_values('cell')
                assert len(rows)==16
                heat=rows.delta_price.to_numpy().reshape(GRID,GRID)
                base=float(original[original.imageid.eq(name)&original.model.eq(label)].prediction.iloc[0])
                axes[i,j].imshow(image)
                artist=axes[i,j].imshow(heat,cmap='RdBu_r',norm=norm,alpha=.62,
                    extent=(-.5,image.width-.5,image.height-.5,-.5),interpolation='nearest',aspect='equal')
                axes[i,j].set_title(f'{label}\nUnmasked prediction {base:,.1f}',fontsize=10)
                for a in range(1,GRID):
                    axes[i,j].axvline(round(a*image.width/GRID)-.5,color='white',lw=.5,alpha=.55)
                    axes[i,j].axhline(round(a*image.height/GRID)-.5,color='white',lw=.5,alpha=.55)
            for ax in axes[i]:
                ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
        fig.suptitle('PE-Core-G regional sensitivity: '+('gray occlusion' if mode=='gray' else 'blur replacement'),
                     fontsize=16,y=.98)
        fig.subplots_adjust(left=.035,right=.90,top=.87,bottom=.20,wspace=.14,hspace=.42)
        cax=fig.add_axes([.92,.28,.015,.50])
        cb=fig.colorbar(artist,cax=cax)
        cb.set_label('Unmasked − perturbed price (thousand USD)',fontsize=9.5)
        fig.text(.5,.045,'4×4 regions in original-image coordinates; same masks for all models. Red: removal lowers prediction; blue: removal raises it.\n'
            'Shared signed color scale across both controls and all panels; overlays show regional sensitivity, not segmentation.\n'
            'Frozen development heads; no training or retuning. Two outcome-selected cases; not a prevalence estimate or the final three-encoder ensemble.',
            ha='center',fontsize=9.0,linespacing=1.6)
        stem='pe_g_occlusion_sensitivity_'+mode
        for suffix in ['png','svg']:
            path=FIGURES/(stem+'.'+suffix);fig.savefig(path,dpi=300,facecolor='white');outputs.append(path.name)
        plt.close(fig)
    consistency=[]
    for name in selection.imageid:
        for model in LABELS:
            sub=table[table.imageid.eq(name)&table.model.eq(model)]
            a=sub[sub['mode'].eq('gray')].sort_values('cell').delta_price.to_numpy()
            b=sub[sub['mode'].eq('blur')].sort_values('cell').delta_price.to_numpy()
            consistency.append(dict(imageid=name,model=model,
                gray_max_abs_change=float(abs(a).max()),blur_max_abs_change=float(abs(b).max()),
                signed_map_pearson=float(np.corrcoef(a,b)[0,1]),
                strongest_cell_gray=int(abs(a).argmax()),strongest_cell_blur=int(abs(b).argmax())))
    pd.DataFrame(consistency).to_csv(OUT/'replacement_consistency.csv',index=False)
    return outputs,consistency


def main():
    import torch
    from pe_core_giant_encoder import make_encoder
    all_cases=pd.read_csv(track(SOURCES/'whole_frame_case_selection.csv'),dtype={'imageid':str})
    selection=all_cases[all_cases.selection.isin(['Largest improvement','Largest worsening'])].reset_index(drop=True)
    assert len(selection)==2
    if (OUT/'status.json').exists():
        raise RuntimeError('Existing occlusion run: inspect status; no automatic repeated GPU work')
    protocol=dict(cases=selection[['imageid','selection','price']].to_dict('records'),
        selection_source='Previously fixed field-of-view comparison; largest absolute-error improvement and worsening among landscape validation images',
        grid=[4,4],coordinates='EXIF-corrected original RGB image; mask before all model preprocessing',
        replacements={'gray':'RGB (128,128,128)','blur':f'Original RGB image GaussianBlur(radius={BLUR_RADIUS}); paste only selected region'},
        models=[BASE,WHOLE,PCA],final_fusion_explained=False,
        metric='clipped unmasked price minus clipped perturbed price; thousand USD',
        perturbations_per_image=32,new_training=False,parameter_selection=False,
        authorization='User explicitly allowed brief GPU inference; prior training pause remains in effect',
        maximum_inference_seconds_after_load=240)
    write(OUT/'protocol.json',protocol)
    write(OUT/'status.json',dict(status='preparing'))
    heads=[load_head(r) for r in [BASE,WHOLE,PCA]]
    with np.load(track(LOCAL/PCA/'feature_transform.npz'),allow_pickle=False) as f:
        transform={k:f[k].copy() for k in f.files}
    assert float(transform['whitening_power'])==0
    manifest=read(LOCAL/ORIGINAL/'data_manifest.json')
    # Resolve the exact folder from its registered kernel, avoiding guessed paths.
    original_registration=read(REPO/'output/experiment_record'/BASE/'registration.json')
    original_folder=CACHE/original_registration['cache_family']/original_registration['feature_cache_key']
    assert digest(original_folder/'manifest.json')==original_registration['manifest_sha256']
    whole_registration=read(REPO/'output/experiment_record'/WHOLE/'registration.json')
    whole_folder=CACHE/whole_registration['cache_family']/whole_registration['feature_cache_key']
    whole_manifest=read(whole_folder/'manifest.json')
    assert digest(whole_folder/'manifest.json')==whole_registration['manifest_sha256']
    references={r:pd.read_csv(track(LOCAL/r/'predictions.csv'),dtype={'imageid':str}).set_index('imageid') for r in [BASE,WHOLE,PCA]}
    split=pd.read_csv(track(Path(SETTINGS['split_path'])),dtype={'imageid':str}).set_index('imageid')
    records=[];feature_rows=[];metadata=[];probes=[];encode=None
    torch.set_num_threads(2);torch.manual_seed(2026)
    assert torch.cuda.is_available()
    free,total=torch.cuda.mem_get_info()
    if free<4*1024**3:raise RuntimeError('Less than 4 GiB GPU memory free; leave other tasks untouched')
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    try:
        write(OUT/'status.json',dict(status='loading_frozen_encoder'))
        print('Loading pinned PE-Core-G; inference only',flush=True)
        encode=make_encoder(manifest)
        deadline=time.monotonic()+240

        def evaluate(image):
            if time.monotonic()>deadline:raise RuntimeError('Fixed inference budget exhausted')
            a=encode([image])
            whole=ImageOps.pad(image,(448,448),method=Image.Resampling.BICUBIC,color=(128,128,128))
            b=encode([whole])
            joint=unit(unit(np.concatenate([a,b],axis=1)).astype(np.float32).astype(np.float64))
            pca=unit((joint-transform['mean'])@transform['vectors'])
            prices=[predict(x,h) for x,h in zip([a,joint,pca],heads)]
            return prices,a,joint

        for _,case in selection.iterrows():
            name=case.imageid;assert split.loc[name,'partition']=='validation'
            with Image.open(track(DATA/'train'/name)) as source:
                image=ImageOps.exif_transpose(source).convert('RGB')
            pixel_sha=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
            assert pixel_sha==split.loc[name,'pixel_hash']
            baseline,a,joint=evaluate(image)
            old_a=cached_row(manifest,original_folder,name)
            old_joint=cached_row(whole_manifest,whole_folder,name)
            np.testing.assert_allclose(a,old_a,rtol=1e-4,atol=1e-5)
            np.testing.assert_allclose(joint,old_joint,rtol=1e-4,atol=1e-5)
            expected=[float(references[r].loc[name,'prediction']) for r in [BASE,WHOLE,PCA]]
            # Same tolerance as archived feature-to-price replay; no relaxed gate.
            np.testing.assert_allclose(baseline,expected,rtol=1e-7,atol=1e-4)
            probes.append(dict(imageid=name,feature_max_abs_error=float(abs(a-old_a).max()),
                joint_feature_max_abs_error=float(abs(joint-old_joint).max()),
                fresh_prices=baseline,archived_prices=expected,max_price_error=float(np.max(np.abs(np.array(baseline)-expected)))))
            for label,p in zip(LABELS,baseline):
                records.append(dict(imageid=name,mode='original',cell=-1,row=-1,column=-1,
                    model=label,prediction=p,delta_price=0.,true_price=float(case.price)))
            feature_rows.append(np.concatenate([a,joint],axis=1)[0]);metadata.append(dict(imageid=name,mode='original',cell=-1))
            blurred=image.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
            for mode in ['gray','blur']:
                for row in range(GRID):
                    for col in range(GRID):
                        box=(round(col*image.width/GRID),round(row*image.height/GRID),
                             round((col+1)*image.width/GRID),round((row+1)*image.height/GRID))
                        changed=image.copy()
                        changed.paste((128,128,128) if mode=='gray' else blurred.crop(box),box)
                        prices,a,joint=evaluate(changed)
                        cell=row*GRID+col
                        for label,p,p0 in zip(LABELS,prices,baseline):
                            records.append(dict(imageid=name,mode=mode,cell=cell,row=row,column=col,
                                model=label,prediction=p,delta_price=p0-p,true_price=float(case.price),
                                left=box[0],top=box[1],right=box[2],bottom=box[3]))
                        feature_rows.append(np.concatenate([a,joint],axis=1)[0]);metadata.append(dict(imageid=name,mode=mode,cell=cell))
                    print(name,mode,'completed grid row',row+1,flush=True)
                    pd.DataFrame(records).to_csv(OUT/'per_region_predictions.csv',index=False)
                    write(OUT/'status.json',dict(status='inference',imageid=name,mode=mode,grid_row=row+1))
        torch.cuda.synchronize()
        peak=int(torch.cuda.max_memory_allocated())
    finally:
        encode=None;gc.collect();torch.cuda.empty_cache()
    table=pd.DataFrame(records)
    assert len(table)==2*33*3 and np.isfinite(table[['prediction','delta_price']]).all().all()
    np.savez_compressed(OUT/'archived_feature_rows.npz',features=np.stack(feature_rows))
    write(OUT/'feature_row_index.json',metadata)
    table.to_csv(OUT/'per_region_predictions.csv',index=False)
    write(OUT/'unmasked_replay_checks.json',dict(passed=True,probes=probes,rtol=1e-7,atol=1e-4))
    write(OUT/'gpu_execution_stats.json',dict(elapsed_seconds=time.monotonic()-started,
        peak_cuda_allocated_bytes=peak,encoder_forward_calls=132,source_sha256=SOURCE_HASHES))
    outputs,consistency=plot(table,selection,protocol)
    report=dict(status='completed',outputs=outputs,elapsed_seconds=time.monotonic()-started,
        peak_cuda_allocated_bytes=peak,images=2,encoder_forward_calls=132,models=LABELS,
        new_training=False,new_test_access=False,unmasked_replay_passed=True,
        replacement_consistency=consistency,source_sha256=SOURCE_HASHES,
        script_sha256=digest(Path(__file__)),
        limitation='Two outcome-selected validation cases and coarse perturbations; no segmentation, causal attribution, population estimate or final fusion explanation')
    write(OUT/'status.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def plot_saved_results():
    """CPU-only recovery/replot; independently replay all saved feature rows."""
    protocol=read(OUT/'protocol.json')
    probes=read(OUT/'unmasked_replay_checks.json')
    assert probes['passed'] and len(probes['probes'])==2
    selection=pd.DataFrame(protocol['cases'])
    table=pd.read_csv(track(OUT/'per_region_predictions.csv'),dtype={'imageid':str})
    assert len(table)==198 and np.isfinite(table[['prediction','delta_price']]).all().all()
    index=read(OUT/'feature_row_index.json')
    with np.load(track(OUT/'archived_feature_rows.npz'),allow_pickle=False) as data:
        features=data['features'].copy()
    assert features.shape==(66,3840) and len(index)==66
    heads=[load_head(rel) for rel in [BASE,WHOLE,PCA]]
    with np.load(track(LOCAL/PCA/'feature_transform.npz'),allow_pickle=False) as data:
        transform={k:data[k].copy() for k in data.files}
    maximum_error=0.
    for values,meta in zip(features,index):
        a=values[:1280][None,:];joint=values[1280:][None,:]
        pca=unit((joint-transform['mean'])@transform['vectors'])
        rows=table[table.imageid.eq(meta['imageid'])&table['mode'].eq(meta['mode'])&table.cell.eq(meta['cell'])].set_index('model')
        assert len(rows)==3
        fresh=np.array([predict(x,h) for x,h in zip([a,joint,pca],heads)])
        expected=rows.loc[LABELS,'prediction'].to_numpy()
        maximum_error=max(maximum_error,float(abs(fresh-expected).max()))
        np.testing.assert_allclose(fresh,expected,rtol=1e-10,atol=1e-7)
    for name in selection.imageid:
        for label in LABELS:
            subset=table[table.imageid.eq(name)&table.model.eq(label)]
            p0=float(subset[subset['mode'].eq('original')].prediction.iloc[0])
            np.testing.assert_allclose(subset.delta_price,p0-subset.prediction,rtol=1e-10,atol=1e-7)
    outputs,consistency=plot(table,selection,protocol)
    stats_path=OUT/'gpu_execution_stats.json'
    stats=read(stats_path) if stats_path.exists() else dict(
        note='GPU inference completed and saved all outputs; initial plotting failed before execution counters were persisted. No extra GPU inference for plotting recovery.')
    write(OUT/'status.json',dict(status='completed',outputs=outputs,images=2,encoder_forward_calls=132,
        models=LABELS,new_training=False,new_test_access=False,unmasked_replay_passed=True,
        all_198_prices_cpu_replayed=True,max_cpu_replay_error=maximum_error,
        replacement_consistency=consistency,execution=stats,source_sha256=SOURCE_HASHES,
        script_sha256=digest(Path(__file__)),
        output_checksums={name:dict(sha256=digest(FIGURES/name),bytes=(FIGURES/name).stat().st_size) for name in outputs},
        limitation='Two outcome-selected cases; coarse regional sensitivity, not segmentation, causal evidence or the final three-encoder ensemble'))
    print(json.dumps(dict(outputs=outputs,max_cpu_replay_error=maximum_error,consistency=consistency),indent=2),flush=True)


if __name__=='__main__':
    try:
        with threadpool_limits(limits=2):
            if '--plot-only' in sys.argv:plot_saved_results()
            else:main()
    except Exception as error:
        write(OUT/'failure.json',dict(error_type=type(error).__name__,message=str(error)))
        raise
