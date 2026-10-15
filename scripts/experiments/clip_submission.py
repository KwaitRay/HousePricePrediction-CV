"""Refit the frozen chosen CLIP ridge head on all 8000 labels; export test CSV."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse, hashlib, json, time, uuid
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from common.config import ROOT, resolve, read_json, write_json, digest
from common.local_paths import local_path
from preparation.split import load_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-run', type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference_run.resolve()
    if read_json(reference/'status.json')['status'] != 'completed':
        raise ValueError('Reference is not completed')
    config = resolve(ROOT/'configs/selected/architecture.json')
    frame, split_meta = load_split(config)
    manifest = read_json(reference/'data_manifest.json')
    if manifest['imageids'] != frame.imageid.tolist() or manifest['pixel_hashes'] != frame.pixel_hash.tolist():
        raise ValueError('Reference data contents or order changed')
    if manifest['split_sha256'] != split_meta['split_sha256']:
        raise ValueError('Reference split differs')
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    cache = local_path('cache_root')/'clip_features'/key
    chunks = []
    for offset in range(0, len(frame), 32):
        n = min(32, len(frame)-offset)
        values = np.load(cache/f'{offset:05d}_{n}.npy', allow_pickle=False)
        if values.shape != (n,512) or not np.isfinite(values).all():
            raise ValueError('Invalid reference feature chunk')
        chunks.append(values)
    features = np.concatenate(chunks)
    if features.shape != (8000,512) or not np.allclose(np.linalg.norm(features,axis=1),1.,atol=1e-5):
        raise ValueError('Invalid full training features')
    original = joblib.load(reference/'head.joblib')
    for partition, folder in [('train',reference/'train_evaluation'),('validation',reference)]:
        mask = frame.partition.eq(partition).to_numpy()
        saved = pd.read_csv(folder/'predictions.csv',dtype={'imageid':str})
        if saved.imageid.tolist() != frame.loc[mask,'imageid'].tolist():
            raise ValueError('Saved reference predictions are misaligned')
        if not np.allclose(original.predict(features[mask])*1000.,saved.raw_prediction,rtol=1e-7,atol=1e-4):
            raise ValueError('Reference feature replay failed')
    alpha = float(read_json(reference/'head_selection.json')['alpha'])
    if original.named_steps['ridge'].alpha != alpha:
        raise ValueError('Chosen regularization differs from reference head')
    root = local_path('data_root')
    test = pd.read_csv(root/'test.csv',dtype={'imageid':str})
    if len(test)!=3000 or test.imageid.isna().any() or test.imageid.duplicated().any():
        raise ValueError('Invalid test IDs')
    out = local_path('records_root')/'08_final/clip_ridge'/(
        time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',{'status':'running'})
    write_json(out/'registration.json',{'method':'Frozen OpenAI CLIP ViT-B/32 + StandardScaler + Ridge',
        'alpha':alpha,'train_n':8000,'test_n':3000,'target_scale':1000.,'prediction_unit':'thousand USD',
        'reference_run':reference.name,'train_csv_sha256':digest(root/'train.csv'),
        'test_csv_sha256':digest(root/'test.csv'),'feature_sha256':hashlib.sha256(features.tobytes()).hexdigest(),
        'scope':'Generated exploratory candidate; course CLIP eligibility unresolved; not uploaded to Kaggle'})
    try:
        head = make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='svd'))
        with threadpool_limits(limits=4):
            head.fit(features,frame.price.to_numpy()/1000.)
        joblib.dump(head,out/'head.joblib')
        print('Full ridge head fitted on 8000 labels; alpha frozen at',alpha,flush=True)
        import open_clip
        checkpoint=local_path('cache_root')/'clip_weights/ViT-B-32.pt'
        expected=open_clip.get_pretrained_cfg('ViT-B-32','openai')
        if digest(checkpoint)!=expected['url'].split('/')[-2]:raise ValueError('Official checkpoint hash mismatch')
        torch.set_num_threads(4);torch.manual_seed(2026);torch.use_deterministic_algorithms(True)
        if not torch.cuda.is_available():raise RuntimeError('Expected CUDA environment')
        model,_,preprocess=open_clip.create_model_and_transforms('ViT-B-32',pretrained=str(checkpoint),
            force_quick_gelu=True,weights_only=False,image_mean=expected['mean'],image_std=expected['std'],
            image_interpolation=expected['interpolation'],image_resize_mode=expected['resize_mode'])
        if repr(preprocess)!=manifest['preprocess_repr']:
            raise ValueError('Preprocessing differs from selected reference')
        model=model.cuda().float().eval().requires_grad_(False)
        # Independent spot-check of cached labeled-image features before test inference.
        with torch.inference_mode():
            indices=np.random.default_rng(2026).choice(8000,8,replace=False)
            probes=[]
            for i in indices:
                with Image.open(root/'train'/frame.iloc[i].imageid) as im:
                    probes.append(preprocess(ImageOps.exif_transpose(im).convert('RGB')))
            recomputed=model.encode_image(torch.stack(probes).cuda(),normalize=True).cpu().numpy()
            if not np.allclose(recomputed,features[indices],rtol=1e-4,atol=2e-6):
                raise ValueError('Cached features fail independent checkpoint replay')
            all_features=[]
            for offset in range(0,3000,32):
                images=[]
                for name in test.imageid.iloc[offset:offset+32]:
                    with Image.open(root/'test'/name) as im:
                        images.append(preprocess(ImageOps.exif_transpose(im).convert('RGB')))
                batch=model.encode_image(torch.stack(images).cuda(),normalize=True).cpu().numpy()
                if not np.isfinite(batch).all():raise ValueError('Nonfinite test features')
                all_features.append(batch)
                if offset%320==0:print(f'Test inference {min(offset+32,3000)}/3000',flush=True)
        test_features=np.concatenate(all_features)
        np.save(out/'test_features.npy',test_features)
        prediction=np.maximum(head.predict(test_features)*1000.,0.)
        if prediction.shape!=(3000,) or not np.isfinite(prediction).all():raise ValueError('Invalid prices')
        result=pd.DataFrame({'imageid':test.imageid,'price':prediction})
        destination=out/'predictions.csv'
        result.to_csv(destination,index=False,encoding='utf-8',mode='x')
        restored=pd.read_csv(destination,dtype={'imageid':str})
        if list(restored)!=['imageid','price'] or restored.imageid.tolist()!=test.imageid.tolist():
            raise ValueError('Export schema/order mismatch')
        if len(restored)!=3000 or restored.imageid.nunique()!=3000 or not np.isfinite(restored.price).all() or (restored.price<0).any():
            raise ValueError('Export coverage/value validation failed')
        write_json(out/'checks.json',{'n':3000,'unique_ids':3000,'test_order_matches':True,'columns':['imageid','price'],
            'finite_nonnegative':True,'min_price':float(prediction.min()),'max_price':float(prediction.max()),
            'csv_sha256':digest(destination),'checkpoint_sha256':digest(checkpoint),
            'labeled_feature_spotcheck_passed':True,'full_reference_predictions_replayed':True})
        write_json(out/'status.json',{'status':'completed','kind':'full_retrain_and_test_inference'})
        print('Verified Kaggle candidate CSV:',destination,flush=True)
    except BaseException as exc:
        write_json(out/'status.json',{'status':'failed','error':str(exc)});raise


if __name__=='__main__':main()
