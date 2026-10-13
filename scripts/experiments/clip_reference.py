"""Frozen OpenAI CLIP image features; train-only grouped CV ridge reference."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid
from importlib.metadata import version

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import joblib
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common.config import resolve, digest, write_json
from common.local_paths import local_path, REPOSITORY
from common.metrics import save_predictions
from preparation.split import load_split

ALPHAS = [0.1, 1., 10., 100., 1000.]


def fit_head(features, prices, groups):
    """Each CV fold fits its own scaler; validation labels never enter fitting."""
    search = GridSearchCV(make_pipeline(StandardScaler(), Ridge(solver='svd')),
                          {'ridge__alpha': ALPHAS}, cv=GroupKFold(3),
                          scoring='neg_mean_squared_error', n_jobs=1)
    with threadpool_limits(limits=4):
        search.fit(features, prices / 1000., groups=groups)
    return search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    torch.use_deterministic_algorithms(True)
    config = resolve(Path(__file__).parent / 'configs/selected/architecture.json')
    frame, meta = load_split(config)
    if (frame.partition == 'train').sum() != 6399 or (frame.partition == 'validation').sum() != 1601:
        raise ValueError('Unexpected frozen split')
    out = local_path('records_root') / '03_clip_reference/runs' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    protocol = {'model': 'ViT-B-32', 'pretrained': 'openai', 'frozen': True,
                'preprocessing': 'OpenAI CLIP validation resize/center-crop/normalization, EXIF transpose, RGB',
                'features': 'L2-normalized image embedding only; no text or prompts',
                'head': 'StandardScaler + Ridge; 3-fold train-only GroupKFold', 'alphas': ALPHAS,
                'seed': 2026, 'scope': 'Exploratory pretrained reference; course eligibility unresolved; not a scratch-model ablation',
                'test_images_used': False}
    write_json(out / 'registration.json', protocol)
    write_json(out / 'status.json', {'status': 'running'})
    start = time.perf_counter()
    try:
        import open_clip
        weights_root = local_path('cache_root') / 'clip_weights'
        checkpoint = weights_root / 'ViT-B-32.pt'
        expected = open_clip.get_pretrained_cfg('ViT-B-32', 'openai')
        if not checkpoint.exists() or digest(checkpoint) != expected['url'].split('/')[-2]:
            raise ValueError('Download the official ViT-B-32.pt first; checkpoint SHA256 must match OpenAI')
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained=str(checkpoint), force_quick_gelu=True,
            weights_only=False,  # Official SHA256-verified checkpoint is a TorchScript archive.
            image_mean=expected['mean'], image_std=expected['std'],
            image_interpolation=expected['interpolation'], image_resize_mode=expected['resize_mode'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).float().eval().requires_grad_(False)
        checkpoints = {p.name: digest(p) for p in weights_root.rglob('*') if p.is_file() and p.suffix in ('.pt', '.bin', '.safetensors')}
        if not checkpoints:
            raise ValueError('Missing checkpoint provenance')
        manifest = {'split_sha256': meta['split_sha256'], 'source_sha256': digest(__file__),
                    'checkpoint_sha256': checkpoints, 'protocol': protocol,
                    'torch': torch.__version__, 'open_clip': version('open_clip_torch'),
                    'torchvision': version('torchvision'), 'pillow': version('Pillow'), 'device': device,
                    'preprocess_repr': repr(preprocess), 'precision': 'float32',
                    'imageids': frame.imageid.tolist(), 'pixel_hashes': frame.pixel_hash.tolist()}
        import hashlib
        key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        cache = local_path('cache_root') / 'clip_features' / key
        cache.mkdir(parents=True, exist_ok=True)
        write_json(out / 'data_manifest.json', manifest)
        chunks = []
        image_root = local_path('data_root') / 'train'
        with torch.inference_mode():
            for offset in range(0, len(frame), args.batch_size):
                names = frame.imageid.iloc[offset:offset + args.batch_size].tolist()
                path = cache / f'{offset:05d}_{len(names)}.npy'
                if path.exists():
                    values = np.load(path, allow_pickle=False)
                else:
                    images = []
                    for name in names:
                        with Image.open(image_root / name) as im:
                            images.append(preprocess(ImageOps.exif_transpose(im).convert('RGB')))
                    values = model.encode_image(torch.stack(images).to(device), normalize=True).cpu().numpy()
                    temporary = path.with_suffix('.tmp')
                    with temporary.open('wb') as stream:
                        np.save(stream, values)
                    temporary.replace(path)
                if values.shape != (len(names), 512) or not np.isfinite(values).all():
                    raise ValueError('Invalid feature cache')
                chunks.append(values)
                if offset % (args.batch_size * 10) == 0:
                    print(f'CLIP features: {offset + len(names)}/{len(frame)} ({device})', flush=True)
                    write_json(out / 'progress.json', {'images': offset + len(names), 'total': len(frame)})
        features = np.concatenate(chunks)
        extraction_seconds = time.perf_counter() - start
        train_mask = frame.partition.eq('train').to_numpy()
        train, val = frame.loc[train_mask].reset_index(drop=True), frame.loc[~train_mask].reset_index(drop=True)
        search = fit_head(features[train_mask], train.price.to_numpy(), train.group_id.to_numpy())
        joblib.dump(search.best_estimator_, out / 'head.joblib')
        metrics = save_predictions(out, val, search.predict(features[~train_mask]) * 1000., meta['tail_threshold'])
        save_predictions(_folder(out / 'train_evaluation'), train,
                         search.predict(features[train_mask]) * 1000., meta['tail_threshold'])
        write_json(out / 'head_selection.json', {'alpha': search.best_params_['ridge__alpha'],
                   'cv_mse_original_units': (-search.cv_results_['mean_test_score'] * 1e6).tolist(), 'alphas': ALPHAS})
        write_json(out / 'cost.json', {'extraction_seconds': extraction_seconds, 'total_seconds': time.perf_counter() - start, 'device': device})
        write_json(out / 'status.json', {'status': 'completed'})
        pred = np.maximum(search.predict(features[~train_mask]) * 1000., 0)
        band_rows = []
        for label, mask in [('≤300', val.price <= 300), ('300–700', (val.price > 300) & (val.price <= 700)),
                            ('700–1300', (val.price > 700) & (val.price <= 1300)), ('>1300', val.price > 1300)]:
            error = pred[mask] - val.price.to_numpy()[mask]
            band_rows.append({'band': label, 'n': int(mask.sum()), 'mse': float(np.mean(error ** 2)),
                              'mae': float(np.mean(abs(error))), 'bias': float(error.mean())})
        write_json(out / 'price_bands.json', band_rows)
        public = REPOSITORY / 'output/experiment_record/03_clip_reference'
        public.mkdir(parents=True, exist_ok=True)
        for name in ('registration.json', 'metrics.json', 'head_selection.json', 'cost.json', 'status.json', 'price_bands.json'):
            import shutil
            shutil.copy2(out / name, public / name)
        improvement = (126810.00739756384 - metrics['mse']) / 126810.00739756384 * 100
        report = f'# CLIP冻结图像特征参照实验\n\n探索性预训练路线，课程适用性未确认；不替换从零训练主线。\n\nOpenAI ViT-B/32冻结图像编码器，官方确定性预处理，L2特征＋训练侧标准化＋岭回归。6399训练／1601验证，原冻结分组划分。正则化仅用训练侧三折分组交叉验证选择，无文本输入、不读取测试图片。\n\n验证MSE {metrics["mse"]:.2f}，MAE {metrics["mae"]:.2f}，尾部MSE {metrics["tail_mse"]:.2f}；相对三个独立翻转AlexNet的平均MSE 126810.01改善{improvement:.2f}%。这是完整方法参照，不是单因素结构消融，亦不是CLIP三种子结果。岭回归为确定性拟合，不重复伪造种子复核。\n\n选择alpha={search.best_params_["ridge__alpha"]}。运行编号：{out.name}。完整逐图预测、头权重、特征缓存和预训练权重仅保留在本地。\n'
        report += '\n| 价位 | 数量 | MSE | MAE | 平均偏差 |\n| --- | ---: | ---: | ---: | ---: |\n'
        for row in band_rows:
            report += f'| {row["band"]} | {row["n"]} | {row["mse"]:.2f} | {row["mae"]:.2f} | {row["bias"]:+.2f} |\n'
        report += '\n官方模型来源：https://github.com/openai/CLIP 。权重SHA256、数据指纹和包版本见本地data_manifest.json。\n'
        (public / '实验报告.md').write_text(report, encoding='utf-8')
        print(json.dumps({'run': str(out), 'metrics': metrics}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        write_json(out / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise


def _folder(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == '__main__':
    main()
