"""Train price attention on frozen patch tokens with properly nested residual targets."""
import argparse
import json
import os
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.linalg import solve
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from common.metrics import save_predictions
from visual_kernel_submission import features_for


def plot_progress(history, path, fold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([row['epoch'] for row in history], [row['residual_training_mse'] for row in history])
    ax.set(title=f'房价注意力训练进度：{fold}', xlabel='训练轮数', ylabel='训练残差均方误差（千美元²）')
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


class PriceAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.keys = nn.Sequential(nn.Linear(1152, 32), nn.GELU(), nn.Linear(32, 4))
        self.regressor = nn.Sequential(nn.LayerNorm(4608), nn.Linear(4608, 64), nn.GELU(), nn.Dropout(.25), nn.Linear(64, 1))
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, tokens, return_attention=False):
        attention = self.keys(tokens).softmax(dim=1)
        pooled = torch.einsum('bth,btd->bhd', attention, tokens).flatten(1)
        result = self.regressor(pooled).flatten()
        return (result, attention) if return_attention else result


def kernel_predict(x, y, held, gamma, alpha):
    mean = float(y.mean())
    with threadpool_limits(limits=4):
        kernel = np.exp(-gamma * np.maximum(2 - 2 * (x @ x.T), 0))
        kernel.flat[::len(y) + 1] += alpha
        coefficients = solve(kernel, y - mean, assume_a='pos', check_finite=False, overwrite_a=True)
        predictions = []
        for offset in range(0, len(held), 128):
            predictions.append(np.exp(-gamma * np.maximum(2 - 2 * (held[offset:offset + 128] @ x.T), 0)) @ coefficients + mean)
    return np.concatenate(predictions), coefficients, mean


def nested_baseline(x, y, groups, held, gamma, alpha):
    oof = np.full(len(y), np.nan)
    for fit, valid in GroupKFold(3).split(x, y, groups):
        if set(groups[fit]) & set(groups[valid]):
            raise ValueError('Inner baseline group leakage')
        oof[valid] = kernel_predict(x[fit], y[fit], x[valid], gamma, alpha)[0]
    if not np.isfinite(oof).all():
        raise ValueError('Incomplete nested residual targets')
    prediction, coefficients, mean = kernel_predict(x, y, held, gamma, alpha)
    return oof, prediction, coefficients, mean


def fit_attention(tokens, fit_ids, residual_target, held_ids, epochs, public, fold, update):
    torch.manual_seed(2026)
    rng = np.random.default_rng(2026)
    model = PriceAttention().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    target = torch.tensor(residual_target, dtype=torch.float32, device='cuda')
    results, history = {}, []
    plotted_at = 0.
    for epoch in range(1, max(epochs) + 1):
        model.train()
        order = rng.permutation(len(fit_ids))
        total = 0.
        started = time.perf_counter()
        for offset in range(0, len(order), 16):
            ids = order[offset:offset + 16]
            batch = torch.from_numpy(np.asarray(tokens[fit_ids[ids]], dtype=np.float32)).cuda()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = ((prediction - target[ids]) ** 2).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite attention loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        history.append({'epoch': epoch, 'residual_training_mse': total / len(order) * 1e6,
            'seconds': time.perf_counter() - started})
        pd.DataFrame(history).to_csv(public / f'{fold}_history.csv', index=False)
        if time.monotonic() - plotted_at >= 60 or epoch == max(epochs):
            plot_progress(history, public / f'{fold}_中文训练曲线.png', fold)
            plotted_at = time.monotonic()
        update('training', fold=fold, epoch=epoch, max_epoch=max(epochs), epoch_seconds=history[-1]['seconds'])
        print(f'Price attention {fold} epoch {epoch}/{max(epochs)}: {history[-1]["seconds"]:.1f}s', flush=True)
        if epoch in epochs:
            model.eval()
            parts = []
            with torch.inference_mode():
                for offset in range(0, len(held_ids), 16):
                    batch = torch.from_numpy(np.asarray(tokens[held_ids[offset:offset + 16]], dtype=np.float32)).cuda()
                    parts.append(model(batch).float().cpu().numpy())
            results[epoch] = np.concatenate(parts)
    return results, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-key', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    source = '19_image_embedding_reference/siglip2_so400m_kernel/20261013T152001Z_34db8dd3'
    root = local_path('records_root')
    source_registration = read_json(REPOSITORY / 'output/experiment_record' / source / 'registration.json')
    cache = local_path('cache_root') / 'siglip_patch_tokens' / args.cache_key
    manifest, checks = read_json(cache / 'manifest.json'), read_json(cache / 'checks.json')
    tokens_path = cache / 'patch_tokens.npy'
    if digest(tokens_path) != checks['array_sha256'] or not checks['all_chunks_finite'] or not checks['pooled_feature_replay_passed']:
        raise ValueError('Patch tokens are not independently verified')
    tokens = np.load(tokens_path, mmap_mode='r', allow_pickle=False)
    if tokens.shape != (8000, 729, 1152) or tokens.dtype != np.float16:
        raise ValueError('Unexpected frozen token shape or precision')
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    train, valid = frame.partition.eq('train').to_numpy(), frame.partition.eq('validation').to_numpy()
    if len(frame) != 8000 or train.sum() != 6399 or valid.sum() != 1601 or frame.imageid.tolist() != manifest['imageids'] or digest(local_path('split_path')) != manifest['split_sha256'] or set(frame.loc[train, 'group_id']) & set(frame.loc[valid, 'group_id']):
        raise ValueError('Frozen split differs')
    x = features_for(source_registration, frame)
    with np.load(root / source / 'head.npz', allow_pickle=False) as previous:
        np.testing.assert_allclose(x[train], previous['train_features'], rtol=0, atol=1e-12)
        gamma, alpha = float(previous['gamma']), float(previous['alpha'])
    train_ids, valid_ids = np.flatnonzero(train), np.flatnonzero(valid)
    y, groups = frame.loc[train, 'price'].to_numpy() / 1000, frame.loc[train, 'group_id'].to_numpy()
    epochs, scales = [5, 15, 30], [0., .25, .5, 1.]
    run = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    relative = '19_image_embedding_reference/siglip2_price_attention/' + run
    out, public = root / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    registration = {'method': 'Frozen full-image SigLIP2 patch tokens; learned 4-head price attention residual over fixed RBF baseline',
        'source_kernel': source, 'source_head_sha256': digest(root / source / 'head.npz'), 'source_sha256': digest(__file__),
        'token_cache_key': args.cache_key, 'token_manifest_sha256': digest(cache / 'manifest.json'), 'token_array_sha256': checks['array_sha256'],
        'split_sha256': digest(local_path('split_path')), 'epochs': epochs, 'residual_scales': scales,
        'attention': '1152->32->4 keys, token softmax, 4 weighted full-token averages; LayerNorm4608->64->1',
        'batch_size': 16, 'lr': .0003, 'weight_decay': .01, 'dropout': .25, 'seed': 2026,
        'price_loss': 'Original unweighted MSE, prices divided by1000 during optimization', 'gamma': gamma, 'alpha': alpha,
        'residual_targets': 'Inner GroupKFold3 baseline OOF on each fitting fold only; never use outer held labels',
        'selection': 'Outer training GroupKFold3 selects epoch and residual scale; validation evaluated once after selection',
        'null_control': 'Residual scale zero must reproduce original baseline training OOF and validation predictions',
        'backbone_frozen': True, 'test_images_used': False, 'text_modules_used': False, 'mllm_components': False,
        'course_scope': 'Generic pretrained image tower; price attention trained only from supplied training images and labels',
        'caveat': 'Source gamma/alpha previously selected on these training folds; selected CV criterion is not unbiased nested model-selection performance; attention alone does not prove causal object use'}
    write_json(public / 'registration.json', registration)
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    def status(state, **extra):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **extra})
    started = time.perf_counter()
    try:
        baseline_oof = np.full(len(y), np.nan)
        residual_oof = {epoch: np.full(len(y), np.nan) for epoch in epochs}
        for fold, (fit, held) in enumerate(GroupKFold(3).split(x[train], y, groups)):
            if set(groups[fit]) & set(groups[held]):
                raise ValueError('Outer group leakage')
            status('fitting_nested_baseline', fold=fold + 1)
            inner, base, _, _ = nested_baseline(x[train][fit], y[fit], groups[fit], x[train][held], gamma, alpha)
            baseline_oof[held] = base
            values, model = fit_attention(tokens, train_ids[fit], y[fit] - inner, train_ids[held], epochs, public, f'fold_{fold + 1}', status)
            for epoch in epochs:
                residual_oof[epoch][held] = values[epoch]
            del model
            torch.cuda.empty_cache()
        old = pd.read_csv(root / source / 'selected_training_oof.csv', dtype={'imageid': str})
        if old.imageid.tolist() != frame.loc[train, 'imageid'].tolist():
            raise ValueError('Baseline OOF IDs differ')
        np.testing.assert_allclose(baseline_oof * 1000, old.raw_oof_prediction, rtol=0, atol=1e-4)
        rows = []
        for epoch in epochs:
            for scale in scales:
                oof = baseline_oof + scale * residual_oof[epoch]
                if not np.isfinite(oof).all():
                    raise ValueError('Incomplete attention OOF predictions')
                rows.append({'epochs': epoch, 'residual_scale': scale, 'training_oof_mse': float(np.mean((np.maximum(oof, 0) - y) ** 2) * 1e6)})
        pd.DataFrame(rows).to_csv(public / 'training_cv.csv', index=False)
        best = min(rows, key=lambda row: row['training_oof_mse'])
        status('refitting_nested_baseline', **best)
        inner, baseline, coefficients, mean = nested_baseline(x[train], y, groups, x[valid], gamma, alpha)
        values, model = fit_attention(tokens, train_ids, y - inner, valid_ids, [best['epochs']], public, 'refit', status)
        prediction = (baseline + best['residual_scale'] * values[best['epochs']]) * 1000
        if best['residual_scale'] == 0:
            previous = pd.read_csv(root / source / 'predictions.csv')
            np.testing.assert_allclose(prediction, previous.raw_prediction, rtol=0, atol=1e-4)
        metrics = save_predictions(out, frame.loc[valid], prediction, 1300)
        # For the unweighted KRR baseline, K*c + mean = y - alpha*c.
        # Evaluate the selected attention head on every training image as well.
        train_residual = []
        with torch.inference_mode():
            for offset in range(0, len(train_ids), 16):
                batch = torch.from_numpy(np.asarray(tokens[train_ids[offset:offset + 16]], dtype=np.float32)).cuda()
                train_residual.append(model(batch).float().cpu().numpy())
        train_prediction = (y - alpha * coefficients + best['residual_scale'] * np.concatenate(train_residual)) * 1000
        train_out = out / 'full_training'
        train_out.mkdir()
        training_metrics = save_predictions(train_out, frame.loc[train], train_prediction, 1300)
        torch.save(model.state_dict(), out / 'attention_head.pt')
        np.savez_compressed(out / 'baseline_head.npz', train_features=x[train], coefficients=coefficients, gamma=gamma, alpha=alpha, target_mean=mean)
        chosen_oof = baseline_oof + best['residual_scale'] * residual_oof[best['epochs']]
        pd.DataFrame({'imageid': frame.loc[train, 'imageid'], 'raw_oof_prediction': chosen_oof * 1000,
            'oof_prediction': np.maximum(chosen_oof, 0) * 1000}).to_csv(out / 'selected_training_oof.csv', index=False)
        write_json(public / 'results.json', {'head': 'price_attention_residual', **best, **metrics,
            'full_training_metrics': training_metrics, 'full_training_role': 'In-fit diagnosis only, never used for model selection',
            'seconds': time.perf_counter() - started})
        status('completed')
        print(json.dumps({'head': 'price_attention_residual', **best, **metrics}), flush=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
