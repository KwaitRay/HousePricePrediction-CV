"""CPU-only residual-attention diagnosis with replayed prices and fixed case rules."""
import argparse
import hashlib
import os
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from siglip_price_attention import PriceAttention


def choose_cases(frame):
    chosen, roles = [], {}
    def add(rows, role):
        for imageid in rows.imageid:
            if imageid not in roles:
                chosen.append(imageid)
                roles[imageid] = role
    high = frame[frame.price > 1300]
    low = frame[frame.price <= 300]
    add(high.sort_values('error').head(4), '高价最大低估')
    add(high.assign(absolute_error=high.error.abs()).sort_values('absolute_error').head(2), '高价较好案例')
    add(low.sort_values('error', ascending=False).head(2), '低价最大高估')
    middle = frame[(frame.price > 300) & (frame.price <= 1300) & ~frame.imageid.isin(chosen)]
    add(middle.sample(n=2, random_state=2026), '中价固定随机')
    add(frame[~frame.imageid.isin(chosen)].sample(n=2, random_state=2026), '全验证固定随机')
    if len(chosen) != 12:
        raise ValueError('Expected twelve distinct diagnosis cases')
    selected = frame.set_index('imageid').loc[chosen].reset_index()
    selected['case_role'] = selected.imageid.map(roles)
    return selected


def image_heatmap(grid, image_size, height, width, patch):
    # The 384px/14px patch embedding covers 378px, leaving a 6px border.
    dense = np.full((height, width), np.nan, dtype=np.float32)
    covered = np.repeat(np.repeat(grid, patch, axis=0), patch, axis=1)
    dense[:covered.shape[0], :covered.shape[1]] = covered
    resized = np.asarray(Image.fromarray(dense).resize(image_size, Image.Resampling.NEAREST))
    return np.ma.masked_invalid(resized)


def main(reference):
    reference = reference.resolve()
    relative = reference.relative_to(local_path('records_root'))
    public_reference = REPOSITORY / 'output/experiment_record' / relative
    if read_json(public_reference / 'status.json')['status'] != 'completed':
        raise ValueError('Attention training must complete before diagnosis')
    registration = read_json(public_reference / 'registration.json')
    result = read_json(public_reference / 'results.json')
    run = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    out = reference / 'attention_diagnostics' / run
    public = public_reference / 'attention_diagnostics' / run
    out.mkdir(parents=True, exist_ok=False)
    public.mkdir(parents=True, exist_ok=False)
    write_json(public / 'status.json', {'status': 'diagnosing', 'pid': os.getpid()})
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    attention_source = Path(__file__).with_name('siglip_price_attention.py')
    if digest(attention_source) != registration['source_sha256']:
        raise ValueError('Attention architecture source changed')
    cache = local_path('cache_root') / 'siglip_patch_tokens' / registration['token_cache_key']
    manifest = read_json(cache / 'manifest.json')
    if digest(cache / 'manifest.json') != registration['token_manifest_sha256'] or read_json(cache / 'checks.json')['array_sha256'] != registration['token_array_sha256']:
        raise ValueError('Verified token provenance changed')
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    if digest(local_path('split_path')) != registration['split_sha256'] or frame.imageid.tolist() != manifest['imageids']:
        raise ValueError('Frozen image order or split changed')
    predictions = pd.read_csv(reference / 'predictions.csv', dtype={'imageid': str})
    if predictions.imageid.tolist() != frame.loc[frame.partition.eq('validation'), 'imageid'].tolist():
        raise ValueError('Validation prediction IDs differ')
    selected = choose_cases(predictions)
    baseline_root = local_path('records_root') / registration['source_kernel']
    if digest(baseline_root / 'head.npz') != registration['source_head_sha256']:
        raise ValueError('Baseline kernel changed')
    baseline = pd.read_csv(baseline_root / 'predictions.csv', dtype={'imageid': str}).set_index('imageid')
    token_array = np.load(cache / 'patch_tokens.npy', mmap_mode='r', allow_pickle=False)
    if token_array.shape != (8000, 729, 1152) or token_array.dtype != np.float16:
        raise ValueError('Unexpected token cache')
    model = PriceAttention().cpu().eval()
    torch.set_num_threads(1)
    model.load_state_dict(torch.load(reference / 'attention_head.pt', map_location='cpu', weights_only=True), strict=True)
    model.requires_grad_(False)
    scale = float(result['residual_scale'])
    diagnostics, arrays = [], {}
    index = {imageid: i for i, imageid in enumerate(frame.imageid)}
    for row in selected.itertuples(index=False):
        tokens = torch.from_numpy(np.asarray(token_array[index[row.imageid]], dtype=np.float32)).unsqueeze(0).requires_grad_(True)
        residual, attention = model(tokens, return_attention=True)
        residual_price = residual[0] * scale * 1000
        residual_price.backward()
        replay = float(baseline.loc[row.imageid, 'raw_prediction']) + float(residual_price.detach())
        np.testing.assert_allclose(replay, row.raw_prediction, rtol=1e-6, atol=1e-3)
        weights = attention[0].detach().numpy()
        gradient_activation = (tokens.grad[0] * tokens.detach()[0]).sum(dim=-1).numpy()
        if not np.isfinite(weights).all() or not np.isfinite(gradient_activation).all():
            raise ValueError('Nonfinite attention or sensitivity')
        np.testing.assert_allclose(weights.sum(axis=0), np.ones(4), rtol=0, atol=1e-6)
        # Independently check the sensitivity on the strongest token of the first case.
        probe = None
        if not diagnostics:
            position = int(np.argmax(np.abs(gradient_activation)))
            direction = torch.zeros_like(tokens)
            direction[:, position] = tokens.detach()[:, position]
            estimates = []
            with torch.no_grad():
                for step in (.01, .003, .001):
                    plus = model(tokens.detach() + step * direction)[0] * scale * 1000
                    minus = model(tokens.detach() - step * direction)[0] * scale * 1000
                    estimates.append({'step': step, 'central_difference_sensitivity': float((plus - minus) / (2 * step))})
            difference = estimates[-1]['central_difference_sensitivity']
            np.testing.assert_allclose(difference, gradient_activation[position], rtol=.01, atol=1e-3)
            if abs(difference - gradient_activation[position]) > abs(estimates[0]['central_difference_sensitivity'] - gradient_activation[position]) + 1e-3:
                raise ValueError('Finite differences did not converge toward the analytic gradient')
            probe = {'token_index': position, 'analytic_sensitivity': float(gradient_activation[position]),
                     'central_difference_steps': estimates, 'passed': True}
        arrays[row.imageid] = (weights.reshape(27, 27, 4), gradient_activation.reshape(27, 27))
        diagnostics.append({'imageid': row.imageid, 'case_role': row.case_role, 'true_price': row.price,
            'token_fp16_sha256': hashlib.sha256(token_array[index[row.imageid]].tobytes()).hexdigest(),
            'finite_difference_probe': probe,
            'prediction': row.prediction, 'raw_prediction': row.raw_prediction, 'baseline_raw_prediction': float(baseline.loc[row.imageid, 'raw_prediction']),
            'residual_contribution': float(residual_price.detach()), 'cpu_replay_absolute_error': abs(replay - row.raw_prediction),
            'normalized_head_entropies': (-np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=0) / np.log(729)).tolist(),
            'head_top_5_percent_mass': np.sort(weights, axis=0)[-37:].sum(axis=0).tolist()})
    selected.to_csv(out / 'selected_validation_cases.csv', index=False, encoding='utf-8-sig')
    write_json(public / 'cases.json', diagnostics)
    write_json(public / 'registration.json', {'source_sha256': digest(__file__), 'attention_head_sha256': digest(reference / 'attention_head.pt'),
        'prediction_sha256': digest(reference / 'predictions.csv'), 'token_manifest_sha256': digest(cache / 'manifest.json'),
        'token_array_sha256_at_training': registration['token_array_sha256'],
        'scope': 'Post-training diagnostics only; cached baseline held constant for sensitivity; gradient is of the learned residual branch, not the entire price model',
        'selection': '4 largest high-price underestimates, 2 best high-price cases, 2 largest low-price overestimates, 2 fixed random middle-price, 2 fixed random validation cases',
        'labels_role': 'Select and describe diagnosis cases only; no training, prompt changes, model selection or calibration',
        'token_position_caveat': 'Last-layer tokens contain global context; maps show nominal patch locations, not independent local pixel effects',
        'test_images_used': False, 'gpu_used': False, 'cpu_prediction_replay_passed': True})
    render(out, public, arrays, diagnostics, manifest)
    write_json(public / 'status.json', {'status': 'completed', 'pid': os.getpid(), 'cases': len(diagnostics)})
    write_json(public_reference / 'attention_diagnostics/latest_run.json', {'relative_run': run, 'status': 'completed'})
    print('Replayed and diagnosed twelve attention validation cases:', out, flush=True)


def render(out, public, arrays, diagnostics, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    attention_max = max(float(weights.max()) for weights, _ in arrays.values())
    sensitivity_limit = float(np.quantile(np.abs(np.concatenate([values.reshape(-1) for _, values in arrays.values()])), .98))
    sensitivity_limit = max(sensitivity_limit, 1e-10)
    height = int(manifest['preprocess']['size']['height'])
    width = int(manifest['preprocess']['size']['width'])
    if height != 384 or width != 384:
        raise ValueError('Diagnosis expects the registered 384px square resize')
    write_json(public / 'visual_scale.json', {'attention_range': [0, attention_max],
        'attention_display_norm': 'Shared logarithmic normalization, lower display bound 1e-6; raw attention weights unchanged',
        'sensitivity_range': [-sensitivity_limit, sensitivity_limit], 'sensitivity_limit_rule': 'Shared 98th percentile absolute residual gradient-times-activation across all twelve cases',
        'patch_grid': [27, 27], 'patch_size': 14, 'processed_size': [height, width], 'uncovered_bottom_right_pixels': 6,
        'overlay': 'Nominal 14px token cells mapped through the inverse square resize onto the original full image; untokenized border masked',
        'color_scales_shared_across_cases': True})
    rows = []
    for number, case in enumerate(diagnostics, 1):
        imageid = case['imageid']
        with Image.open(local_path('data_root') / 'train' / imageid) as source:
            original = ImageOps.exif_transpose(source).convert('RGB')
        weights, sensitivity = arrays[imageid]
        np.savez_compressed(out / f'case_{number:02d}_maps.npz', attention=weights,
            residual_gradient_times_activation=sensitivity)
        fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.6))
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('原始全图参照')
        attention_plot = None
        for head, ax in enumerate([axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]):
            ax.imshow(original)
            attention_plot = ax.imshow(image_heatmap(weights[:, :, head], original.size, height, width, 14),
                cmap='magma', alpha=.55, norm=LogNorm(vmin=1e-6, vmax=max(attention_max, 1.001e-6)))
            ax.set_title(f'残差注意力头 {head + 1}；熵 {case["normalized_head_entropies"][head]:.2f}')
        axes[1, 2].imshow(original)
        sensitivity_plot = axes[1, 2].imshow(image_heatmap(sensitivity, original.size, height, width, 14),
            cmap='RdBu_r', alpha=.6, vmin=-sensitivity_limit, vmax=sensitivity_limit)
        axes[1, 2].set_title('残差分支敏感度（非全模型归因）')
        for ax in axes.flat:
            ax.axis('off')
        fig.suptitle(f'{imageid}｜{case["case_role"]}｜真实 {case["true_price"]:.1f}；基线 {case["baseline_raw_prediction"]:.1f} → 预测 {case["prediction"]:.1f}\n'
            f'残差贡献 {case["residual_contribution"]:+.1f} 千美元；注意力与敏感度不证明对象因果作用', fontsize=11)
        fig.subplots_adjust(left=.025, right=.91, bottom=.025, top=.88, wspace=.12, hspace=.19)
        fig.colorbar(attention_plot, cax=fig.add_axes([.93, .57, .015, .27]), label='注意力权重（对数色标）')
        fig.colorbar(sensitivity_plot, cax=fig.add_axes([.93, .08, .015, .27]), label='残差一阶敏感度')
        path = out / f'case_{number:02d}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        rows.append(f'| {imageid} | {case["case_role"]} | {case["true_price"]:.1f} | {case["baseline_raw_prediction"]:.1f} | {case["prediction"]:.1f} | {case["residual_contribution"]:+.1f} | [诊断图](<{path.as_posix()}>) |')
    report = '# 价格残差注意力诊断\n\n'
    report += '选择 12 张固定规则的验证案例，逐例复现已保存预测后，展示四个注意力头及残差分支 token 梯度乘激活。原始全图始终保留；名义 patch 网格按真实预处理映射回原图，未被 14px 卷积覆盖的底部和右侧 6px 边界不着色。\n\n'
    report += '所有案例共用色标。敏感度图仅针对新增的价格残差分支，计算时缓存的核回归基线保持常量，因此不是完整价格模型的归因。最后层 token 包含全局上下文，梯度乘激活是一阶敏感度近似，不是独立局部像素贡献；注意力也不是泳池/建筑检测或因果证据。\n\n'
    report += '案例包括较好和固定随机样本，既有价格仅用于误差描述及诊断样本选择，不用于新训练、校准或方法选择。价格单位为千美元；raw prediction 复现容差 rtol=1e-6、atol=1e-3，用于 CPU/GPU 浮点差异。原图与图像衍生诊断图保存在 .local，不打包原始课程图片。\n\n'
    report += '| 图片 | 案例角色 | 真实价格 | 基线原始预测 | 最终预测 | 残差贡献 | 图像 |\n| --- | --- | ---: | ---: | ---: | ---: | --- |\n'
    report += '\n'.join(rows) + '\n\n原始四头权重、敏感度网格保存在各 case_maps.npz；元数据见 cases.json、registration.json、visual_scale.json。本诊断不涉及 Kaggle 测试图片或标签。\n'
    (public / '实验报告.md').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.reference_dir)
    except BaseException as exc:
        try:
            relative = args.reference_dir.resolve().relative_to(local_path('records_root'))
            diagnosis_root = REPOSITORY / 'output/experiment_record' / relative / 'attention_diagnostics'
            candidates = sorted(diagnosis_root.glob('*/status.json'))
            target = candidates[-1] if candidates else diagnosis_root / 'status.json'
            write_json(target, {'status': 'failed', 'pid': os.getpid(), 'error': repr(exc)})
        except ValueError:
            pass
        raise
