"""Compare bounded ordinal readouts with an exact continuous-regression null control."""
import os
from pathlib import Path
import shutil
import time
import uuid
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import eigh, solve
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import isotonic_regression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from common.metrics import save_predictions
from visual_kernel_components import feature_registration, kernel_for
from visual_kernel_submission import features_for

SOURCE = '19_image_embedding_reference/siglip2_price_metric_kernel/20261013T155603Z_4129a7fd'
THRESHOLDS = np.arange(0, 2000, 100, dtype=np.float64) / 1000
STEP = 0.1


def targets(y):
    survival = (y[:, None] > THRESHOLDS[None, :]).astype(np.float64)
    residual = y - STEP * survival.sum(axis=1)
    result = np.column_stack([survival, residual])
    np.testing.assert_allclose(STEP * result[:, :-1].sum(axis=1) + result[:, -1], y, rtol=0, atol=1e-14)
    return result


def reconstruct(outputs, variant):
    probabilities = outputs[:, :-1]
    if variant == 'bounded':
        probabilities = np.clip(probabilities, 0, 1)
    elif variant == 'isotonic':
        probabilities = np.stack([isotonic_regression(row, increasing=False, y_min=0, y_max=1) for row in probabilities])
    elif variant != 'null':
        raise ValueError('Unknown ordinal readout')
    return STEP * probabilities.sum(axis=1) + outputs[:, -1]


def main():
    root = local_path('records_root')
    source = root / SOURCE
    source_public = REPOSITORY / 'output/experiment_record' / SOURCE
    if read_json(source_public / 'status.json')['status'] != 'completed':
        raise ValueError('Source metric kernel is not completed')
    registration_source = read_json(source_public / 'registration.json')
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    train, valid = frame[frame.partition.eq('train')], frame[frame.partition.eq('validation')]
    if len(train) != 6399 or len(valid) != 1601 or set(train.group_id) & set(valid.group_id):
        raise ValueError('Unexpected frozen group split')
    x = features_for(feature_registration(registration_source), frame)
    xt, xv = x[frame.partition.eq('train')], x[frame.partition.eq('validation')]
    y = train.price.to_numpy() / 1000
    groups = train.group_id.to_numpy()
    labels = targets(y)
    prior = pd.read_csv(source / 'selected_training_oof.csv', dtype={'imageid': str})
    if prior.imageid.tolist() != train.imageid.tolist():
        raise ValueError('Source OOF order differs')
    previous = {k: v for k, v in np.load(source / 'head.npz', allow_pickle=False).items()}
    np.testing.assert_allclose(xt, previous['train_features'], rtol=0, atol=1e-12)
    image_gamma = float(previous['image_gamma'])
    pls_gamma, components = float(previous['pls_gamma']), int(previous['components'])
    null_alpha = float(previous['alpha'])
    if components != 4 or image_gamma != 4 or pls_gamma != 0.1 or null_alpha != 1:
        raise ValueError('The preregistered source parameters changed')
    out = root / '19_image_embedding_reference/siglip2_ordinal_price_kernel' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    public = REPOSITORY / 'output/experiment_record' / out.relative_to(root)
    registration = {'method': 'Bounded ordinal survival readout over fold-fitted PLS product kernel',
        'source': SOURCE, 'source_head_sha256': digest(source / 'head.npz'), 'source_metric_sha256': digest(source / 'feature_metric.joblib'),
        'source_registration_sha256': digest(source_public / 'registration.json'), 'source_sha256': digest(__file__),
        'split_sha256': digest(local_path('split_path')), 'thresholds_thousand_usd': (THRESHOLDS * 1000).tolist(),
        'bin_step_thousand_usd': 100, 'targets': '20 survival indicators plus exact within-bin residual; unprojected sum exactly equals continuous price',
        'image_gamma': image_gamma, 'pls_gamma': pls_gamma, 'pls_fit_components': 16, 'used_components': components,
        'alphas': [0.1, 1, 10], 'variants': ['bounded', 'isotonic'],
        'null_control': 'Existing continuous PLS product kernel at alpha=1; exact unprojected ordinal reconstruction replay on every OOF and validation image',
        'selection': 'Original unweighted training GroupKFold3 MSE, including existing null baseline; validation once after selection',
        'caveat': 'Source metric hyperparameters were already selected on these training folds; this is not a nested unbiased estimate',
        'mllm_components': False, 'test_images_used': False, 'test_labels_used': False,
        'course_scope': 'Provided training labels and existing generic frozen image features only'}
    write_json(public / 'registration.json', registration)
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    def status(state, **extra):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **extra})
    started = time.perf_counter()
    try:
        status('fitting', fold=0)
        candidates = {(variant, alpha): np.full(len(y), np.nan) for variant in ('bounded', 'isotonic') for alpha in (0.1, 1., 10.)}
        null_replay = np.full(len(y), np.nan)
        with threadpool_limits(limits=2):
            image_kernel = np.exp(-image_gamma * np.maximum(2 - 2 * (xt @ xt.T), 0))
            for fold, (fit, held) in enumerate(GroupKFold(3).split(xt, y, groups), 1):
                if set(groups[fit]) & set(groups[held]):
                    raise ValueError('Fold group leakage')
                status('fitting', fold=fold)
                pls = PLSRegression(n_components=16, scale=True, max_iter=500).fit(xt[fit], y[fit])
                raw_scores = pls.transform(xt)
                scaled = StandardScaler().fit(raw_scores[fit]).transform(raw_scores)[:, :components]
                a = image_kernel[np.ix_(fit, fit)] * np.exp(-pls_gamma * cdist(scaled[fit], scaled[fit], 'sqeuclidean') / components)
                b = image_kernel[np.ix_(held, fit)] * np.exp(-pls_gamma * cdist(scaled[held], scaled[fit], 'sqeuclidean') / components)
                eigen, vectors = eigh(a, check_finite=False, overwrite_a=True)
                mean = labels[fit].mean(axis=0)
                projected = vectors.T @ (labels[fit] - mean)
                held_vectors = b @ vectors
                for alpha in (0.1, 1., 10.):
                    outputs = held_vectors @ (projected / (np.maximum(eigen, 0)[:, None] + alpha)) + mean
                    for variant in ('bounded', 'isotonic'):
                        candidates[(variant, alpha)][held] = reconstruct(outputs, variant)
                    if alpha == null_alpha:
                        null_replay[held] = reconstruct(outputs, 'null')
                print(f'Ordinal price kernel fold {fold}/3 completed', flush=True)
                del a, b, eigen, vectors, projected, held_vectors
            np.testing.assert_allclose(null_replay * 1000, prior.raw_oof_prediction.to_numpy(), rtol=1e-7, atol=1e-4)
            best_oof = prior.raw_oof_prediction.to_numpy() / 1000
            best = {'variant': 'null', 'alpha': null_alpha, 'training_oof_mse': float(np.mean((np.maximum(best_oof, 0) - y) ** 2) * 1e6)}
            rows = [best.copy()]
            for (variant, alpha), prediction in candidates.items():
                if not np.isfinite(prediction).all():
                    raise ValueError('Incomplete ordinal OOF predictions')
                row = {'variant': variant, 'alpha': alpha, 'training_oof_mse': float(np.mean((np.maximum(prediction, 0) - y) ** 2) * 1e6)}
                rows.append(row)
                if row['training_oof_mse'] < best['training_oof_mse']:
                    best, best_oof = row.copy(), prediction.copy()
            pd.DataFrame(rows).to_csv(public / 'training_cv.csv', index=False)
            metric = joblib.load(source / 'feature_metric.joblib')
            kernel = kernel_for(previous, xt, metric)
            validation_kernel = kernel_for(previous, xv, metric)
            alpha = best['alpha']
            kernel.flat[::len(y) + 1] += alpha
            mean = labels.mean(axis=0)
            coefficients = solve(kernel, labels - mean, assume_a='pos', check_finite=False, overwrite_a=True)
            outputs = validation_kernel @ coefficients + mean
            old_prediction = pd.read_csv(source / 'predictions.csv', dtype={'imageid': str})
            if old_prediction.imageid.tolist() != valid.imageid.tolist():
                raise ValueError('Source validation order differs')
            if alpha == null_alpha:
                np.testing.assert_allclose(reconstruct(outputs, 'null') * 1000, old_prediction.raw_prediction, rtol=1e-7, atol=1e-4)
            else:
                # Replay the original alpha independently, without using validation to choose it.
                direct = (validation_kernel @ previous['coefficients'] + float(previous['target_mean'])) * 1000
                np.testing.assert_allclose(direct, old_prediction.raw_prediction, rtol=1e-7, atol=1e-4)
            prediction = reconstruct(outputs, best['variant']) * 1000
        metrics = save_predictions(out, valid, prediction, 1300)
        if best['variant'] == 'null':
            np.testing.assert_allclose(prediction, old_prediction.raw_prediction, rtol=1e-7, atol=1e-4)
            shutil.copyfile(source / 'head.npz', out / 'head.npz')
        else:
            np.savez_compressed(out / 'head.npz', train_features=xt, train_pls_scores=previous['train_pls_scores'],
                image_gamma=image_gamma, pls_gamma=pls_gamma, components=components, alpha=alpha,
                coefficients=coefficients, target_mean=mean, thresholds=THRESHOLDS, step=STEP, ordinal_variant=best['variant'])
        shutil.copyfile(source / 'feature_metric.joblib', out / 'feature_metric.joblib')
        pd.DataFrame({'imageid': train.imageid, 'raw_oof_prediction': best_oof * 1000, 'oof_prediction': np.maximum(best_oof, 0) * 1000}).to_csv(out / 'selected_training_oof.csv', index=False)
        result = {'head': 'ordinal_kernel_readout', **best, **metrics, 'source_validation_mse': read_json(source_public / 'results.json')['mse'],
            'training_null_replay_passed': True, 'validation_source_replay_passed': True, 'seconds': time.perf_counter() - started}
        write_json(public / 'results.json', result)
        status('completed')
        (public / '实验报告.md').write_text('# 价格有序概率读出实验\n\n'
            f'训练侧选择：{best["variant"]}，alpha={alpha}；OOF MSE={best["training_oof_mse"]:.5f}。\n\n'
            f'验证 MSE={metrics["mse"]:.5f}，MAE={metrics["mae"]:.5f}，高价组 MSE={metrics["tail_mse"]:.5f}。\n\n'
            '无约束累计概率加区间残差能精确还原连续价格；全部训练 OOF 和源验证预测核验通过。PLS 与概率目标均只在拟合折训练，投影模式和 alpha 按完整未加权训练 OOF MSE 选择；验证未用于选择。固定阈值为 0 到 1900 千美元、间隔 100，最终保留原价非负下限。\n\n'
            '若选择 null，则有序投影未显示训练侧优势，保持现有连续回归，不登记为模型提升。基础核超参数此前已选择，本次 OOF 不代表嵌套无偏评估。该候选尚无 Kaggle 公榜核验，也未生成课程提交 CSV。\n', encoding='utf-8')
        print(result, flush=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
