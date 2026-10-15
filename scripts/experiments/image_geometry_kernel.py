"""Test native image geometry as an added covariate, without listing metadata."""
import json
import os
import time
import uuid
import numpy as np
import pandas as pd
from PIL import Image
from scipy.linalg import eigh, solve
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from common.metrics import save_predictions
from visual_kernel_submission import features_for


def main():
    root = local_path('records_root')
    source = '13_clip_optimization/clip_l14_multiview_kernel/20261013T120554Z_e2a41622'
    config = read_json(REPOSITORY / 'output/experiment_record' / source / 'registration.json')
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    train, valid = frame.partition.eq('train').to_numpy(), frame.partition.eq('validation').to_numpy()
    x = features_for(config, frame)
    geometry = []
    for name in frame.imageid:
        with Image.open(local_path('data_root') / 'train' / name) as im:
            w, h = im.size
            if im.getexif().get(274, 1) in [5, 6, 7, 8]:
                w, h = h, w
        geometry.append([np.log(w), np.log(h)])
    geometry = np.asarray(geometry, dtype=np.float64)
    with np.load(root / source / 'head.npz', allow_pickle=False) as previous:
        np.testing.assert_allclose(x[train], previous['train_features'], rtol=0, atol=1e-12)
        image_gamma = float(previous['gamma'])
    y = frame.loc[train, 'price'].to_numpy() / 1000
    groups = frame.loc[train, 'group_id'].to_numpy()
    assert len(frame) == 8000 and train.sum() == 6399 and valid.sum() == 1601
    assert not set(groups) & set(frame.loc[valid, 'group_id'])
    gammas, alphas = [0., .01, .1, 1.], [.01, .1, 1., 10.]
    run = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    relative = '13_clip_optimization/clip_multiview_geometry_kernel/' + run
    out, public = root / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    registration = {'method': 'RBF image kernel multiplied by RBF native-image-geometry kernel',
                    'geometry': ['log(native width)', 'log(native height)'], 'geometry_gammas': gammas,
                    'image_gamma': image_gamma, 'alphas': alphas, 'source': source,
                    'source_head_sha256': digest(root / source / 'head.npz'), 'source_sha256': digest(__file__),
                    'split_sha256': digest(local_path('split_path')), 'selection': 'Original unweighted training GroupKFold3 MSE',
                    'geometry_scaling': 'StandardScaler fit separately on each fitting fold; never validation',
                    'geometry_gamma_zero': 'Image-only null control; must reproduce original selected training OOF predictions',
                    'test_images_used': False, 'filename_or_order_features': False,
                    'course_scope': 'Provided image dimensions only, no EXIF GPS, listing metadata or MLLM',
                    'caveat': 'Geometry is photographic framing/resolution, not physical house area; source gamma selected in previous training CV'}
    write_json(public / 'registration.json', registration)
    write_json(out / 'registration.json', registration)
    def status(state, **kw):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **kw})
    started = time.perf_counter()
    try:
        rows, best = [], None
        folds = list(GroupKFold(3).split(x[train], y, groups))
        with threadpool_limits(limits=4):
            image_kernel = np.exp(-image_gamma * np.maximum(2 - 2 * (x[train] @ x[train].T), 0))
            for gamma in gammas:
                oof = {a: np.full(len(y), np.nan) for a in alphas}
                for fold, (fit, held) in enumerate(folds):
                    assert not set(groups[fit]) & set(groups[held])
                    status('fitting', geometry_gamma=gamma, fold=fold + 1)
                    scaler = StandardScaler().fit(geometry[train][fit])
                    gf = scaler.transform(geometry[train][fit])
                    gh = scaler.transform(geometry[train][held])
                    kernel = image_kernel[np.ix_(fit, fit)] * np.exp(-gamma * cdist(gf, gf, metric='sqeuclidean'))
                    eigen, vectors = eigh(kernel, check_finite=False, overwrite_a=True)
                    held_kernel = image_kernel[np.ix_(held, fit)] * np.exp(-gamma * cdist(gh, gf, metric='sqeuclidean'))
                    mean = float(y[fit].mean())
                    projected, held_vectors = vectors.T @ (y[fit] - mean), held_kernel @ vectors
                    for alpha in alphas:
                        oof[alpha][held] = held_vectors @ (projected / (np.maximum(eigen, 0) + alpha)) + mean
                    print(f'Native geometry gamma={gamma} fold={fold + 1}/3', flush=True)
                if gamma == 0:
                    original = pd.read_csv(root / source / 'selected_training_oof.csv', dtype={'imageid': str})
                    original = original.set_index('imageid').loc[frame.loc[train, 'imageid'], 'oof_prediction'].to_numpy()
                    np.testing.assert_allclose(np.maximum(oof[1.], 0) * 1000, original, rtol=1e-7, atol=1e-4)
                    write_json(public / 'null_control.json', {'passed': True, 'original_oof_reproduced': True})
                for alpha in alphas:
                    score = float(np.mean((np.maximum(oof[alpha], 0) - y)**2) * 1e6)
                    row = {'geometry_gamma': gamma, 'alpha': alpha, 'training_oof_mse': score}
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(public / 'training_cv.csv', index=False)
                    if best is None or score < best['training_oof_mse']:
                        best, best_oof = row.copy(), oof[alpha].copy()
            scaler = StandardScaler().fit(geometry[train])
            gt, gv = scaler.transform(geometry[train]), scaler.transform(geometry[valid])
            gamma, alpha, mean = best['geometry_gamma'], best['alpha'], float(y.mean())
            kernel = image_kernel * np.exp(-gamma * cdist(gt, gt, metric='sqeuclidean'))
            kernel.flat[::len(y) + 1] += alpha
            coefficients = solve(kernel, y - mean, assume_a='pos', check_finite=False, overwrite_a=True)
            vk = np.exp(-image_gamma * np.maximum(2 - 2 * (x[valid] @ x[train].T), 0))
            vk *= np.exp(-gamma * cdist(gv, gt, metric='sqeuclidean'))
            raw = (vk @ coefficients + mean) * 1000
        metrics = save_predictions(out, frame.loc[valid], raw, 1300)
        np.savez_compressed(out / 'head.npz', train_features=x[train], train_geometry=gt, geometry_mean=scaler.mean_,
                    geometry_scale=scaler.scale_, coefficients=coefficients, image_gamma=image_gamma,
                    geometry_gamma=gamma, alpha=alpha, target_mean=mean)
        pd.DataFrame({'imageid': frame.loc[train, 'imageid'], 'raw_oof_prediction': best_oof * 1000,
                    'oof_prediction': np.maximum(best_oof, 0) * 1000}).to_csv(out / 'selected_training_oof.csv', index=False)
        result = {'head': 'image_geometry_kernel', **best, **metrics, 'seconds': time.perf_counter() - started}
        write_json(public / 'results.json', result)
        status('completed')
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
