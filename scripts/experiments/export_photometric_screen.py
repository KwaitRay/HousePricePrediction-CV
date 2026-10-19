"""Audit a completed local screen and export only portable, aggregate evidence.

No training, model promotion, test inference, or raw-image export is performed.
Run from the repository root:
  python scripts/experiments/export_photometric_screen.py --workdir PATH --output PATH
The output must be a new directory. Original run files remain untouched.
"""
import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from common.config import ROOT, differences, digest, read_json, write_json
from photometric_screen import NAMES, check


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(frame, threshold=1300.0):
    """Compute original-price-unit errors, independently of saved metrics."""
    require(len(frame) > 0, 'Empty prediction frame')
    require(not frame.imageid.duplicated().any(), 'Duplicate image IDs')
    values = frame[['price', 'prediction']].to_numpy(dtype=float)
    require(np.isfinite(values).all(), 'Non-finite predictions or prices')
    error = values[:, 1] - values[:, 0]
    tail = values[:, 0] > threshold
    require(tail.any(), 'Missing high-price tail')
    return dict(mse=float(np.mean(error**2)), mae=float(np.mean(abs(error))),
                tail_mse=float(np.mean(error[tail]**2)), n=len(frame),
                tail_n=int(tail.sum()), tail_threshold=threshold)


def price_slices(frame, group):
    # Fixed descriptive bins; no parameter/model selection uses these slices.
    labels = ('<=300', '(300,700]', '(700,1300]', '>1300')
    bins = pd.cut(frame.price, [-np.inf, 300, 700, 1300, np.inf], labels=labels)
    rows = []
    for label in labels:
        error = (frame.prediction - frame.price)[bins == label].to_numpy()
        require(len(error) > 0, f'Empty price bin: {label}')
        rows.append(dict(group=group, price_bin=label, n=len(error),
                         mse=float(np.mean(error**2)), mae=float(np.mean(abs(error))),
                         contribution_to_overall_mse=float(np.sum(error**2) / len(frame))))
    return rows


def portable_config(config):
    """Retain all experiment settings, replace only machine-local metadata."""
    config = {k: v for k, v in config.items() if not k.startswith('_')}
    config['output_root'] = 'artifacts/photometric_screen'
    # Effective data locations come from local_paths.json / CV_* overrides.
    config['data'] = dict(config['data'], root='<local_paths.data_root>',
                          split='<local_paths.split_path>')
    return config


def audit(work):
    registration = check(work)
    require(read_json(work / 'status.json')['status'] == 'completed', 'Queue incomplete')
    rows, slices, runs, histories, hashes = [], [], {}, {}, {}
    reference = reference_manifest = reference_init = reference_config = None
    for name in NAMES:
        run = Path(read_json(work / f'run_{name}.json')['path'])
        require(run.is_relative_to(work), 'Run outside screen directory')
        smoke = Path(read_json(work / f'smoke_{name}.json')['path'])
        require(read_json(smoke / 'status.json')['status'] == 'completed', 'Smoke incomplete')
        require(read_json(run / 'status.json') == {'status': 'completed', 'kind': 'development'},
                'Run incomplete or not a development run')
        config = read_json(run / 'config.json')
        manifest = read_json(run / 'data_manifest.json')
        init = read_json(run / 'model_summary.json')['initialization_sha256']
        prediction = pd.read_csv(run / 'predictions.csv')
        history = pd.read_csv(run / 'history.csv')
        stopping = read_json(run / 'stopping.json')
        expected_aug = {} if name == 'control' else {name: {'probability': 1., 'range': [.9, 1.1]}}
        require(config['augmentation'] == expected_aug and config['seed'] == 2026,
                'Unexpected augmentation or seed')
        if reference is None:
            reference, reference_manifest = prediction, manifest
            reference_init, reference_config = init, config
        else:
            pd.testing.assert_frame_equal(prediction[['imageid', 'group_id', 'price']],
                                          reference[['imageid', 'group_id', 'price']])
            require(manifest == reference_manifest, 'Data manifests differ')
            require(init == reference_init, 'Model initializations differ')
            require(all(k == 'augmentation' or k.startswith('augmentation.')
                        for k in differences(reference_config, config)), 'Non-augmentation difference')
        require(manifest['split_sha256'] == registration['split_sha256'], 'Split hash differs')
        require(len(manifest['train_ids']) == 6399 and len(prediction) == 1601, 'Wrong split counts')
        require(not set(manifest['train_ids']) & set(prediction.imageid), 'Train/validation overlap')
        require(history.samples.eq(6399).all() and history.updates.eq(200).all(), 'Incomplete epochs')
        require(len(history) == stopping['epochs'] == 40, 'Unexpected epoch count')
        require(history.epoch.tolist() == list(range(1, 41)), 'Missing/repeated epochs')
        require(stopping['reason'] == 'early_stopping', 'Unexpected stop reason')
        computed = metrics(prediction, manifest['tail_threshold'])
        saved = read_json(run / 'metrics.json')
        for key, value in computed.items():
            np.testing.assert_allclose(value, saved[key], rtol=1e-10, atol=1e-8)
        best = history.loc[history.val_mse.idxmin()]
        require(int(best.epoch) == stopping['best_epoch'], 'Wrong best checkpoint epoch')
        np.testing.assert_allclose(best.val_mse, computed['mse'], rtol=1e-10)
        events = read_json(run / 'augmentation_history.json')
        require(len(events) == len(history), 'Missing augmentation events')
        for event in events:
            require(set(event['operations']) == set(expected_aug), 'Unexpected operation')
            if name != 'control':
                stats = event['operations'][name]
                require(stats['samples'] == stats['triggered'] == 6399 and stats['skipped'] == 0,
                        'Augmentation not applied to all training samples')
                low, high = stats['observed_ranges']['factor']
                require(.9 <= low <= high <= 1.1, 'Factor outside registered range')
        rows.append(dict(group=name, run_id=run.name, **computed, epochs=len(history),
                         best_epoch=stopping['best_epoch']))
        slices.extend(price_slices(prediction, name))
        runs[name], histories[name] = run, history
        hashes[name] = {p.relative_to(run).as_posix(): digest(p) for p in run.rglob('*')
                        if p.is_file() and p.suffix in ('.json', '.csv')}
    for row in rows:
        row['delta_mse'] = row['mse'] - rows[0]['mse']
        row['delta_mse_percent'] = 100 * row['delta_mse'] / rows[0]['mse']
        row['delta_tail_mse_percent'] = 100 * (row['tail_mse'] / rows[0]['tail_mse'] - 1)
    return registration, rows, slices, runs, histories, hashes


def plot_curves(histories, output, scale):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (name, history) in zip(axes, histories.items()):
        ax.plot(history.epoch, history.data_loss * scale**2, label='Online train MSE')
        ax.plot(history.epoch, history.val_mse, label='Unaugmented validation MSE')
        best = history.loc[history.val_mse.idxmin()]
        ax.scatter([best.epoch], [best.val_mse], color='black', s=24, zorder=5,
                   label=f'Best epoch: {int(best.epoch)}')
        ax.set(title=name.capitalize(), xlabel='Epoch')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('MSE in original price units')
    fig.suptitle('Scratch AlexNet photometric screen | seed 2026 | same split and initialization')
    fig.text(.5, .015, 'Train: online augmented/dropout-active batches; validation: evaluation mode. '
             'Curves are not identical evaluation conditions.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(output / 'learning_curves.png', dpi=180)
    plt.close(fig)


def export(work, output):
    require(not output.exists(), 'Refusing to overwrite an existing evidence directory')
    registration, rows, slices, runs, histories, hashes = audit(work)
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', registration)
    write_json(output / 'results.json', dict(rows=rows, screening_only=True,
        automatic_promotion=False, price_unit='thousand USD; MSE in squared price units',
        conclusion='Brightness: effectively unchanged MSE. Contrast: lower overall MSE/MAE, '
                   'but higher high-price-tail MSE. No change to final ensemble or Kaggle submission.',
        limitations=['One seed; not a statistical significance claim.',
                     'Best epochs selected on the repeatedly reused validation split; not a fresh test.',
                     'Tail threshold 1300 comes from training data; slices are descriptive.',
                     'No conclusion about augmenting the frozen-encoder final model.']))
    pd.DataFrame(rows).to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(slices).to_csv(output / 'price_slices.csv', index=False)
    shutil.copyfile(work / 'input_checks.json', output / 'input_checks.json')
    for name, run in runs.items():
        dest = output / name
        dest.mkdir()
        for filename in ('history.csv', 'metrics.json', 'stopping.json', 'cost.json',
                         'augmentation_history.json', 'normalization.json', 'model_summary.json'):
            shutil.copyfile(run / filename, dest / filename)
        write_json(dest / 'config.json', portable_config(read_json(run / 'config.json')))
        environment = read_json(run / 'environment.json')
        environment.pop('executable', None)
        write_json(dest / 'environment.json', environment)
        shutil.copyfile(run / 'train_evaluation/metrics.json', dest / 'train_evaluation_metrics.json')
    plot_curves(histories, output, registration['target']['scale'])
    # Match Git's LF text storage; Windows checkouts may need CRLF normalization.
    for path in output.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.csv'):
            path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    write_json(output / 'audit.json', dict(
        passed=True, original_run_file_sha256=hashes,
        checks=['Frozen source/config/split hashes', 'Completed formal and smoke runs',
                'Same data manifest, initialization, validation IDs and labels',
                'Only augmentation differs', 'All 40 epochs consume 6399 samples and 200 updates',
                'Augmentation factors and counts checked for every epoch',
                'MSE/MAE/tail MSE independently recomputed from local predictions',
                'Best checkpoint matches minimum validation MSE'],
        export_note='Raw predictions, images, checkpoints and feature caches stay local. '
                    'Hashes identify original files, not sanitized copies. Public aggregates alone '
                    'cannot reproduce the per-image audit. Source hashes are original file bytes; '
                    'Git line-ending conversion can change them on another machine.',
        exporter_sha256=digest(Path(__file__)),
        exported_hash_policy='SHA256 of LF-normalized JSON/CSV bytes; binary files unchanged. '
                             'Normalize CRLF to LF when verifying Windows checkout files.',
        exported_file_sha256={p.relative_to(output).as_posix(): digest(p)
                              for p in output.rglob('*') if p.is_file()}))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.workdir.resolve(), args.output.resolve())
