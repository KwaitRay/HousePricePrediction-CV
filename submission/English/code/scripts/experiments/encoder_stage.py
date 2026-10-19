"""Scratch mean-head and encoder experiments after the CLIP reference."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from copy import deepcopy
from pathlib import Path
import sys
import subprocess
import time
import numpy as np
import pandas as pd
from common.config import ROOT, resolve, validate, write_json, read_json, digest, differences
from common.local_paths import local_path, REPOSITORY

SEEDS = (2026, 2027, 2028)


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this registered queue')
    public = REPOSITORY / 'output/experiment_record/07_encoder'
    public.mkdir(parents=True, exist_ok=True)
    work = local_path('records_root') / '07_encoder' / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    work.mkdir(parents=True, exist_ok=False)
    import msvcrt
    handle = (public / 'queue.lock').open('a+b')
    handle.write(b'0'); handle.flush(); handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    # CLIP must have completed before scratch training starts.
    if read_json(REPOSITORY / 'output/experiment_record/03_clip_reference/status.json')['status'] != 'completed':
        raise RuntimeError('CLIP reference has not completed')
    baseline = resolve(ROOT / 'configs/selected/convolution.json')
    base_state = read_json(local_path('records_root') / '05_augmentation/geometry_20261004/state.json')
    controls = {s: Path(base_state['runs'][f'augment_01_flip_{s}']) for s in SEEDS}
    state = {'status': 'running', 'runs': {}, 'reviews': [], 'decisions': [],
             'protocol': 'scratch mean head; one-layer screen versus both mean head and retained AlexNet; replicate winners; two layers only after one layer accepted',
             'cuda': torch.version.cuda, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
    sources = {str(p): digest(p) for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts}
    write_json(work / 'registration.json', {'sources': sources, 'baseline': baseline, 'seeds': SEEDS, 'max_new_runs': 9})

    def publish():
        write_json(work / 'state.json', state)
        write_json(public / 'status.json', {k: v for k, v in state.items() if k != 'runs'})
        lines = ['# 卷积特征处理阶段实验报告', f'当前状态：{state["status"]}。',
                 '全部从零训练，不使用CLIP权重或特征。沿用翻转AlexNet的划分、输入、增强、优化器与40/15/0.1%/60早停协议。小头替换原回归头；编码器只在小头基础上添加。预训练CLIP仅为独立方法参照。',
                 '| 实验 | 种子 | 验证MSE | MAE | 尾部MSE | 相对AlexNet MSE | 最佳轮 |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in state['reviews']:
            lines.append(f'| {r["head"]} | {r["seed"]} | {r["mse"]:.2f} | {r["mae"]:.2f} | {r["tail_mse"]:.2f} | {r["delta_mse"]:+.2f} | {r["best_epoch"]} |')
        lines += state['decisions']
        (public / '实验报告.md').write_text('\n\n'.join(lines[:3]) + '\n\n' + '\n'.join(lines[3:]) + '\n', encoding='utf-8')

    def execute(head, seed):
        key = f'{head}_{seed}'
        if key in state['runs']:
            return Path(state['runs'][key])
        for path, value in sources.items():
            if digest(path) != value:
                raise RuntimeError(f'Source changed during queue: {path}')
        c = deepcopy(baseline)
        c['seed'] = seed
        c['model']['head'] = 'mean' if head == 'mean' else 'encoder'
        c['model']['encoder_layers'] = 2 if head == 'two_layers' else 1
        c['experiment'].update(id=f'encoder_{head}', name=f'卷积特征处理：{head}', stage='05_encoder', enforce_changes=False)
        c['_parent_source'] = None
        validate(c)
        ref = read_json(controls[seed] / 'config.json')
        allowed = {'model.head', 'model.encoder_layers', 'run_kind'}
        if set(differences(ref, c)) - allowed:
            raise ValueError('Unexpected change to baseline conditions')
        # Rebuild the historical initialization and pair all retained conv layers.
        from models.alexnet import AlexNetRegressor, initialize
        from initialization import hashes
        from common.metrics import transform_target
        from preparation.images import HouseImages
        torch.set_num_threads(c['training']['cpu_threads'])
        split = pd.read_csv(local_path('split_path'))
        training_frame = split.loc[split.partition == 'train']
        target_mean = float(np.mean(transform_target(training_frame.price.to_numpy(), c['target'])))
        original = AlexNetRegressor(ref['model'])
        initialize(original, seed, target_mean, ref['training']['initialization'])
        fingerprint, original_parts = hashes(original)
        if fingerprint != read_json(controls[seed] / 'model_summary.json')['initialization_sha256']:
            raise ValueError('Historical baseline initialization cannot be reproduced')
        del original
        candidate = AlexNetRegressor(c['model'])
        initialize(candidate, seed, target_mean, c['training']['initialization'])
        _, candidate_parts = hashes(candidate)
        retained = {k: v for k, v in original_parts.items() if not k.startswith(('regressor.', 'output.'))}
        if retained != {k: v for k, v in candidate_parts.items() if not k.startswith(('regressor.', 'output.'))}:
            raise ValueError('Shared convolution initialization changed')
        probe = HouseImages(local_path('data_root'), training_frame.sample(2, random_state=2026), c)
        candidate.eval()
        prediction = candidate(torch.stack([probe[i][0] for i in range(2)]))
        prediction.square().mean().backward()
        if prediction.shape != (2,) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in candidate.parameters()):
            raise ValueError('Invalid head forward/backward')
        write_json(work / f'{key}_preflight.json', {'baseline_initialization_sha256': fingerprint,
                   'shared_convolution_identical': True, 'forward_backward_passed': True,
                   'candidate_parameters': sum(p.numel() for p in candidate.parameters())})
        del candidate
        path = work / f'{key}.json'
        write_json(path, c)
        parent = local_path('records_root') / '07_encoder/runs' / c['experiment']['id']
        before = set(parent.glob('*'))
        state['status'] = f'running {key}'
        publish()
        with (work / f'{key}.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-X', 'utf8', str(ROOT / 'run.py'), 'train', str(path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    if (work / 'STOP').exists():
                        raise RuntimeError('Stop requested')
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError(f'Training failed; inspect {key}.log')
            except BaseException:
                if child.poll() is None:
                    subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
                raise
        created = list(set(parent.glob('*')) - before)
        if len(created) != 1:
            raise ValueError('Ambiguous run output')
        out = created[0]
        if read_json(out / 'status.json')['status'] != 'completed':
            raise ValueError('Incomplete run')
        actual = pd.read_csv(out / 'predictions.csv')
        expected = pd.read_csv(controls[seed] / 'predictions.csv')
        if not actual[['imageid', 'group_id', 'price']].equals(expected[['imageid', 'group_id', 'price']]):
            raise ValueError('Validation samples differ')
        history = pd.read_csv(out / 'history.csv')
        if not history.samples.eq(6399).all() or len(actual) != 1601:
            raise ValueError('Incomplete sample coverage')
        m = read_json(out / 'metrics.json')
        if not np.isclose(m['mse'], actual.squared_error.mean()):
            raise ValueError('Prediction metric mismatch')
        state['runs'][key] = str(out)
        r = {'head': head, 'seed': seed, **m, 'delta_mse': m['mse'] - read_json(controls[seed] / 'metrics.json')['mse'],
             'best_epoch': read_json(out / 'cost.json')['best_epoch']}
        state['reviews'].append(r)
        import shutil
        dest = public / 'runs' / c['experiment']['id'] / out.name
        dest.mkdir(parents=True, exist_ok=True)
        for filename in ('metrics.json', 'cost.json', 'history.csv', 'stopping.json', 'status.json', 'model_summary.json', 'config.json'):
            if filename == 'config.json':
                exported = {k:v for k,v in read_json(out / filename).items() if not k.startswith('_')}
                write_json(dest / filename, exported)
            else:
                shutil.copy2(out / filename, dest / filename)
        publish()
        return out

    def mse(path):
        return read_json(path / 'metrics.json')['mse']

    def accepted(runs, references):
        a, b = np.array([mse(p) for p in runs]), np.array([mse(p) for p in references])
        return bool(a.mean() < b.mean() and (a < b).sum() >= 2)

    try:
        publish()
        mean = {2026: execute('mean', 2026)}
        if mse(mean[2026]) < mse(controls[2026]):
            for s in SEEDS[1:]:
                mean[s] = execute('mean', s)
            state['decisions'].append('平均汇总小头三种子复核：' + ('优于AlexNet，保留为候选。' if accepted(list(mean.values()), list(controls.values())) else '未满足采用条件，保留原AlexNet。'))
            if accepted(list(mean.values()), list(controls.values())):
                selected = {k:v for k,v in read_json(mean[2026] / 'config.json').items() if not k.startswith('_') and k not in ('run_kind', 'selection')}
                selected['experiment']['enforce_changes'] = False
                selected['selection'] = {'reason': state['decisions'][-1], 'seeds': list(SEEDS),
                                          'mean_mse': float(np.mean([mse(p) for p in mean.values()]))}
                write_json(ROOT / 'configs/selected/head.json', selected)
            publish()
        encoder = {2026: execute('one_layer', 2026)}
        if mse(encoder[2026]) < min(mse(mean[2026]), mse(controls[2026])):
            for s in SEEDS[1:]:
                mean[s] = execute('mean', s)
                encoder[s] = execute('one_layer', s)
            if accepted(list(encoder.values()), list(mean.values())) and accepted(list(encoder.values()), list(controls.values())):
                state['decisions'].append('单层编码器三种子同时优于小头和AlexNet，开展两层对照。')
                two = [execute('two_layers', 2026)]
                if mse(two[0]) < mse(encoder[2026]):
                    two += [execute('two_layers', s) for s in SEEDS[1:]]
                    state['decisions'].append('两层对照：' + ('采用条件通过。' if accepted(two, list(encoder.values())) else '未通过，保留单层。'))
                else:
                    state['decisions'].append('两层初筛未优于单层，不追加种子。')
            else:
                state['decisions'].append('单层复核未同时优于两个对照，不开展两层。')
        else:
            state['decisions'].append('单层初筛未同时优于小头和AlexNet，不追加种子、不开展两层；结论仅限初筛。')
        state['status'] = 'completed'
        publish()
    except BaseException as exc:
        state['status'] = 'failed'
        state['error'] = str(exc)
        publish()
        raise


if __name__ == '__main__':
    main()
