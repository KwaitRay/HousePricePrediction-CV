"""Registered sequential Adam replication followed by cosine scheduling study."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
import traceback
import shutil
import numpy as np
import pandas as pd
import torch
from common.config import ROOT, read_json, write_json, resolve, validate, digest, differences
from common.local_paths import REPOSITORY, local_path

SEEDS = (2026, 2027, 2028)
RECORD = REPOSITORY / 'output/experiment_record/04_training'
REPORT = RECORD / '实验报告.md'
MARKER = '\n## 2026年10月3日：Adam复核与学习率调度\n'


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    write_json(path.with_suffix('.tmp'), value)
    os.replace(path.with_suffix('.tmp'), path)


def function_ast(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    return ast.dump(next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name))


def migration(legacy, config, work):
    torch.set_num_threads(config['training']['cpu_threads'])
    old = legacy / 'scripts/experiments'
    checks = {}
    for relative in ('preparation/images.py', 'models/alexnet.py', 'models/encoder.py', 'training/loading.py', 'common/metrics.py'):
        checks[relative] = ast.dump(ast.parse((old / relative).read_text(encoding='utf-8-sig'))) == ast.dump(ast.parse((ROOT / relative).read_text(encoding='utf-8-sig')))
    for name in ('train_epoch', '_fit_neural', 'predict', 'loader', 'worker_seed', 'EarlyStop', 'regularizer'):
        checks[name] = function_ast(old / 'training/engine.py', name) == function_ast(ROOT / 'training/engine.py', name)
    if not all(checks.values()):
        raise ValueError(f'Training semantics changed: {checks}')
    pointer = read_json(legacy / 'output/experiment_record/04_training/cache_optimization/cache_pointer.json')
    os.environ['CV_STRETCH_CACHE_DIRECTORY'] = pointer['directory']
    from training.cached_input import verify, CachedHouseImages
    from preparation.images import HouseImages
    from preparation.split import load_split
    meta = verify(config)
    frame, split = load_split(config)
    sample = frame.sample(32, random_state=2026)
    raw = HouseImages(local_path('data_root'), sample, config)
    cached = CachedHouseImages(local_path('data_root'), sample, config)
    for i in range(len(sample)):
        a, b = raw[i], cached[i]
        if not torch.equal(a[0], b[0]) or a[1:3] != b[1:3]:
            raise ValueError('Migrated cache input mismatch')
    result = {'passed': True, 'utc': now(), 'semantic_checks': checks, 'cache_key': meta['key'],
              'cache_sha256': meta['inputs_sha256'], 'all_source_images_checked': len(frame),
              'independent_tensor_recomputations': len(sample), 'split_sha256': split['split_sha256'],
              'scope': 'Core loop/model/loading AST equality plus exact cache input checks; optimizer extension separately tested on CPU.'}
    atomic(work / 'migration.json', result)
    os.environ['CV_MIGRATION_EVIDENCE'] = str(work / 'migration.json')
    return result


def plot(run):
    if not (run / 'history.csv').exists():
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h = pd.read_csv(run / 'history.csv')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(h.epoch, h.data_loss); axes[0].set_title('Training loss')
    axes[1].plot(h.epoch, h.probe_mse, label='Fixed train probe')
    axes[1].plot(h.epoch, h.val_mse, label='Validation'); axes[1].legend(); axes[1].set_title('Original-price MSE')
    axes[2].plot(h.epoch, h.lr); axes[2].set_title('Learning rate')
    for ax in axes: ax.set_xlabel('Epoch')
    fig.tight_layout(); fig.savefig(run / 'learning_curves.png', dpi=130); plt.close(fig)


def export(run):
    dest = RECORD / 'runs' / run.parent.name / run.name
    dest.mkdir(parents=True, exist_ok=True)
    for name in ('metrics.json', 'cost.json', 'stopping.json', 'status.json', 'history.csv', 'learning_curves.png', 'review.json'):
        if (run / name).exists(): shutil.copy2(run / name, dest / name)


def review(run, control, allowed):
    c, base = read_json(run / 'config.json'), read_json(control / 'config.json')
    diff = differences(base, c)
    if set(diff) - set(allowed): raise ValueError(f'Unexpected configuration changes: {diff}')
    if read_json(run / 'status.json')['status'] != 'completed': raise ValueError('Run did not complete')
    for file in ('data_manifest.json',):
        if read_json(run / file) != read_json(control / file): raise ValueError('Data manifest mismatch')
    init = read_json(run / 'model_summary.json')['initialization_sha256']
    if init != read_json(control / 'model_summary.json')['initialization_sha256']: raise ValueError('Initialization mismatch')
    a, b = pd.read_csv(run / 'predictions.csv'), pd.read_csv(control / 'predictions.csv')
    if not a[['imageid', 'price', 'group_id']].equals(b[['imageid', 'price', 'group_id']]): raise ValueError('Prediction sample mismatch')
    if not np.isfinite(a[['prediction', 'squared_error']].to_numpy()).all(): raise ValueError('Nonfinite predictions')
    h = pd.read_csv(run / 'history.csv')
    if not h.samples.eq(6399).all() or len(a) != 1601: raise ValueError('Incomplete samples')
    m, ref = read_json(run / 'metrics.json'), read_json(control / 'metrics.json')
    if not np.isclose(a.squared_error.mean(), m['mse']): raise ValueError('Metric mismatch')
    groups = []
    for label, mask in [('低价≤300', a.price <= 300), ('中价300–700', (a.price > 300) & (a.price <= 700)), ('高价700–1300', (a.price > 700) & (a.price <= 1300)), ('尾部>1300', a.price > 1300)]:
        groups.append({'group': label, 'n': int(mask.sum()), 'mean_signed_error': float(a.loc[mask, 'error'].mean()), 'delta_mse': float((a.loc[mask, 'squared_error'] - b.loc[mask, 'squared_error']).mean()), 'contribution': float((a.loc[mask, 'squared_error'] - b.loc[mask, 'squared_error']).sum() / len(a))})
    dominant = max(groups, key=lambda x: abs(x['contribution']))
    gap = m['mse'] - read_json(run / 'train_evaluation/metrics.json')['mse']
    delta = m['mse'] - ref['mse']
    result = {'seed': c['seed'], 'optimizer': c['training']['optimizer'], 'schedule': c['training']['schedule'],
              'mse': m['mse'], 'control_mse': ref['mse'], 'delta_mse': delta, 'price_groups': groups,
              'prediction_std': float(a.prediction.std()), 'target_std': float(a.price.std()),
              'train_validation_gap': gap, 'best_epoch': read_json(run / 'cost.json')['best_epoch'],
              'text': f"种子{c['seed']}，{c['training']['optimizer']}／{c['training']['schedule']}：验证均方误差{m['mse']:.2f}，相对同种子对照变化{delta:+.2f}。主要误差变化来自{dominant['group']}组（对总误差贡献{dominant['contribution']:+.2f}）；最佳模型训练与验证误差差距{gap:.2f}。分组贡献定位结果来源，不能单凭相关性证明优化机制。先保留该结果，按预登记种子完成复核，不临时改参。"}
    if a.prediction.std() < .25 * a.price.std():
        result['text'] += '预测离散程度不足真实价格的四分之一，提示预测向均值收缩；训练验证差距小并不意味着学到了充分的图像信息，后续调度需同时检查这一现象。'
    atomic(run / 'review.json', result)
    export(run)
    return result


def publish(state):
    original = REPORT.read_text(encoding='utf-8').split(MARKER)[0]
    header_end = original.find('\n---\n')
    if header_end >= 0 and original.startswith('# 当前状态'):
        original = '# 当前状态：Adam复核与实验三按序执行\n\n以文末本轮状态为准。此前暂停记录保留在历史证据中。\n' + original[header_end:]
    text = MARKER + '\n用户已授权恢复：先复核Adam 0.001，再开展学习率调度。Muon剩余候选不继续。数据划分、输入、结构、批量、正则化和停止协议保持一致。\n\n'
    text += '选择规则：三种子平均误差降低且至少两次同种子改善才采用Adam，否则保留AdamW。实验三按相同规则比较固定学习率与余弦衰减，周期60轮、最低学习率为初始值1%，不加预热。早停可能截断衰减，因此结论限定于共同停止协议。\n\n'
    text += f"当前状态：{state['status']}；更新时间：{state['updated_utc']}。\n\n"
    for item in state['reviews']: text += item['text'] + '\n\n'
    for key in ('optimizer_decision', 'schedule_decision'):
        if key in state: text += state[key]['text'] + '\n\n'
    temporary = REPORT.with_suffix('.tmp'); temporary.write_text(original + text, encoding='utf-8'); os.replace(temporary, REPORT)
    public = {k: v for k, v in state.items() if k not in ('runs', 'pid')}
    atomic(RECORD / 'continuation_status.json', public)


def decide(candidates, controls, label):
    a = np.array([read_json(p / 'metrics.json')['mse'] for p in candidates])
    b = np.array([read_json(p / 'metrics.json')['mse'] for p in controls])
    wins = int((a < b).sum()); accepted = bool(a.mean() < b.mean() and wins >= 2)
    return {'accepted': accepted, 'mean_mse': float(a.mean()), 'control_mean_mse': float(b.mean()), 'wins': wins,
            'text': f"{label}：候选三种子均值{a.mean():.2f}，对照{b.mean():.2f}，同种子获胜{wins}/3；{'保留候选' if accepted else '保留对照'}。"}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--legacy-root', type=Path, required=True)
    args = parser.parse_args(); legacy = args.legacy_root.resolve()
    work = local_path('records_root') / '04_training/continuation_20261003'
    work.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = (work / 'queue.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    if (work / 'state.json').exists(): raise RuntimeError('Existing continuation state: review before restarting')
    state = {'status': '正在核验迁移和缓存', 'updated_utc': now(), 'reviews': [], 'runs': {}, 'pid': os.getpid()}
    def save():
        state['updated_utc'] = now(); atomic(work / 'state.json', state); publish(state)
    def execute(config, seed, reference, allowed):
        candidate = deepcopy(config); candidate['seed'] = seed
        candidate['_parent_source'] = None
        validate(candidate)
        comparison = deepcopy(candidate); comparison['run_kind'] = 'development'
        diff = differences(read_json(reference / 'config.json'), comparison)
        if set(diff) - set(allowed): raise ValueError(f'Pre-run control mismatch: {diff}')
        path = work / f"{candidate['experiment']['id']}_{seed}.json"; write_json(path, candidate)
        parent = local_path('records_root') / '04_training/runs' / candidate['experiment']['id']
        existing = set(parent.glob('*'))
        state['status'] = f"运行{candidate['experiment']['id']}，种子{seed}"; save()
        with (work / f'{path.stem}.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-X', 'utf8', str(ROOT / 'train_cached.py'), 'train', str(path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    if (work / 'STOP').exists(): raise RuntimeError('User stop requested')
                    for out in set(parent.glob('*')) - existing:
                        try:
                            plot(out); export(out)
                        except (pd.errors.EmptyDataError, pd.errors.ParserError, PermissionError):
                            pass  # Retry after an in-progress history write.
                    time.sleep(30)
                if child.returncode: raise RuntimeError(f'Training failed, see {path.stem}.log')
            except BaseException:
                if child.poll() is None:
                    subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
                raise
        created = list(set(parent.glob('*')) - existing)
        if len(created) != 1: raise ValueError('Ambiguous output directory')
        out = created[0]; plot(out)
        state['reviews'].append(review(out, reference, allowed))
        state['runs'][path.stem] = str(out); state['status'] = '本组复核完成，准备下一组'; save()
        return out
    try:
        save()
        config = resolve(ROOT / 'configs/04_training_recorded/tune_02_adam_0p001.json')
        config = {k: v for k, v in config.items() if not k.startswith('_')}
        config['experiment']['enforce_changes'] = False
        migration(legacy, config, work)
        oldrecord = legacy / 'output/experiment_record'
        mapping = read_json(oldrecord / '01_preprocessing/pipeline_state_v2.json')['runs']
        controls = [oldrecord / mapping[f'preprocess_01_stretch:{s}'] for s in SEEDS]
        previous = [p for p in (oldrecord / '04_training/runs/tune_02_adam_0p001').glob('*') if (p / 'metrics.json').exists() and read_json(p / 'status.json')['status'] == 'completed']
        if len(previous) != 1: raise ValueError('Expected exactly one completed historical Adam run')
        adam = [previous[0]]
        state['reviews'].append(review(adam[0], controls[0], {'training.optimizer', 'training.lr'})); save()
        for seed, control in zip(SEEDS[1:], controls[1:]):
            adam.append(execute(config, seed, control, {'training.optimizer', 'training.lr'}))
        decision = decide(adam, controls, '优化器选择'); state['optimizer_decision'] = decision
        chosen = adam if decision['accepted'] else controls
        base = read_json(chosen[0] / 'config.json')
        base = {k: v for k, v in base.items() if not k.startswith('_') and k not in ('run_kind', 'selection')}
        base['experiment']['enforce_changes'] = False
        base['selection'] = {'reason': decision['text'], 'seeds': list(SEEDS), 'new_runtime_migration_verified': True}
        write_json(ROOT / 'configs/selected/optimizer.json', base); save()
        cosine = deepcopy(base); cosine['training']['schedule'] = 'cosine'
        cosine['experiment'].update(id='tune_03_cosine', name='学习率余弦衰减', stage='02_training', changed_fields=['training.schedule'], comparison='同种子固定学习率', question='共同停止协议下衰减是否改善验证误差')
        runs = [execute(cosine, seed, ref, {'training.schedule'}) for seed, ref in zip(SEEDS, chosen)]
        state['schedule_decision'] = decide(runs, chosen, '学习率调度选择')
        selected = deepcopy(cosine if state['schedule_decision']['accepted'] else base)
        selected['selection'] = {'reason': state['schedule_decision']['text'], 'seeds': list(SEEDS)}
        selected['seed'] = 2026
        write_json(ROOT / 'configs/selected/schedule.json', selected)
        state['status'] = 'Adam复核与实验三完成；不自动进入实验四'; save()
    except BaseException:
        state['status'] = '异常停止，需定位后再继续'; state['error'] = traceback.format_exc(); save(); raise


if __name__ == '__main__':
    main()
