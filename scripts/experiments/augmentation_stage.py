"""Registered independent flip/geometry screens, then at most two replications."""
from continue_training import *
import continue_training as shared

RECORD = REPOSITORY / 'output/experiment_record/05_augmentation'
REPORT = RECORD / '实验报告.md'
shared.RECORD = RECORD
CANDIDATES = ('augment_01_flip', 'augment_05_rotation', 'augment_06_translation', 'augment_07_crop')
NAMES = dict(zip(CANDIDATES, ('实验一：水平翻转', '实验五：小角度旋转', '实验六：小幅平移', '实验七：轻度裁剪')))
WORK_NAME = 'geometry_20261004'
PRIOR_WINNERS = []
COMPLETE_STATUS = '实验一、五、六、七及必要复核完成；队列结束'


def review_augmented(run, control, allowed):
    result = shared.review(run, control, allowed)
    config = read_json(run / 'config.json')
    events = read_json(run / 'augmentation_history.json')
    op = next(iter(config['augmentation']))
    history = pd.read_csv(run / 'history.csv')
    if len(events) != len(history): raise ValueError('Missing augmentation epochs')
    for epoch in events:
        stats = epoch['operations']
        if set(stats) != {op} or stats[op]['samples'] != 6399:
            raise ValueError('Incorrect augmentation sample coverage')
        if not 0 <= stats[op]['skipped'] <= stats[op]['triggered'] <= 6399:
            raise ValueError('Invalid augmentation trigger counters')
    result['experiment'] = config['experiment']['id']
    result['trigger_rate'] = sum(e['operations'][op]['triggered'] for e in events) / (6399 * len(events))
    result['skipped_count'] = sum(e['operations'][op]['skipped'] for e in events)
    result['epochs'] = len(history)
    result['train_mse'] = read_json(run / 'train_evaluation/metrics.json')['mse']
    positive = [g for g in result['price_groups'] if g['contribution'] > 0]
    negative = [g for g in result['price_groups'] if g['contribution'] < 0]
    lines = []
    if positive:
        g = max(positive, key=lambda x: x['contribution'])
        lines.append(f"主要退步来源：{g['group']}，对总误差变化贡献{g['contribution']:+.2f}")
    if negative:
        g = min(negative, key=lambda x: x['contribution'])
        lines.append(f"主要改善来源：{g['group']}，贡献{g['contribution']:+.2f}")
    result['analysis'] = '；'.join(lines) + '。这是误差分解，不是机制的因果证明。'
    atomic(run / 'review.json', result)
    shared.export(run)
    shutil.copy2(run / 'augmentation_history.json', RECORD / 'runs' / run.parent.name / run.name / 'augmentation_history.json')
    return result


def publish(state):
    lines = ['# 数据增强阶段实验报告',
             f"当前进度：{state['status']}。更新时间（UTC）：{state['updated_utc']}。",
             '## 一、研究问题与固定条件',
             '训练调优结束后，无增强对照三种子平均验证均方误差127774.42。当前存在训练验证差距与价格段误差抵消，研究轻微构图变化是否改善泛化，不预先认定增强有效。',
             '固定直接拉伸224×224、从零训练AlexNet、默认Kaiming初始化、AdamW学习率0.0001固定、有效批量32（4×8）、随机失活0.5、权重衰减0.0001，无显式惩罚。冻结6399张训练、1601张验证及相同种子；测试集不参与选择。最少40轮、耐心15、阈值0.1%、最多60轮，恢复最低验证误差检查点。',
             '只对训练图增强：原图→随机增强→拉伸→归一化。验证和训练误差评估均关闭增强，复用已核验的无增强缓存。训练图片不固定缓存随机增强，以保持逐轮变化；实际耗时包含原图解码与变换。',
             '## 二、单因素实验及依据',
             '| 实验 | 参数 | 研究依据与风险 |', '| --- | --- | --- |',
             '| 一：水平翻转 | 概率50% | 检验左右构图依赖，可能改变文字方向 |',
             '| 五：小角度旋转 | 概率30%，−3°至3° | 检验轻微倾斜；双线性插值、灰色填角、不扩展画布 |',
             '| 六：小幅平移 | 概率30%，横纵各±3% | 检验居中依赖；灰色填边，注意边缘截断 |',
             '| 七：轻度裁剪 | 概率30%，保留90%–100%面积 | 尽量保持宽高比（整数取整有误差），检查周边信息损失 |',
             '四项分别与同一无增强对照比较，不累积前一项。先依次跑种子2026；完整性通过且严格优于同种子对照的候选中，按误差选最多两项补2027/2028。预计4–8次训练。最终采用须三种子平均误差更低且至少两个配对改善；多个通过则选均值最低。单种子未晋级只说明本次筛选未通过。筛选存在选择偏差，最终效果还需独立评价。',
             '## 三、输入核验与图片预览',
             '每项首次训练前，对固定32张训练图片生成5种随机视图，另生成4张强制触发预览供人工检查。预览与逐图事件保存在本地，不上传数据图片。检查同种子同轮重现、跨轮变化、形状及有限值、全局随机流不受增强影响、验证侧与原始确定性输入逐位相同。',
             '## 四、实验结果与逐组归因',
             '| 实验 | 种子 | 验证均方误差 | 相对同种子对照 | 最佳轮／总轮 | 实际触发率 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in state['reviews']:
        lines.append(f"| {NAMES[r['experiment']]} | {r['seed']} | {r['mse']:.2f} | {r['delta_mse']:+.2f} | {r['best_epoch']}/{r['epochs']} | {r['trigger_rate']:.1%} |")
    if not state['reviews']: lines.append('尚无完整训练结果，不能判断是否有效。')
    for r in state['reviews']:
        lines.append(f"**{NAMES[r['experiment']]}，种子{r['seed']}：** {r['analysis']} 最佳模型无增强训练误差{r['train_mse']:.2f}，验证与训练差距{r['train_validation_gap']:.2f}。{'初筛改善；仅达到后续复核资格。' if r['delta_mse'] < 0 else '本次未优于对照；不因训练误差下降而采用。'}")
    lines.extend(['## 五、选取与下一步', *[d['text'] for d in state['decisions']],
                  '每组完成配置、初始化、数据指纹、样本完整性、增强触发统计、价格段贡献及训练验证差距检查后，才开始下一组；异常即停。曲线见本阶段runs目录。完成本轮四项及必要复核后停止，不自动进入颜色增强或组合实验。'])
    if state.get('error'): lines.append('执行异常已记录到状态文件，定位后才能继续。')
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    import re
    rendered = re.sub(r'(?m)(\|[^\n]*\|)\n\n(?=\|)', r'\1\n', '\n\n'.join(lines))
    temp = REPORT.with_suffix('.tmp'); temp.write_text(rendered + '\n', encoding='utf-8'); os.replace(temp, REPORT)
    atomic(RECORD / 'augmentation_status.json', {k:v for k,v in state.items() if k not in ('pid', 'runs')})


def prepare(legacy, work):
    from preparation.images import HouseImages, preview
    from train_augmented import AugmentedHouseImages
    base = validate(resolve(ROOT / 'configs/selected/training.json'))
    migration(legacy, base, work)
    frame = pd.read_csv(local_path('split_path'))
    train = frame[frame.partition == 'train']
    probe = train.sample(8, random_state=2026)
    val = frame[frame.partition == 'validation']
    if len(val) != 1601:
        val = frame[frame.partition != 'train']
    if len(train) != 6399 or len(val) != 1601: raise ValueError('Unexpected split')
    summary = {}
    for name in CANDIDATES:
        c = validate(resolve(ROOT / f'configs/03_augmentation/{name}.json'))
        folder = work / 'previews' / name; folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / 'events.json', preview(local_path('data_root'), train, c, folder / 'preview.png'))
        forced = deepcopy(c)
        next(iter(forced['augmentation'].values()))['probability'] = 1.0
        preview(local_path('data_root'), train, forced, folder / 'inspection.png', count=4, views=2)
        ds = AugmentedHouseImages(local_path('data_root'), probe, c, training=True)
        reference = HouseImages(local_path('data_root'), probe, c, training=True)
        changed = 0
        for i in range(len(probe)):
            rng_before = torch.random.get_rng_state().clone()
            a = ds[i]; b = ds[i]
            if not torch.equal(a[0], b[0]) or a[1:] != b[1:]: raise ValueError('Nonreproducible augmentation')
            if not torch.equal(a[0], reference[i][0]): raise ValueError('Training semantics changed')
            if not torch.equal(rng_before, torch.random.get_rng_state()): raise ValueError('Global RNG modified')
            if a[0].shape != (3,224,224) or not torch.isfinite(a[0]).all(): raise ValueError('Invalid augmented input')
            ds.epoch = 1
            changed += int(not torch.equal(a[0], ds[i][0])); ds.epoch = 0
        if not changed: raise ValueError('No epoch variation')
        sample = val.sample(8, random_state=2026)
        cached = AugmentedHouseImages(local_path('data_root'), sample, c)
        raw = HouseImages(local_path('data_root'), sample, c)
        for i in range(len(sample)):
            if not torch.equal(cached[i][0], raw[i][0]) or cached[i][3] != '{}': raise ValueError('Validation augmentation leakage')
        summary[name] = {'passed': True, 'train_samples_checked': 8, 'validation_samples_checked': 8, 'changed_across_epochs': changed, 'preview_images': 32, 'views_per_image': 5}
    atomic(work / 'preflight.json', summary)
    atomic(RECORD / f'analyses/{WORK_NAME}/preflight.json', summary)
    return base


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--legacy-root', type=Path, required=True); parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); legacy = args.legacy_root.resolve()
    work = local_path('records_root') / '05_augmentation' / WORK_NAME; work.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        prepare(legacy, work); return
    import msvcrt
    lock = (work / 'queue.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    if (work / 'state.json').exists(): raise RuntimeError('Existing queue state; review before restarting')
    state = {'status':'准备增强实验', 'updated_utc':now(), 'reviews':[], 'decisions':[], 'runs':{}, 'pid':os.getpid()}
    sources = {str(p):digest(p) for p in ROOT.rglob('*.py') if '.venv' not in p.parts and '__pycache__' not in p.parts}
    def save():
        state['updated_utc'] = now(); atomic(work / 'state.json', state); publish(state)
    def execute(config, seed, reference, allowed):
        for filename, expected in sources.items():
            if digest(filename) != expected: raise ValueError(f'Source changed during queue: {filename}')
        candidate = deepcopy(config); candidate['seed'] = seed
        candidate['_parent_source'] = None
        validate(candidate)
        comparison = deepcopy(candidate); comparison['run_kind'] = 'development'
        diff = differences(read_json(reference / 'config.json'), comparison)
        if set(diff) - set(allowed): raise ValueError(f'Pre-run control mismatch: {diff}')
        path = work / f"{candidate['experiment']['id']}_{seed}.json"; write_json(path, candidate)
        parent = local_path('records_root') / '05_augmentation/runs' / candidate['experiment']['id']
        existing = set(parent.glob('*'))
        state['status'] = f"运行{candidate['experiment']['id']}，种子{seed}"; save()
        with (work / f'{path.stem}.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-X', 'utf8', str(ROOT / 'train_augmented.py'), 'train', str(path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
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
        state['reviews'].append(review_augmented(out, reference, allowed))
        state['runs'][path.stem] = str(out); state['status'] = '本组复核完成，准备下一组'; save()
        return out
    try:
        save()
        base = prepare(legacy, work)
        atomic(work / 'registration.json', {'sources':sources, 'candidates':list(CANDIDATES), 'screen_seed':2026, 'max_replications':2, 'selection':'lower mean and at least two paired wins'})
        mapping = read_json(legacy / 'output/experiment_record/01_preprocessing/pipeline_state_v2.json')['runs']
        controls = [legacy / 'output/experiment_record' / mapping[f'preprocess_01_stretch:{s}'] for s in SEEDS]
        screened = []
        for name in CANDIDATES:
            c = validate(resolve(ROOT / f'configs/03_augmentation/{name}.json'))
            allowed = {f'augmentation.{op}' for op in c['augmentation']}
            out = execute(c, 2026, controls[0], allowed)
            mse = read_json(out / 'metrics.json')['mse']
            if mse < read_json(controls[0] / 'metrics.json')['mse']: screened.append((mse, name, c, out, allowed))
            state['decisions'].append({'text':f"{NAMES[name]}种子2026复核完成：{'进入候选排序' if mse < read_json(controls[0] / 'metrics.json')['mse'] else '未达到追加种子条件'}。"}); save()
        finalists = sorted(screened, key=lambda x:x[0])[:2]
        winners = list(PRIOR_WINNERS)
        state['decisions'].append({'text':'追加种子候选：' + ('、'.join(NAMES[x[1]] for x in finalists) or '无') + '。'}); save()
        for _, name, c, out, allowed in finalists:
            runs = [out] + [execute(c, seed, ref, allowed) for seed,ref in zip(SEEDS[1:], controls[1:])]
            decision = decide(runs, controls, NAMES[name]); state['decisions'].append(decision); save()
            if decision['accepted']: winners.append((decision['mean_mse'], name, c))
        selected = deepcopy(min(winners, key=lambda x:x[0])[2] if winners else base)
        selected = {k:v for k,v in selected.items() if not k.startswith('_') and k not in ('run_kind','selection')}
        selected['seed'] = 2026
        selected['selection'] = {'reason':WORK_NAME + ' screening complete', 'seeds':list(SEEDS), 'scope':list(CANDIDATES), 'retained_prior_candidates':[x[1] for x in PRIOR_WINNERS]}
        write_json(ROOT / 'configs/selected/augmentation.json', selected)
        state['decisions'].append({'text':'本轮推荐：' + (NAMES[min(winners,key=lambda x:x[0])[1]] if winners else '保留无增强方案') + '。未执行组合，不推断单项叠加有效。'})
        state['status'] = COMPLETE_STATUS; save()
    except BaseException:
        state['status'] = '增强实验异常停止'; state['error'] = traceback.format_exc(); save(); raise


if __name__ == '__main__':
    main()
