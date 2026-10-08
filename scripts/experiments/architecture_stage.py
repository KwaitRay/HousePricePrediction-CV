"""Independent AlexNet structural ablations with paired shared initialization."""
from continue_training import *
import continue_training as shared
from initialization import hashes
from models.alexnet import AlexNetRegressor, initialize
from common.metrics import transform_target
from preparation.images import HouseImages

RECORD = REPOSITORY / 'output/experiment_record/06_architecture'
REPORT = RECORD / '实验报告.md'
shared.RECORD = RECORD
CANDIDATES = ('structure_01_small_kernels','structure_02_inception','structure_03_residual')
FLAGS = dict(zip(CANDIDATES, ('small_kernels','inception','residual')))
NAMES = dict(zip(CANDIDATES, ('实验一：小卷积核堆叠','实验二：多尺度卷积','实验三：残差连接')))
WORK_NAME = 'architecture_20261005'
PRIOR_WINNERS = []
COMPLETE_STATUS = '三项结构初筛及必要种子复核完成；本阶段结束'
PREFLIGHT = {}


def control_runs():
    mapping = read_json(local_path('records_root') / '05_augmentation/geometry_20261004/state.json')['runs']
    return [Path(mapping[f'augment_01_flip_{s}']) for s in SEEDS]


def prepare(legacy, work):
    global PREFLIGHT
    base = validate(resolve(ROOT / 'configs/selected/augmentation.json'))
    plain = deepcopy(base); plain['augmentation'] = {}
    migration(legacy, plain, work)
    controls = control_runs()
    frame = pd.read_csv(local_path('split_path'))
    train = frame[frame.partition == 'train']
    mean = float(np.mean(transform_target(train.price.to_numpy(), base['target'])))
    probe = train.sample(4,random_state=2026)
    ds = HouseImages(local_path('data_root'), probe, plain)
    x = torch.stack([ds[i][0] for i in range(4)])
    target = torch.tensor(transform_target(probe.price.to_numpy(), base['target']),dtype=torch.float32)
    result = {'passed': True, 'seeds': {}, 'scope': 'Shared-layer bitwise initialization; CPU forward/backward on four training images; unchanged cached evaluation inputs'}
    configs = {name:validate(resolve(ROOT / f'configs/04_architecture/{name}.json')) for name in CANDIDATES}
    for seed, control in zip(SEEDS,controls):
        if read_json(control / 'status.json')['status'] != 'completed': raise ValueError('Incomplete historical control')
        cbase = deepcopy(base); cbase['seed'] = seed; cbase['run_kind'] = 'development'
        if differences(cbase,read_json(control / 'config.json')): raise ValueError('Selected configuration differs from control')
        baseline = AlexNetRegressor(base['model']); initialize(baseline,seed,mean,base['training']['initialization'])
        whole, baseline_parts = hashes(baseline)
        if whole != read_json(control / 'model_summary.json')['initialization_sha256']: raise ValueError('Control hash mismatch')
        item = {'baseline':{'initialization_sha256':whole,'parameters':sum(p.numel() for p in baseline.parameters())}}
        baseline.eval()
        with torch.no_grad(): baseline_features = baseline.features(x)
        del baseline
        for name,c in configs.items():
            if set(differences(base,c)) != {f'model.{FLAGS[name]}'}: raise ValueError('Confounded architecture config')
            model = AlexNetRegressor(c['model']); initialize(model,seed,mean,c['training']['initialization'])
            full, parts = hashes(model)
            prefix = {'small_kernels':'conv2.', 'inception':'conv3.', 'residual':'shortcut.'}[FLAGS[name]]
            untouched = {k:v for k,v in parts.items() if not k.startswith(prefix)}
            expected = {k:v for k,v in baseline_parts.items() if not k.startswith(prefix)}
            if untouched != expected: raise ValueError('Shared layer initialization changed')
            entry = {'initialization_sha256':full,'parameters':sum(p.numel() for p in model.parameters()),'shared_parameters_identical':True,'changed_region':prefix}
            if seed == 2026:
                model.eval(); shapes = {}; hooks = []
                for key in ('conv1','conv2','conv3','conv4','conv5','spatial_pool'):
                    def capture(module,inputs,output,key=key):
                        shapes[key] = list(output.shape)
                        if not torch.isfinite(output).all(): raise ValueError('Nonfinite activation')
                    hooks.append(getattr(model,key).register_forward_hook(capture))
                pred = model(x); loss = (pred-target).square().mean(); loss.backward()
                if pred.shape != (4,) or not torch.isfinite(loss): raise ValueError('Invalid forward output')
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError('Invalid gradients')
                gradients = {n:float(p.grad.norm()) for n,p in model.named_parameters() if n.startswith(prefix)}
                if not any(v > 0 for v in gradients.values()): raise ValueError('Changed region receives no gradient')
                if shapes['spatial_pool'] != [4,256,6,6]: raise ValueError('Unexpected head input')
                entry['diagnostic'] = {'shapes':shapes,'scaled_loss':float(loss.detach()),'changed_region_gradient_norms':gradients}
                for hook in hooks: hook.remove()
                if FLAGS[name] == 'residual':
                    with torch.no_grad():
                        model.shortcut.weight.zero_(); model.shortcut.bias.zero_()
                        if not torch.equal(model.features(x),baseline_features): raise ValueError('Zero residual branch does not recover baseline')
                    entry['zero_shortcut_recovers_baseline'] = True
            item[name] = entry
            del model
        result['seeds'][str(seed)] = item
    PREFLIGHT = result
    atomic(work / 'preflight.json', result)
    atomic(RECORD / f'analyses/{WORK_NAME}/preflight.json',result)
    return base


def publish(state):
    lines = ['# 结构改进阶段实验报告',
        f"当前状态：{state['status']}。更新时间（UTC）：{state['updated_utc']}。",
        '## 一、研究依据与共同对照',
        '现有模型有预测向均值收缩、低价高估与高价低估，以及训练误差下降而验证误差反弹的问题。本阶段检验局部特征结构是否改善泛化，不预设更复杂结构更好。三项均从零训练，分别对比相同的水平翻转AlexNet；不累积上一项改动，也不是直接替换为完整VGG、GoogLeNet或ResNet。',
        '复用增强阶段已完成且配置、划分、初始化核验一致的三种子对照：2026为126601.97、2027为125020.35、2028为128807.71；平均验证均方误差126810.01。复用避免重复训练，耗时只作历史参照。',
        '## 二、冻结条件与实验顺序',
        '固定6399训练、1601验证，划分不随种子变化；外部测试集不参与选择。直接拉伸224×224，归一化均值和标准差均为0.5；训练原图50%水平翻转后拉伸，评估无增强并复用已核验缓存。AdamW固定学习率0.0001、衰减0.0001、有效批量32（4×8）、随机失活0.5、原初始化规则。至少40轮、耐心15轮、相对改善阈值0.1%、最多60轮；保存严格最低验证均方误差检查点。',
        '| 顺序 | 唯一结构改动 | 依据与局限 |',
        '| --- | --- | --- |',
        '| 一：小卷积核堆叠 | 第二层5×5、64→192改为两层3×3、64→96→192，中间加ReLU | 相同理论感受野下增加非线性并减少此处参数；首层11×11保持不变 |',
        '| 二：多尺度卷积 | 第三层改为1×1、降维后3×3、降维后5×5、池化后1×1四分支，输出96+128+96+64=384通道 | 检验不同尺度信息；分支、参数量和非线性共同改变，不能归因于单一尺度 |',
        '| 三：残差连接 | 第四、五层两层卷积外增加384→256的1×1投影捷径，相加后ReLU | 检验保留输入和梯度传播；通道不同需投影，参数和计算量增加，不是恒等捷径 |',
        '先按一、二、三顺序各跑2026。仅从严格优于同种子对照者中选误差最低的最多两项追加2027、2028，总计3–7次新训练。最终采用需三种子均值更低且至少两次配对改善；多项达标取均值最低者，否则保留AlexNet。单种子落选不等于跨种子证实无效。不同结构沿用同一训练配方，结论限于该配方，不等于各架构独立调优后的性能上限。',
        '## 三、实施核验与成本',
        '训练前核验原训练循环、固定划分、全部源图及缓存指纹；逐种子重建对照初始权重，并要求结构变更区域之外的全部参数逐位一致。四张训练图检查前向、梯度和回归头输入尺寸；残差投影置零必须恢复对照特征。这里四张仅用于预检，正式训练每轮使用全部6399张。完整模型指纹随结构变化，不能误要求全模型指纹相同。参数量及每组实际训练成本单独记录；固定轮数不表示相同计算量。',
    ]
    if PREFLIGHT:
        lines += ['| 模型 | 参数量 | 相对AlexNet |','| --- | ---: | ---: |']
        s = PREFLIGHT['seeds']['2026']; base_n = s['baseline']['parameters']
        for name in ('baseline', *CANDIDATES):
            n = s[name]['parameters']; lines.append(f"| {NAMES.get(name,'AlexNet')} | {n:,} | {n-base_n:+,} |")
    lines += ['## 四、结果与逐组误差定位','| 实验 | 种子 | 验证均方误差 | 相对同种子AlexNet | 最佳轮／总轮 |','| --- | ---: | ---: | ---: | ---: |']
    for r in state['reviews']:
        lines.append(f"| {NAMES[r['experiment']]} | {r['seed']} | {r['mse']:.2f} | {r['delta_mse']:+.2f} | {r['best_epoch']}/{r['epochs']} |")
    if not state['reviews']: lines.append('尚无完成的训练结果；当前不能判断结构是否有效。')
    for r in state['reviews']:
        lines += [f"**{NAMES[r['experiment']]}，种子{r['seed']}**：{r['analysis']} 最佳模型训练误差{r['train_mse']:.2f}，验证与训练差距{r['train_validation_gap']:.2f}，预测标准差{r['prediction_std']:.2f}（真实价格{r['target_std']:.2f}）。",
                  f"训练及逐轮评估耗时{r['cost']['train_seconds']/60:.1f}分钟；其中训练{r['cost']['training_seconds']/60:.1f}分钟、验证{r['cost']['validation_seconds']/60:.1f}分钟。峰值张量显存{r['cost']['peak_cuda_bytes']/1024**3:.2f}GiB，参数量{r['parameters']:,}。",
                  '| 价位 | 样本数 | 平均有符号误差 | 相对对照均方误差变化 | 对总体变化贡献 |','| --- | ---: | ---: | ---: | ---: |']
        for g in r['price_groups']: lines.append(f"| {g['group']} | {g['n']} | {g['mean_signed_error']:+.2f} | {g['delta_mse']:+.2f} | {g['contribution']:+.2f} |")
        lines.append('建议：' + ('进入复核候选排序；先确认种子稳定性，不立即替换对照。' if r['delta_mse']<0 else '本轮不采用；结合价位贡献与训练验证差距定位退步。暂不改动学习率或增强，以免引入混杂因素。'))
    lines += ['## 五、选择与后续',*[d['text'] for d in state['decisions']],
        '每组结束先核验样本覆盖、配置、指纹和预测，分解四个价位的误差贡献，再启动下一组。指标变化能定位来源，不能仅凭相关性断言视觉机制。训练损失、训练探针和验证误差、学习率曲线实时更新在本阶段runs目录；最佳模型全量训练误差用于最终归因。状态异常则停止。全部预登记实验完成后停止，不自动进入编码器或结构组合。',
        '反复使用同一验证集存在选择偏差；本阶段不是独立测试成绩。视觉类型稀缺结论仍来自400张训练侧分层样本，本轮不更改采样、损失或数据划分。']
    if state.get('error'): lines.append('执行异常已记录；当前队列已停止，需定位后继续。')
    import re
    rendered = re.sub(r'(?m)(\|[^\n]*\|)\n\n(?=\|)',r'\1\n','\n\n'.join(lines))
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    temp = REPORT.with_suffix('.tmp'); temp.write_text(rendered+'\n',encoding='utf-8'); os.replace(temp,REPORT)
    atomic(RECORD / 'architecture_status.json',{k:v for k,v in state.items() if k not in ('pid','runs')})


def review_architecture_base(run, control, allowed):
    c, base = read_json(run / 'config.json'), read_json(control / 'config.json')
    diff = differences(base, c)
    if set(diff) - set(allowed): raise ValueError(f'Unexpected configuration changes: {diff}')
    if read_json(run / 'status.json')['status'] != 'completed': raise ValueError('Run did not complete')
    for file in ('data_manifest.json',):
        if read_json(run / file) != read_json(control / file): raise ValueError('Data manifest mismatch')
    init = read_json(run / 'model_summary.json')['initialization_sha256']
    expected = PREFLIGHT['seeds'][str(c['seed'])]
    if init != expected[c['experiment']['id']]['initialization_sha256']: raise ValueError('Candidate initialization mismatch')
    if read_json(control / 'model_summary.json')['initialization_sha256'] != expected['baseline']['initialization_sha256']: raise ValueError('Control initialization mismatch')
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


def review_architecture(run, control, allowed):
    result = review_architecture_base(run, control, allowed)
    config = read_json(run / 'config.json')
    events = read_json(run / 'augmentation_history.json')
    control_events = read_json(control / 'augmentation_history.json')
    overlap = min(len(events), len(control_events))
    if events[:overlap] != control_events[:overlap]:
        raise ValueError('Paired augmentation counters differ from baseline')
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
    result['parameters'] = read_json(run / 'model_summary.json')['parameters']
    result['cost'] = read_json(run / 'cost.json')
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


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--legacy-root', type=Path, required=True); parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); legacy = args.legacy_root.resolve()
    work = local_path('records_root') / '06_architecture' / WORK_NAME; work.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        prepare(legacy, work); return
    import msvcrt
    lock = (work / 'queue.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    if (work / 'state.json').exists(): raise RuntimeError('Existing queue state; review before restarting')
    state = {'status':'准备结构改进实验', 'updated_utc':now(), 'reviews':[], 'decisions':[], 'runs':{}, 'pid':os.getpid()}
    sources = {str(p):digest(p) for p in ROOT.rglob('*.py') if '.venv' not in p.parts and '__pycache__' not in p.parts}
    for p in [ROOT / 'configs/selected/augmentation.json', *(ROOT / f'configs/04_architecture/{name}.json' for name in CANDIDATES)]:
        sources[str(p)] = digest(p)
    for control in control_runs():
        for name in ('config.json','model_summary.json','data_manifest.json','metrics.json','predictions.csv','augmentation_history.json','status.json'):
            sources[str(control / name)] = digest(control / name)
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
        parent = local_path('records_root') / '06_architecture/runs' / candidate['experiment']['id']
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
        state['reviews'].append(review_architecture(out, reference, allowed))
        state['runs'][path.stem] = str(out); state['status'] = '本组复核完成，准备下一组'; save()
        return out
    try:
        save()
        base = prepare(legacy, work)
        atomic(work / 'registration.json', {'sources':sources, 'candidates':list(CANDIDATES), 'screen_seed':2026, 'max_replications':2, 'selection':'lower mean and at least two paired wins'})
        controls = control_runs()
        screened = []
        for name in CANDIDATES:
            c = validate(resolve(ROOT / f'configs/04_architecture/{name}.json'))
            allowed = {f'model.{FLAGS[name]}'}
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
        write_json(ROOT / 'configs/selected/architecture.json', selected)
        state['decisions'].append({'text':'本轮推荐：' + (NAMES[min(winners,key=lambda x:x[0])[1]] if winners else '保留水平翻转AlexNet') + '。未执行组合，不推断单项叠加有效。'})
        state['status'] = COMPLETE_STATUS; save()
    except BaseException:
        state['status'] = '结构实验异常停止'; state['error'] = traceback.format_exc(); save(); raise


if __name__ == '__main__':
    main()
