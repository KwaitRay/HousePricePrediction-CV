"""Controlled convolution-initialization screening, preserving head initialization."""
from continue_training import *
import hashlib
from models.alexnet import AlexNetRegressor, initialize
from common.metrics import transform_target
from preparation.images import HouseImages

PREFLIGHT = {}
MARKER8 = '\n## 十一、实验八：卷积初始化检查\n'


def hashes(model):
    whole=hashlib.sha256();parts={}
    for name,p in model.state_dict().items():
        raw=p.detach().cpu().numpy().tobytes();whole.update(name.encode());whole.update(raw)
        parts[name]=hashlib.sha256(raw).hexdigest()
    return whole.hexdigest(),parts


def preflight(config, controls):
    torch.set_num_threads(config['training']['cpu_threads'])
    frame=pd.read_csv(local_path('split_path'))
    train=frame[frame.partition=='train']
    target_mean=float(np.mean(transform_target(train.price.to_numpy(),config['target'])))
    probe=train.sample(4,random_state=2026)
    ds=HouseImages(local_path('data_root'),probe,config)
    x=torch.stack([ds[i][0] for i in range(len(ds))])
    target=torch.tensor(transform_target(probe.price.to_numpy(),config['target']),dtype=torch.float32)
    result={'passed':True,'seeds':{},'scope':'CPU training-only initial activation and gradient checks; no optimization or validation-based selection'}
    for seed,control in zip(SEEDS,controls):
        item={};snapshots={};expected_changed=None
        for variant in ('default','xavier_convolution'):
            model=AlexNetRegressor(config['model']);initialize(model,seed,target_mean,variant)
            full,parts=hashes(model);snapshots[variant]=parts
            item['default_hash' if variant=='default' else 'xavier_hash']=full
            expected_changed={name+'.weight' for name,m in model.named_modules() if isinstance(m,torch.nn.Conv2d)}
            if seed==2026:
                activations={};hooks=[]
                for name,module in model.named_modules():
                    if isinstance(module,torch.nn.Conv2d):
                        def capture(module,inputs,output,key=name):
                            activations[key]={'mean':float(output.detach().mean()),'std':float(output.detach().std()),'finite':bool(torch.isfinite(output).all())}
                        hooks.append(module.register_forward_hook(capture))
                model.eval();pred=model(x);loss=(pred-target).square().mean();loss.backward()
                norms={name:float(p.grad.norm()) for name,p in model.named_parameters() if name in expected_changed}
                if not torch.isfinite(loss) or not all(np.isfinite(v) and v>0 for v in norms.values()):raise ValueError('Invalid initial gradients')
                if not all(v['finite'] for v in activations.values()):raise ValueError('Invalid activations')
                item[variant+'_diagnostic']={'scaled_loss':float(loss.detach()),'convolution_gradient_norms':norms,'activations':activations}
                for hook in hooks:hook.remove()
            del model
        changed={name for name,value in snapshots['default'].items() if value!=snapshots['xavier_convolution'][name]}
        if changed!=expected_changed:raise ValueError(f'Unexpected initialized layers: {changed}')
        if item['default_hash']!=read_json(control/'model_summary.json')['initialization_sha256']:raise ValueError('Default initialization does not reproduce control')
        item['changed_parameters']=sorted(changed);item['all_other_parameters_identical']=True
        result['seeds'][str(seed)]=item
    return result


def review_initialization(run, control, allowed):
    c, base = read_json(run / 'config.json'), read_json(control / 'config.json')
    diff = differences(base, c)
    if set(diff) - set(allowed): raise ValueError(f'Unexpected configuration changes: {diff}')
    if read_json(run / 'status.json')['status'] != 'completed': raise ValueError('Run did not complete')
    for file in ('data_manifest.json',):
        if read_json(run / file) != read_json(control / file): raise ValueError('Data manifest mismatch')
    init = read_json(run / 'model_summary.json')['initialization_sha256']
    expected = PREFLIGHT['seeds'][str(c['seed'])]
    if init != expected['xavier_hash']: raise ValueError('Candidate initialization differs from registered fingerprint')
    if read_json(control / 'model_summary.json')['initialization_sha256'] != expected['default_hash']: raise ValueError('Control initialization differs from registered fingerprint')
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




def publish8(state):
    original=REPORT.read_text(encoding='utf-8').split(MARKER8)[0].rstrip()
    lines=[original,MARKER8,'按用户要求复用实验七证据后开展初始化检查。仅把五层卷积的Kaiming正态初始化换成Xavier均匀初始化（实现默认增益1）；全连接层、偏置、输出均值及其他训练条件不变。',
           '该比较用于探查初始化敏感性，不把已有种子差异当成初始化不良的证明。先验证各层指纹：卷积权重应不同，其余参数应逐位相同；只在训练侧4张图片上检查初始激活和梯度有限性。',
           '先运行种子2026；其验证误差严格优于同种子对照且完整性检查通过，才追加2027/2028。最终仍要求三种子均值降低且至少两次配对改善；单种子落选不代表跨种子已证伪。本实验只比较一种预登记Xavier变体，不把收益归因于某个单独的分布或方差因素。',
           f"当前状态：{state['status']}；更新：{state['updated_utc']}。"]
    lines.extend(x['text'] for x in state['reviews']);lines.extend(x['text'] for x in state['decisions'])
    temp=REPORT.with_suffix('.tmp');temp.write_text('\n\n'.join(lines).rstrip()+'\n',encoding='utf-8');os.replace(temp,REPORT)
    atomic(RECORD/'initialization_status.json',{k:v for k,v in state.items() if k not in ('runs','pid')})


def main():
    global PREFLIGHT
    parser=argparse.ArgumentParser();parser.add_argument('--legacy-root',type=Path,required=True)
    legacy=parser.parse_args().legacy_root.resolve()
    work=local_path('records_root')/'04_training/initialization_20261004';work.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(work/'queue.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (work/'state.json').exists():raise RuntimeError('Existing initialization state: review before restart')
    state={'status':'正在检查初始化及复用对照','updated_utc':now(),'reviews':[],'decisions':[],'runs':{},'pid':os.getpid()}
    sources={str(p):digest(p) for p in ROOT.rglob('*.py') if '.venv' not in p.parts and '__pycache__' not in p.parts}
    def save():
        state['updated_utc']=now();atomic(work/'state.json',state);publish8(state)
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
        state['reviews'].append(review_initialization(out, reference, allowed))
        state['runs'][path.stem] = str(out); state['status'] = '本组复核完成，准备下一组'; save()
        return out
    try:
        save()
        base=resolve(ROOT/'configs/selected/training.json')
        candidate=validate(resolve(ROOT/'configs/02_training/tune_08_initialization.json'))
        candidate['experiment']['enforce_changes']=False
        migration(legacy,candidate,work)
        mapping=read_json(legacy/'output/experiment_record/01_preprocessing/pipeline_state_v2.json')['runs']
        controls=[legacy/'output/experiment_record'/mapping[f'preprocess_01_stretch:{s}'] for s in SEEDS]
        PREFLIGHT=preflight(candidate,controls)
        atomic(work/'preflight.json',PREFLIGHT);atomic(RECORD/'analyses/initialization/preflight.json',PREFLIGHT)
        atomic(work/'registration.json',{'source_hashes':sources,'candidate':candidate,'protocol':'single-seed screen then at most two additional seeds'})
        runs=[execute(candidate,2026,controls[0],{'training.initialization'})]
        accepted=False
        if read_json(runs[0]/'metrics.json')['mse'] < read_json(controls[0]/'metrics.json')['mse']:
            state['decisions'].append({'text':'种子2026优于对照，追加2027和2028复核。'});save()
            runs.extend(execute(candidate,s,ref,{'training.initialization'}) for s,ref in zip(SEEDS[1:],controls[1:]))
            decision=decide(runs,controls,'卷积初始化选择');state['decisions'].append(decision);accepted=decision['accepted']
        else:
            state['decisions'].append({'text':'种子2026未优于对照，按省算力协议停止追加种子，保留原初始化；结论仅限本次初筛。'})
        selected=deepcopy(candidate if accepted else base)
        selected={k:v for k,v in selected.items() if not k.startswith('_') and k not in ('run_kind','selection')}
        selected['seed']=2026;selected['experiment']['enforce_changes']=False
        selected['selection']={'reason':state['decisions'][-1]['text'],'screen_seed':2026,'replicated':len(runs)==3,'early_stopping_evidence':'reused historical replay and implementation verification'}
        write_json(ROOT/'configs/selected/initialization.json',selected)
        write_json(ROOT/'configs/selected/training.json',selected)
        state['status']='实验八完成；不自动进入增强阶段';save()
    except BaseException:
        state['status']='实验八异常停止，需定位后继续';state['error']=traceback.format_exc();save();raise


if __name__=='__main__':main()
