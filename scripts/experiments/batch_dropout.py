"""Controlled batch-size selection followed by dropout selection."""
from continue_training import *
import continue_training as shared

MARKER45 = '\n## 实验四与实验五：批量与随机失活\n'

def publish45(state):
    original = REPORT.read_text(encoding='utf-8').split(MARKER45)[0]
    original = original.replace('# 当前状态：Adam复核与实验三按序执行', '# 当前状态：实验四与实验五按序执行')
    lines = [original, MARKER45, '按用户授权继续实验四，然后实验五。此前阶段已完成；以下为当前队列状态。',
             '实验四实际批量固定4，有效批量16/32/64分别累积4/8/16次，不按批量缩放学习率。不同批量会改变每轮更新次数，单独记录，不宣称计算预算完全相等。',
             '实验五继承选定有效批量，仅比较随机失活0/0.2/0.5，两处全连接层共同改变。',
             '两阶段均复核2026/2027/2028，候选平均误差降低且至少两次同种子改善才保留；多个达标候选选均值最低者，否则保留对照。',
             '固定直接拉伸、AdamW 0.0001、固定学习率、权重衰减0.0001和40轮起/耐心15/最多60轮。已有完全相同配置的三种子对照复用。每组检查配置、划分、初始化、完整样本和误差分组，更新本报告后继续。',
             f"当前状态：{state['status']}；更新时间：{state['updated_utc']}。"]
    lines += [x['text'] for x in state['reviews']]
    lines += [x['text'] for x in state.get('decisions', [])]
    temp = REPORT.with_suffix('.tmp'); temp.write_text('\n\n'.join(lines).rstrip()+'\n',encoding='utf-8'); os.replace(temp,REPORT)
    atomic(RECORD/'batch_dropout_status.json',{k:v for k,v in state.items() if k not in ('runs','pid')})

def configuration(base, stage, value):
    c = deepcopy(base)
    c = {k:v for k,v in c.items() if not k.startswith('_') and k not in ('run_kind','selection')}
    field = 'training.accumulation' if stage == 'batch' else 'model.dropout'
    if stage == 'batch':
        c['training']['accumulation'] = value // 4
        identity = f'tune_04_batch_{value}'
    else:
        c['model']['dropout'] = value
        identity = 'tune_05_dropout_'+str(value).replace('.','p')
    c['experiment'].update(id=identity,name=identity,stage='02_training',changed_fields=[field],enforce_changes=False,
                           comparison='same-seed selected control',question=field)
    c['_parent_source'] = None
    validate(c)
    return c

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--legacy-root',type=Path,required=True)
    legacy=parser.parse_args().legacy_root.resolve()
    work=local_path('records_root')/'04_training/batch_dropout_20261003'
    work.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(work/'queue.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (work/'state.json').exists():raise RuntimeError('Existing state: review before restart')
    state={'status':'核验实验四前置条件','updated_utc':now(),'reviews':[],'decisions':[],'runs':{},'pid':os.getpid()}
    sources={str(p):digest(p) for p in ROOT.rglob('*.py') if '.venv' not in p.parts and '__pycache__' not in p.parts}
    atomic(work/'registration.json',{'utc':now(),'source_hashes':sources,'seeds':list(SEEDS),'batch_candidates':[16,32,64],'dropout_candidates':[0,.2,.5]})
    def save():
        state['updated_utc']=now();atomic(work/'state.json',state);publish45(state)
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
        state['reviews'].append(review(out, reference, allowed))
        state['runs'][path.stem] = str(out); state['status'] = '本组复核完成，准备下一组'; save()
        return out
    try:
        save()
        base=resolve(ROOT/'configs/selected/schedule.json')
        assert base['training']['optimizer']=='adamw' and base['training']['lr']==.0001 and base['training']['schedule']=='constant'
        assert base['training']['batch_size']==4 and base['training']['accumulation']==8 and base['model']['dropout']==.5
        migration(legacy,base,work)
        old=legacy/'output/experiment_record'
        mapping=read_json(old/'01_preprocessing/pipeline_state_v2.json')['runs']
        controls=[old/mapping[f'preprocess_01_stretch:{s}'] for s in SEEDS]
        for stage,values,slot in [('batch',[16,64],'batch'),('dropout',[0,.2],'dropout')]:
            allowed={'training.accumulation'} if stage=='batch' else {'model.dropout'}
            qualifying=[]
            for value in values:
                config=configuration(base,stage,value)
                candidates=[execute(config,seed,reference,allowed) for seed,reference in zip(SEEDS,controls)]
                decision=decide(candidates,controls,f'{stage}={value}')
                for out in candidates:
                    cost=read_json(out/'cost.json')
                    decision.setdefault('costs',[]).append({k:cost[k] for k in ('epochs','optimizer_updates','train_seconds')})
                state['decisions'].append(decision);save()
                if decision['accepted']:qualifying.append((decision['mean_mse'],config,candidates))
            if qualifying:
                _,base,controls=min(qualifying,key=lambda x:x[0])
            selected=deepcopy(base)
            selected={k:v for k,v in selected.items() if not k.startswith('_') and k not in ('run_kind','selection')}
            selected['seed']=2026;selected['experiment']['enforce_changes']=False
            selected['selection']={'reason':f'{slot}: lowest mean among candidates meeting paired-win rule; otherwise retain control','seeds':list(SEEDS)}
            write_json(ROOT/f'configs/selected/{slot}.json',selected)
            actual=selected['training']['batch_size']*selected['training']['accumulation'] if stage=='batch' else selected['model']['dropout']
            state['decisions'].append({'text':f"{'实验四' if stage=='batch' else '实验五'}完成：选定{'有效批量' if stage=='batch' else '随机失活概率'}{actual}。"})
            base=selected;save()
        state['status']='实验四与实验五完成；不自动进入实验六';save()
    except BaseException:
        state['status']='异常停止，需定位后再继续';state['error']=traceback.format_exc();save();raise

if __name__=='__main__':main()
