"""Finish-current-run handoff to bounded single-seed screening."""
from regularization import *
import regularization as original_queue


def shortlist(scores, control_mse, no_penalty_mse):
    eligible = [r for r in scores if np.isfinite(r['mse']) and r['mse'] < min(control_mse,no_penalty_mse)]
    return sorted(eligible,key=lambda r:(r['mse'],r['index']))[:2]


def publish_screen(state):
    original_queue.publish6(state)
    text=REPORT.read_text(encoding='utf-8')
    old='无惩罚、L1系数1e-7/1e-6、L2系数1e-6/1e-5、解耦衰减1e-5/1e-3各跑三个种子；原衰减1e-4的三种子复用，共21次新训练。'
    new=('用户已授权缩减：无惩罚三种子和当前组完整保留；六个正则化候选先各跑种子2026，'
         '仅给优于同种子原最好配置及无惩罚对照、且诊断通过的最低误差前两项补跑2027/2028；'
         '若只有一项达标仅复核一项，无达标则结束。总量由21次降为9–13次，包含已完成结果。'
         '单种子落选仅表示筛选未晋级，不代表已证明其跨种子无效；可能漏掉对种子敏感的候选。')
    text=text.replace(old,new)
    temporary=REPORT.with_suffix('.tmp');temporary.write_text(text,encoding='utf-8');os.replace(temporary,REPORT)


def await_boundary(oldwork, work):
    import msvcrt
    while True:
        if (work/'STOP').exists():raise RuntimeError('Stop requested before handoff')
        old=read_json(oldwork/'state.json')
        error=old.get('error','')
        if error:
            if 'Source changed during queue' not in error or 'regularization.py' not in error:
                raise RuntimeError('Original queue failed unexpectedly: '+error)
            handle=(oldwork/'queue.lock').open('r+b');handle.seek(0)
            try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:
                handle.close();time.sleep(5);continue
            # Owning the original lock prevents any old-controller restart overlap.
            old['handoff_guard_error']=old.pop('error')
            old['status']='按用户授权在完整运行及复核后交接至单种子筛选'
            old['updated_utc']=now();atomic(oldwork/'state.json',old)
            return old,handle
        time.sleep(15)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--legacy-root',type=Path,required=True)
    legacy=parser.parse_args().legacy_root.resolve()
    oldwork=local_path('records_root')/'04_training/regularization_20261004'
    work=local_path('records_root')/'04_training/regularization_screen_20261004'
    work.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(work/'queue.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (work/'state.json').exists():raise RuntimeError('Existing screening state: review before restarting')
    state={'status':'等待当前组完成及复核后交接','updated_utc':now(),'reviews':[],'decisions':[],'runs':{},'pid':os.getpid()}
    atomic(work/'state.json',state)
    old,old_lock=await_boundary(oldwork,work)
    state.update(reviews=old['reviews'],decisions=old['decisions'],runs=old['runs'])
    state['decisions'].append({'text':'已按用户授权由全候选三种子改为先筛选后复核；当前运行完整结束，未截断训练。'})
    sources={str(p):digest(p) for p in ROOT.rglob('*.py') if '.venv' not in p.parts and '__pycache__' not in p.parts}
    configs=[resolve(ROOT/'configs/02_training'/name) for name in FILES]
    for c,label in zip(configs,LABELS):
        validate(c);c['experiment']['name']=label;c['experiment']['enforce_changes']=False
    atomic(work/'registration.json',{'utc':now(),'source_hashes':sources,'screen_seed':2026,'max_finalists':2,'new_run_budget_total':[9,13],'configs':configs})
    def save():
        state['updated_utc']=now();atomic(work/'state.json',state);publish_screen(state)
    def execute(config, seed, reference, allowed):
        for filename, expected in sources.items():
            if digest(filename) != expected: raise ValueError(f'Source changed during queue: {filename}')
        key = f"{config['experiment']['id']}_{seed}"
        if key in state['runs']:
            out = Path(state['runs'][key])
            if read_json(out/'status.json')['status'] != 'completed': raise ValueError('Cannot reuse unfinished run')
            return out
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
            child = subprocess.Popen([sys.executable, '-X', 'utf8', str(ROOT / 'train_regularized.py'), 'train', str(path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
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
        state['reviews'].append(review_regularization(out, reference, allowed))
        state['runs'][path.stem] = str(out); state['status'] = '本组复核完成，准备下一组'; save()
        return out
    try:
        state['status']='交接完成，核验筛选协议';save()
        base=resolve(ROOT/'configs/selected/dropout.json')
        migration(legacy,base,work)
        mapping=read_json(legacy/'output/experiment_record/01_preprocessing/pipeline_state_v2.json')['runs']
        controls=[legacy/'output/experiment_record'/mapping[f'preprocess_01_stretch:{s}'] for s in SEEDS]
        none=[Path(state['runs'][f"{configs[0]['experiment']['id']}_{s}"]) for s in SEEDS]
        scores=[];screen_runs={}
        for index,c in enumerate(configs[1:],1):
            run=execute(c,2026,none[0],ALLOWED)
            # Reused current run must have its successful post-run review recorded.
            check=read_json(run/'review.json')
            if check['seed']!=2026:raise ValueError('Invalid screening seed')
            mse=read_json(run/'metrics.json')['mse']
            scores.append({'index':index,'mse':mse});screen_runs[index]=run
        finalists=shortlist(scores,read_json(controls[0]/'metrics.json')['mse'],read_json(none[0]/'metrics.json')['mse'])
        state['screening_scores']=scores;state['finalists']=[r['index'] for r in finalists]
        names='、'.join(LABELS[x['index']] for x in finalists) or '无'
        state['decisions'].append({'text':f'六项单种子初筛结束，晋级候选：{names}。只对晋级候选追加两个种子。'});save()
        qualifying=[]
        decision=decide(none,controls,'无惩罚相对原最好配置')
        if decision['accepted']:qualifying.append((decision['mean_mse'],configs[0]))
        for row in finalists:
            index=row['index'];c=configs[index]
            runs=[screen_runs[index]]+[execute(c,seed,ref,ALLOWED) for seed,ref in zip(SEEDS[1:],none[1:])]
            state['decisions'].append(decide(runs,none,LABELS[index]+'相对无惩罚'))
            decision=decide(runs,controls,LABELS[index]+'相对原最好配置');state['decisions'].append(decision)
            if decision['accepted']:qualifying.append((decision['mean_mse'],c))
            save()
        selected=deepcopy(min(qualifying,key=lambda x:x[0])[1] if qualifying else base)
        selected={k:v for k,v in selected.items() if not k.startswith('_') and k not in ('run_kind','selection')}
        selected['seed']=2026;selected['experiment']['enforce_changes']=False;cfg=selected['training']
        conclusion=f"实验六筛选与必要复核结束：选定惩罚{cfg['penalty']}、系数{cfg['penalty_coefficient']}、解耦衰减{cfg['weight_decay']}。单种子未晋级者未进行三种子确认。"
        selected['selection']={'reason':conclusion,'seeds':list(SEEDS),'protocol':'seed2026 screening; at most 2 finalists; lower mean and >=2 paired wins to replace incumbent'}
        write_json(ROOT/'configs/selected/regularization.json',selected)
        state['decisions'].append({'text':conclusion});state['status']='实验六完成；不自动进入实验七';save()
    except BaseException:
        state['status']='筛选队列异常停止，需定位后继续';state['error']=traceback.format_exc();save();raise


if __name__=='__main__':main()
