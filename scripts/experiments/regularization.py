from continue_training import *
import continue_training as shared

MARKER6 = '\n## 九、实验六：权重正则化（本轮执行记录）\n'
ALLOWED = {'training.weight_decay', 'training.penalty', 'training.penalty_coefficient'}
FILES = ['tune_06_no_penalty.json', 'tune_06_l1_1e-07.json', 'tune_06_l1_1e-06.json',
         'tune_06_l2_1e-06.json', 'tune_06_l2_1e-05.json',
         'tune_06_decay_1e-05.json', 'tune_06_decay_0.001.json']
LABELS = ['无惩罚', 'L1，系数1e-7', 'L1，系数1e-6', 'L2，系数1e-6', 'L2，系数1e-5',
          '解耦权重衰减1e-5', '解耦权重衰减1e-3']


def publish6(state):
    original = REPORT.read_text(encoding='utf-8').split(MARKER6)[0]
    original = original.replace('实验一至五的本轮研究已结束，实验六尚未启动，无自动续跑任务。',
                                '实验一至五的本轮研究已结束；实验六已获用户授权，最新状态见第九节。')
    original = original.replace('实验六尚未获得本轮启动指令，因此本次只整理，不运行。',
                                '实验六随后获得用户授权，执行登记见第九节。')
    lines = [original.rstrip(), MARKER6,
             '继承直接拉伸、AdamW 0.0001固定学习率、有效批量32、随机失活0.5和既有停止协议。先建立无惩罚对照，再分别检验L1、L2与解耦权重衰减；不混用多个机制。',
             '无惩罚、L1系数1e-7/1e-6、L2系数1e-6/1e-5、解耦衰减1e-5/1e-3各跑三个种子；原衰减1e-4的三种子复用，共21次新训练。L1按权重绝对值求和，L2按权重平方和的一半计算；仅约束卷积和线性权重，不含偏置与归一化参数。',
             '机制比较以无惩罚为共同对照；最终替换还必须相对原有最好配置，三种子均值降低且至少两次同种子获胜。达标者取均值最低，否则保留原有1e-4衰减。关闭惩罚也可作为最终候选。',
             '每组结束核对完整样本、划分、初始化和配置差异，再查看价格分组、完整训练验证差距、数据损失、惩罚值及权重范数，写入本文后继续。图表中的损失监控合计是样本平均数据损失与更新平均惩罚之和，不是原价验证误差；不用于选模型。',
             f"当前状态：{state['status']}；更新：{state['updated_utc']}。"]
    lines.extend(x['text'] for x in state['reviews'])
    lines.extend(x['text'] for x in state['decisions'])
    temporary = REPORT.with_suffix('.tmp')
    temporary.write_text('\n\n'.join(lines).rstrip()+'\n', encoding='utf-8')
    os.replace(temporary, REPORT)
    atomic(RECORD / 'regularization_status.json', {k:v for k,v in state.items() if k not in ('runs','pid')})


def review_regularization(run, reference, allowed):
    result = shared.review(run, reference, allowed)
    c = read_json(run/'config.json'); cfg = c['training']
    positive = [x for x in result['price_groups'] if x['contribution'] > 0]
    negative = [x for x in result['price_groups'] if x['contribution'] < 0]
    bad = max(positive, key=lambda x:x['contribution']) if positive else None
    good = min(negative, key=lambda x:x['contribution']) if negative else None
    result['text'] = (f"{c['experiment']['name']}，种子{c['seed']}：验证均方误差{result['mse']:.2f}，"
                      f"相对本组对照{result['delta_mse']:+.2f}；最佳模型训练验证差距{result['train_validation_gap']:.2f}。")
    for label, item in [('主要退步来源',bad),('主要改善来源',good)]:
        if item: result['text'] += f"{label}为{item['group']}，对总变化贡献{item['contribution']:+.2f}。"
    h = pd.read_csv(run/'history.csv')
    if not np.isfinite(h[['penalty','weight_l1','weight_l2','total_loss_monitor']].to_numpy()).all():
        raise ValueError('Nonfinite regularization diagnostics')
    if cfg['penalty']=='none' and not h.penalty.eq(0).all(): raise ValueError('Unexpected explicit penalty')
    result['last_weight_l1'] = float(h.iloc[-1].weight_l1)
    result['last_weight_l2'] = float(h.iloc[-1].weight_l2)
    result['last_penalty'] = float(h.iloc[-1].penalty)
    result['text'] += f"最后一轮平均显式惩罚{result['last_penalty']:.6g}，权重L2范数{result['last_weight_l2']:.3f}。误差分解用于定位，不单独证明机制；完成预登记复核后选择。"
    atomic(run/'review.json',result);export(run)
    return result


def plot(run):
    shared.plot(run)
    if not (run/'history.csv').exists(): return
    h = pd.read_csv(run/'history.csv')
    if 'weight_l1' not in h: return
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for field,label in [('data_loss','数据损失'),('penalty','平均显式惩罚'),('total_loss_monitor','监控合计')]:
        axes[0].plot(h.epoch,h[field],label=label)
    axes[0].legend();axes[0].set_title('目标缩放后的训练诊断')
    for field,ax,title in [('weight_l1',axes[1],'权重绝对值之和'),('weight_l2',axes[2],'权重L2范数')]:
        ax.plot(h.epoch,h[field]);ax.set_title(title)
    for ax in axes:ax.set_xlabel('轮次')
    fig.tight_layout();fig.savefig(run/'regularization_curves.png',dpi=130);plt.close(fig)


def export(run):
    shared.export(run)
    dest = RECORD/'runs'/run.parent.name/run.name
    for name in ('regularization_curves.png','regularization_runtime.json'):
        if (run/name).exists():shutil.copy2(run/name,dest/name)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--legacy-root',type=Path,required=True)
    legacy=parser.parse_args().legacy_root.resolve()
    work=local_path('records_root')/'04_training/regularization_20261004'
    work.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(work/'queue.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (work/'state.json').exists():raise RuntimeError('Existing state: inspect before restarting')
    state={'status':'正在核验实验六前置条件','updated_utc':now(),'reviews':[],'decisions':[],'runs':{},'pid':os.getpid()}
    sources={str(p):digest(p) for p in ROOT.rglob('*.py') if '.venv' not in p.parts and '__pycache__' not in p.parts}
    configs=[resolve(ROOT/'configs/02_training'/name) for name in FILES]
    for c,label in zip(configs,LABELS):
        validate(c);c['experiment']['name']=label;c['experiment']['enforce_changes']=False
    atomic(work/'registration.json',{'utc':now(),'source_hashes':sources,'seeds':list(SEEDS),'configs':configs})
    def save():
        state['updated_utc']=now();atomic(work/'state.json',state);publish6(state)
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
        save()
        base=resolve(ROOT/'configs/selected/dropout.json')
        assert base['training']['optimizer']=='adamw' and base['training']['lr']==.0001
        assert base['training']['schedule']=='constant' and base['model']['dropout']==.5
        assert base['training']['batch_size']==4 and base['training']['accumulation']==8
        migration(legacy,base,work)
        mapping=read_json(legacy/'output/experiment_record/01_preprocessing/pipeline_state_v2.json')['runs']
        controls=[legacy/'output/experiment_record'/mapping[f'preprocess_01_stretch:{s}'] for s in SEEDS]
        all_runs=[];qualifying=[]
        for index,c in enumerate(configs):
            references=controls if index==0 else all_runs[0]
            runs=[execute(c,seed,ref,ALLOWED) for seed,ref in zip(SEEDS,references)]
            all_runs.append(runs)
            if index:
                state['decisions'].append(decide(runs,all_runs[0],LABELS[index]+'相对无惩罚'))
            else:
                state['decisions'].append(decide(controls,runs,'原衰减1e-4相对无惩罚'))
            decision=decide(runs,controls,LABELS[index]+'相对原最好配置')
            state['decisions'].append(decision)
            if decision['accepted']:qualifying.append((decision['mean_mse'],c))
            save()
        selected=deepcopy(min(qualifying,key=lambda x:x[0])[1] if qualifying else base)
        selected={k:v for k,v in selected.items() if not k.startswith('_') and k not in ('run_kind','selection')}
        selected['seed']=2026;selected['experiment']['enforce_changes']=False
        cfg=selected['training']
        text=f"实验六结束：保留显式惩罚{cfg['penalty']}、系数{cfg['penalty_coefficient']}、解耦衰减{cfg['weight_decay']}；未同时启用两种机制。"
        selected['selection']={'reason':text,'seeds':list(SEEDS),'rule':'Lower mean and at least 2 paired wins against previous selected configuration'}
        write_json(ROOT/'configs/selected/regularization.json',selected)
        state['decisions'].append({'text':text});state['status']='实验六完成；不自动进入实验七';save()
    except BaseException:
        state['status']='异常停止，需定位后继续';state['error']=traceback.format_exc();save();raise


if __name__=='__main__':main()


# Authorized handoff: after active run, controller digest guard yields to regularization_screen.py.
