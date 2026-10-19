"""Independent blur/noise/erasing screening against the frozen unaugmented model."""
import augmentation_stage as stage
from continue_training import *

RECORD = stage.RECORD
REPORT = stage.REPORT
WORK_NAME = 'robustness_20261004'
CANDIDATES = ('augment_08_blur', 'augment_09_noise', 'augment_10_erase')
NEW_NAMES = dict(zip(CANDIDATES, ('实验八：高斯模糊', '实验九：高斯噪声', '实验十：局部擦除')))
NAMES = {**stage.NAMES, **NEW_NAMES}
ORIGINAL_PREPARE = stage.prepare


def prepare(legacy, work):
    base = ORIGINAL_PREPARE(legacy, work)
    from preparation.images import HouseImages
    frame = pd.read_csv(local_path('split_path'))
    sample = frame[frame.partition == 'train'].sample(32, random_state=2026)
    plain = HouseImages(local_path('data_root'), sample, base)
    effects = {}
    for name in CANDIDATES:
        c = resolve(ROOT / f'configs/03_augmentation/{name}.json')
        next(iter(c['augmentation'].values()))['probability'] = 1.0
        transformed = HouseImages(local_path('data_root'), sample, c, training=True)
        deltas = []; skipped = 0; minimum = 1.0; maximum = 0.0
        for i in range(len(sample)):
            a, _ = plain.view(i); b, events = transformed.view(i)
            minimum = min(minimum, float(b.min())); maximum = max(maximum, float(b.max()))
            # Float32 convolution/resizing can overshoot by a few ulps; do not
            # clamp or otherwise change the registered training tensors.
            if not torch.isfinite(b).all() or minimum < -1e-6 or maximum > 1 + 1e-6:
                raise ValueError('Invalid augmented pixel range')
            deltas.append(float((a-b).abs().mean()))
            skipped += sum(int(e['skipped']) for e in events.values())
        effects[name] = {'mean_absolute_pixel_change_after_resize':float(np.mean(deltas)),
                         'minimum_sample_change':min(deltas), 'maximum_sample_change':max(deltas),
                         'samples':len(sample), 'forced_probability_for_diagnostic_only':1.0, 'skipped':skipped,
                         'observed_minimum':minimum, 'observed_maximum':maximum, 'range_check_tolerance':1e-6}
    atomic(work/'input_effects.json',effects)
    atomic(RECORD/f'analyses/{WORK_NAME}/input_effects.json',effects)
    return base


def historical():
    value = read_json(local_path('records_root') / '05_augmentation/geometry_20261004/state.json')
    if len(value['reviews']) != 6 or '队列结束' not in value['status']:
        raise ValueError('Previous geometry batch is not complete')
    return value


def publish(state):
    old = historical()
    lines = ['# 数据增强阶段实验报告',
             f"当前进度：{state['status']}。更新时间（UTC）：{state['updated_utc']}。",
             '## 一、结论与阅读说明',
             '**已完成的几何类实验推荐水平翻转，概率50%。** 三种子平均验证均方误差126810.01，相比无增强127774.42改善0.75%，三次配对中两次改善。第三个种子退步，不能解释为稳定或普遍有效。旋转、平移、裁剪均在单种子初筛未通过，不追加训练。',
             '本次继续单独研究高斯模糊、高斯噪声与局部擦除，尚未完成前不改变已选翻转方案。每项仍与无增强对照比较，不叠加翻转；这样才能定位单项贡献。组合需要另行登记。本文是增强阶段唯一阅读报告，逐次曲线和机器记录在runs及analyses中。',
             '## 二、固定条件与判定规则',
             '冻结训练6399张、验证1601张及分组索引，种子2026/2027/2028不改变划分。AlexNet从零训练；直接拉伸224×224、固定归一化、默认Kaiming初始化、AdamW固定学习率0.0001、有效批量32（4×8）、随机失活0.5、衰减0.0001、无显式惩罚。最少40轮、耐心15、有效改善阈值0.1%、最多60轮；选择严格最低验证误差的检查点。测试集不参与选择。',
             '仅训练图增强，顺序为原图→增强→拉伸→归一化；验证、固定训练探针和最佳模型的完整训练集评价均关闭增强，使用核验后的确定性缓存。不会固化一次随机增强并逐轮重复。',
             '新增三项依次用2026初筛，完整性通过且严格优于同种子无增强对照的候选中，最多取误差最低两项追加2027/2028；预计3–7组。采用仍需三种子均值下降且至少两次配对改善。最终在符合条件的新候选与已验证的翻转中按均值推荐一个单项；不会因本轮新候选失败而丢弃翻转。候选间的小差异不代表统计显著，反复使用同一验证集存在选择偏差。',
             '## 三、已完成的翻转与几何实验',
             '| 实验 | 参数 | 种子2026验证误差 | 相对无增强 | 处理 |',
             '| --- | --- | ---: | ---: | --- |',
             '| 无增强 | 固定对照 | 128268.67 | — | 三种子均值127774.42 |',
             '| 一：水平翻转 | 概率50% | 126601.97 | −1666.70 | 三种子复核后采用 |',
             '| 五：小角度旋转 | 概率30%，±3° | 129526.18 | +1257.51 | 初筛未通过 |',
             '| 六：小幅平移 | 概率30%，横纵各±3% | 129162.73 | +894.06 | 初筛未通过 |',
             '| 七：轻度裁剪 | 概率30%，保留90%–100%面积 | 130222.98 | +1954.31 | 初筛未通过 |',
             '### 水平翻转的复核与归因',
             '| 种子 | 翻转验证误差 | 同种子对照 | 差值 | 最佳轮／总轮 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for r in old['reviews']:
        if r['experiment'] == 'augment_01_flip':
            lines.append(f"| {r['seed']} | {r['mse']:.2f} | {r['control_mse']:.2f} | {r['delta_mse']:+.2f} | {r['best_epoch']}/{r['epochs']} |")
    lines += ['2026的高价和尾部组改善抵消了低价组退步；2027尾部与高价组贡献分别改善6129.88和4547.77，超过低价与中价组退步；2028则主要因中价组退步1214.04而整体变差。收益来自价格段间的取舍，并非所有房价区间都改善。水平翻转可能降低左右构图依赖，但这里只测量了误差变化，未直接证明特征机制。',
              '### 其他几何方案为什么没有保留',
              '旋转的尾部与低价组退步贡献分别为3770.96、2145.58，中价改善4337.63不足抵消。平移的低价组退步3045.92及尾部退步1601.01抵消了中高价改善。裁剪的低价与中价退步贡献为6205.57、2989.07，虽然高价与尾部改善，整体仍更差。裁剪最佳模型训练误差127504.88、验证130222.98，两者接近但均较高，因此不能把小泛化差距当作效果好。',
              '这些变换可能损失边缘环境、改变构图或引入填充，但当前证据不能区分各机制。建议维持原幅度的失败记录、不叠加到翻转；若以后重试，应先明确诊断依据，再单独研究更弱幅度或边界处理。单种子落选不能证明所有幅度、种子都无效。',
              '## 四、新增实验：依据、参数与风险',
              '| 实验 | 固定参数 | 研究问题与检查重点 |',
              '| --- | --- | --- |',
              '| 八：高斯模糊 | 概率20%，5×5核，标准差0.3–0.8原图像素 | 检验对锐利细节的依赖；可能削弱屋面与外墙纹理 |',
              '| 九：高斯噪声 | 概率20%，零均值，标准差0.01 | 在0–1原图张量上加噪并截断至0–1；检验微弱像素扰动的鲁棒性 |',
              '| 十：局部擦除 | 概率15%，面积1%–3%，宽高比0.5–2 | 填充灰色0.5，最多尝试10次；检验局部遮挡，记录触发与放置失败 |',
              '本轮依据是已有训练验证差距，探索纹理和局部区域依赖，不声称数据已经证实存在模糊或噪声问题。模糊与噪声发生在原图尺度，随后缩小可能明显减弱扰动；若无收益，只能否定当前位置与幅度的候选，不能直接推断操作本身无效。擦除可能覆盖房屋关键部位，因此先检查固定训练图片预览。',
              '## 五、新增结果与逐组分析',
              '| 实验 | 种子 | 验证均方误差 | 相比同种子无增强 | 最佳轮／总轮 | 触发率 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in state['reviews']:
        lines.append(f"| {NAMES[r['experiment']]} | {r['seed']} | {r['mse']:.2f} | {r['delta_mse']:+.2f} | {r['best_epoch']}/{r['epochs']} | {r['trigger_rate']:.1%} |")
    if not state['reviews']: lines.append('尚无新增完整训练结果，不预判是否有效。')
    for r in state['reviews']:
        lines.append(f"**{NAMES[r['experiment']]}，种子{r['seed']}：** {r['analysis']} 无增强完整训练集误差{r['train_mse']:.2f}，训练验证差距{r['train_validation_gap']:.2f}。{'本次验证误差改善，是否采用以三种子决策为准。' if r['delta_mse']<0 else '本次验证未改善；训练损失下降不能单独支持采用。'}")
        if r['experiment'] == 'augment_10_erase':
            lines.append(f"局部擦除因无法放置而跳过{r.get('skipped_count',0)}次；触发率与成功擦除率分别检查。")
    effects_path = RECORD/f'analyses/{WORK_NAME}/input_effects.json'
    if effects_path.exists():
        lines.append('输入扰动诊断：以下仅在固定32张训练图上强制触发，用于检查缩放后的扰动强度；正式训练保持登记概率，不用这项指标选择模型。')
        for name,value in read_json(effects_path).items():
            lines.append(f"{NAMES[name]}：缩放后0–1像素尺度的平均绝对变化{value['mean_absolute_pixel_change_after_resize']:.6f}。")
    lines += ['## 六、选取建议与后续', *[d['text'] for d in state['decisions']],
              '每组训练结束先检查数据、配置、初始化、全部样本、增强统计和分价格段误差，写入本报告后才继续。当前授权范围内的三项及必要复核结束后停止，不自动加入亮度、颜色或组合实验。',
              '## 七、证据与复现',
              '旧几何阶段状态保留在analyses/geometry_completed.json。新增预检见analyses/robustness_20261004/preflight.json，含图片预览只留本地。runs按实验与运行编号保存损失、验证误差和学习率曲线；固定32张训练探针不等于训练集只有32张。完整权重、逐图预测和日志保存在.local/experiment_record/05_augmentation。']
    if state.get('error'): lines.append('本轮异常停止，诊断见robustness_status.json，排查后才能继续。')
    import re
    text = re.sub(r'(?m)(\|[^\n]*\|)\n\n(?=\|)', r'\1\n', '\n\n'.join(lines)) + '\n'
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temp=REPORT.with_suffix('.tmp');temp.write_text(text,encoding='utf-8');os.replace(temp,REPORT)
    public={k:v for k,v in state.items() if k not in ('pid','runs')}
    atomic(RECORD/'robustness_status.json',public)
    atomic(RECORD/'augmentation_status.json',public)


def configure():
    old = historical()
    prior = resolve(ROOT / 'configs/03_augmentation/augment_01_flip.json')
    mean = float(np.mean([r['mse'] for r in old['reviews'] if r['experiment']=='augment_01_flip']))
    stage.CANDIDATES = CANDIDATES
    stage.NAMES = NAMES
    stage.WORK_NAME = WORK_NAME
    stage.PRIOR_WINNERS = [(mean, 'augment_01_flip', prior)]
    stage.COMPLETE_STATUS = '实验八、九、十及必要复核完成；队列结束'
    stage.publish = publish
    stage.prepare = prepare


if __name__ == '__main__':
    configure()
    stage.main()
