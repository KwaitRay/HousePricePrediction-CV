"""Aggregate authoritative development results without replacing unfinished runs."""
import json
from pathlib import Path
from common.local_paths import REPOSITORY
from common.config import write_json

def main():
    root=REPOSITORY/'output/experiment_record'
    rows=[]
    for stage in ['11_pretrained_cnn','12_finetune','13_clip_optimization','14_blip_reference','15_fixed_ensembles','16_price_contrastive','17_geo_visual_reference','19_image_embedding_reference','20_price_lora']:
        for path in sorted((root/stage).glob('*/*/status.json')):
            folder=path.parent;state=json.loads(path.read_text(encoding='utf-8'))
            result=folder/'results.json'
            if result.exists():
                values=json.loads(result.read_text(encoding='utf-8'))
                for item in values if isinstance(values,list) else [values]:
                    rows.append({'model':folder.parent.name,'run':folder.name,'head':item.get('head','kernel' if 'gamma' in item else 'residual_mlp' if folder.parent.name=='clip_residual' else 'fine_tune'),
                        'status':state['status'],'validation_mse':item.get('mse',item.get('best_mse')),
                        'training_cv_mse':item.get('training_cv_mse',item.get('training_oof_mse')),
                        'mae':item.get('mae'),'tail_mse':item.get('tail_mse'),'artifact':str(folder.relative_to(root))})
                if state.get('head') and not any(r['run']==folder.name and r['head']==state['head'] for r in rows):
                    rows.append({'model':folder.parent.name,'run':folder.name,'head':state['head'],'status':state['status'],
                        'validation_mse':None,'artifact':str(folder.relative_to(root))})
            elif 'best_mse' in state:
                rows.append({'model':folder.parent.name,'run':folder.name,'head':'fine_tune','status':state['status'],
                    'validation_mse':state['best_mse'],'best_epoch':state.get('best_epoch'),'current_epoch':state.get('epoch'),
                    'artifact':str(folder.relative_to(root))})
            else:
                pipeline=state['status']=='completed' and ((folder/'kernel_queue.json').exists() or (folder/'training_queue.json').exists())
                rows.append({'model':folder.parent.name,'run':folder.name,'head':state.get('head','feature_extraction_pipeline' if pipeline else 'pending'),
                    'status':state['status'],'validation_mse':None,'artifact':str(folder.relative_to(root))})
    destination=root/'improvement_to_50000';destination.mkdir(exist_ok=True)
    feedback=[]
    for path in sorted((root/'08_final').glob('*/*/kaggle_feedback.json')):
        value=json.loads(path.read_text(encoding='utf-8'))
        feedback.append({**value,'record':str(path.relative_to(root))})
    best_public=min((v['public_mse'] for v in feedback),default=None)
    primary_met=best_public is not None and best_public<50000
    current_met=best_public is not None and best_public<40000
    write_json(destination/'summary.json',{'target_mse':50000,'target_verified':primary_met,
        'current_target_mse':40000,'current_target_verified':current_met,'best_public_mse':best_public,
        'scope':'Development validation and separately identified user-reported Kaggle scores',
        'existing_clip_reference_mse':91820.73,'existing_kaggle_feedback':91993.03622,'kaggle_feedback':feedback,'attempts':rows})
    report='# 预测改进尝试总览\n\n'
    report+=f'已记录最佳用户反馈公榜MSE：{best_public}。原50000目标已达到：{primary_met}；当前40000目标已达到：{current_met}。暂停与未完成任务以各运行status及交接文档为准，本脚本只汇总，不启动实验。\n\n'
    report+='本地开发验证与用户反馈公榜分开记录，分数绑定具体CSV哈希，未独立访问Kaggle；公开结果不保证私榜表现。原始status保留失败、暂停和未完成状态，LoRA结果中的保留基线不冒充新模型成绩。通用视觉主线不使用MLLM、额外房源数据或元数据。完整设计与逻辑链见[唯一设计文档](../../../design/model_and_experiment_plan.md)，当前交接见[最新进展](最新进展与接续.md)。\n'
    if feedback:
        report+='\n\n用户反馈的 Kaggle 公榜结果（与具体 CSV 校验值绑定）：\n\n'
        for value in feedback:
            report+=f"- 公榜 MSE **{value['public_mse']:.5f}**；对应开发验证 MSE {value['reference_validation_mse']:.2f}，记录：[{value['csv_sha256'][:12]}](../{value['record'].replace(chr(92),'/')})。\n"
    report+='\n| 模型 | 回归方法 | 验证MSE | 状态 | 记录 |\n| --- | --- | ---: | --- | --- |\n'
    for r in rows:
        score=('子实验见对应记录' if r['head']=='feature_extraction_pipeline' else '未完成') if r['validation_mse'] is None else f"{r['validation_mse']:.2f}"
        relative='../'+r['artifact'].replace('\\','/')
        report+=f"| {r['model']} | {r['head']} | {score} | {r['status']} | [运行产物]({relative}) |\n"
    report+='\n只从训练分区选择回归超参数；微调检查点按原完整未加权验证MSE选择。方法之间反复比较仍存在开发选择偏差。权重、特征和逐图预测在.local，完整候选CV成绩与配置公开聚合保存。\n'
    (destination/'实验总览.md').write_text(report,encoding='utf-8')
    print(report)

if __name__=='__main__':main()
