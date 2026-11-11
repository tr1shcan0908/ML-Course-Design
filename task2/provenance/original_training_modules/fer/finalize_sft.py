"""Re-evaluate supervised fine-tuning candidates, select a recipe and export."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FERDataset, ImagePipeline
from .io import atomic_json, atomic_output, read_json, update_status
from .models import make_model
from .selection import recipe_candidates
from .train import classification_metrics, evaluate, seed_everything


STATUS=Path('artifacts/sft_finalization_status.json')


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(4*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def save_npz(path, **arrays):
    with atomic_output(path) as temp:
        with temp.open('wb') as file:
            np.savez_compressed(file,**arrays)


def load_model(profile, manifest, classes):
    checkpoint=torch.load(profile['checkpoint'],map_location='cpu',weights_only=True)
    config=checkpoint['config']
    if config['manifest_sha256']!=manifest or checkpoint['classes']!=classes:
        raise RuntimeError(f'Checkpoint data/label mismatch: {profile["checkpoint"]}')
    model,_=make_model(config,load_pretrained=False)
    model.load_state_dict(checkpoint['model'],strict=True)
    del checkpoint
    return model.to(device='cuda',memory_format=torch.channels_last).eval(),config


def validation_predictions(profile, dataset, source_hashes, cache_directory='artifacts/sft_validation'):
    cache=Path(cache_directory)
    cache.mkdir(parents=True,exist_ok=True)
    key=profile['key']
    metadata={**profile,'manifest_sha256':dataset.split['manifest_sha256'],
              'eval_precision':'float32','eval_batch_size':32,'source_sha256':source_hashes}
    npz_path=cache/(key+'.npz')
    meta_path=cache/(key+'.json')
    if npz_path.exists() and meta_path.exists() and read_json(meta_path)==metadata:
        with np.load(npz_path) as saved:
            labels=saved['labels'].copy()
            probs=saved['probabilities'].copy()
            indices=saved['indices'].copy()
    else:
        model,config=load_model(profile,dataset.split['manifest_sha256'],dataset.split['classes'])
        loader=DataLoader(dataset,batch_size=32,num_workers=0,pin_memory=True)
        pipeline=ImagePipeline('cuda',size=config.get('image_size',224))
        labels,probs,indices=evaluate(model,loader,pipeline,config['model'],'cuda',
                                      tta=profile['flip_tta'],precision='float32')
        save_npz(npz_path,labels=labels,probabilities=probs,indices=indices)
        atomic_json(meta_path,metadata)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    np.testing.assert_array_equal(indices,dataset.indices)
    np.testing.assert_array_equal(labels,dataset.labels)
    if probs.shape!=(len(dataset),6) or not np.isfinite(probs).all():
        raise RuntimeError('Invalid validation probabilities.')
    np.testing.assert_allclose(probs.sum(1),1.,atol=1e-6)
    return probs


def main():
    global STATUS
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',default='experiments/sft_plan.json')
    args=parser.parse_args()
    plan=read_json(args.plan)
    prefix=plan.get('result_prefix','sft')
    STATUS=Path(f'artifacts/{prefix}_finalization_status.json')
    cache_directory=f'artifacts/{prefix}_validation'
    seed_everything(plan['seed'])
    validation=FERDataset(plan['data'],'val')
    classes=validation.split['classes']
    manifest=file_hash(Path(plan['data'])/'manifest.jsonl')
    if manifest!=validation.split['manifest_sha256']:
        raise RuntimeError('Data manifest was modified.')
    source_hashes={name:file_hash(name) for name in ['fer/train.py','fer/models.py','fer/data.py',
                  'fer/emotieff.py','fer/losses.py','fer/finalize_sft.py','fer/selection.py']}
    profiles={}
    probabilities={}
    records=[]
    for path in plan['reference_configs']+plan.get('comparison_configs',[])+plan['configs']:
        config=read_json(path)
        run=Path(config['output'])
        status=read_json(run/'status.json')
        if status.get('status')!='complete':
            raise RuntimeError(f'Training has not completed: {run}')
        history=[json.loads(line) for line in (run/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        expected=status.get('epochs_completed',config['epochs'])
        if [row['epoch'] for row in history]!=list(range(1,expected+1)):
            raise RuntimeError(f'Incomplete or duplicated epoch history: {run}')
        checkpoint=run/'best.pt'
        profile=dict(key=config['name'],run=config['name'],config_path=path,model=config['model'],
                     checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),flip_tta=False)
        profiles[profile['key']]=profile
        update_status(STATUS,dict(status='validating',current=config['name']))
        probs=validation_predictions(profile,validation,source_hashes,cache_directory)
        probabilities[profile['key']]=probs
        metrics=classification_metrics(validation.labels,probs,classes)
        record=dict(name=config['name'],phase=prefix if path in plan['configs'] else 'reference',
                    accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'],
                    best_epoch=read_json(run/'best_metrics.json')['epoch'],epochs_completed=expected,
                    early_stopped=status.get('stopped_early',False),
                    training_seconds=sum(row['epoch_seconds'] for row in history),
                    per_class=metrics['per_class'])
        records.append(record)
        atomic_json(f'artifacts/{prefix}_revalidation.json',records)
        print(json.dumps(record),flush=True)
    best_plain=max(records,key=lambda r:(r['accuracy'],r['macro_f1']))
    flip={**profiles[best_plain['name']],'key':best_plain['name']+'_flip','flip_tta':True}
    profiles[flip['key']]=flip
    update_status(STATUS,dict(status='checking_flip_tta',current=best_plain['name']))
    probabilities[flip['key']]=validation_predictions(flip,validation,source_hashes,cache_directory)
    reference_keys=[read_json(path)['name'] for path in plan['reference_configs']]
    new_keys=[read_json(path)['name'] for path in plan['configs']]
    candidates=recipe_candidates(probabilities,validation.labels,profiles,plan['ensemble_weights'],reference_keys,
                                 new_keys,plan.get('reference_blend_weights',[]))
    selected=candidates[0]
    selected_probs=sum(probabilities[c['profile']]*c['weight'] for c in selected['components'])
    selected_metrics=classification_metrics(validation.labels,selected_probs,classes)
    reference_probs=sum(probabilities[key]/len(reference_keys) for key in reference_keys)
    reference_metrics=classification_metrics(validation.labels,reference_probs,classes)
    previous=None
    if plan.get('previous_result'):
        previous=read_json(plan['previous_result'])['selected_validation_metrics']
        if abs(reference_metrics['accuracy']-previous['accuracy'])>1/len(validation):
            raise RuntimeError('Previous best ensemble could not be reproduced to within one validation image.')
    best_single=next(c for c in candidates if len(c['components'])==1)
    best_sft=max((row for row in records if row['phase']==prefix),key=lambda r:(r['accuracy'],r['macro_f1']))
    baseline=next(row for row in records if row['name']=='convnext_tiny_s42')
    output=Path(plan.get('prediction_directory','predictions/sft'))
    output.mkdir(parents=True,exist_ok=True)
    save_npz(output/'selected_validation_predictions.npz',labels=validation.labels,
             probabilities=selected_probs,indices=np.array(validation.indices))
    testing=FERDataset(plan['data'],'test')
    test_probs=np.zeros((len(testing),6),dtype=np.float32)
    components=[]
    for component in selected['components']:
        profile=profiles[component['profile']]
        update_status(STATUS,dict(status='predicting_test',current=profile['run'],recipe=selected['name']))
        model,config=load_model(profile,manifest,classes)
        loader=DataLoader(testing,batch_size=32,num_workers=0,pin_memory=True)
        _,probs,indices=evaluate(model,loader,ImagePipeline('cuda',size=config.get('image_size',224)),
                                 config['model'],'cuda',tta=profile['flip_tta'],precision='float32')
        np.testing.assert_array_equal(indices,testing.indices)
        test_probs+=probs*component['weight']
        components.append({**profile,'weight':component['weight'],'image_size':config.get('image_size',224)})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if test_probs.shape!=(7065,6) or not np.isfinite(test_probs).all():
        raise RuntimeError('Invalid test probabilities.')
    np.testing.assert_allclose(test_probs.sum(1),1.,atol=1e-5)
    with atomic_output(output/'best_predictions.csv') as temp:
        with temp.open('w',encoding='utf-8',newline='') as file:
            writer=csv.writer(file)
            writer.writerow(['image','predicted_label','class_id','confidence'])
            for index,probs in zip(testing.indices,test_probs):
                label=int(probs.argmax())
                writer.writerow([Path(testing.rows[index]['path']).name,classes[label],label,f'{float(probs[label]):.8f}'])
    with (output/'best_predictions.csv').open(encoding='utf-8',newline='') as file:
        exported=list(csv.DictReader(file))
    assert len(exported)==len({row['image'] for row in exported})==7065
    save_npz(output/'test_probabilities.npz',probabilities=test_probs,indices=np.array(testing.indices),classes=np.array(classes))
    recipe=dict(name=selected['name'],components=components,classes=classes,eval_precision='float32',
                manifest_sha256=manifest,selection_metric='validation_accuracy',source_sha256=source_hashes)
    atomic_json(output/'inference_recipe.json',recipe)
    summary=dict(completed_at=datetime.now(timezone.utc).isoformat(),plan=args.plan,
                 selected_model=selected['name'],selected_recipe=recipe,selected_validation_metrics=selected_metrics,
                 best_single=best_single,best_sft_single=best_sft,baseline=baseline,runs=records,candidates=candidates,
                 accuracy_gain_percentage_points=100*(selected_metrics['accuracy']-baseline['accuracy']),
                 validation_images=len(validation),test_images=len(testing),test_labels_available=False,
                 predictions=str(output/'best_predictions.csv'),
                 limitations=['One existing validation split is reused for tuning, early stopping and recipe selection.',
                              'Validation improvements are not independent labeled test results.'])
    summary['reference_ensemble_metrics']=reference_metrics
    summary['gain_over_reference_ensemble_pp']=100*(selected_metrics['accuracy']-reference_metrics['accuracy'])
    if previous is not None:
        summary['previous_reported_accuracy']=previous['accuracy']
    atomic_json(f'artifacts/{prefix}_final_results.json',summary)
    write_report(plan,summary)
    update_status(STATUS,dict(status='complete',selected_model=selected['name'],
                  validation_accuracy=selected_metrics['accuracy'],validation_macro_f1=selected_metrics['macro_f1'],
                  predictions=summary['predictions']),required=True)
    print(json.dumps(dict(event='complete',selected=selected['name'],accuracy=selected_metrics['accuracy'],
                         macro_f1=selected_metrics['macro_f1'],gain_pp=summary['accuracy_gain_percentage_points'])),flush=True)


def write_report(plan,summary):
    metric=summary['selected_validation_metrics']
    lines=['# '+plan.get('report_title','监督微调与准确率优化结果'),'',
           f'最佳推理方案：**{summary["selected_model"]}**。验证 Accuracy **{metric["accuracy"]:.2%}**，Macro-F1 **{metric["macro_f1"]:.2%}**。',
           f'相对同精度复核的原 ConvNeXt 基线，准确率变化 **{summary["accuracy_gain_percentage_points"]:+.2f} 个百分点**。','',
           '| 单模型 | 类别 | 最佳轮次 | 已训轮数 | 验证 Accuracy | Macro-F1 |',
           '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in sorted(summary['runs'],key=lambda r:(r['accuracy'],r['macro_f1']),reverse=True):
        lines.append(f'| {row["name"]} | {row["phase"]} | {row["best_epoch"]} | {row["epochs_completed"]} | {row["accuracy"]:.2%} | {row["macro_f1"]:.2%} |')
    lines+=['',f'相对原三模型融合（{summary["reference_ensemble_metrics"]["accuracy"]:.2%}），准确率变化 **{summary["gain_over_reference_ensemble_pp"]:+.2f} 个百分点**。',
            '上表统一重载权重，以 FP32、无翻转进行验证。最佳轮次 0 表示保留微调前的初始权重。','',
            f'最佳单模型推理方案：`{summary["best_single"]["name"]}`，Accuracy {summary["best_single"]["accuracy"]:.2%}。','',
            '最终方案组成：']
    for item in summary['selected_recipe']['components']:
        lines.append(f'- `{item["run"]}`：权重 {item["weight"]:.3f}，水平翻转 TTA {"启用" if item["flip_tta"] else "关闭"}。')
    lines+=['','| 类别 | Precision | Recall | F1 | 样本数 |','| --- | ---: | ---: | ---: | ---: |']
    for name in summary['selected_recipe']['classes']:
        row=metric['per_class'][name]
        lines.append(f'| {name} | {row["precision"]:.2%} | {row["recall"]:.2%} | {row["f1-score"]:.2%} | {int(row["support"])} |')
    if plan.get('result_prefix')=='transfer':
        by_name={row['name']:row for row in summary['runs']}
        ce=by_name['transfer_emotieff_b0_ce_s42']
        eac=by_name['transfer_emotieff_b0_eac_s42']
        lines+=['',f'B0 配对对照：增加 EAC 后单模型 Accuracy 变化 {100*(eac["accuracy"]-ce["accuracy"]):+.2f} 个百分点。仅一个种子，不能据此判断统计显著性。',
                '', '| 重点类别 | 原融合 Recall | 本轮最终 Recall | Recall 变化（百分点） |',
                '| --- | ---: | ---: | ---: |']
        for name in ('Fear','Sad'):
            before=summary['reference_ensemble_metrics']['per_class'][name]['recall']
            after=metric['per_class'][name]['recall']
            lines.append(f'| {name} | {before:.2%} | {after:.2%} | {100*(after-before):+.2f} |')
    lines+=['','全部 7,065 张无标签测试图已生成预测，并检查文件名唯一性、类别映射和概率归一化。',
            f'预测：[best_predictions.csv](../{summary["predictions"]})；推理配置：[inference_recipe.json](../{plan.get("prediction_directory","predictions/sft")}/inference_recipe.json)。',
            '本轮验证集同时用于调参、早停和选择融合方案，因此成绩是开发验证结果；独立测试准确率仍需真实测试标签。',
            '完整配置与方法参考：['+plan.get('plan_document','SFT_PLAN.md')+']('+plan.get('plan_document','SFT_PLAN.md')+')；'
            f'详细候选分数：[{plan.get("result_prefix","sft")}_final_results.json](../artifacts/{plan.get("result_prefix","sft")}_final_results.json)。',
            f'生成时间（北京时间）：{datetime.now(timezone(timedelta(hours=8))).isoformat()}']
    with atomic_output(plan['report']) as temp:
        temp.write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        update_status(STATUS,dict(status='failed',error=repr(error)))
        traceback.print_exc()
        raise
