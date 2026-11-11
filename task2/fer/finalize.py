"""Audit completed validation runs, select a model, and predict the unlabeled test set."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FERDataset, ImagePipeline
from .io import atomic_json, read_json, update_status
from .models import make_model
from .train import classification_metrics, evaluate, seed_everything


def completed_configs(plan):
    configs=[read_json(path) for path in plan['configs']]
    return all((Path(c['output'])/'status.json').exists()
               and read_json(Path(c['output'])/'status.json').get('status')=='complete' for c in configs)


def write_progress(plan):
    lines=['# FER 实验进行中','',
           '下表是训练中的最佳验证成绩，完整对照尚未结束。每约 20 秒更新，重新打开文件可查看。','',
           '| 配置 | 状态 | 轮次 | 当前最佳 Accuracy | 当前最佳 Macro-F1 |',
           '| --- | --- | --- | ---: | ---: |']
    for path in plan['configs']:
        config=read_json(path)
        run=Path(config['output'])
        status=read_json(run/'status.json') if (run/'status.json').exists() else {}
        metrics=read_json(run/'best_metrics.json') if (run/'best_metrics.json').exists() else None
        accuracy=f'{metrics["accuracy"]:.2%}' if metrics else '—'
        f1=f'{metrics["macro_f1"]:.2%}' if metrics else '—'
        epoch=status.get('epoch',config['epochs'] if status.get('status')=='complete' else 0)
        lines.append(f'| {config["name"]} | {status.get("status","pending")} | {epoch}/{config["epochs"]} | {accuracy} | {f1} |')
    lines.extend(['','统一验证集 5,409 张，原 test 无标签；没有正式测试准确率。',
                  '全部训练完成后，将自动重载最佳权重、汇总每类指标并生成测试图预测。','',
                  '配置、数据排除规则与复现说明见 [README.md](README.md)。',
                  f'更新时间（UTC）：{datetime.now(timezone.utc).isoformat()}'])
    Path('experiments/RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',default='experiments/plan.json')
    parser.add_argument('--wait-for-queue',action='store_true')
    args=parser.parse_args()
    plan=read_json(args.plan)
    status_path=Path('artifacts/finalization_status.json')
    if args.wait_for_queue:
        print('Waiting for all four training runs to complete.',flush=True)
        while not completed_configs(plan):
            queue=read_json('artifacts/queue_status.json') if Path('artifacts/queue_status.json').exists() else {}
            state='waiting_for_failed_run_to_resume' if queue.get('status')=='failed' else 'waiting_for_training'
            update_status(status_path,dict(status=state))
            write_progress(plan)
            time.sleep(20)
    elif not completed_configs(plan):
        raise RuntimeError('All planned training runs must finish before final evaluation.')
    update_status(status_path,dict(status='auditing_validation'))
    results=[]
    for path in plan['configs']:
        config=read_json(path)
        run=Path(config['output'])
        metrics=read_json(run/'best_metrics.json')
        split=read_json(Path(config['data'])/'split.json')
        saved=np.load(run/'val_predictions.npz')
        np.testing.assert_array_equal(saved['indices'],split['val'])
        recalculated=classification_metrics(saved['labels'],saved['probabilities'],split['classes'])
        assert abs(recalculated['accuracy']-metrics['accuracy'])<1e-12
        assert abs(recalculated['macro_f1']-metrics['macro_f1'])<1e-12
        history=[json.loads(line) for line in (run/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(history)==config['epochs']
        initialization=read_json(run/'initialization.json')
        results.append(dict(name=config['name'],model=config['model'],config_path=path,
                            epoch=metrics['epoch'],accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'],
                            per_class=metrics['per_class'],confusion_matrix=metrics['confusion_matrix'],
                            parameters=initialization['parameters'],pretraining=initialization['pretrained']['pretraining'],
                            training_seconds=sum(row['epoch_seconds'] for row in history),
                            peak_memory_gib=max(row['peak_gpu_memory_gib'] for row in history)))
    winner=max(results,key=lambda row:(row['macro_f1'],row['accuracy']))
    config=read_json(winner['config_path'])
    run=Path(config['output'])
    seed_everything(config['seed'])
    model,_=make_model(config,load_pretrained=False)
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model'],strict=True)
    model=model.to(device='cuda',memory_format=torch.channels_last).eval()
    pipeline=ImagePipeline('cuda')
    validation=FERDataset(config['data'],'val')
    val_loader=DataLoader(validation,batch_size=config['eval_batch_size'],num_workers=0,pin_memory=True)
    update_status(status_path,dict(status='checking_winner',winner=winner['name']))
    y,probs,indices=evaluate(model,val_loader,pipeline,config['model'],'cuda',tta=False)
    reloaded=classification_metrics(y,probs,validation.split['classes'])
    saved=np.load(run/'val_predictions.npz')
    max_delta=float(np.max(np.abs(probs-saved['probabilities'])))
    if abs(reloaded['accuracy']-winner['accuracy'])>.001 or abs(reloaded['macro_f1']-winner['macro_f1'])>.001:
        raise RuntimeError('Reloaded checkpoint differs materially from saved validation predictions.')
    _,tta_probs,_=evaluate(model,val_loader,pipeline,config['model'],'cuda',tta=True)
    tta_metrics=classification_metrics(y,tta_probs,validation.split['classes'])
    use_tta=(tta_metrics['macro_f1'],tta_metrics['accuracy'])>(reloaded['macro_f1'],reloaded['accuracy'])
    selected=tta_metrics if use_tta else reloaded
    output=Path('predictions')
    output.mkdir(exist_ok=True)
    np.savez_compressed(output/'selected_validation_predictions.npz',labels=y,probabilities=tta_probs if use_tta else probs,indices=indices)
    testing=FERDataset(config['data'],'test')
    test_loader=DataLoader(testing,batch_size=config['eval_batch_size'],num_workers=0,pin_memory=True)
    update_status(status_path,dict(status='predicting_test',winner=winner['name'],flip_tta=use_tta))
    _,test_probs,test_indices=evaluate(model,test_loader,pipeline,config['model'],'cuda',tta=use_tta)
    assert test_probs.shape==(7065,6)
    assert np.all(np.isfinite(test_probs))
    np.testing.assert_allclose(test_probs.sum(1),1.,atol=1e-5)
    classes=testing.split['classes']
    with (output/'best_model_predictions.csv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['image','predicted_label','class_id','confidence'])
        for index,probability in zip(test_indices,test_probs):
            label=int(probability.argmax())
            writer.writerow([Path(testing.rows[int(index)]['path']).name,classes[label],label,f'{float(probability[label]):.8f}'])
    np.savez_compressed(output/'test_probabilities.npz',probabilities=test_probs,indices=test_indices,classes=np.array(classes))
    atomic_json(output/'label_mapping.json',{str(i):name for i,name in enumerate(classes)})
    summary=dict(completed_at=datetime.now(timezone.utc).isoformat(),selection_metric='validation_macro_f1',
                 validation_images=len(validation),test_images=len(testing),test_labels_available=False,
                 manifest_sha256=validation.split['manifest_sha256'],runs=results,
                 selected_model=winner['name'],selected_epoch=winner['epoch'],flip_tta=use_tta,
                 selected_validation_metrics=selected,tta_validation_metrics=tta_metrics,
                 checkpoint_reload_max_probability_delta=max_delta,
                 best_checkpoint=str(run/'best.pt'),predictions=str(output/'best_model_predictions.csv'),
                 limitations=['One split and one training seed; 12-epoch initial comparison, not a convergence claim.',
                              'NLA is transferred to EmoNeXt; QCS uses different pretraining and compute.',
                              'Validation is used for checkpoint/model/TTA selection; no independent labeled test result.'])
    atomic_json('artifacts/final_results.json',summary)
    lines=['# 第一轮本机 FER 实验结果','',
           '所有模型完成 12 轮，使用同一固定验证集。表中为验证集选出的最佳轮次，未使用无标签测试集选模。','',
           '| 模型 | Accuracy | Macro-F1 | 最佳轮次 | 参数量 | 训练+验证时间 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in sorted(results,key=lambda row:row['macro_f1'],reverse=True):
        lines.append(f'| {row["name"]} | {row["accuracy"]:.2%} | {row["macro_f1"]:.2%} | {row["epoch"]} | {row["parameters"]/1e6:.2f}M | {row["training_seconds"]/60:.1f} 分钟 |')
    lines.extend(['',f'按 Macro-F1 选择 `{winner["name"]}`。水平翻转 TTA {"启用" if use_tta else "未启用"}；最终验证 Accuracy **{selected["accuracy"]:.2%}**，Macro-F1 **{selected["macro_f1"]:.2%}**。','',
                  '已为全部 7,065 张原测试图生成预测。其真实标签不可用，因此这里没有正式测试准确率；CSV 的标签编号是本项目内部映射，尚未核对外部提交格式。','',
                  '| 类别 | Precision | Recall | F1 | 验证样本数 |','| --- | ---: | ---: | ---: | ---: |'])
    for name in classes:
        values=selected['per_class'][name]
        lines.append(f'| {name} | {values["precision"]:.2%} | {values["recall"]:.2%} | {values["f1-score"]:.2%} | {int(values["support"])} |')
    lines.extend(['','训练曲线与混淆矩阵（若绘图环境可用）：','',
                  '![训练曲线](../artifacts/figures/learning_curves.png)','',
                  '![混淆矩阵](../artifacts/figures/confusion_matrix.png)','',
                  '这是一轮单划分、单种子的初步实验。NLA 是在 EmoNeXt 上的损失迁移，QCS 的预训练和计算预算也不同；不能据此宣称方法普遍优劣或达到公开 SOTA。','',
                  f'权重：`{run.as_posix()}/best.pt`；预测：`predictions/best_model_predictions.csv`；详细指标：`artifacts/final_results.json`。','',
                  '模型重载后再次计算验证指标，并检查保存的逐样本预测、标签映射、完整测试文件数和概率归一化。'])
    Path('experiments/RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    update_status(status_path,dict(status='complete',selected_model=winner['name'],
                                validation_accuracy=selected['accuracy'],validation_macro_f1=selected['macro_f1'],
                                completed_at=datetime.now(timezone.utc).isoformat()),required=True)
    print(json.dumps(summary,ensure_ascii=True),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        update_status('artifacts/finalization_status.json',dict(status='failed',error=repr(error),
                    updated_at=datetime.now(timezone.utc).isoformat()))
        traceback.print_exc()
        raise SystemExit(1)
