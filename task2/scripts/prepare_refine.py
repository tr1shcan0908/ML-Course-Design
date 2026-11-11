"""Pin the current best recipe and prepare a bounded component refinement plan."""
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from fer.data import FERDataset
from fer.io import atomic_json, read_json
from fer.refinement import ReplacementReference, file_sha256


def main():
    plan_path = ROOT/'experiments/refine_plan.json'
    if plan_path.exists():
        raise FileExistsError('Refinement plan already exists; inspect rather than overwrite.')
    previous = read_json('artifacts/transfer_final_results.json')
    recipe = previous['selected_recipe']
    validation = FERDataset('artifacts/data','val')
    assert recipe['classes'] == validation.split['classes']
    assert recipe['manifest_sha256'] == file_sha256('artifacts/data/manifest.jsonl')
    assert len(recipe['components']) == 4
    references, arrays = [], {}
    for component in recipe['components']:
        assert component['weight'] == .25 and not component['flip_tta']
        name = component['run']
        assert component['checkpoint_sha256'] == file_sha256(component['checkpoint'])
        npz = f'artifacts/transfer_validation/{name}.npz'
        meta = f'artifacts/transfer_validation/{name}.json'
        row = dict(name=name,weight=.25,checkpoint=component['checkpoint'],
                   checkpoint_sha256=component['checkpoint_sha256'],probabilities=npz,
                   probabilities_sha256=file_sha256(npz),metadata=meta,metadata_sha256=file_sha256(meta))
        references.append(row)
        with np.load(npz,allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['indices'],validation.indices)
            np.testing.assert_array_equal(saved['labels'],validation.labels)
            arrays[name] = saved['probabilities'].copy()
    baseline = sum(arrays[row['name']]*row['weight'] for row in references)
    accuracy = float(np.mean(baseline.argmax(1)==validation.labels))
    assert accuracy == previous['selected_validation_metrics']['accuracy']

    def selection_spec(parent):
        replaced = next(row for row in references if row['name']==parent)
        spec = dict(candidate_weight=.25,components=[r for r in references if r['name']!=parent],
                    replaces_checkpoint=replaced['checkpoint'],
                    replaces_checkpoint_sha256=replaced['checkpoint_sha256'],
                    manifest_sha256=recipe['manifest_sha256'],classes=recipe['classes'],baseline_accuracy=accuracy)
        scorer=ReplacementReference(spec,validation.indices,validation.labels,recipe['classes'],recipe['manifest_sha256'])
        recombined=scorer.combine(validation.labels,arrays[parent],validation.indices)
        np.testing.assert_array_equal(recombined.argmax(1),baseline.argmax(1))
        return spec

    b2parent='transfer_emotieff_b2_ce_s42'
    emoparent='emonext_tiny_s42'
    b2=read_json(f'experiments/configs/{b2parent}.json')
    b2.update(name='refine_b2_ce_s52',output='runs/refine_b2_ce_s52',seed=52,epochs=14,
              lr=8e-6,head_lr_multiplier=2.,weight_decay=5e-4,warmup_epochs=1,
              early_stopping_patience=4,min_epochs=5,finetune_from=f'runs/{b2parent}/best.pt',
              selection_reference=selection_spec(b2parent))
    eac=copy.deepcopy(b2)
    eac.update(name='refine_b2_eac_s52',output='runs/refine_b2_eac_s52',loss='eac',eac_weight=.05)
    emo=read_json(f'experiments/configs/{emoparent}.json')
    emo.update(name='refine_emonext_s53',output='runs/refine_emonext_s53',seed=53,epochs=12,
               lr=1.5e-5,weight_decay=.02,label_smoothing=.15,warmup_epochs=1,
               early_stopping_patience=4,min_epochs=5,image_size=224,
               selection_metric='validation_accuracy',eval_precision='float32',eval_batch_size=32,
               evaluate_initial=True,finetune_from=f'runs/{emoparent}/best.pt',
               freeze_stages=1,freeze_modules=['localization','fc_loc'],drop_path_rate=.2,layer_decay=.75,
               mixup_alpha=.2,mixup_probability=.5,
               augmentation=dict(degrees=10,scale_range=.05,translation=.03,jitter=.08,erase_probability=.25),
               selection_reference=selection_spec(emoparent))
    paths=[]
    for config in (b2,eac,emo):
        path=f'experiments/configs/{config["name"]}.json'
        atomic_json(path,config)
        paths.append(path)
    plan=dict(phase='best_ensemble_component_refinement',configs=paths,
              reference_configs=[c['config_path'] for c in recipe['components']],comparison_configs=[],
              data='artifacts/data',seed=20261014,eval_precision='float32',ensemble_weights=[],
              reference_blend_weights=[],replacement_parents={b2['name']:b2parent,eac['name']:b2parent,emo['name']:emoparent},
              result_prefix='refine',prediction_directory='predictions/refine',
              report='experiments/REFINE_RESULTS.md',report_title='最佳融合模型继续微调',
              plan_document='REFINE_PLAN.md',queue_status='artifacts/refine_queue_status.json',
              controller_lock='artifacts/refine_controller.lock',finalization_log='artifacts/refine_finalization_console.log',
              previous_result='artifacts/transfer_final_results.json',
              progress_notes=['基准为四模型等权融合，验证 Accuracy 72.3794%。',
                              '新模型按替换相应组件后的四模型融合 Accuracy 保存最佳权重；Macro-F1 用于同分比较。',
                              'B2 CE / EAC 配对低学习率微调；EmoNeXt 增加 Mixup、随机深度并冻结第一阶段与 STN。',
                              '最终保留原四模型候选，比较三个单组件替换和两个双组件替换；全部新输出放在 refine 目录。'])
    atomic_json(plan_path,plan)
    atomic_json('artifacts/refine_reference_audit.json',dict(baseline_accuracy=accuracy,classes=recipe['classes'],
                manifest_sha256=recipe['manifest_sha256'],validation_images=len(validation),references=references,
                previous_result_sha256=file_sha256('artifacts/transfer_final_results.json'),
                existing_submission_sha256=file_sha256('submission_best_7238.csv')))
    print(f'Prepared {len(paths)} refinement trials; reference accuracy {accuracy:.8%}.',flush=True)


if __name__=='__main__':
    main()
