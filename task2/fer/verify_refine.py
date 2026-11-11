"""Check exact initialization, frozen state and real accumulated GPU updates."""
import gc
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .data import FERDataset, ImagePipeline
from .io import atomic_json, read_json
from .models import make_model
from .refinement import ReplacementReference, file_sha256, freeze_modules
from .train import EMA, build_optimizer, check_gradient_finite, evaluate, forward_loss, mixup_batch, seed_everything


def main():
    plan=read_json('experiments/refine_plan.json')
    configs=[read_json(path) for path in plan['configs']]
    ce,eac=configs[:2]
    differences={key for key in ce.keys()|eac.keys() if ce.get(key)!=eac.get(key)}
    assert differences=={'name','output','loss','eac_weight'},differences
    training,validation=(FERDataset(plan['data'],split) for split in ('train','val'))
    assert set(training.indices).isdisjoint(validation.indices)
    assert {training.rows[i]['pixel_sha256'] for i in training.indices}.isdisjoint(
        {validation.rows[i]['pixel_sha256'] for i in validation.indices})
    manifest=file_sha256('artifacts/data/manifest.jsonl')
    assert manifest==training.split['manifest_sha256']
    results=[]
    for config in configs:
        started=time.perf_counter()
        print(f'Checking {config["name"]}: load checkpoint and reproduce full validation baseline.',flush=True)
        seed_everything(config['seed'])
        checkpoint=torch.load(config['finetune_from'],map_location='cpu',weights_only=True)
        assert checkpoint['classes']==validation.split['classes']
        assert checkpoint['config']['manifest_sha256']==manifest
        model,_=make_model(config,load_pretrained=False)
        model.load_state_dict(checkpoint['model'],strict=True)
        del checkpoint
        frozen=freeze_modules(model,config)
        model=model.to(device='cuda',memory_format=torch.channels_last)
        optimizer=build_optimizer(model,config)
        ema=EMA(model)
        pipeline=ImagePipeline('cuda',size=config['image_size'],augmentation=config['augmentation'])
        scorer=ReplacementReference(config['selection_reference'],validation.indices,validation.labels,
                                     validation.split['classes'],manifest)
        labels,probs,indices=evaluate(model,DataLoader(validation,batch_size=32,num_workers=0,pin_memory=True),
                                       pipeline,config['model'],'cuda',precision='float32')
        fused=scorer.combine(labels,probs,indices)
        initial_accuracy=float(np.mean(fused.argmax(1)==labels))
        assert initial_accuracy==config['selection_reference']['baseline_accuracy'],initial_accuracy
        print(f'Initial fusion {initial_accuracy:.8%}; testing accumulated gradients and EMA.',flush=True)
        rng=np.random.default_rng(config['seed'])
        positions=rng.choice(len(training),size=config['batch_size']*6,replace=False)
        assert len(np.unique(training.labels[positions]))==6
        loader=DataLoader(Subset(training,positions.tolist()),batch_size=config['batch_size'])
        model.train()
        for module in frozen.values():
            module.eval()
        frozen_before={name:{key:value.detach().cpu().clone() for key,value in module.state_dict().items()}
                       for name,module in frozen.items()}
        head=model.backbone.classifier if config['model']=='emotieff' else model.head
        before=head.weight.detach().clone()
        torch.cuda.reset_peak_memory_stats()
        losses,timings,loss_parts=[],[],[]
        optimizer.zero_grad(set_to_none=True)
        for batch_index,(raw,labels,_) in enumerate(loader):
            tick=time.perf_counter()
            images=pipeline(raw.to('cuda'),training=True)
            labels=labels.to('cuda')
            images,targets=mixup_batch(images,labels,config.get('mixup_alpha',0.),config.get('mixup_probability',0.))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss,logits=forward_loss(model,images,targets,config,0)
            assert torch.isfinite(loss) and logits.shape==(len(labels),6)
            ce_loss=F.cross_entropy(logits.float(),targets,label_smoothing=config['label_smoothing'])
            loss_parts.append(dict(ce=float(ce_loss.detach()),weighted_regularizer=float((loss-ce_loss).detach())))
            (loss/config['accumulation_steps']).backward()
            if (batch_index+1)%config['accumulation_steps']==0:
                check_gradient_finite(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            torch.cuda.synchronize()
            losses.append(float(loss.detach()))
            timings.append(time.perf_counter()-tick)
        assert not torch.equal(before,head.weight)
        assert ema.updates==3
        for name,module in frozen.items():
            for key,value in module.state_dict().items():
                torch.testing.assert_close(value.cpu(),frozen_before[name][key],rtol=0,atol=0)
            assert all(parameter.grad is None for parameter in module.parameters())
        check_labels,check_probs,check_indices=evaluate(ema.model,
                DataLoader(Subset(validation,range(32)),batch_size=32),pipeline,config['model'],'cuda',precision='float32')
        assert np.isfinite(check_probs).all()
        np.testing.assert_allclose(check_probs.sum(1),1.,atol=1e-6)
        np.testing.assert_array_equal(check_indices,validation.indices[:32])
        result=dict(name=config['name'],status='passed',initial_ensemble_accuracy=initial_accuracy,
                    losses=losses,loss_parts=loss_parts,frozen_modules=list(frozen),
                    seconds_per_microbatch=float(np.median(timings[1:])),
                    peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                    elapsed_seconds=time.perf_counter()-started,ema_updates=ema.updates,
                    config_sha256=file_sha256(f'experiments/configs/{config["name"]}.json'))
        results.append(result)
        atomic_json('artifacts/refine_integration_checks.json',dict(manifest_sha256=manifest,
                    split_pixel_overlap=0,paired_config_differences=sorted(differences),runs=results))
        print(result,flush=True)
        del model,optimizer,ema,head,before,images,labels,targets,logits,loss,ce_loss,frozen,frozen_before
        gc.collect()
        torch.cuda.empty_cache()
    print('All refinement GPU checks passed. Temporary updates were discarded.',flush=True)


if __name__=='__main__':
    main()
