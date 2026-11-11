"""Exercise every planned fine-tuning configuration on real training batches."""
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import FERDataset, ImagePipeline
from .io import atomic_json, read_json
from .models import make_model
from .train import build_optimizer, check_gradient_finite, evaluate, forward_loss, mixup_batch, seed_everything


def main():
    plan=read_json('experiments/sft_plan.json')
    data=FERDataset(plan['data'],'train')
    val=FERDataset(plan['data'],'val')
    assert set(data.indices).isdisjoint(val.indices)
    assert {data.rows[i]['pixel_sha256'] for i in data.indices}.isdisjoint({val.rows[i]['pixel_sha256'] for i in val.indices})
    manifest=hashlib.sha256((Path(plan['data'])/'manifest.jsonl').read_bytes()).hexdigest()
    assert manifest==data.split['manifest_sha256']
    results=[]
    for path in plan['configs']:
        config=read_json(path)
        seed_everything(config['seed'])
        model,info=make_model(config,load_pretrained=not bool(config.get('finetune_from')))
        if config.get('finetune_from'):
            source=torch.load(config['finetune_from'],map_location='cpu',weights_only=True)
            assert source['config']['manifest_sha256']==manifest
            assert source['classes']==data.split['classes']
            model.load_state_dict(source['model'],strict=True)
            del source
        for stage in range(config.get('freeze_stages',0)):
            model.stages[stage].requires_grad_(False)
            model.downsample_layers[stage].requires_grad_(False)
        model=model.to(device='cuda',memory_format=torch.channels_last).train()
        optimizer=build_optimizer(model,config)
        pipeline=ImagePipeline('cuda',augmentation=config.get('augmentation'))
        loader=DataLoader(Subset(data,range(config['batch_size']*4)),batch_size=config['batch_size'])
        before=model.head.weight.detach().clone()
        frozen_before=model.downsample_layers[0][0].weight.detach().clone()
        torch.cuda.reset_peak_memory_stats()
        timings=[]
        for raw,labels,_ in loader:
            started=time.perf_counter()
            labels=labels.to('cuda')
            images=pipeline(raw.to('cuda'),training=True)
            images,targets=mixup_batch(images,labels,config.get('mixup_alpha',0.),1.)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss,logits=forward_loss(model,images,targets,config,0)
            assert torch.isfinite(loss) and logits.shape==(len(labels),6)
            loss.backward()
            check_gradient_finite(model)
            optimizer.step()
            torch.cuda.synchronize()
            timings.append(time.perf_counter()-started)
        assert not torch.equal(before,model.head.weight)
        if config.get('freeze_stages',0):
            torch.testing.assert_close(frozen_before,model.downsample_layers[0][0].weight,rtol=0,atol=0)
            assert model.downsample_layers[0][0].weight.grad is None
        val_loader=DataLoader(Subset(val,range(16)),batch_size=16)
        y,probs,indices=evaluate(model,val_loader,pipeline,config['model'],'cuda',precision='float32')
        assert np.isfinite(probs).all()
        np.testing.assert_allclose(probs.sum(1),1.,atol=1e-6)
        np.testing.assert_array_equal(indices,val.indices[:16])
        result=dict(name=config['name'],status='passed',pretrained=info,
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    seconds_per_microbatch=float(np.median(timings[1:])),
                    peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/2**30)
        results.append(result)
        print(json.dumps(result),flush=True)
        atomic_json('artifacts/sft_integration_checks.json',dict(manifest_sha256=manifest,split_overlap=0,runs=results))
        del model,optimizer,images,targets,loss,logits,before,frozen_before
        gc.collect()
        torch.cuda.empty_cache()
    print('All fine-tuning GPU checks passed.',flush=True)


if __name__=='__main__':
    main()
