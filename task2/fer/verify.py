"""Integration checks: split integrity, pretrained loading, loss parity, GPU steps."""
import gc
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from .data import FERDataset, ImagePipeline
from .losses import nla_loss
from .models import make_model
from .train import seed_everything, build_optimizer, forward_loss, check_gradient_finite, atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--only', choices=['qcs','all'], default='all')
    parser.add_argument('--qcs-batch', type=int, default=8)
    args_cli=parser.parse_args()
    seed_everything(42)
    directory=Path('artifacts/data')
    train=FERDataset(directory,'train')
    val=FERDataset(directory,'val')
    assert hashlib.sha256((directory/'manifest.jsonl').read_bytes()).hexdigest()==train.split['manifest_sha256']
    assert set(train.indices).isdisjoint(val.indices)
    assert {train.rows[i]['pixel_sha256'] for i in train.indices}.isdisjoint(
        {val.rows[i]['pixel_sha256'] for i in val.indices})
    assert set(train.labels)==set(range(6)) and set(val.labels)==set(range(6))
    from third_party.nla.loss import NAW, jensen_shannon_divergence
    logits=torch.randn(7,6,device='cuda',requires_grad=True)
    flipped=torch.randn(7,6,device='cuda',requires_grad=True)
    targets=torch.tensor([0,1,2,3,4,5,0],device='cuda')
    epoch,epochs=2,12
    args=SimpleNamespace(eps=1-np.exp(-15*epoch/epochs))
    original=NAW(args,.5,.5,.3,.15,
                 f_minor=.91**2-(.91/6)**2,f_major=.91**2+(.91/6)**2,
                 t_minor=-.8**2+(.8/2)**2,t_major=.8**2+(.8/2)**2)
    reference=.5*F.cross_entropy(logits,targets)+.5*original(logits,targets).mean()
    reference+=.5*jensen_shannon_divergence(logits,flipped)*32/len(targets)
    adapted=nla_loss(logits,flipped,targets,epoch,epochs)
    torch.testing.assert_close(adapted,reference,rtol=2e-5,atol=2e-5)
    adapted.backward()
    assert torch.isfinite(logits.grad).all()
    pipeline=ImagePipeline('cuda')
    results=[]
    for name,kind,loss_name,batch in [('convnext_tiny','convnext','ce',16),
                                     ('emonext_tiny','emonext','ce',16),
                                     ('emonext_nla','emonext','nla',16),
                                     ('qcs','qcs','ce',args_cli.qcs_batch)]:
        if args_cli.only=='qcs' and kind!='qcs':
            continue
        seed_everything(42)
        config=dict(name=name,model=kind,loss=loss_name,epochs=12,lr=1e-4,
                    effective_batch_size=32,label_smoothing=.2,weight_decay=1e-4,
                    pretrained='artifacts/weights/'+('ir50_msceleb.pth' if kind=='qcs' else 'convnext_tiny_22k_224.pth'))
        model,info=make_model(config)
        model=model.to(device='cuda',memory_format=torch.channels_last)
        optimizer=build_optimizer(model,config)
        labels=torch.tensor([i%6 for i in range(batch)],device='cuda')
        raw=torch.stack([train[i][0] for i in range(batch)]).to('cuda')
        images=pipeline(raw,training=True)
        negative=(labels+1)%6
        input_images=[images,images.flip(-1),images,images.flip(-1)] if kind=='qcs' else images
        torch.cuda.reset_peak_memory_stats()
        timings=[]
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss,logits=forward_loss(model,input_images,labels,config,0,negative)
            assert logits.shape==(batch,6) and torch.isfinite(loss)
            loss.backward()
            if step==0: check_gradient_finite(model)
            if kind=='qcs':
                optimizer.first_step(zero_grad=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    second,_=forward_loss(model,input_images,labels,config,0,negative)
                second.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()
            torch.cuda.synchronize()
            timings.append(time.perf_counter()-started)
        result=dict(name=name,batch_size=batch,pretrained=info,parameters=sum(p.numel() for p in model.parameters()),
                    seconds_per_step=float(np.median(timings[1:])),last_loss=loss.item(),
                    peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                    integration_status='passed')
        print(json.dumps(result),flush=True)
        results.append(result)
        atomic_json('artifacts/integration_checks'+('_qcs' if args_cli.only=='qcs' else '')+'.json',dict(split_overlap=0,nla_loss_parity='passed',runs=results))
        del model,optimizer,images,input_images,loss,logits
        if kind=='qcs': del second
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json('artifacts/integration_checks'+('_qcs' if args_cli.only=='qcs' else '')+'.json',dict(split_overlap=0,nla_loss_parity='passed',runs=results))
    print('All integration checks passed.',flush=True)


if __name__=='__main__':
    main()
