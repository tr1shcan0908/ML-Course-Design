"""Preflight on real train batches, without retaining these temporary updates."""
import argparse
import gc
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from .data import FERDataset, ImagePipeline
from .io import atomic_json, read_json
from .losses import eac_consistency
from .models import make_model
from .train import build_optimizer, check_gradient_finite, evaluate, forward_loss, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only',choices=['ce','eac'])
    args = parser.parse_args()
    plan = read_json('experiments/transfer_plan.json')
    training, validation = (FERDataset(plan['data'], split) for split in ('train','val'))
    assert set(training.indices).isdisjoint(validation.indices)
    assert {training.rows[i]['pixel_sha256'] for i in training.indices}.isdisjoint(
        {validation.rows[i]['pixel_sha256'] for i in validation.indices})
    manifest = hashlib.sha256((Path(plan['data'])/'manifest.jsonl').read_bytes()).hexdigest()
    assert manifest == training.split['manifest_sha256']
    configs = [read_json(path) for path in plan['configs']]
    ce, eac = configs[:2]
    different = {key for key in ce.keys() | eac.keys() if ce.get(key) != eac.get(key)}
    assert different == {'name','output','loss','eac_weight'}, different
    results = []
    for config in configs:
        if args.only and config['loss'] != args.only:
            continue
        seed_everything(config['seed'])
        model, _ = make_model(config)
        model = model.to(device='cuda',memory_format=torch.channels_last).train()
        optimizer = build_optimizer(model,config)
        pipeline = ImagePipeline('cuda',size=config['image_size'],augmentation=config['augmentation'])
        loader = DataLoader(Subset(training,range(config['batch_size']*5)),batch_size=config['batch_size'])
        before = model.backbone.classifier.weight.detach().clone()
        timings, losses = [], []
        torch.cuda.reset_peak_memory_stats()
        first_loss_parts = None
        for raw,labels,_ in loader:
            started = time.perf_counter()
            images, labels = pipeline(raw.to('cuda'),training=True), labels.to('cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss, logits = forward_loss(model,images,labels,config,0)
            if first_loss_parts is None:
                ce_value = F.cross_entropy(logits.float(),labels,label_smoothing=config['label_smoothing'])
                first_loss_parts = dict(ce=float(ce_value.detach()),weighted_consistency=float((loss-ce_value).detach()))
            assert torch.isfinite(loss) and logits.shape == (len(labels),6)
            loss.backward()
            check_gradient_finite(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            timings.append(time.perf_counter()-started)
            losses.append(float(loss.detach()))
        assert not torch.equal(before,model.backbone.classifier.weight)
        labels,probs,indices = evaluate(model,DataLoader(Subset(validation,range(32)),batch_size=32),
                                        pipeline,config['model'],'cuda',precision='float32')
        assert np.isfinite(probs).all()
        np.testing.assert_allclose(probs.sum(1),1.,atol=1e-6)
        np.testing.assert_array_equal(indices,validation.indices[:32])
        result = dict(name=config['name'],status='passed',initial_loss_parts=first_loss_parts,
                      losses=losses,seconds_per_microbatch=float(np.median(timings[1:])),
                      peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                      parameters=sum(p.numel() for p in model.parameters()))
        results.append(result)
        print(result,flush=True)
        output = f'artifacts/transfer_integration_{args.only}.json' if args.only else 'artifacts/transfer_integration_checks.json'
        atomic_json(output,dict(manifest_sha256=manifest,
                    split_pixel_overlap=0,paired_config_differences=sorted(different),runs=results))
        del model, optimizer, images, labels, logits, loss, before, ce_value
        gc.collect()
        torch.cuda.empty_cache()
    print('All transfer GPU checks passed.',flush=True)


if __name__ == '__main__':
    main()
