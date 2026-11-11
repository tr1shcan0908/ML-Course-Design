"""Train the controlled six-class FER experiments and retain validation evidence."""
import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import FERDataset, ImagePipeline
from .io import atomic_json, atomic_output, update_status
from .losses import eac_consistency, nla_loss
from .models import make_model
from .refinement import ReplacementReference, freeze_modules


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def atomic_checkpoint(path, value):
    with atomic_output(path) as temp:
        torch.save(value, temp)


class EMA:
    def __init__(self, model):
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(.999, (1+self.updates)/(10+self.updates))
        original = model.state_dict()
        averaged, current = [], []
        for key, target in self.model.state_dict().items():
            source = original[key].detach()
            if target.is_floating_point():
                averaged.append(target)
                current.append(source)
            else:
                target.copy_(source)
        torch._foreach_lerp_(averaged, current, 1-decay)


def classification_metrics(labels, probabilities, classes):
    prediction = probabilities.argmax(1)
    return dict(accuracy=float(accuracy_score(labels,prediction)),
                macro_f1=float(f1_score(labels,prediction,labels=range(6),average='macro',zero_division=0)),
                confusion_matrix=confusion_matrix(labels,prediction,labels=range(6)).tolist(),
                per_class=classification_report(labels,prediction,labels=range(6),target_names=classes,
                                                output_dict=True,zero_division=0))


@contextmanager
def evaluation_precision(precision):
    if precision not in ('bf16','float32'):
        raise ValueError(f'Unknown evaluation precision: {precision}')
    if precision=='bf16':
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(benchmark=False,deterministic=True,allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@torch.inference_mode()
def evaluate(model, loader, pipeline, model_name, device, tta=False, precision='bf16'):
    model.eval()
    probabilities, targets, indices = [], [], []
    with evaluation_precision(precision):
        for raw, labels, row_indices in loader:
            images = pipeline(raw.to(device, non_blocking=True), training=False)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision=='bf16'):
                logits = model(images,None,None,None) if model_name=='qcs' else model(images)
                probs = logits.float().softmax(1)
                if tta:
                    flipped = images.flip(-1)
                    logits2 = model(flipped,None,None,None) if model_name=='qcs' else model(flipped)
                    probs = .5*(probs+logits2.float().softmax(1))
            probabilities.append(probs.cpu().numpy())
            targets.append(labels.numpy())
            indices.append(row_indices.numpy())
    return np.concatenate(targets), np.concatenate(probabilities), np.concatenate(indices)


def build_optimizer(model, config):
    if config['model'] == 'emotieff':
        groups = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            scale = config.get('head_lr_multiplier',1.) if name.startswith('backbone.classifier.') else 1.
            decay = 0. if parameter.ndim <= 1 or name.endswith('.bias') else config['weight_decay']
            group = groups.setdefault((scale,decay),dict(params=[],lr=config['lr']*scale,
                                                        lr_scale=scale,weight_decay=decay))
            group['params'].append(parameter)
        return torch.optim.AdamW(list(groups.values()),lr=config['lr'])
    if config['model'] == 'qcs':
        from third_party.qcs.sam import SAM
        return SAM(model.parameters(), torch.optim.Adam, lr=config['lr'], rho=.05,
                   weight_decay=config.get('weight_decay',0))
    if 'layer_decay' not in config:
        return torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    groups = {}
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        layer = int(name.split('.')[1]) if name.startswith(('stages.','downsample_layers.')) else 4
        scale = config['layer_decay']**(4-layer)
        decay = 0. if parameter.ndim<=1 or name.endswith('.bias') else config['weight_decay']
        group = groups.setdefault((layer,decay),dict(params=[],lr=config['lr']*scale,
                                                    lr_scale=scale,weight_decay=decay))
        group['params'].append(parameter)
    return torch.optim.AdamW(list(groups.values()),lr=config['lr'])


def mixup_batch(images, labels, alpha, probability):
    if alpha<=0 or len(labels)<2 or np.random.random()>=probability:
        return images, labels
    weight = float(np.random.beta(alpha,alpha))
    permutation = torch.randperm(len(labels),device=labels.device)
    targets = F.one_hot(labels,num_classes=6).float()
    return (images*weight+images[permutation]*(1-weight),
            targets*weight+targets[permutation]*(1-weight))


def selection_key(metrics, selection_metric):
    if selection_metric=='validation_accuracy':
        return metrics['accuracy'],metrics['macro_f1']
    if selection_metric=='validation_macro_f1':
        return metrics['macro_f1'],metrics['accuracy']
    raise ValueError(f'Unknown selection metric: {selection_metric}')


def save_best(run, model, config, epoch, metrics, labels, probabilities, indices, classes,
              selection_metrics=None):
    selection_metrics = selection_metrics or metrics
    atomic_checkpoint(run/'best.pt',dict(model=model.state_dict(),config=config,
                      epoch=epoch,metrics=metrics,classes=classes,selection_metrics=selection_metrics))
    with atomic_output(run/'val_predictions.npz') as temp:
        with temp.open('wb') as file:
            np.savez_compressed(file,labels=labels,probabilities=probabilities,indices=indices)
    atomic_json(run/'best_metrics.json',dict(epoch=epoch,**metrics,selection_metrics=selection_metrics))


def forward_loss(model, images, labels, config, epoch, negative_labels=None):
    if config.get('loss') == 'eac':
        logits, attention = model(images,return_attention=True)
        # The unused flipped classifier must not draw a second dropout mask:
        # keep the augmentation/dropout random stream aligned with the CE run.
        flipped_attention = model(images.flip(-1),attention_only=True)
        loss = F.cross_entropy(logits.float(),labels,label_smoothing=config['label_smoothing'])
        return loss + config['eac_weight'] * eac_consistency(attention,flipped_attention), logits
    if config['model'] == 'qcs':
        outputs = model(images[0],images[1],images[2],images[3])
        all_labels = [labels,labels,negative_labels,negative_labels]*2
        loss = sum(F.cross_entropy(logits.float(),target) for logits,target in zip(outputs,all_labels))/8
        return loss, outputs[0]
    logits, regularizer = model(images,return_aux=True)
    if config['loss'] == 'nla':
        flipped = model(images.flip(-1))
        loss = nla_loss(logits,flipped,labels,epoch,config['epochs'],config['effective_batch_size'])
    else:
        loss = F.cross_entropy(logits.float(),labels,label_smoothing=config['label_smoothing'])
    return loss + regularizer.float(), logits


def check_gradient_finite(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f'Nonfinite gradient in {name}')


def train(config_path, resume=False):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    run = Path(config['output'])
    if (run/'metrics.jsonl').exists() and not resume:
        raise FileExistsError(f'Existing experiment: {run}; use --resume or a new run name.')
    run.mkdir(parents=True,exist_ok=True)
    seed_everything(config['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for this experiment budget.')
    device = torch.device('cuda:0')
    training = FERDataset(config['data'],'train',quadruplets=config['model']=='qcs')
    validation = FERDataset(config['data'],'val')
    manifest_bytes = (Path(config['data'])/'manifest.jsonl').read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == training.split['manifest_sha256']
    config['manifest_sha256'] = training.split['manifest_sha256']
    atomic_json(run/'config.json',config)
    loader_generator = torch.Generator().manual_seed(config['seed'])
    train_loader = DataLoader(training,batch_size=config['batch_size'],shuffle=True,
                              num_workers=0,pin_memory=True,drop_last=False,generator=loader_generator)
    val_loader = DataLoader(validation,batch_size=config.get('eval_batch_size',64),shuffle=False,
                            num_workers=0,pin_memory=True)
    pipeline = ImagePipeline(device,size=config.get('image_size',224),augmentation=config.get('augmentation'))
    source_path = config.get('finetune_from')
    model, pretrained_info = make_model(config,load_pretrained=not (source_path or resume))
    if config['model']=='emotieff' and pretrained_info['classes']!=training.split['classes']:
        raise RuntimeError('Training class order differs from the mapped EmotiEff classifier.')
    if source_path and not resume:
        source = torch.load(source_path,map_location='cpu',weights_only=True)
        if source['config']['manifest_sha256']!=config['manifest_sha256'] or source['classes']!=training.split['classes']:
            raise RuntimeError('Fine-tuning checkpoint has a different data manifest or label mapping.')
        model.load_state_dict(source['model'],strict=True)
        pretrained_info.update(finetune_from=source_path,source_epoch=source['epoch'],
                               checkpoint_sha256=hashlib.sha256(Path(source_path).read_bytes()).hexdigest())
        del source
    frozen_modules = freeze_modules(model,config)
    reference = None
    if config.get('selection_reference'):
        if config.get('eval_precision') != 'float32' or config.get('eval_batch_size') != 32:
            raise ValueError('Replacement selection requires FP32 evaluation with batch size 32.')
        spec = config['selection_reference']
        if not source_path or Path(source_path).resolve() != Path(spec['replaces_checkpoint']).resolve():
            raise ValueError('Replacement parent must match the fine-tuning source.')
        reference = ReplacementReference(spec,validation.indices,validation.labels,
                                         validation.split['classes'],config['manifest_sha256'])
    def selection_metrics_for(labels,probs,indices,metrics):
        if reference is None:
            return metrics
        return classification_metrics(labels,reference.combine(labels,probs,indices),training.split['classes'])
    model = model.to(device=device,memory_format=torch.channels_last)
    optimizer = build_optimizer(model,config)
    ema = EMA(model) if config.get('ema',False) else None
    parameters = sum(p.numel() for p in model.parameters())
    atomic_json(run/'initialization.json',dict(pretrained=pretrained_info,parameters=parameters,
                gpu=torch.cuda.get_device_name(0),torch=torch.__version__,numpy=np.__version__,
                config_file=str(config_path),started_at=datetime.now(timezone.utc).isoformat()))
    start_epoch, best_f1, best_acc, best_epoch = 0, -1., -1., -1
    best_selection = dict(accuracy=-1.,macro_f1=-1.)
    selection_scope = 'replacement_ensemble' if reference is not None else 'single_model'
    total_steps = 0
    bad_epochs = 0
    selection_metric = config.get('selection_metric','validation_macro_f1')
    eval_precision = config.get('eval_precision','bf16')
    selection_key(dict(accuracy=0.,macro_f1=0.),selection_metric)
    if resume:
        checkpoint = torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        if checkpoint['config']['manifest_sha256'] != config['manifest_sha256']:
            raise RuntimeError('Cannot resume with a different manifest.')
        if checkpoint['config'] != config:
            raise RuntimeError('Resume configuration changed; create a new fine-tuning run instead.')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if ema is not None:
            ema.model.load_state_dict(checkpoint['ema'])
            ema.updates = checkpoint['ema_updates']
        start_epoch=checkpoint['epoch']+1
        best_f1,best_acc,best_epoch=checkpoint['best_f1'],checkpoint['best_acc'],checkpoint['best_epoch']
        best_selection=checkpoint.get('best_selection',dict(accuracy=best_acc,macro_f1=best_f1))
        total_steps=checkpoint['total_steps']
        bad_epochs=checkpoint.get('bad_epochs',0)
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        loader_generator.set_state(checkpoint['loader_rng'])
        del checkpoint
    elif source_path or config.get('evaluate_initial',False):
        update_status(run/'status.json',dict(status='initial_evaluation',run=config['name'],
                      eval_precision=eval_precision))
        labels,probs,indices=evaluate(ema.model if ema is not None else model,val_loader,pipeline,
                                      config['model'],device,precision=eval_precision)
        metrics=classification_metrics(labels,probs,training.split['classes'])
        best_selection=selection_metrics_for(labels,probs,indices,metrics)
        if reference is not None and abs(best_selection['accuracy']-config['selection_reference']['baseline_accuracy'])>1e-12:
            raise RuntimeError('Initial replacement ensemble does not reproduce the reference accuracy.')
        best_f1,best_acc,best_epoch=metrics['macro_f1'],metrics['accuracy'],0
        save_best(run,ema.model if ema is not None else model,config,0,metrics,
                  labels,probs,indices,training.split['classes'],best_selection)
        atomic_json(run/'initial_metrics.json',dict(**metrics,selection_metrics=best_selection))
        print(json.dumps(dict(event='initial_evaluation',run=config['name'],accuracy=best_acc,
                              macro_f1=best_f1,selection_accuracy=best_selection['accuracy'],
                              selection_scope=selection_scope)),flush=True)
    started = time.time()
    completed_epochs = start_epoch
    stopped_early = False
    print(json.dumps(dict(event='start',run=config['name'],train=len(training),val=len(validation),
                          parameters=parameters,start_epoch=start_epoch)),flush=True)
    try:
        for epoch in range(start_epoch,config['epochs']):
            model.train()
            for module in frozen_modules.values():
                module.eval()
            warmup = min(1.,(epoch+1)/config.get('warmup_epochs',1))
            if config['model']=='qcs':
                lr = config['lr'] * .98**epoch
            else:
                fraction=max(0.,(epoch-config.get('warmup_epochs',1)+1)/max(1,config['epochs']-config.get('warmup_epochs',1)))
                lr=config['lr']*warmup*(.05+.95*.5*(1+math.cos(math.pi*fraction)))
            for group in optimizer.param_groups:
                group['lr']=lr*group.get('lr_scale',1.)
            epoch_start=time.time()
            loss_sum=correct=count=0
            optimizer.zero_grad(set_to_none=True)
            accumulation=config.get('accumulation_steps',1)
            torch.cuda.reset_peak_memory_stats()
            for batch_index,batch in enumerate(train_loader):
                if config['model']=='qcs':
                    raw,labels,negative_labels=batch
                    labels=labels.to(device,non_blocking=True)
                    negative_labels=negative_labels.to(device,non_blocking=True)
                    raw=raw.to(device,non_blocking=True)
                    images=[pipeline(raw[:,i],training=True) for i in range(4)]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss,logits=forward_loss(model,images,labels,config,epoch,negative_labels)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite QCS first-step loss')
                    loss.backward()
                    if batch_index==0: check_gradient_finite(model)
                    optimizer.first_step(zero_grad=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        sam_loss,_=forward_loss(model,images,labels,config,epoch,negative_labels)
                    if not torch.isfinite(sam_loss): raise FloatingPointError('Nonfinite QCS SAM loss')
                    sam_loss.backward()
                    if batch_index==0: check_gradient_finite(model)
                    optimizer.second_step(zero_grad=True)
                    total_steps+=1
                else:
                    raw,labels,_=batch
                    labels=labels.to(device,non_blocking=True)
                    images=pipeline(raw.to(device,non_blocking=True),training=True)
                    images,loss_labels=mixup_batch(images,labels,config.get('mixup_alpha',0.),
                                                   config.get('mixup_probability',0.))
                    group_start=(batch_index//accumulation)*accumulation
                    group_samples=min(config['batch_size']*accumulation,len(training)-group_start*config['batch_size'])
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss,logits=forward_loss(model,images,loss_labels,config,epoch)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                    (loss*len(labels)/group_samples).backward()
                    if (batch_index+1)%accumulation==0 or batch_index+1==len(train_loader):
                        if batch_index<accumulation: check_gradient_finite(model)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        total_steps+=1
                        if ema is not None: ema.update(model)
                loss_sum+=loss.item()*len(labels)
                prediction=logits.detach().argmax(1)
                if config['model']!='qcs' and loss_labels.ndim==2:
                    correct+=loss_labels.gather(1,prediction[:,None]).sum().item()
                else:
                    correct+=prediction.eq(labels).sum().item()
                count+=len(labels)
                if batch_index==0 or (batch_index+1)%100==0:
                    status=dict(status='training',run=config['name'],epoch=epoch+1,epochs=config['epochs'],
                                batch=batch_index+1,batches=len(train_loader),train_loss=loss_sum/count,
                                train_accuracy=correct/count,epoch_seconds=time.time()-epoch_start,
                                best_val_macro_f1=best_f1,best_val_accuracy=best_acc,
                                best_selection_accuracy=best_selection['accuracy'],selection_scope=selection_scope,
                                updated_at=datetime.now(timezone.utc).isoformat())
                    update_status(run/'status.json',status)
                    print(json.dumps(status),flush=True)
            eval_model=ema.model if ema is not None else model
            labels,probs,indices=evaluate(eval_model,val_loader,pipeline,config['model'],device,
                                          precision=eval_precision)
            metrics=classification_metrics(labels,probs,training.split['classes'])
            scoring=selection_metrics_for(labels,probs,indices,metrics)
            record=dict(epoch=epoch+1,lr=lr,train_loss=loss_sum/count,train_accuracy=correct/count,
                        val_accuracy=metrics['accuracy'],val_macro_f1=metrics['macro_f1'],
                        selection_accuracy=scoring['accuracy'],selection_macro_f1=scoring['macro_f1'],
                        selection_scope=selection_scope,
                        epoch_seconds=time.time()-epoch_start,total_seconds=time.time()-started,
                        peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                        train_accuracy_kind='mixup_weighted' if config.get('mixup_alpha',0.) else 'hard_labels')
            with (run/'metrics.jsonl').open('a',encoding='utf-8') as file:
                file.write(json.dumps(record)+'\n')
            is_best=selection_key(scoring,selection_metric)>selection_key(best_selection,selection_metric)
            bad_epochs=0 if is_best else bad_epochs+1
            if is_best:
                best_f1,best_acc,best_epoch=metrics['macro_f1'],metrics['accuracy'],epoch+1
                best_selection=scoring
                save_best(run,eval_model,config,epoch+1,metrics,labels,probs,indices,training.split['classes'],scoring)
            checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,
                            epoch=epoch,best_f1=best_f1,best_acc=best_acc,best_epoch=best_epoch,total_steps=total_steps,
                            bad_epochs=bad_epochs,best_selection=best_selection,
                            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=loader_generator.get_state())
            if ema is not None: checkpoint.update(ema=ema.model.state_dict(),ema_updates=ema.updates)
            atomic_checkpoint(run/'last.pt',checkpoint)
            del checkpoint
            update_status(run/'status.json',dict(status='epoch_complete',run=config['name'],**record,
                        best_epoch=best_epoch,best_val_accuracy=best_acc,best_val_macro_f1=best_f1,
                        best_selection_accuracy=best_selection['accuracy']))
            print(json.dumps(dict(event='epoch_complete',run=config['name'],**record,is_best=is_best)),flush=True)
            completed_epochs=epoch+1
            patience=config.get('early_stopping_patience',0)
            if patience and bad_epochs>=patience and completed_epochs>=config.get('min_epochs',0):
                stopped_early=True
                print(json.dumps(dict(event='early_stopping',epoch=completed_epochs,best_epoch=best_epoch)),flush=True)
                break
        update_status(run/'status.json',dict(status='complete',run=config['name'],epochs=config['epochs'],
                    best_epoch=best_epoch,best_val_accuracy=best_acc,best_val_macro_f1=best_f1,
                    best_selection_accuracy=best_selection['accuracy'],selection_scope=selection_scope,
                    epochs_completed=completed_epochs,stopped_early=stopped_early,selection_metric=selection_metric,
                    total_seconds=time.time()-started,finished_at=datetime.now(timezone.utc).isoformat()),required=True)
        print(json.dumps(dict(event='complete',run=config['name'],best_epoch=best_epoch,
                              best_val_accuracy=best_acc,best_val_macro_f1=best_f1)),flush=True)
    except BaseException as error:
        update_status(run/'status.json',dict(status='failed',run=config['name'],error=repr(error),
                    updated_at=datetime.now(timezone.utc).isoformat()))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    train(args.config,args.resume)


if __name__=='__main__':
    main()
