"""Fixed five-crop inference around held-out coarse predictions, without refitting."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
from pathlib import Path

import numpy as np
import torch

from .data import ROOT, FoveaDataset
from .experiment import load_checkpoint, loader, predict, read_json, score_records, write_csv, write_json


def crop_offsets(roi):
    delta=roi/8
    return np.array([[0,0],[-delta,0],[delta,0],[0,-delta],[0,delta]],dtype=float)


def average_crops(predictions):
    ids=[r['id'] for r in predictions[0]]
    if any([r['id'] for r in rows]!=ids for rows in predictions):
        raise ValueError('Crop prediction IDs are not aligned')
    points=np.array([[[r['pred_x'],r['pred_y']] for r in rows] for rows in predictions])
    mean=points.mean(axis=0)
    spread=np.sqrt(((points-mean[None])**2).sum(axis=-1).mean(axis=0))
    return [dict(id=r['id'],width=r['width'],height=r['height'],x=r['x'],y=r['y'],
                 pred_x=float(mean[i,0]),pred_y=float(mean[i,1]),crop_dispersion_rms=float(spread[i]))
            for i,r in enumerate(predictions[0])]


def shifted_predictions(model,rows,anchors,config,device,batch_size):
    predictions=[]
    for shift in crop_offsets(config['roi']):
        shifted={key:np.asarray(value)+shift for key,value in anchors.items()}
        dataset=FoveaDataset(rows,config['size'],'fine',False,config['roi'],shifted)
        predictions.append(predict(model,loader(dataset,batch_size),device,config['size'],tta=False))
        del dataset
    return average_crops(predictions)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args()
    source=Path(args.run).resolve()
    output=Path(args.output).resolve()
    config=read_json(source/'config.json')
    if config['stage']!='fine': raise ValueError('This experiment requires a trained fine model')
    if not read_json(source/'metrics.json')['complete_five_fold']: raise ValueError('Finish source five folds')
    manifest=read_json(ROOT/config['manifest'])
    train_rows=[r for r in manifest['records'] if r['split']=='train']
    test_rows=[r for r in manifest['records'] if r['split']=='test']
    experiment=dict(source_run=args.run,dataset_hash=manifest['dataset_hash'],variant='pred',
                    offsets=crop_offsets(config['roi']).tolist(),aggregation='equal mean in original pixels',
                    source_checkpoint_sha256={str(f):hashlib.sha256((source/f'fold_{f}'/'best.pth').read_bytes()).hexdigest() for f in range(5)})
    if (output/'config.json').exists() and read_json(output/'config.json')!=experiment:
        raise ValueError('Inference configuration differs from existing output')
    write_json(output/'config.json',experiment)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    all_validation=[]
    all_test=[]
    for fold in range(5):
        directory=output/f'fold_{fold}'
        validation=[r for r in train_rows if r['fold']==fold]
        coarse_dir=ROOT/config['coarse_run']/f'fold_{fold}'
        with (coarse_dir/'val_predictions.csv').open(encoding='utf-8') as handle:
            anchors={int(r['id']):np.array([float(r['pred_x']),float(r['pred_y'])]) for r in csv.DictReader(handle)}
        assert set(anchors)=={r['id'] for r in validation}
        coarse,checkpoint=load_checkpoint(coarse_dir/'best.pth',device)
        assert checkpoint['fold']==fold and checkpoint['config']['dataset_hash']==manifest['dataset_hash']
        coarse_config=checkpoint['config']
        predicted_test=predict(coarse,loader(FoveaDataset(test_rows,coarse_config['size']),args.batch_size),
                               device,coarse_config['size'])
        test_anchors={r['id']:np.array([r['pred_x'],r['pred_y']]) for r in predicted_test}
        del coarse,checkpoint
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
        model,checkpoint=load_checkpoint(source/f'fold_{fold}'/'best.pth',device)
        assert checkpoint['fold']==fold and checkpoint['config']['dataset_hash']==manifest['dataset_hash']
        values=shifted_predictions(model,validation,anchors,config,device,args.batch_size)
        for row in values: row['fold']=fold
        write_csv(directory/'val_predictions.csv',values)
        all_validation.extend(values)
        test=shifted_predictions(model,test_rows,test_anchors,config,device,args.batch_size)
        for row in test: row['fold']=fold
        write_csv(directory/'test_predictions.csv',test)
        all_test.append(test)
        metrics=score_records(values)
        write_json(directory/'complete.json',dict(metrics=metrics,source_epoch=checkpoint['epoch'],trained=False))
        print(f"fold {fold}: five-crop RMSE {metrics['rmse']:.4f}",flush=True)
        del model,checkpoint
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
    all_validation.sort(key=lambda r:r['id'])
    write_csv(output/'oof_predictions.csv',all_validation)
    write_json(output/'metrics.json',dict(complete_five_fold=True,completed_folds=list(range(5)),
               selected_variant='pred',scores=dict(pred=score_records(all_validation)),
               selection_note='Fixed five-crop inference on previously selected checkpoints; no additional training.'))
    final_test=[]
    for i,row in enumerate(test_rows):
        assert all(values[i]['id']==row['id'] for values in all_test)
        point=np.mean([[values[i]['pred_x'],values[i]['pred_y']] for values in all_test],axis=0)
        point=np.clip(point,[0,0],[row['width']-1,row['height']-1])
        final_test.append(dict(id=row['id'],x=float(point[0]),y=float(point[1]),width=row['width'],height=row['height']))
    write_csv(output/'test_predictions.csv',final_test)
    lookup={r['id']:r for r in final_test}
    with (ROOT/'sample_submission.csv').open(encoding='utf-8-sig') as handle:
        template=list(csv.DictReader(handle))
    submission=[]
    for row in template:
        image_id,_,axis=row['ImageID'].split('_')
        submission.append(dict(ImageID=row['ImageID'],value=f"{lookup[int(image_id)][axis.lower()]:.6f}"))
    write_csv(output/'submission.csv',submission)
    write_json(output/'submission_metadata.json',dict(variant='pred',folds=5,rows=len(submission),
                visibility='No validated invisible class; crop disagreement is not a visibility score.'))
    print(f"Five-crop OOF RMSE: {score_records(all_validation)['rmse']:.4f}",flush=True)


if __name__=='__main__': main()
