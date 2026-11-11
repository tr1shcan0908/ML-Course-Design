"""Reproducible grouped validation, paired coarse/fine inference, and submission export."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ROOT, FoveaDataset, prepare
from .model import FoveaNet, decode, localization_loss, to_original


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("w",encoding="utf-8") as handle:
        handle.write(json.dumps(value,ensure_ascii=False,indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary,path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary,path)


def metric(target, prediction):
    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite predictions")
    error = prediction-target
    radius = np.linalg.norm(error, axis=1)
    return dict(rmse=float(np.sqrt(np.mean(error**2))),
                rmse_x=float(np.sqrt(np.mean(error[:,0]**2))),
                rmse_y=float(np.sqrt(np.mean(error[:,1]**2))),
                aed=float(radius.mean()), median_distance=float(np.median(radius)),
                p95_distance=float(np.percentile(radius,95)), max_distance=float(radius.max()),
                count=len(target))


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key,value in batch.items()}


def coordinates(output, size, local=False):
    return (output["coordinate"].float() if "coordinate" in output else
            decode(output["heatmap"], size, local=local))


@torch.inference_mode()
def predict(model, loader, device, size, tta=False):
    model.eval()
    results = []
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(batch["image"])
        variants = {"pred": coordinates(output, size),
                    "local": coordinates(output, size, local=True)}
        if tta:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                flipped = model(batch["image"].flip(-1))
            for name, use_local in (("pred",False),("local",True)):
                coordinate = coordinates(flipped, size, local=use_local)
                coordinate[:,0] = size-1-coordinate[:,0]
                variants["tta" if name=="pred" else "local_tta"] = (variants[name]+coordinate)/2
        variants = {name: to_original(value,batch["inverse"]).cpu().numpy()
                    for name,value in variants.items()}
        for index, image_id in enumerate(cpu_batch["id"].tolist()):
            row = dict(id=image_id, width=int(cpu_batch["width"][index]),
                       height=int(cpu_batch["height"][index]),
                       x=float(cpu_batch["original"][index,0]),
                       y=float(cpu_batch["original"][index,1]))
            for name, values in variants.items():
                row[f"{name}_x"], row[f"{name}_y"] = map(float, values[index])
            results.append(row)
    return results


def score_records(records, variant="pred"):
    return metric([[r["x"],r["y"]] for r in records],
                  [[r[f"{variant}_x"],r[f"{variant}_y"]] for r in records])


def loader(dataset, batch_size, shuffle=False):
    # Windows workers duplicate the Python process and are expensive on a busy 16GB laptop.
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=0,pin_memory=False)


def load_checkpoint(path, device):
    checkpoint = torch.load(path,map_location="cpu",weights_only=False)
    config = checkpoint["config"]
    model = FoveaNet(config["architecture"],pretrained=False,auxiliary=config["region_weight"]>0)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), checkpoint


def train(args):
    manifest = read_json(args.manifest)
    rows = [r for r in manifest["records"] if r["split"]=="train"]
    if args.stage=="coarse" and args.size!=manifest["size"]:
        raise ValueError("Prepare a manifest with the requested coarse resolution")
    if args.stage=="fine" and not args.coarse_run:
        raise ValueError("Fine training requires a coarse run for predicted validation anchors")
    run = Path(args.run).resolve()
    run.mkdir(parents=True,exist_ok=True)
    config = vars(args).copy()
    config["dataset_hash"] = manifest["dataset_hash"]
    config["pretraining"] = "torchvision ResNet18 ImageNet1K V1; no retinal task weights"
    signature_payload = {k:v for k,v in config.items() if k not in ("folds", "command")}
    if signature_payload.get("heatmap_loss") == "gaussian":
        signature_payload.pop("heatmap_loss")  # Keep original experiment signatures compatible.
    signature = hashlib.sha256(json.dumps(signature_payload,sort_keys=True).encode()).hexdigest()
    if (run/"config.json").exists():
        previous = read_json(run/"config.json")
        if previous["signature"]!=signature:
            raise ValueError("Run directory belongs to a different configuration; choose a new --run")
    config["signature"]=signature
    config["source_sha256"] = {path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted(Path(__file__).parent.glob("*.py"))}
    write_json(run/"config.json",config)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.set_num_threads(args.threads)
    if device.type=="cuda":
        torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True
    print(json.dumps(dict(device=str(device),gpu=torch.cuda.get_device_name() if device.type=="cuda" else None,
                          run=str(run),stage=args.stage,size=args.size)),flush=True)
    for fold in args.folds:
        fold_dir = run/f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        if (fold_dir/"complete.json").exists():
            print(f"fold {fold}: already complete",flush=True)
            continue
        seed_all(args.seed+fold)
        fit = [r for r in rows if r["fold"]!=fold]
        validation = [r for r in rows if r["fold"]==fold]
        anchors = None
        warm_checkpoint = None
        if args.stage=="fine":
            coarse_dir=Path(args.coarse_run)/f"fold_{fold}"
            warm_checkpoint=torch.load(coarse_dir/"best.pth",map_location="cpu",weights_only=False)
            if warm_checkpoint["fold"]!=fold or warm_checkpoint["config"]["dataset_hash"]!=manifest["dataset_hash"]:
                raise ValueError("Coarse fold or dataset mismatch")
            with (coarse_dir/"val_predictions.csv").open(encoding="utf-8") as handle:
                coarse_predictions=list(csv.DictReader(handle))
            anchors={int(r["id"]):np.array([float(r["pred_x"]),float(r["pred_y"])]) for r in coarse_predictions}
            if set(anchors)!={r["id"] for r in validation}:
                raise ValueError("Fine validation anchors are not the matching held-out coarse predictions")
        train_set=FoveaDataset(fit,args.size,args.stage,True,args.roi,auxiliary=args.region_weight>0)
        val_set=FoveaDataset(validation,args.size,args.stage,False,args.roi,anchors)
        train_loader=loader(train_set,args.batch_size,True)
        val_loader=loader(val_set,args.batch_size)
        model=FoveaNet(args.architecture,pretrained=warm_checkpoint is None,
                       auxiliary=args.region_weight>0)
        if warm_checkpoint is not None:
            model.load_state_dict(warm_checkpoint["state_dict"],strict=True)
            del warm_checkpoint
        model.to(device)
        encoder_parameters=list(model.encoder.parameters())
        encoder_ids={id(p) for p in encoder_parameters}
        head_parameters=[p for p in model.parameters() if id(p) not in encoder_ids]
        optimizer=torch.optim.AdamW([{"params":encoder_parameters,"lr":args.encoder_lr},
                                     {"params":head_parameters,"lr":args.lr}],weight_decay=args.weight_decay)
        def learning_rate_factor(epoch):
            if epoch<3:
                return 0.25+0.75*(epoch+1)/3
            progress=(epoch-3)/max(args.epochs-3,1)
            return 0.05+0.95*(1+math.cos(math.pi*progress))/2
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,learning_rate_factor)
        scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
        best, best_epoch, stale = float("inf"), 0, 0
        start=time.monotonic()
        epoch_log=(fold_dir/"epochs.jsonl").open("w",encoding="utf-8",buffering=1)
        initial=score_records(predict(model,val_loader,device,args.size))
        print(f"fold {fold}: fit={len(fit)} validation={len(validation)} initial_rmse={initial['rmse']:.3f}",flush=True)
        for epoch in range(1,args.epochs+1):
            model.train()
            loss_sum=0
            for cpu_batch in train_loader:
                batch=move_batch(cpu_batch,device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
                    output=model(batch["image"])
                loss,_=localization_loss(output,batch,args.size,args.sigma,
                                         args.coordinate_weight,args.region_weight,args.heatmap_loss)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss in fold {fold} epoch {epoch}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),5)
                scaler.step(optimizer)
                scaler.update()
                loss_sum+=float(loss.detach())*len(cpu_batch["id"])
            scheduler.step()
            validation_records=predict(model,val_loader,device,args.size)
            metrics=score_records(validation_records)
            improved=metrics["rmse"]<best-args.min_delta
            if improved:
                best,best_epoch,stale=metrics["rmse"],epoch,0
                temporary = fold_dir/"best.tmp.pth"
                with temporary.open("wb") as handle:
                    torch.save(dict(state_dict=model.state_dict(),config=config,fold=fold,
                                    epoch=epoch,metrics=metrics),handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary,fold_dir/"best.pth")
            else:
                stale+=1
            entry=dict(epoch=epoch,train_loss=loss_sum/len(fit),**metrics,
                       local_rmse=score_records(validation_records,"local")["rmse"],
                       best_rmse=best,best_epoch=best_epoch,seconds=round(time.monotonic()-start,1))
            epoch_log.write(json.dumps(entry)+"\n")
            epoch_log.flush()
            os.fsync(epoch_log.fileno())
            if epoch==1 or epoch%5==0:
                print(f"fold {fold} epoch {epoch:03d}: loss={entry['train_loss']:.4f} rmse={metrics['rmse']:.3f} local={entry['local_rmse']:.3f} best={best:.3f}@{best_epoch} seconds={entry['seconds']}",flush=True)
            if epoch>=args.min_epochs and stale>=args.patience:
                print(f"fold {fold}: early stop at epoch {epoch}",flush=True)
                break
        epoch_log.close()
        del model,optimizer,scaler,scheduler
        gc.collect()
        if device.type=="cuda":
            torch.cuda.empty_cache()
        model,checkpoint=load_checkpoint(fold_dir/"best.pth",device)
        records=predict(model,val_loader,device,args.size,tta=True)
        for record in records:
            record["fold"]=fold
        write_csv(fold_dir/"val_predictions.csv",records)
        variants={name:score_records(records,name) for name in ("pred","local","tta","local_tta")}
        if args.stage=="fine":
            hits=[max(abs(anchors[r["id"]]-np.array(r["point"])))<=args.roi/2 for r in validation]
            variants["coarse_roi_coverage"]=float(np.mean(hits))
        write_json(fold_dir/"complete.json",dict(best_epoch=checkpoint["epoch"],metrics=variants,
                                                elapsed_seconds=time.monotonic()-start))
        print(f"fold {fold} complete: {json.dumps(variants)}",flush=True)
        del model,checkpoint,train_set,val_set,train_loader,val_loader
        gc.collect()
        if device.type=="cuda":
            torch.cuda.empty_cache()
    summarize(run)


def summarize(run):
    run=Path(run)
    records=[]
    complete=[]
    for fold in range(5):
        path=run/f"fold_{fold}"/"val_predictions.csv"
        if path.exists() and (path.parent/"complete.json").exists():
            with path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    records.append({k:(int(v) if k in ("id","fold","width","height") else float(v))
                                    for k,v in row.items()})
            complete.append(fold)
    if not records:
        return
    records.sort(key=lambda r:r["id"])
    scores={name:score_records(records,name) for name in ("pred","local","tta","local_tta")}
    best=min(scores,key=lambda name:scores[name]["rmse"])
    summary=dict(completed_folds=complete,complete_five_fold=len(complete)==5,
                 scores=scores,selected_variant=best,
                 selection_note="Epochs and decoding/TTA are selected on validation data; these scores are internal model-selection estimates, not an independent test result.")
    write_csv(run/("oof_predictions.csv" if len(complete)==5 else "partial_validation.csv"),records)
    write_json(run/"metrics.json",summary)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def predict_test(args):
    run=Path(args.run).resolve()
    config=read_json(run/"config.json")
    summary=read_json(run/"metrics.json")
    if not summary["complete_five_fold"]:
        raise ValueError("Finish all five folds before exporting a test submission")
    manifest=read_json(config["manifest"])
    rows=[r for r in manifest["records"] if r["split"]=="test"]
    device=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.set_num_threads(args.threads)
    all_predictions=[]
    for fold in range(5):
        anchors=None
        if config["stage"]=="fine":
            coarse_model,coarse_checkpoint=load_checkpoint(Path(config["coarse_run"])/f"fold_{fold}"/"best.pth",device)
            coarse_config=coarse_checkpoint["config"]
            ds=FoveaDataset(rows,coarse_config["size"])
            anchor_rows=predict(coarse_model,loader(ds,args.batch_size),device,coarse_config["size"])
            anchors={r["id"]:np.array([r["pred_x"],r["pred_y"]]) for r in anchor_rows}
            del coarse_model,coarse_checkpoint,ds
            gc.collect()
            if device.type=="cuda": torch.cuda.empty_cache()
        model,checkpoint=load_checkpoint(run/f"fold_{fold}"/"best.pth",device)
        dataset=FoveaDataset(rows,config["size"],config["stage"],False,config["roi"],anchors)
        predictions=predict(model,loader(dataset,args.batch_size),device,config["size"],tta=True)
        all_predictions.append(predictions)
        del model,checkpoint,dataset
        gc.collect()
        if device.type=="cuda": torch.cuda.empty_cache()
    variant=args.variant or summary["selected_variant"]
    averaged=[]
    for index,row in enumerate(rows):
        xy=np.mean([[p[index][f"{variant}_x"],p[index][f"{variant}_y"]] for p in all_predictions],axis=0)
        xy=np.clip(xy,[0,0],[row["width"]-1,row["height"]-1])
        averaged.append(dict(id=row["id"],x=float(xy[0]),y=float(xy[1]),
                             width=row["width"],height=row["height"]))
    lookup={r["id"]:r for r in averaged}
    with (ROOT/"sample_submission.csv").open(encoding="utf-8-sig") as f:
        template=list(csv.DictReader(f))
    submission=[]
    for row in template:
        image_id,_,axis=row["ImageID"].split("_")
        value=lookup[int(image_id)][axis.lower()]
        if not np.isfinite(value): raise ValueError("Invalid submission value")
        submission.append(dict(ImageID=row["ImageID"],value=f"{value:.6f}"))
    write_csv(run/"test_predictions.csv",averaged)
    write_csv(run/"submission.csv",submission)
    write_json(run/"submission_metadata.json",dict(variant=variant,folds=5,rows=len(submission),
        visibility="No real invisible labels were provided. These predictions localize a visible fovea and do not claim to validate an invisible/zero-coordinate classifier.",
        external_training_data="ImageNet only; no external retinal labels or test labels used."))
    print(f"Saved {len(submission)} submission rows to {run/'submission.csv'}",flush=True)


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare")
    prep.add_argument("--size",type=int,default=512)
    prep.add_argument("--roi",type=int,default=768)
    fit=sub.add_parser("train")
    fit.add_argument("--manifest",default="artifacts/manifest_512_768.json")
    fit.add_argument("--run",required=True)
    fit.add_argument("--stage",choices=["coarse","fine"],default="coarse")
    fit.add_argument("--architecture",choices=["heatmap","regressor"],default="heatmap")
    fit.add_argument("--coarse-run")
    fit.add_argument("--folds",type=int,nargs="+",default=[0,1,2,3,4])
    fit.add_argument("--size",type=int,default=512)
    fit.add_argument("--roi",type=int,default=768)
    fit.add_argument("--epochs",type=int,default=120)
    fit.add_argument("--min-epochs",type=int,default=40)
    fit.add_argument("--patience",type=int,default=25)
    fit.add_argument("--min-delta",type=float,default=0.02)
    fit.add_argument("--batch-size",type=int,default=4)
    fit.add_argument("--encoder-lr",type=float,default=1e-4)
    fit.add_argument("--lr",type=float,default=1e-3)
    fit.add_argument("--weight-decay",type=float,default=1e-4)
    fit.add_argument("--sigma",type=float,default=2.)
    fit.add_argument("--heatmap-loss",choices=["gaussian","marginal_multiscale"],default="gaussian")
    fit.add_argument("--coordinate-weight",type=float,default=1000.)
    fit.add_argument("--region-weight",type=float,default=0.)
    fit.add_argument("--seed",type=int,default=2026)
    fit.add_argument("--threads",type=int,default=2)
    fit.add_argument("--cpu",action="store_true")
    pred=sub.add_parser("predict")
    pred.add_argument("--run",required=True)
    pred.add_argument("--variant",choices=["pred","local","tta","local_tta"])
    pred.add_argument("--batch-size",type=int,default=4)
    pred.add_argument("--threads",type=int,default=2)
    pred.add_argument("--cpu",action="store_true")
    report=sub.add_parser("summarize")
    report.add_argument("--run",required=True)
    args=parser.parse_args()
    if args.command=="prepare": prepare(args.size,args.roi)
    elif args.command=="train": train(args)
    elif args.command=="predict": predict_test(args)
    else: summarize(args.run)


if __name__=="__main__": main()
