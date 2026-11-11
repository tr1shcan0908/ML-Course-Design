"""YOLO pose comparison with checkpoints selected by original-coordinate RMSE."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from .data import ROOT
from .experiment import metric, read_json, write_csv, write_json

local_settings = ROOT / "artifacts" / "ultralytics_settings"
local_settings.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(local_settings))


def prepare_pose(manifest):
    import yaml
    folder=ROOT/"artifacts"/"pose"
    images=folder/"images"/"all"
    labels=folder/"labels"/"all"
    images.mkdir(parents=True,exist_ok=True)
    labels.mkdir(parents=True,exist_ok=True)
    for row in manifest["records"]:
        destination=images/f'{row["id"]:04d}.jpg'
        if not destination.exists(): shutil.copyfile(ROOT/row["path"],destination)
        if row["split"]=="train":
            a,b,c,d=row["box"]
            width,height=row["width"],row["height"]
            x,y=row["point"]
            values=[0,(a+c)/2/width,(b+d)/2/height,(c-a)/width,(d-b)/height,x/width,y/height,2]
            (labels/f'{row["id"]:04d}.txt').write_text(" ".join(f"{v:.9f}" for v in values)+"\n",encoding="utf-8")
    train=[r for r in manifest["records"] if r["split"]=="train"]
    for fold in range(5):
        for name,is_val in (("train",False),("val",True)):
            items=[r for r in train if (r["fold"]==fold)==is_val]
            (folder/f"{name}_{fold}.txt").write_text("\n".join((images/f'{r["id"]:04d}.jpg').as_posix() for r in items)+"\n",encoding="utf-8")
        settings=dict(path=folder.as_posix(),train=f"train_{fold}.txt",val=f"val_{fold}.txt",
                      names={0:"fovea_region"},kpt_shape=[1,3],flip_idx=[0])
        (folder/f"fold_{fold}.yaml").write_text(yaml.safe_dump(settings,allow_unicode=True),encoding="utf-8")
    return folder


def pose_predictions(model,rows,size,batch):
    result=[]
    # Ultralytics loads a list source eagerly, so chunk it explicitly on this 16GB host.
    for start in range(0,len(rows),batch):
        chunk=rows[start:start+batch]
        predictions=model.predict(source=[str(ROOT/r["path"]) for r in chunk],imgsz=size,batch=batch,
                                  device=0,conf=0.001,max_det=1,verbose=False,save=False,stream=True)
        for row,prediction in zip(chunk,predictions,strict=True):
            missed=prediction.keypoints is None or len(prediction.keypoints)==0
            xy=None
            if not missed:
                xy=prediction.keypoints.xy[0,0].detach().cpu().numpy()
                missed=not np.isfinite(xy).all() or np.all(xy==0)
            if missed:
                # Declared, image-only fallback; never reinterpret a missed point as invisible.
                xy=np.array([row["width"]/2,row["height"]/2])
            point=row.get("point",[0,0])
            result.append(dict(id=row["id"],width=row["width"],height=row["height"],
                               x=point[0],y=point[1],pred_x=float(xy[0]),pred_y=float(xy[1]),
                               missed=int(missed)))
    return result


def score(records):
    values=metric([[r["x"],r["y"]] for r in records],[[r["pred_x"],r["pred_y"]] for r in records])
    values["missed"]=sum(r["missed"] for r in records)
    return values


class PixelCheckpoint:
    def __init__(self,folder,rows,size,batch,min_epochs=40,patience=20,min_delta=0.02):
        self.folder,self.rows,self.size,self.batch=folder,rows,size,batch
        self.best=float("inf")
        self.best_epoch=0
        self.min_epochs,self.patience,self.min_delta=min_epochs,patience,min_delta
        (self.folder/"pixel_epochs.jsonl").write_text("",encoding="utf-8")

    def __call__(self,trainer):
        from ultralytics import YOLO
        # on_model_save fires after last.pt has been fully written. Evaluate a separate
        # inference model so predictor fusion cannot mutate the training model.
        model=YOLO(str(trainer.last))
        records=pose_predictions(model,self.rows,self.size,self.batch)
        metrics=score(records)
        epoch=trainer.epoch+1
        with (self.folder/"pixel_epochs.jsonl").open("a",encoding="utf-8") as handle:
            handle.write(json.dumps(dict(epoch=epoch,**metrics))+"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if metrics["rmse"]<self.best-self.min_delta:
            self.best=metrics["rmse"]
            self.best_epoch=epoch
            temporary=self.folder/"best_rmse.tmp.pt"
            shutil.copyfile(trainer.last,temporary)
            with temporary.open("rb+") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary,self.folder/"best_rmse.pt")
            for record in records: record["fold"]=self.rows[0]["fold"]
            write_csv(self.folder/"val_predictions.csv",records)
            write_json(self.folder/"best_metric.json",dict(epoch=epoch,**metrics))
        if epoch==1 or epoch%5==0:
            print(f'POSE original-pixel epoch {epoch}: rmse={metrics["rmse"]:.3f} best={self.best:.3f} missed={metrics["missed"]}',flush=True)
        if epoch>=self.min_epochs and epoch-self.best_epoch>=self.patience:
            trainer.stop=True
            print(f"POSE early stop by pixel RMSE at epoch {epoch}",flush=True)
        del model
        gc.collect()


class ConfigurePointLoss:
    def __init__(self,sigma):
        self.sigma=sigma

    def __call__(self,trainer):
        # Ultralytics 8.3.253 uses sigma=1/n for non-COCO keypoints. For one point,
        # sigma=1 makes a small coordinate error almost cost-free inside a large box.
        trainer.model.criterion=trainer.model.init_criterion()
        trainer.model.criterion.keypoint_loss.sigmas.fill_(self.sigma)
        print(f"Fovea keypoint loss sigma={self.sigma}; checkpoint selection uses pixel RMSE",flush=True)


def summarize(run):
    records=[]
    completed=[]
    for fold in range(5):
        folder=run/f"fold_{fold}"
        if (folder/"complete.json").exists():
            completed.append(fold)
            with (folder/"val_predictions.csv").open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    records.append({k:(int(v) if k in ("id","fold","width","height","missed") else float(v)) for k,v in row.items()})
    if not records: return
    records.sort(key=lambda r:r["id"])
    complete=len(completed)==5
    write_csv(run/("oof_predictions.csv" if complete else "partial_validation.csv"),records)
    report=dict(completed_folds=completed,complete_five_fold=complete,scores={"pred":score(records)},
                selected_variant="pred",selection_note="Epochs selected by validation RMSE, not an independent test result.")
    write_json(run/"metrics.json",report)
    print(json.dumps(report,indent=2),flush=True)


def train(args):
    from ultralytics import YOLO, settings
    # Disable connected experiment logging; this experiment stays local.
    settings.update({key:False for key in ("sync","wandb","mlflow","clearml","comet",
                                         "hub","neptune","dvc","raytune")})
    manifest=read_json(args.manifest)
    data_folder=prepare_pose(manifest)
    run=Path(args.run).resolve()
    run.mkdir(parents=True,exist_ok=True)
    config=dict(vars(args),dataset_hash=manifest["dataset_hash"],architecture="yolo11n-pose",
                pretraining="COCO generic pose weights, no retinal task labels",image_cache="ram_resized")
    if (run/"config.json").exists():
        comparable=lambda value:{key:item for key,item in value.items() if key not in ("folds","command")}
        if comparable(read_json(run/"config.json"))!=comparable(config):
            raise ValueError("Pose run configuration differs; use a new run directory")
    write_json(run/"config.json",config)
    torch.set_num_threads(2)
    rows=[r for r in manifest["records"] if r["split"]=="train"]
    for fold in args.folds:
        folder=run/f"fold_{fold}"
        folder.mkdir(exist_ok=True)
        if (folder/"complete.json").exists(): continue
        validation=[r for r in rows if r["fold"]==fold]
        callback=PixelCheckpoint(folder,validation,args.size,args.batch_size,
                                 args.min_epochs,args.patience,args.min_delta)
        model=YOLO("yolo11n-pose.pt")
        model.add_callback("on_train_start",ConfigurePointLoss(args.kpt_sigma))
        model.add_callback("on_model_save",callback)
        model.train(data=str(data_folder/f"fold_{fold}.yaml"),project=str(run),name=f"trainer_{fold}",
                    exist_ok=True,imgsz=args.size,epochs=args.epochs,batch=args.batch_size,
                    device=0,workers=0,amp=True,cache="ram",plots=False,
                    seed=2026+fold,deterministic=False,optimizer="AdamW",lr0=0.001,lrf=0.05,
                    warmup_epochs=3,weight_decay=0.0001,patience=0,
                    mosaic=0.,mixup=0.,degrees=15.,translate=0.035,scale=0.1,
                    fliplr=0.5,flipud=0.,hsv_h=0.005,hsv_s=0.1,hsv_v=0.15,
                    close_mosaic=0,verbose=False)
        if not (folder/"best_rmse.pt").exists(): raise RuntimeError("Pixel checkpoint callback did not run")
        write_json(folder/"complete.json",read_json(folder/"best_metric.json"))
        del model,callback
        gc.collect()
        torch.cuda.empty_cache()
    summarize(run)


def predict_test(args):
    from ultralytics import YOLO
    run=Path(args.run).resolve()
    config=read_json(run/"config.json")
    if not read_json(run/"metrics.json")["complete_five_fold"]: raise ValueError("Finish five folds first")
    manifest=read_json(config["manifest"])
    rows=[r for r in manifest["records"] if r["split"]=="test"]
    all_predictions=[]
    for fold in range(5):
        model=YOLO(str(run/f"fold_{fold}"/"best_rmse.pt"))
        all_predictions.append(pose_predictions(model,rows,config["size"],args.batch_size))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    averaged=[]
    for index,row in enumerate(rows):
        point=np.mean([[p[index]["pred_x"],p[index]["pred_y"]] for p in all_predictions],axis=0)
        point=np.clip(point,[0,0],[row["width"]-1,row["height"]-1])
        averaged.append(dict(id=row["id"],x=float(point[0]),y=float(point[1]),width=row["width"],height=row["height"]))
    lookup={r["id"]:r for r in averaged}
    with (ROOT/"sample_submission.csv").open(encoding="utf-8-sig") as f:
        template=list(csv.DictReader(f))
    submission=[]
    for row in template:
        image_id,_,axis=row["ImageID"].split("_")
        submission.append(dict(ImageID=row["ImageID"],value=f'{lookup[int(image_id)][axis.lower()]:.6f}'))
    write_csv(run/"test_predictions.csv",averaged)
    write_csv(run/"submission.csv",submission)
    write_json(run/"submission_metadata.json",dict(variant="pred",folds=5,rows=len(submission),
        visibility="All provided training keypoints are visible; invisible classification is not validated.",
        pretraining=config["pretraining"]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["train","predict"])
    parser.add_argument("--manifest",default="artifacts/manifest_512_768.json")
    parser.add_argument("--run",default="runs/pose_768")
    parser.add_argument("--folds",type=int,nargs="+",default=[0,1,2,3,4])
    parser.add_argument("--size",type=int,default=768)
    parser.add_argument("--epochs",type=int,default=80)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--kpt-sigma",type=float,default=0.1)
    parser.add_argument("--min-epochs",type=int,default=40)
    parser.add_argument("--patience",type=int,default=20)
    parser.add_argument("--min-delta",type=float,default=0.02)
    args=parser.parse_args()
    if args.command=="train": train(args)
    else: predict_test(args)


if __name__=="__main__": main()
