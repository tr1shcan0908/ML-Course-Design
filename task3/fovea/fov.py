"""Image-only field-of-view cropping, with an exact original-coordinate transform."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json

import cv2
import numpy as np

from .data import ROOT, letterbox, read_rgb, save_rgb, transform_points


def field_of_view(image,size=512):
    height,width=image.shape[:2]
    scale=min(1.,600/max(width,height))
    sw,sh=round(width*scale),round(height*scale)
    small=cv2.resize(image,(sw,sh),interpolation=cv2.INTER_AREA)
    mask=(small.max(axis=2)>20).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    count,_,stats,_=cv2.connectedComponentsWithStats(mask)
    bounds=[0,0,width,height]
    fallback=True
    if count>1:
        index=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
        x,y,w,h,area=map(int,stats[index])
        if area>=0.05*sw*sh:
            sx,sy=sw/width,sh/height
            # Convert pixel edges, then add 2% of the larger dimension on each side.
            pad=0.02*max(w/sx,h/sy)
            bounds=[max(0,int(np.floor(x/sx-pad))),max(0,int(np.floor(y/sy-pad))),
                    min(width,int(np.ceil((x+w)/sx+pad))),min(height,int(np.ceil((y+h)/sy+pad)))]
            fallback=False
    left,top,right,bottom=bounds
    result,local=letterbox(image[top:bottom,left:right],size)
    translation=np.array([[1,0,-left],[0,1,-top],[0,0,1]],dtype=float)
    return result,local@translation,dict(bounds=bounds,fallback=fallback)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',default='artifacts/manifest_512_768.json')
    parser.add_argument('--size',type=int,default=512)
    args=parser.parse_args()
    base=json.loads((ROOT/args.base).read_text(encoding='utf-8'))
    rows=copy.deepcopy(base['records'])
    coverage=[]
    for row in rows:
        small,matrix,description=field_of_view(read_rgb(ROOT/row['path']),args.size)
        cached=ROOT/'artifacts'/f'coarse_fov_{args.size}'/f"{row['split']}_{row['id']:04d}.jpg"
        save_rgb(cached,small)
        row.update(cache=cached.relative_to(ROOT).as_posix(),transform=matrix.tolist(),fov=description)
        if row['split']=='train':
            point=transform_points(np.array(row['point']),matrix)
            coverage.append(bool(np.all(point>=0) and np.all(point<args.size)))
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    manifest={**base,'size':args.size,'records':rows,'dataset_hash':digest,
              'parent_dataset_hash':base['dataset_hash'],'parent_manifest':args.base,
              'preprocessing':'Largest component of max-channel >20 at max side 600, close 9x9, pad 2%, crop then letterbox; image-only.',
              'training_point_coverage_audit':float(np.mean(coverage))}
    destination=ROOT/'artifacts'/f'manifest_fov_{args.size}.json'
    destination.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(manifest=str(destination),records=len(rows),training_point_coverage=float(np.mean(coverage)),
                         fallbacks=sum(r['fov']['fallback'] for r in rows),dataset_hash=digest),indent=2))


if __name__=='__main__': main()
