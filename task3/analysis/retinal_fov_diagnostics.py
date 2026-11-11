"""Descriptive field-of-view geometry; no model input or labels are modified."""
from pathlib import Path
import csv
import json

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'artifacts/manifest_512_768.json').read_text(encoding='utf-8'))
    output=ROOT/'analysis/round2'
    output.mkdir(exist_ok=True)
    records=[]
    previews=[]
    for row in manifest['records']:
        bgr=cv2.imdecode(np.frombuffer((ROOT/row['path']).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        width,height=row['width'],row['height']
        scale=600/max(width,height)
        sw,sh=round(width*scale),round(height*scale)
        small=cv2.resize(bgr,(sw,sh),interpolation=cv2.INTER_AREA)
        # Fixed intensity threshold is only a rough FOV heuristic, not a tissue segmentation.
        mask=(small.max(axis=2)>20).astype(np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
        count,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
        if count<2: raise ValueError(f'No FOV candidate in {row["id"]}')
        index=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
        x,y,w,h,area=map(int,stats[index])
        sx,sy=sw/width,sh/height
        bbox_width,bbox_height=w/sx,h/sy
        diameter=2*np.sqrt((area/(sx*sy))/np.pi)
        records.append(dict(id=row['id'],split=row['split'],width=width,height=height,
            bbox_width=bbox_width,bbox_height=bbox_height,area_equivalent_diameter=diameter,
            roi512_over_diameter=512/diameter,roi768_over_diameter=768/diameter,
            occupied_fraction=area/(sw*sh)))
        if row['id'] in [1,16,34,44,49,72,81,93] or width!=2992:
            preview=cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
            previews.append((row,preview,(x,y,w,h)))
    with (output/'fov_geometry.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary={}
    for width in sorted({r['width'] for r in records}):
        values=[r for r in records if r['width']==width]
        diameter=np.array([r['area_equivalent_diameter'] for r in values])
        summary[str(width)]=dict(count=len(values),median_diameter=float(np.median(diameter)),
                                 min_diameter=float(diameter.min()),max_diameter=float(diameter.max()),
                                 median_512_ratio=float(np.median(512/diameter)))
    (output/'fov_summary.json').write_text(json.dumps(dict(groups=summary,
        note='Fixed max-channel intensity >20, morphological closing, largest component. Descriptive proxy only; no optical scale calibration.'),indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(int(np.ceil(len(previews)/4)),4,figsize=(12,3*int(np.ceil(len(previews)/4))),layout='constrained')
    for ax in axes.flat: ax.axis('off')
    for ax,(row,preview,box) in zip(axes.flat,previews):
        ax.imshow(preview)
        x,y,w,h=box
        ax.add_patch(Rectangle((x,y),w,h,fill=False,color='#22d3ee',linewidth=1))
        ax.set_title(f"{row['id']:04d} | {row['width']} x {row['height']}")
    fig.suptitle('Heuristic FOV bounding boxes (descriptive only)')
    fig.savefig(output/'fov_examples.png',dpi=130)
    plt.close(fig)
    view_path=ROOT/'artifacts/manifest_fov_512.json'
    if view_path.exists():
        from PIL import Image
        normalized=json.loads(view_path.read_text(encoding='utf-8'))
        original_lookup={r['id']:r for r in manifest['records']}
        new_lookup={r['id']:r for r in normalized['records']}
        fig,axes=plt.subplots(3,2,figsize=(8,12),layout='constrained')
        for index,image_id in enumerate([1,49,72]):
            a,b=original_lookup[image_id],new_lookup[image_id]
            magnification=b['transform'][0][0]/a['transform'][0][0]
            for ax,row,title in [(axes[index,0],a,'Original full-frame input'),
                                  (axes[index,1],b,f'FOV input: {magnification:.2f}x scale')]:
                ax.imshow(Image.open(ROOT/row['cache']))
                ax.set_title(f'{image_id:04d} | {title}')
                ax.axis('off')
        fig.suptitle('Both inputs are 512 x 512; no annotation is used to crop')
        fig.savefig(output/'fov_input_comparison.png',dpi=140)
        plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
