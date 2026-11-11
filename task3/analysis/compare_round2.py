"""Evaluate the ten predeclared candidates, including paired group sensitivity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rmse(target,prediction):
    return float(np.sqrt(np.mean((prediction-target)**2)))


def plots(output,stats,predictions,target,folds,rows,test,selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({'font.size':9,'figure.dpi':150})
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    ordered=sorted(stats,key=lambda r:r['rmse'])
    names=[r['method'] for r in ordered]
    axes[0].barh(names,[r['rmse'] for r in ordered],
                 color=['#0f766e' if n==selected else '#94a3b8' for n in names])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('OOF coordinate RMSE (px)')
    baseline=predictions['round1_best']
    base_errors=np.linalg.norm(baseline-target,axis=1)
    final_errors=np.linalg.norm(predictions[selected]-target,axis=1)
    axes[1].scatter(base_errors,final_errors,c=folds,cmap='tab10',vmin=0,vmax=9,s=28)
    limit=max(float(base_errors.max()),float(final_errors.max()))*1.1
    axes[1].plot([0,limit],[0,limit],ls='--',color='#94a3b8')
    axes[1].set(xlabel='Round 1 radial error (px)',ylabel='Selected radial error (px)',xlim=(0,limit),ylim=(0,limit))
    label_offsets={44:(6,9),72:(6,-13),49:(-28,8),16:(6,3),34:(6,-8)}
    for index in np.argsort(base_errors)[-5:]:
        axes[1].annotate(f"{rows[index]['id']:04d}",(base_errors[index],final_errors[index]),fontsize=8,
                         xytext=label_offsets.get(rows[index]['id'],(6,4)),textcoords='offset points')
    fig.savefig(output/'comparison.png')
    plt.close(fig)
    chosen=predictions[selected]
    radial=np.linalg.norm(chosen-target,axis=1)
    fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    for ax,index in zip(axes.flat,np.argsort(radial)[-8:][::-1]):
        row=rows[index]
        ax.imshow(Image.open(ROOT/row['path']))
        ax.scatter(*target[index],marker='+',s=120,c='#22d3ee')
        ax.scatter(*baseline[index],marker='o',s=45,facecolors='none',edgecolors='#fbbf24')
        ax.scatter(*chosen[index],marker='x',s=80,c='#f472b6')
        center=target[index]
        ax.set(xlim=(center[0]-150,center[0]+150),ylim=(center[1]+150,center[1]-150),
               title=f"{row['id']:04d}: {base_errors[index]:.1f} -> {radial[index]:.1f} px")
        ax.axis('off')
    fig.suptitle('Cyan +: label; yellow circle: round 1; pink x: selected')
    fig.savefig(output/'worst_examples.png')
    plt.close(fig)
    lookup={r['id']:r for r in test}
    fig,axes=plt.subplots(4,5,figsize=(15,9),layout='constrained')
    for ax,record in zip(axes.flat,test):
        ax.imshow(Image.open(ROOT/'test'/f"{record['id']:04d}.jpg"))
        ax.scatter(record['x'],record['y'],marker='+',c='#22d3ee',s=80)
        ax.set_title(f"{record['id']:04d}: ({record['x']:.1f}, {record['y']:.1f})")
        ax.axis('off')
    fig.suptitle('Round 2 test predictions: no ground truth available')
    fig.savefig(output/'test_preview.png')
    plt.close(fig)
    runs=['fine_context768','marginal_ms512']
    if 'fov512' in predictions: runs.append('fov_heatmap512')
    fig,axes=plt.subplots(len(runs),1,figsize=(10,3*len(runs)),squeeze=False,layout='constrained')
    for ax,run in zip(axes[:,0],runs):
        for fold in range(5):
            history=[json.loads(line) for line in (ROOT/'runs'/run/f'fold_{fold}'/'epochs.jsonl').read_text().splitlines() if line]
            ax.plot([r['epoch'] for r in history],[r['rmse'] for r in history],label=f'fold {fold}')
        ax.set(title=run,xlabel='Epoch',ylabel='Validation RMSE (px)',yscale='log')
        ax.legend(ncol=5,fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(output/'learning_curves.png')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='outputs/round2')
    parser.add_argument('--include-fov',action='store_true')
    args=parser.parse_args()
    output=ROOT/args.output
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'artifacts/manifest_512_768.json').read_text(encoding='utf-8'))
    rows=sorted([r for r in manifest['records'] if r['split']=='train'],key=lambda r:r['id'])
    tests=sorted([r for r in manifest['records'] if r['split']=='test'],key=lambda r:r['id'])
    target=np.array([r['point'] for r in rows])
    ids=[r['id'] for r in rows]
    folds=np.array([r['fold'] for r in rows])
    groups=np.array([r['group'] for r in rows])
    unique_groups=np.unique(groups)
    original_labels={int(r['data']):[float(r['Fovea_X']),float(r['Fovea_Y'])]
                     for r in read_csv(ROOT/'fovea_localization_train_GT.csv')}
    np.testing.assert_allclose([original_labels[i] for i in ids],target,atol=1e-9,rtol=0)
    for row in manifest['records']:
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['group']
    for group in unique_groups:
        assert len(set(folds[groups==group]))==1
    sources=['heatmap_512','fine_512','fine_context768','marginal_ms512','fine_shift5']
    if args.include_fov: sources.append('fov_heatmap512')
    base_oof,base_test,run_hashes={},{},{}
    for name in sources:
        directory=ROOT/'runs'/name
        config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
        summary=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
        assert summary['complete_five_fold']
        if config['dataset_hash']!=manifest['dataset_hash']:
            assert name=='fov_heatmap512'
            view=json.loads((ROOT/config['manifest']).read_text(encoding='utf-8'))
            assert view['parent_dataset_hash']==manifest['dataset_hash']
            assert view['dataset_hash']==config['dataset_hash']
            assert hashlib.sha256(json.dumps(view['records'],sort_keys=True).encode()).hexdigest()==view['dataset_hash']
            assert len(view['records'])==len(manifest['records'])
            for raw,changed in zip(manifest['records'],view['records'],strict=True):
                for key in ('id','split','path','width','height','group','fold','point','box'):
                    assert raw.get(key)==changed.get(key), f'FOV preprocessing changed original identity: {key}'
        values=sorted(read_csv(directory/'oof_predictions.csv'),key=lambda r:int(r['id']))
        assert [int(r['id']) for r in values]==ids
        assert [int(r['fold']) for r in values]==folds.tolist()
        np.testing.assert_allclose([[float(r['x']),float(r['y'])] for r in values],target,atol=0.001,rtol=0)
        base_oof[name]=np.array([[float(r['pred_x']),float(r['pred_y'])] for r in values])
        assert np.isfinite(base_oof[name]).all()
        np.testing.assert_allclose(rmse(target,base_oof[name]),summary['scores']['pred']['rmse'],atol=0.001)
        test_values=sorted(read_csv(directory/'test_predictions.csv'),key=lambda r:int(r['id']))
        assert [int(r['id']) for r in test_values]==[r['id'] for r in tests]
        metadata=json.loads((directory/'submission_metadata.json').read_text(encoding='utf-8'))
        assert metadata['variant']=='pred', 'Round 2 predeclares the pred decoder for every source'
        base_test[name]=np.array([[float(r['x']),float(r['y'])] for r in test_values])
        assert np.isfinite(base_test[name]).all()
        run_hashes[name]={p:hashlib.sha256((directory/p).read_bytes()).hexdigest()
                          for p in ('config.json','oof_predictions.csv','test_predictions.csv')}
    candidates={'round1_best':{'heatmap_512':0.5,'fine_512':0.5}}
    proposals=[('fine_context768','context768'),('marginal_ms512','axis_ms'),('fine_shift5','shift5')]
    if args.include_fov: proposals.append(('fov_heatmap512','fov512'))
    for name,label in proposals:
        candidates[label]={name:1.0}
        candidates[f'heatmap+{label}']={'heatmap_512':0.5,name:0.5}
        candidates[f'heatmap+fine+{label}']={'heatmap_512':1/3,'fine_512':1/3,name:1/3}
    assert len(candidates)==(13 if args.include_fov else 10)
    predictions={name:sum(base_oof[source]*weight for source,weight in weights.items()) for name,weights in candidates.items()}
    test_predictions={name:sum(base_test[source]*weight for source,weight in weights.items()) for name,weights in candidates.items()}
    old=sorted(read_csv(ROOT/'outputs/oof_predictions.csv'),key=lambda r:int(r['id']))
    np.testing.assert_allclose(predictions['round1_best'],[[float(r['pred_x']),float(r['pred_y'])] for r in old],atol=1e-7)
    squared={name:np.square(values-target).sum(axis=1) for name,values in predictions.items()}
    baseline_error=squared['round1_best']
    rng=np.random.default_rng(20261022)
    sampled=rng.integers(0,len(unique_groups),size=(5000,len(unique_groups)))
    group_sizes=np.array([(groups==group).sum() for group in unique_groups])
    denominators=2*group_sizes[sampled].sum(axis=1)
    group_sums={name:np.array([error[groups==group].sum() for group in unique_groups]) for name,error in squared.items()}
    bootstrap={name:np.sqrt(sums[sampled].sum(axis=1)/denominators) for name,sums in group_sums.items()}
    stats=[]
    sensitivities=[]
    for name,values in predictions.items():
        radial=np.sqrt(squared[name])
        delta=bootstrap[name]-bootstrap['round1_best']
        item=dict(method=name,rmse=rmse(target,values),delta_rmse=rmse(target,values)-rmse(target,predictions['round1_best']),
                  aed=float(radial.mean()),median_radial=float(np.median(radial)),p95_radial=float(np.percentile(radial,95)),
                  max_radial=float(radial.max()),bootstrap_delta_low=float(np.percentile(delta,2.5)),
                  bootstrap_delta_high=float(np.percentile(delta,97.5)),images_improved=int((squared[name]<baseline_error).sum()))
        for fold in range(5): item[f'fold_{fold}_rmse']=rmse(target[folds==fold],values[folds==fold])
        item['folds_improved']=sum(item[f'fold_{f}_rmse']<rmse(target[folds==f],predictions['round1_best'][folds==f]) for f in range(5))
        leave_deltas=[]
        for group in unique_groups:
            keep=groups!=group
            difference=rmse(target[keep],values[keep])-rmse(target[keep],predictions['round1_best'][keep])
            leave_deltas.append(difference)
            sensitivities.append(dict(method=name,excluded_ids=';'.join(str(i) for i,g in zip(ids,groups) if g==group),delta_rmse=difference))
        item['leave_one_group_delta_min']=min(leave_deltas)
        item['leave_one_group_delta_max']=max(leave_deltas)
        stats.append(item)
    stats.sort(key=lambda r:r['rmse'])
    chosen=stats[0]['method']
    selected=predictions[chosen]
    changes=[]
    for index,row in enumerate(rows):
        changes.append(dict(id=row['id'],fold=row['fold'],x=row['point'][0],y=row['point'][1],
            pred_x=float(selected[index,0]),pred_y=float(selected[index,1]),
            baseline_radial=float(np.sqrt(baseline_error[index])),selected_radial=float(np.sqrt(squared[chosen][index])),
            sse_change=float(squared[chosen][index]-baseline_error[index])))
    final_test=[]
    for row,point in zip(tests,test_predictions[chosen],strict=True):
        point=np.clip(point,[0,0],[row['width']-1,row['height']-1])
        final_test.append(dict(id=row['id'],x=float(point[0]),y=float(point[1]),width=row['width'],height=row['height']))
    lookup={r['id']:r for r in final_test}
    template=read_csv(ROOT/'sample_submission.csv')
    submission=[]
    for row in template:
        image_id,_,axis=row['ImageID'].split('_')
        submission.append(dict(ImageID=row['ImageID'],value=f"{lookup[int(image_id)][axis.lower()]:.6f}"))
    assert len(submission)==40 and [r['ImageID'] for r in submission]==[r['ImageID'] for r in template]
    write_csv(output/'comparison.csv',stats)
    write_csv(output/'leave_one_group_diagnostics.csv',sensitivities)
    write_csv(output/'oof_predictions.csv',changes)
    write_csv(output/'test_predictions.csv',final_test)
    write_csv(output/'submission.csv',submission)
    selection=dict(selected=chosen,weights=candidates[chosen],metrics=stats[0],candidate_definitions=candidates,
        baseline_rmse=rmse(target,predictions['round1_best']),dataset_hash=manifest['dataset_hash'],run_files=run_hashes,
        adaptive_fov_addendum=args.include_fov,
        bootstrap=dict(repeats=5000,seed=20261022,groups=len(unique_groups),unit='exact image content',
                       note='Conditional on fixed predictions; does not correct validation/model selection bias.'),
        checks=dict(train_rows=len(rows),test_rows=len(tests),submission_rows=len(submission),
                    original_image_hashes_unchanged=True,no_exact_duplicate_leakage=True,template_order_matches=True),
        limitations=['All training labels visible; invisible classifier remains unvalidated.',
                     'Same validation labels select epochs and candidates; these are internal model-selection estimates.',
                     'No patient/eye IDs are available; exact-duplicate grouping cannot exclude all related-eye leakage.'],
        source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [*sorted((ROOT/'fovea').glob('*.py')),Path(__file__)]},
        submission_sha256=hashlib.sha256((output/'submission.csv').read_bytes()).hexdigest())
    (output/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    plots(output,stats,predictions,target,folds,rows,final_test,chosen)
    first=stats[0]
    gain=100*(selection['baseline_rmse']-first['rmse'])/selection['baseline_rmse']
    lines=['# 第二轮改进实验结果','',
           f"本轮选择 **{chosen}**，80 张折外图的原图坐标 RMSE 为 **{first['rmse']:.4f} px**。第一轮为 {selection['baseline_rmse']:.4f} px，相对变化为降低 {gain:.2f}%。这些是内部选型结果，没有测试真值或平台排名。",'',
           '| 候选 | RMSE | 相对第一轮变化 | 改善折数 | 改善图数 | 配对 bootstrap 差值 95% 区间 |',
           '|---|---:|---:|---:|---:|---:|']
    for row in stats:
        lines.append(f"| {row['method']} | {row['rmse']:.4f} | {row['delta_rmse']:+.4f} | {row['folds_improved']}/5 | {row['images_improved']}/80 | [{row['bootstrap_delta_low']:+.3f}, {row['bootstrap_delta_high']:+.3f}] |")
    lines+=['','context768 为更大视野精修；axis_ms 为多尺度 X/Y 边缘概率监督；shift5 为原精修权重的固定五裁剪平均。所有候选均用全局 DSNT，不在本轮重新挑选解码器。','',
            '## 稳定性与解释','',
            f"所选候选在 {first['folds_improved']}/5 折、{first['images_improved']}/80 张图上改善。移除任意一个图像内容组后，其相对第一轮的 RMSE 差值范围为 [{first['leave_one_group_delta_min']:+.4f}, {first['leave_one_group_delta_max']:+.4f}] px；这只是敏感性诊断，正式指标始终包含全部 80 张图。",'',
            'bootstrap 将 0039/0077 视为同一抽样组，保留其原评价权重；共 79 组、5,000 次配对重抽样。这只描述当前固定预测的抽样波动，不校正选 epoch、候选及此前解码选择带来的偏差。不能把这个区间当作独立测试上的显著性证明。','',
            '## 分折比较','',
            '| 候选 | 折 0 | 折 1 | 折 2 | 折 3 | 折 4 |','|---|---:|---:|---:|---:|---:|']
    for row in stats:
        lines.append('| '+row['method']+' | '+' | '.join(f"{row[f'fold_{f}_rmse']:.3f}" for f in range(5))+' |')
    lines+=['','## 最大误差样本的变化','',
            '| 图像 | 第一轮欧氏误差 | 第二轮欧氏误差 |','|---|---:|---:|']
    for row in sorted(changes,key=lambda r:r['baseline_radial'],reverse=True)[:8]:
        lines.append(f"| {row['id']:04d} | {row['baseline_radial']:.2f} | {row['selected_radial']:.2f} |")
    lines+=['','## 实验范围与文件','',
        '- R2-A、R2-B 各完成五折训练；R2-C 只做固定推理增强。初始参数和十个候选在第二轮结果产生前写入 `../../analysis/round2_plan.md`。',
        '- 原提交 `../submission.csv` 保留；本目录 `submission.csv` 是本轮所选候选的 40 行本地文件，未上传。',
        '- 逐图误差见 `oof_predictions.csv`，单组敏感性见 `leave_one_group_diagnostics.csv`，精确权重及来源哈希见 `selection.json`。',
        '- 全部训练标签均可见；五裁剪分歧不是不可见概率，本轮仍未验证 (0,0) 判定。',
        '- 设计、论文来源、与原论文的实现差别及后续发散方向见 `../../analysis/round2_plan.md`。']
    if args.include_fov:
        lines+=['','## 追加的 FOV 实验','',
            'fov512 将图像自身阈值掩码的最大连通域加边界后裁剪，再缩放到 512；使用原 Gaussian 损失。它提高了宽幅图的有效视野分辨率。五折全部重训，标签、原图、折和其余训练设置保持相同；80 个训练点均仍在裁剪视野内。',
            '', '此项源于训练期间的无标签视野几何审计，之后在 FOV 训练前记录到 `../../analysis/round2_addendum_fov.md`。追加三个固定候选，共十三个；不是在最初方案中预先列出的十项。主比较的十项结果仍保留在 `../round2/`。适应性追加进一步增加选型偏差。']
        resolution_rows=[]
        diagnostic_methods={'original_heatmap':base_oof['heatmap_512'],
                            'fov_heatmap':base_oof['fov_heatmap512'],
                            'round1_best':predictions['round1_best'],
                            'selected':selected}
        lines+=['','## 按原始分辨率诊断','',
                '这是在固定预测上检查视野标准化假设的事后诊断，不新增选型候选，也不按分辨率分别选模型。较小分辨率仅有 5 张图，不能据此确立相机或人群层面的结论。',
                '', '| 原始分辨率 | 样本数 | 方法 | 坐标 RMSE |', '|---|---:|---|---:|']
        for width,height in sorted({(r['width'],r['height']) for r in rows}):
            mask=np.array([(r['width'],r['height'])==(width,height) for r in rows])
            for method,values in diagnostic_methods.items():
                item=dict(width=width,height=height,count=int(mask.sum()),method=method,
                          rmse=rmse(target[mask],values[mask]))
                resolution_rows.append(item)
                lines.append(f"| {width}×{height} | {item['count']} | {method} | {item['rmse']:.4f} |")
        write_csv(output/'resolution_diagnostics.csv',resolution_rows)
    (output/'experiment_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(selected=chosen,weights=candidates[chosen],metrics=first),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
