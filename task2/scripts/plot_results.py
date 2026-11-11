"""Render static scientific plots from completed, audited experiment results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    result=json.loads((root/'artifacts/final_results.json').read_text(encoding='utf-8'))
    output=root/'artifacts/figures'
    output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for run in result['runs']:
        rows=[json.loads(line) for line in (root/'runs'/run['name']/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        epochs=[row['epoch'] for row in rows]
        axes[0].plot(epochs,[100*row['val_accuracy'] for row in rows],marker='.',label=run['name'])
        axes[1].plot(epochs,[100*row['val_macro_f1'] for row in rows],marker='.',label=run['name'])
    for axis,title in zip(axes,['Validation accuracy (%)','Validation macro-F1 (%)']):
        axis.set(xlabel='Epoch',ylabel=title)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('FER: fixed duplicate-free development split, seed 42')
    fig.savefig(output/'learning_curves.png',dpi=170)
    plt.close(fig)
    labels=['Angry','Fear','Happy','Neutral','Sad','Surprise']
    counts=np.array(result['selected_validation_metrics']['confusion_matrix'])
    proportions=counts/counts.sum(1,keepdims=True)
    fig,axis=plt.subplots(figsize=(7,6),layout='constrained')
    plot=axis.imshow(proportions,cmap='Blues',vmin=0,vmax=1)
    axis.set_xticks(range(6),labels,rotation=35,ha='right')
    axis.set_yticks(range(6),labels)
    axis.set(xlabel='Predicted label',ylabel='Dataset label',title=result['selected_model']+' — validation')
    for row in range(6):
        for col in range(6):
            axis.text(col,row,f'{counts[row,col]}\n{proportions[row,col]:.1%}',ha='center',va='center',
                      color='white' if proportions[row,col]>.55 else '#17202a',fontsize=9)
    fig.colorbar(plot,ax=axis,label='Fraction of each true class',shrink=.8)
    fig.savefig(output/'confusion_matrix.png',dpi=170)
    plt.close(fig)
    print('Saved learning_curves.png and confusion_matrix.png',flush=True)


if __name__=='__main__':
    main()
