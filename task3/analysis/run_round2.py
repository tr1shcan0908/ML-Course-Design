"""Run the predeclared second-round GPU experiments sequentially."""
from pathlib import Path
import datetime as dt
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'analysis/round2'


def save_status(value):
    path=LOG/'status.json'
    temporary=path.with_suffix('.tmp')
    with temporary.open('w',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary,path)


def main():
    LOG.mkdir(exist_ok=True)
    commands=[
        ('fine_context768_train',['-m','fovea.experiment','train','--run','runs/fine_context768',
            '--stage','fine','--coarse-run','runs/heatmap_512','--size','256','--roi','768',
            '--epochs','100','--encoder-lr','0.00005','--lr','0.0003']),
        ('fine_context768_predict',['-m','fovea.experiment','predict','--run','runs/fine_context768','--variant','pred']),
        ('marginal_ms512_train',['-m','fovea.experiment','train','--run','runs/marginal_ms512',
            '--size','512','--epochs','120','--batch-size','4','--heatmap-loss','marginal_multiscale']),
        ('marginal_ms512_predict',['-m','fovea.experiment','predict','--run','runs/marginal_ms512','--variant','pred']),
        ('fine_shift5',['-m','fovea.multicrop','--run','runs/fine_512','--output','runs/fine_shift5']),
    ]
    state=dict(started=dt.datetime.now().astimezone().isoformat(),pid=os.getpid(),steps=[],status='running')
    for name,command in commands:
        state['active_step']=name
        save_status(state)
        print(f'Starting {name}',flush=True)
        with (LOG/f'{name}.log').open('w',encoding='utf-8') as handle:
            result=subprocess.run([sys.executable,'-u',*command],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        state['steps'].append(dict(name=name,returncode=result.returncode,finished=dt.datetime.now().astimezone().isoformat()))
        if result.returncode:
            state['status']='failed'
            save_status(state)
            raise SystemExit(result.returncode)
    state.update(status='complete',active_step=None,finished=dt.datetime.now().astimezone().isoformat())
    save_status(state)
    print('All predeclared experiments complete',flush=True)


if __name__=='__main__':
    main()
