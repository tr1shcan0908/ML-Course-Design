"""Wait for the original GPU queue, then run the image-only FOV addendum."""
from pathlib import Path
import datetime as dt
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'analysis/round2'


def save(state):
    destination=LOG/'fov_status.json'
    temporary=destination.with_suffix('.tmp')
    with temporary.open('w',encoding='utf-8') as handle:
        json.dump(state,handle,ensure_ascii=False,indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary,destination)


def main():
    state=dict(status='awaiting_round2',started=dt.datetime.now().astimezone().isoformat(),pid=os.getpid(),steps=[])
    save(state)
    start=time.monotonic()
    while True:
        status=json.loads((LOG/'status.json').read_text(encoding='utf-8'))
        if status['status']=='complete': break
        if status['status']=='failed' or time.monotonic()-start>3600:
            state.update(status='blocked_by_failed_or_stale_queue')
            save(state)
            raise SystemExit(1)
        time.sleep(10)
    commands=[
        ('primary_comparison',['analysis/compare_round2.py']),
        ('fov_heatmap512_train',['-m','fovea.experiment','train','--run','runs/fov_heatmap512',
             '--manifest','artifacts/manifest_fov_512.json','--size','512','--epochs','120','--batch-size','4']),
        ('fov_heatmap512_predict',['-m','fovea.experiment','predict','--run','runs/fov_heatmap512','--variant','pred']),
        ('expanded_comparison',['analysis/compare_round2.py','--include-fov','--output','outputs/round2_fov']),
    ]
    for name,args in commands:
        state.update(status='running',active_step=name)
        save(state)
        with (LOG/f'{name}.log').open('w',encoding='utf-8') as handle:
            result=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        state['steps'].append(dict(name=name,returncode=result.returncode,finished=dt.datetime.now().astimezone().isoformat()))
        if result.returncode:
            state.update(status='failed')
            save(state)
            raise SystemExit(result.returncode)
    state.update(status='complete',active_step=None,finished=dt.datetime.now().astimezone().isoformat())
    save(state)


if __name__=='__main__': main()
