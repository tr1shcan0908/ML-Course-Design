"""Run the GPU experiments sequentially, logging and resuming at epoch boundaries."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fer.io import read_json, update_status


def write_status(value):
    path=ROOT/'artifacts'/'queue_status.json'
    update_status(path,value,required=value['status']=='training_complete')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',default='experiments/plan.json')
    args=parser.parse_args()
    os.chdir(ROOT)
    (ROOT/'artifacts'/'controller.pid').write_text(str(os.getpid()),encoding='utf-8')
    plan=read_json(args.plan)
    env=os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['PYTHONUTF8']='1'
    completed=[]
    for config_path in plan['configs']:
        config=read_json(config_path)
        output=ROOT/config['output']
        output.mkdir(parents=True,exist_ok=True)
        status_path=output/'status.json'
        if status_path.exists() and read_json(status_path).get('status')=='complete':
            completed.append(config['name'])
            continue
        command=[sys.executable,'-B','-u','-m','fer.train','--config',config_path]
        if (output/'last.pt').exists(): command.append('--resume')
        with (output/'console.log').open('a',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            print(f'Started {config["name"]}: PID {child.pid}',flush=True)
            while child.poll() is None:
                write_status(dict(status='running',current=config['name'],pid=child.pid,
                                  completed=completed,plan=args.plan,controller_pid=os.getpid()))
                time.sleep(5)
            if child.returncode:
                write_status(dict(status='failed',current=config['name'],exit_code=child.returncode,
                                  completed=completed,log=str(output/'console.log')))
                raise SystemExit(child.returncode)
        completed.append(config['name'])
        write_status(dict(status='between_runs',completed=completed))
    write_status(dict(status='training_complete',completed=completed))
    print('All training runs completed.',flush=True)


if __name__=='__main__':
    main()
