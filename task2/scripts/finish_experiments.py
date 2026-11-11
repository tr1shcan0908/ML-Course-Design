"""Wait for training, finalize predictions and render plots using available runtimes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    env=os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['PYTHONUTF8']='1'
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    with (ROOT/'artifacts'/'finalization_console.log').open('a',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-B','-u','-m','fer.finalize','--wait-for-queue'],
                              stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT,creationflags=flags)
        if result.returncode:
            print(f'Final evaluation failed; see artifacts/finalization_console.log ({result.returncode})',flush=True)
            raise SystemExit(result.returncode)
        plot_python=Path.home()/'AppData/Local/Programs/Python/Python313/python.exe'
        plot=subprocess.run([str(plot_python),'-B','-u','scripts/plot_results.py'],
                            stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT,creationflags=flags)
        print(f'Experiment results ready. Plot exit code: {plot.returncode}',flush=True)


if __name__=='__main__':
    main()
