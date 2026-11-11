"""Run the bounded supervised fine-tuning plan and then select/export results."""
import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fer.io import atomic_output, read_json, update_status


def write_progress(plan, state):
    lines=['# '+plan.get('report_title','监督微调实验进度'),'',f'当前阶段：{state}。按固定验证集 Accuracy 选模，Macro-F1 用于同分比较。','',
           '| 模型 | 状态 | 完成轮数 / 上限 | 所选轮次单模型 Accuracy | Macro-F1 | 替换后融合 Accuracy |',
           '| --- | --- | ---: | ---: | ---: | ---: |']
    for path in plan['reference_configs']+plan.get('comparison_configs',[])+plan['configs']:
        config=read_json(path)
        run=ROOT/config['output']
        status=read_json(run/'status.json') if (run/'status.json').exists() else {}
        metrics=read_json(run/'best_metrics.json') if (run/'best_metrics.json').exists() else {}
        history=(run/'metrics.jsonl').read_text(encoding='utf-8').splitlines() if (run/'metrics.jsonl').exists() else []
        accuracy=f'{metrics["accuracy"]:.2%}' if metrics else '—'
        macro=f'{metrics["macro_f1"]:.2%}' if metrics else '—'
        fusion=f'{metrics["selection_metrics"]["accuracy"]:.2%}' if metrics and config.get('selection_reference') else '—'
        label='已完成参照' if path not in plan['configs'] else status.get('status','pending')
        lines.append(f'| {config["name"]} | {label} | {len(history)} / {config["epochs"]} | {accuracy} | {macro} | {fusion} |')
    lines+=['']+plan.get('progress_notes',['前两组从已有最佳 ConvNeXt 权重微调；第三组从 ImageNet-22K 的 ConvNeXt-Small 权重开始监督训练。'])+[
            '采用早停，完成轮数可能少于上限。最佳轮次为 0 表示初始权重仍优于后续训练。',
            '本阶段训练结束后统一以 FP32 复核各候选，再比较少量预定融合权重与翻转 TTA，导出最优方案。',
            '验证集有 5,409 张图，用于调参和选模；原测试集 7,065 张无标签。',
            '方案说明：['+plan.get('plan_document','SFT_PLAN.md')+']('+plan.get('plan_document','SFT_PLAN.md')+')。',
            f'更新时间（北京时间）：{datetime.now(timezone(timedelta(hours=8))).isoformat()}']
    try:
        with atomic_output(ROOT/plan['report']) as temp:
            temp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    except OSError as error:
        print(f'Progress report update skipped: {error}',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',default='experiments/sft_plan.json')
    args=parser.parse_args()
    os.chdir(ROOT)
    plan=read_json(args.plan)
    lock=(ROOT/plan.get('controller_lock','artifacts/sft_controller.lock')).open('a+b')
    if lock.tell()==0:
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    if os.name=='nt':
        import msvcrt
        try:
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as error:
            lock.close()
            raise RuntimeError('A fine-tuning controller is already running.') from error
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1'}
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    completed=[]
    child=None
    current=None
    try:
        for path in plan['configs']:
            config=read_json(path)
            current=config['name']
            run=ROOT/config['output']
            run.mkdir(parents=True,exist_ok=True)
            if (run/'status.json').exists() and read_json(run/'status.json').get('status')=='complete':
                completed.append(current)
                continue
            command=[sys.executable,'-B','-u','-m','fer.train','--config',path]
            if (run/'last.pt').exists():
                command.append('--resume')
            with (run/'console.log').open('a',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
                print(f'Started {current}: PID {child.pid}',flush=True)
                while child.poll() is None:
                    update_status(plan['queue_status'],dict(status='running',current=current,pid=child.pid,
                                  controller_pid=os.getpid(),completed=completed,plan=args.plan))
                    write_progress(plan,'训练中')
                    time.sleep(10)
                if child.returncode:
                    raise RuntimeError(f'{current} exited with {child.returncode}; see {run / "console.log"}')
            completed.append(current)
        current='finalization'
        update_status(plan['queue_status'],dict(status='finalizing',completed=completed,controller_pid=os.getpid()))
        write_progress(plan,'训练完成，正在复核和导出')
        finalization_log=plan.get('finalization_log','artifacts/sft_finalization_console.log')
        with (ROOT/finalization_log).open('a',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-B','-u','-m','fer.finalize_sft','--plan',args.plan],
                                   cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
            code=child.wait()
        if code:
            raise RuntimeError(f'Finalization failed with {code}; see {finalization_log}')
        update_status(plan['queue_status'],dict(status='complete',completed=completed,plan=args.plan),required=True)
        print('Fine-tuning, validation comparison and prediction export completed.',flush=True)
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        update_status(plan['queue_status'],dict(status='paused',current=current,completed=completed))
        raise
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        update_status(plan['queue_status'],dict(status='failed',current=current,completed=completed,error=repr(error)))
        traceback.print_exc()
        raise
    finally:
        lock.close()


if __name__=='__main__':
    main()
