"""Screen four models, finish five folds for the top two, then evaluate fusion."""

from .runtime import now, write_json

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("experiments/model_search_v2.json"))
    parser.add_argument("--output", type=Path, default=Path("optimization_v2"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / args.plan).read_text(encoding="utf-8"))
    directory = root / args.output
    directory.mkdir(exist_ok=True)
    if (directory / "status.json").exists():
        previous = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        age = (datetime.fromisoformat(now()) - datetime.fromisoformat(previous["updated_at"])).total_seconds()
        if previous["state"] in ("screening", "full_five_fold", "recovering_five_fold", "evaluating_oof_and_fusion") and age < 120:
            raise RuntimeError("A model search has a recent live heartbeat; do not launch a duplicate queue")
    write_json(directory / "plan.json", plan)
    start = now()
    status = dict(state="screening", started_at=start, updated_at=start, models=plan["models"],
                  screen_folds=plan["screen_folds"], screen_stop_step=plan["screen_stop_step"],
                  completed_screen_tasks=0, total_screen_tasks=len(plan["models"]) * len(plan["screen_folds"]),
                  failures=[])

    def publish(**values):
        status.update(values, updated_at=now())
        write_json(directory / "status.json", status)

    def launch(candidate, fold, screening):
        if (directory / "STOP").exists():
            publish(state="stopped")
            return False
        if shutil.disk_usage(root).free < 1_500_000_000:
            raise RuntimeError("Less than 1.5GB free disk space; stopping before another experiment")
        if screening:
            previous = root / candidate["run_dir"] / f"fold_{fold}/status.json"
            recorded_config = root / candidate["run_dir"] / "configuration.json"
            expected_config = root / candidate["config"]
            if previous.exists() and recorded_config.exists():
                previous_state = json.loads(previous.read_text(encoding="utf-8"))
                if (previous_state["step"] >= plan["screen_stop_step"] and
                    json.loads(recorded_config.read_text(encoding="utf-8")) == json.loads(expected_config.read_text(encoding="utf-8"))):
                    return True
        command = [sys.executable, "-u", "-m", "vessel_lab.resilient_io", "--config", candidate["config"],
                   "--run-dir", candidate["run_dir"], "--fold", str(fold), "--resume"]
        if screening:
            command += ["--stop-after", str(plan["screen_stop_step"])]
        log = directory / f"{candidate['name']}_{'screen_'+str(fold) if screening else 'full'}.log"
        publish(current_model=candidate["name"], current_fold=fold, current_run=candidate["run_dir"], current_log=str(log.relative_to(root)))
        print(json.dumps(dict(time=now(), event="launch", model=candidate["name"], fold=fold, screening=screening)), flush=True)
        with log.open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if (directory / "STOP").exists():
                    (root / candidate["run_dir"] / "STOP").touch()
                current = root / candidate["run_dir"] / "status.json"
                if current.exists():
                    try:
                        publish(current_training=json.loads(current.read_text(encoding="utf-8")))
                    except (OSError, json.JSONDecodeError):
                        pass
                time.sleep(5)
        if (directory / "STOP").exists():
            publish(state="stopped")
            return False
        if process.returncode:
            status["failures"].append(dict(model=candidate["name"], fold=fold, return_code=process.returncode, log=str(log.relative_to(root))))
            publish()
            print(json.dumps(dict(event="candidate_failed", model=candidate["name"], log=str(log))), flush=True)
            return False
        return True

    screening_rows = []
    try:
        for candidate in plan["models"]:
            scores = []
            for fold in plan["screen_folds"]:
                if not launch(candidate, fold, True):
                    if status["state"] == "stopped":
                        return
                    break
                metrics = json.loads((root / candidate["run_dir"] / f"fold_{fold}/best_metrics.json").read_text(encoding="utf-8"))
                scores.append(dict(fold=fold, dice=metrics["mean_dice_at_0_5"], best_step=metrics["step"]))
                publish(completed_screen_tasks=status["completed_screen_tasks"] + 1)
            if len(scores) == len(plan["screen_folds"]):
                row = dict(**candidate, screen_mean_dice=sum(s["dice"] for s in scores) / len(scores), folds=scores)
                screening_rows.append(row)
                write_json(directory / "screening_results.json", sorted(screening_rows, key=lambda r: r["screen_mean_dice"], reverse=True))
                print(json.dumps(dict(event="screened", **row)), flush=True)
        if not screening_rows:
            raise RuntimeError("No candidate completed both screening folds")
        ranking = sorted(screening_rows, key=lambda r: r["screen_mean_dice"], reverse=True)
        promoted = ranking[:plan["promote_top_n"]]
        write_json(directory / "promoted.json", promoted)
        publish(state="full_five_fold", promoted=[r["name"] for r in promoted])
        completed = []
        for candidate in promoted:
            if launch(candidate, "all", False):
                completed.append(candidate)
                write_json(directory / "completed_models.json", completed)
            elif status["state"] == "stopped":
                return
        publish(state="evaluating_oof_and_fusion", completed_models=[r["name"] for r in completed])
        from .export_study import compare_and_export
        result = compare_and_export(root, directory, completed, root / plan["final_export_dir"])
        publish(state="completed", result=result, current_training=None)
        print(json.dumps(dict(event="study_completed", **result)), flush=True)
    except BaseException as error:
        publish(state="failed", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
