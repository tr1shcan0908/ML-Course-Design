"""Wait for the active search, resume interrupted finalists, then compare all OOF."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .resilient_io import install

install()
from .runtime import now, source_digest, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("optimization_v2"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study = root / args.study
    recovery = dict(state="waiting_for_current_search", started_at=now(), updated_at=now(),
                    reason="Resume selected finalists after a transient Windows status-file sharing failure",
                    frozen_training_source_sha256=source_digest(), recovered_models=[],
                    recovery_sources_sha256={name: hashlib.sha256((root / "vessel_lab" / name).read_bytes()).hexdigest()
                                             for name in ("resilient_io.py", "recover_study.py")})
    status = None

    def publish(**values):
        recovery.update(values, updated_at=now())
        write_json(study / "recovery_status.json", recovery)

    def main_status(**values):
        status.update(values, updated_at=now(), recovery=recovery.copy())
        write_json(study / "status.json", status)

    publish()
    try:
        while True:
            status = read(study / "status.json")
            if (study / "STOP").exists() or status["state"] == "stopped":
                publish(state="stopped")
                return
            if status["state"] in ("completed", "failed"):
                break
            publish(current_search_state=status["state"])
            time.sleep(5)
        write_json(study / "primary_search_result_before_recovery.json", status)
        promoted = read(study / "promoted.json")
        for candidate in promoted:
            run = root / candidate["run_dir"]
            if (run / "status.json").exists() and read(run / "status.json")["state"] == "completed_all_folds":
                continue
            if (run / "STOP").exists():
                publish(state="stopped", current_model=candidate["name"])
                main_status(state="stopped")
                return
            if read(run / "environment.json")["source_sha256"] != source_digest():
                raise ValueError("Training source differs from the interrupted run; refusing recovery")
            publish(state="resuming_finalist", current_model=candidate["name"])
            log = study / f"{candidate['name']}_recovery.log"
            main_status(state="recovering_five_fold", current_model=candidate["name"],
                        current_run=candidate["run_dir"], current_log=log.relative_to(root).as_posix())
            command = [sys.executable, "-u", "-m", "vessel_lab.resilient_io", "--config", candidate["config"],
                       "--run-dir", candidate["run_dir"], "--fold", "all", "--resume"]
            with log.open("a", encoding="utf-8") as stream:
                child = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    if (study / "STOP").exists():
                        (run / "STOP").touch()
                    try:
                        current = read(run / "status.json")
                    except (OSError, json.JSONDecodeError):
                        current = None
                    publish()
                    main_status(current_training=current)
                    time.sleep(5)
            if (study / "STOP").exists():
                publish(state="stopped")
                main_status(state="stopped")
                return
            if child.returncode or read(run / "status.json")["state"] != "completed_all_folds":
                raise RuntimeError(f"Finalist recovery failed: {candidate['name']}; see {log}")
            recovery["recovered_models"].append(candidate["name"])
        completed = [c for c in promoted if read(root / c["run_dir"] / "status.json")["state"] == "completed_all_folds"]
        write_json(study / "completed_models.json", completed)
        publish(state="evaluating_all_finalists")
        main_status(state="evaluating_oof_and_fusion", completed_models=[c["name"] for c in completed], current_training=None)
        from .export_study import compare_and_export
        plan = read(study / "plan.json")
        result = compare_and_export(root, study, completed, root / plan["final_export_dir"])
        if hashlib.sha256((root / "submission.csv").read_bytes()).hexdigest() != plan["original_submission_sha256"]:
            raise ValueError("Original submission changed")
        publish(state="completed", result=result)
        main_status(state="completed", result=result, current_training=None,
                    completed_models=[c["name"] for c in completed])
        print(json.dumps(dict(event="recovery_and_comparison_completed", **result)), flush=True)
    except BaseException as error:
        publish(state="failed", error=str(error), traceback=traceback.format_exc())
        if status is not None and status["state"] in ("recovering_five_fold", "evaluating_oof_and_fusion"):
            main_status(state="failed", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
