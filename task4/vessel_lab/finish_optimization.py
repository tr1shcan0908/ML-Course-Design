"""Continue from the live recovery queue through refinement and portable delivery."""

from .resilient_io import install
install()
from .runtime import now, write_json

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-python", required=True)
    parser.add_argument("--resume-delivery", action="store_true", help="Reuse completed training/refinement; regenerate report and package")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root / "optimization_v3"
    directory.mkdir(exist_ok=True)
    status = dict(state="waiting_for_training", started_at=now(), updated_at=now())

    def publish(**values):
        status.update(values, updated_at=now())
        write_json(directory / "workflow_status.json", status)

    def launch(stage, command):
        if (root / "optimization_v2/STOP").exists():
            publish(state="stopped")
            return False
        log = directory / f"{stage}.log"
        publish(state=stage, current_log=log.relative_to(root).as_posix())
        with log.open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                publish()
                time.sleep(5)
        if process.returncode:
            raise RuntimeError(f"{stage} failed; inspect {log}")
        return True

    try:
        publish()
        while not args.resume_delivery:
            if (root / "optimization_v2/STOP").exists():
                publish(state="stopped")
                return
            recovery = read(root / "optimization_v2/recovery_status.json")
            if recovery["state"] == "completed":
                break
            if recovery["state"] in ("failed", "stopped"):
                raise RuntimeError("Training recovery did not complete: " + recovery["state"])
            publish(current_training_model=recovery.get("current_model"))
            time.sleep(5)
        if args.resume_delivery:
            if read(directory / "status.json")["state"] != "completed":
                raise ValueError("Delivery resume requires completed inference refinement")
        else:
            if not launch("refining_inference", [sys.executable, "-u", "-m", "vessel_lab.refine_inference"]):
                return
        v2, v3 = read(root / "optimization_v2/result.json"), read(directory / "result.json")
        selected = "submission_v3" if v3["new_submission_exported"] else ("submission_v2" if v2["new_submission_exported"] else "submission")
        if selected != "submission":
            if not launch("reporting_results", [args.report_python, "-u", "-m", "tools.report_optimized_results", "--submission", selected]):
                return
            if not launch("packaging_and_verifying", [sys.executable, "-u", "-m", "tools.package_optimized_submission", "--submission", selected]):
                return
        chosen = v3 if selected == "submission_v3" else v2
        score = chosen["best_calibrated_oof_dice"]
        csv_name = chosen.get("submission_csv", "submission.csv")
        manifest = read(root / selected / "deliverable_manifest.json")
        links = ["# 本轮优化最终状态", "", f"最终选择：`{csv_name}`；本地校准OOF Dice：{score:.6f}。", "",
                 f"v2最佳：{v2['best_calibrated_oof_dice']:.6f}；补充推理比较最佳：{v3['best_calibrated_oof_dice']:.6f}。", "",
                 "这些数值参与checkpoint、模型、TTA、阈值和融合选择，不是独立平台成绩。", "",
                 f"[结果与提交说明](../{selected}/README.md)；[CSV](../{csv_name})。", "",
                 "已完成完整五折、补充推理比较及复现检查；原v1提交和源码保持不变。未上传平台。"]
        (directory / "README.md").write_text("\n".join(links)+"\n", encoding="utf-8")
        main_readme = root / "README.md"
        lines = main_readme.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if line.startswith("当前阶段（"):
                lines[index] = (f"当前阶段（2026-11-04）：第二轮两个入围模型五折训练、八方向/裁块推理与固定权重融合比较已完成。"
                                f"当前选择的本地校准OOF Dice为{score:.6f}，提交文件为[{csv_name}]({csv_name})。"
                                f"[结果与复现说明]({selected}/README.md)；尚未线上提交。")
        main_readme.write_text("\n".join(lines)+"\n", encoding="utf-8")
        publish(state="completed", selected_submission=csv_name, selected_directory=selected,
                selected_oof_dice=score, gain_over_original_v1=score-0.8161414689099453,
                portable_reproduction_verified=manifest.get("extracted_package_execution_passed", manifest.get("portable_zip_rebuild_passed",False)))
        print(json.dumps(status, indent=2), flush=True)
    except BaseException as error:
        publish(state="failed", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
