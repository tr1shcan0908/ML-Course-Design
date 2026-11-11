# Task 3：中心凹定位 heatmap+fov512 最终源码

最终记录为原工程 outputs/round2_fov，选择 **heatmap_512 0.5 + fov_heatmap512 0.5**。80 张 OOF 坐标 RMSE=12.4893407615 px，AED=11.1238787297 px；20 张测试图导出 40 行坐标。该目录没有对应的隐藏测试成绩。

fovea 下收录最终选择记录所指纹约束的原模块，以及必要的 context 依赖。8 个 source_sha256 均与最终 selection.json 一致。训练为 ImageNet ResNet-18 热图网络、Gaussian 目标交叉熵加归一化坐标 MSE、DSNT 期望解码；FOV 流按图像内容裁剪后 letterbox 到 512。

从原始数据重训（在本任务目录中提供原 train/test、fovea_localization_train_GT.csv）：

    python -m pip install -r requirements-training.txt
    python -m fovea.experiment prepare --size 512 --roi 768
    python -m fovea.experiment train --run runs/heatmap_512 --size 512
    python -m fovea.experiment predict --run runs/heatmap_512 --variant pred
    python -m fovea.fov --size 512
    python -m fovea.experiment train --run runs/fov_heatmap512 --manifest artifacts/manifest_fov_512.json --size 512 --epochs 120 --batch-size 4
    python -m fovea.experiment predict --run runs/fov_heatmap512 --variant pred

最后按 selection.json 对两流原图坐标各占 0.5 融合；本次新增的 replay_submission.py 可在无需训练和第三方库的情况下，从已保存两流预测重建并核对最终提交：

    python replay_submission.py --output reproduced/submission.csv

replay_submission.py 是此次源码归档新增的校验工具，未冒充历史训练代码。analysis/compare_round2.py 原样保留完整 13 候选评估逻辑，运行该历史完整比较还需原图、六个候选来源的预测、完整训练日志；当前仓库保留选定双流的记录。

provenance/EXPORT_MANIFEST.json 记录每个文件的来源与 SHA-256。原图、预训练权重和五折模型权重需另行准备。这里归档用户指定的 round2_fov，后续 round3–round13 并未替换此最终版本。
