# ML-Course-Design

四个课程项目的指定最终提交源码归档，整理日期 2026-11-11。

| 目录 | 最终方案 | 最终结果 | 原记录 |
|---|---|---|---|
| [task1](task1) | DINOv2-B 第 8 层微调 + SVC，seed 142 | OOF Accuracy 96.20% | [Kaggle Notebook](https://www.kaggle.com/code/changyangli0908/notebook4a34c3d49e/notebook) |
| [task2](task2) | ConvNeXt / QCS / EmoNeXt / EmotiEff-B2 各 0.25 | Public 0.742 | [指定 V2](https://www.kaggle.com/code/changyangli0908/notebooke316458626?scriptVersionId=359418547) |
| [task3](task3) | heatmap_512 + fov_heatmap512 各 0.5 | OOF 坐标 RMSE 12.4893 px | 原 outputs/round2_fov |
| [task4](task4) | v9 双域融合 0.65/0.35，阈值 0.42 | Public Dice 0.86017 | [Kaggle Notebook](https://www.kaggle.com/code/changyangli0908/notebookdfa66d62eb) |

各目录包含最终提交对应的训练/推断源代码、Notebook（Task 1/2/4）、配置、依赖、最终 CSV 和来源指纹，运行方法见各自 README。原图及大型模型权重需另行准备。Task 2/4 的云端最终执行属于缓存概率重放，完整模型训练和推断实现也随源码归档。

`submission_index.json` 列出四个最终版本与 CSV SHA-256；每个任务的 `provenance/EXPORT_MANIFEST.json` 记录文件来源、字节数、指纹和已完成的源码一致性检查。原始代码按最终 Notebook / 源码归档包 / 指纹匹配记录提取；新增 README、路径可移植配置和重放校验工具明确标注。

Task 1/3 的本地 OOF 与 Task 2/4 的平台 Public 是不同评估口径，不能互相当作隐藏测试成绩。源码归档不声称重新训练可逐字节复现，具体依赖环境与缺少的外部资产见各任务说明。

第三方实现的作者署名、来源和已有许可证保留在各任务目录中；没有为所有第三方代码统一改授许可证。

下载或克隆仓库后，在根目录运行 `python verify_sources.py` 可检查归档文件与四个最终 CSV 的 SHA-256。`.gitattributes` 保留原始文件换行字节，以便 Windows 上核对原指纹。
