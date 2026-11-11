# Task 4：视网膜血管分割最终 v9 源码

对应 [notebookdfa66d62eb](https://www.kaggle.com/code/changyangli0908/notebookdfa66d62eb)，归档 Notebook v1。最终文件 submission_v9_calibrated.csv，Kaggle Public Dice=0.86017，本地同口径 OOF Dice=0.851748。

本目录源文件从最终 v9_reproduce.zip 原样提取，42 个 Python 文件及归档元数据均按原 package_manifest 校验。notebookdfa66d62eb_v1.ipynb 为该最终方案的原 Notebook，默认 replay，支持 infer。原始归档还包含权重与概率文件；本源码仓库保留其路径/指纹和对应实现。

最终流程：旧 v3 原尺寸概率按题目脚本转换到提交尺度，与五折轻量双域高分辨率网络的提交头概率按 0.65/0.35 融合，阈值 0.42。v3 是 CSA U-Net、ResNet18 U-Net、EfficientNet-B0 U-Net 三架构、每架构五折；vendor 中保留的 FR-UNet 属于可选实现，不是最终 v3 成员。

核心入口：

- vessel_lab/dual_domain.py：网络与双域损失。
- vessel_lab/dual_domain_study.py：五折训练、数据处理和 D4 推断。
- vessel_lab/submission_domain.py、dual_domain_fusion.py：提交尺度变换与融合。
- vessel_lab/fusion_calibration.py：固定候选网格的留一折校准。
- tools/reproduce_v8.py：重放及重新推断入口，文件名沿用 v8，传入 v9 配方即可。

按原 REPRODUCE.md 放置图像、五个新 checkpoint 和旧 v3 概率后，在本目录执行：

    python -m tools.reproduce_v8 --kind dual --output reproduced_model
    python -m tools.reproduce_v8 --kind fusion --fusion-recipe submission_v9_calibrated/inference_recipe.json --candidate-probabilities reproduced_model/test_predictions.npz --output reproduced_final

完整重训还需原 train/test、experiments 中的原始数据及折指纹，以及配方要求的预训练文件。依赖版本见 requirements.lock.txt。原 v3 的 15 份旧 checkpoint 未包含于 v9 包；如需重算其概率，需提供原 v3 归档权重。Kaggle Notebook 重放也需要原 v9 输入资产。

原始源码包同时带有历史辅助模块，此次按归档完整保留，最终选型以 submission_v9_calibrated/inference_recipe.json 为准。上游出处与许可证保存在 vendor。最终 CSV SHA-256：8d49aa138f5ccbbe1929601d4044b38dd3c3cb3e0ef61258d93382d31cf61cc9。
