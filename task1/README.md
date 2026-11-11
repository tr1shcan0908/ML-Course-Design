# Task 1：植物幼苗分类最终源码

对应 [notebook4a34c3d49e](https://www.kaggle.com/code/changyangli0908/notebook4a34c3d49e/notebook)，归档版本 2。最终方案为 DINOv2-B/14（4 register tokens）第 8 层表征、仅第 7/8 block 微调、前景加权特征、训练折内 SVC，seed=142。五折 OOF Accuracy=96.20%，Macro-F1=96.1981%。

`notebook4a34c3d49e.ipynb` 为实际导入目标 Notebook 的原始完整源码，包含训练、推断、原始五折划分与 DINOv2 源码。它是首选运行入口。原记录为 Quick Save，未把本地成绩表述为云端重训成绩。

1. 安装与原实验兼容的 PyTorch/CUDA，或在 Kaggle 上打开 Notebook。
2. 提供原任务的 500 张训练图、378 张测试图与提交模板；按配置单元格设置 DATA_ROOT。
3. 运行全部单元格：336×336、batch 4、累积 4、预热 2 轮、微调 6 轮，最终五折测试概率平均。

本目录 `.py` 是本地实验原脚本及其本地导入依赖；`vendor/dinov2` 从最终 Notebook 内嵌包原样提取。脚本版 `load_model` 使用 torch.hub 本地缓存中的 `facebookresearch_dinov2_main`；Notebook 会自行准备相同源码并校验官方预训练权重。权重文件为 `dinov2_vitb14_reg4_pretrain.pth`，SHA-256 见 provenance/source_manifest.json。

`reference` 保存最终结果、OOF 与审计。config.portable.json 只将机器绝对路径改成相对路径，模型参数未改。原 7 个实验源文件、DINOv2 源码和固定折划分均按原指纹核验。依赖源码还包括原实验的辅助模型模块，它们用于满足导入，并不代表最终选型。

本仓库收录源码、配置及小型结果记录。原图和模型权重需自行准备。DINOv2 的 Apache 2.0 许可证保留于 vendor。
