# Task 2：人脸表情识别最终 V2 源码

对应 [notebooke316458626，V2](https://www.kaggle.com/code/changyangli0908/notebooke316458626?scriptVersionId=359418547)，提交编号 57317740，Public=0.742。最终为 ConvNeXt-Tiny、QCS、EmoNeXt-Tiny、EmotiEff-B2 各 0.25；FP32 推断、无 Flip TTA。开发验证 Accuracy=72.3794%，Macro-F1=70.7087%。

`notebooke316458626_v2.ipynb` 是最终 V2 的完整源码。根目录 fer、scripts、third_party 从该次云端输出的 fer_project_source.zip 提取，其 39 个内嵌源文件与 Notebook 逐字节相同。保留了上游许可证与 third_party/SOURCES.json 的来源记录。

Notebook 使用 Kaggle 输入资产，GPU 可重新推断，CPU 默认明确使用已保存概率重放；该 V2 的云端实际执行属于缓存概率重放。四个模型的真实训练代码在 fer/train.py，原训练配置位于 experiments/configs。训练入口示例（先在本任务目录准备原始 train/test 和预训练权重）：

    python -m pip install -r requirements-experiments.txt
    python -m fer.prepare --root . --output artifacts/data
    python -m fer.train --config experiments/configs/convnext_tiny_s42.json

另外三个成员使用各自同名配置运行。准备权重的脚本在 scripts/download_weights.py、scripts/prepare_emotieff.py；准确路径见配置和源代码。运行完整最终推断与提交导出使用 Notebook，而非 fer.selection（后者没有 CLI 导出入口）。

原训练配方的 source_sha256 与最终部署源码存在已记录差异（包括加载完整 QCS checkpoint 时跳过单独 IR50 初始化等适配）。provenance/source_comparison.json 同时记录两组指纹；能找到原实验指纹的模块额外原样保存在 provenance/original_training_modules，不能用部署文件的指纹冒充训练快照。

reference 保存最终融合配方、数据清洗计数、验证结果、实际提交 CSV 和平台记录。最终 CSV SHA-256：3b0b3912924dab7d1866673e91daddf36e904ef167fb73deb1f3e80db2353788。

本仓库收录源码、配置及小型结果记录，运行 Notebook 仍需原 Kaggle 数据/权重资产。最终选型为 0.742 方案；后续本地 72.53% 的 refine 方案不是本目录的最终提交。
