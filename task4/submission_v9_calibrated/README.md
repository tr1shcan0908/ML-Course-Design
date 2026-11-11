# 当前最佳提交：Public Dice 0.86017

用户于 2026-11-06 确认 [submission_v9_calibrated.csv](../submission_v9_calibrated.csv) 的 Public Dice 为 **0.86017**。本地同口径五折 OOF Dice 为 **0.851748**。这是当前已确认的最佳提交。

| 阶段 | Public Dice | 相对前一步 |
| --- | ---: | ---: |
| 最初 submission.csv | 0.74323 | — |
| 原 submission_v3.csv | 0.75625 | +0.01302 |
| v3 脚本兼容导出 | 0.85697 | +0.10072 |
| 新训练模型 v8 融合 | 0.85803 | +0.00106 |
| v9 融合校准 | **0.86017** | **+0.00214** |

冻结方案：先将原 v3 原尺寸概率通过题目脚本的单调掩码转换搬运到 512×512 提交空间，并将原 0.54 阈值平移为 0.5；再将该概率与新五折双域模型的提交头概率按 **65% / 35%** 融合，最终使用 **0.42** 阈值。编码仍为血管前景、按行展开、从 1 开始，20 个 ID 按样例排列。最终 512×512 PNG 通过题目原函数编码，且 RLE 回解与 PNG 完全一致。

校准仅比较预先固定的 9 组参数。留一折选择参数后的 OOF Dice 为 **0.851545**，相对 v8 固定组合增加 0.000783；五次选择均取阈值 0.42，权重在 0.25 和 0.35 间变化。最终参数取全 OOF 最优点。反复使用这 20 张验证图存在选择偏差，因此本地估计与用户确认的平台成绩分开记录。

## 复现当前 CSV

使用项目的 Python 3.11 / PyTorch CUDA 环境。在项目根目录先从新网络的五个最佳权重重新推理，再执行固定融合：

```powershell
python -m tools.reproduce_v8 --kind dual --output submission_v8_dual_domain/rebuilt
python -m tools.reproduce_v8 --kind fusion --fusion-recipe submission_v9_calibrated/inference_recipe.json --candidate-probabilities submission_v8_dual_domain/rebuilt/test_predictions.npz --output submission_v9_calibrated/rebuilt
```

默认使用已冻结的 `submission_v3/test_probabilities_native.npz`。若需从原 v3 的 15 个检查点重建这部分概率，再执行：

```powershell
python -m vessel_lab.export_study --recipe submission_v3/inference_recipe.json --output submission_v3/reproduced
python -m tools.reproduce_v8 --kind fusion --fusion-recipe submission_v9_calibrated/inference_recipe.json --probabilities submission_v3/reproduced/test_probabilities_native.npz --candidate-probabilities submission_v8_dual_domain/rebuilt/test_predictions.npz --output submission_v9_calibrated/rebuilt_from_all_weights
```

重建工具核验检查点和配方指纹，并要求重建 CSV 与交付 CSV 的 SHA-256 完全相同。

[冻结配方](inference_recipe.json) · [逐图提交检查](submission_checks.json) · [平台记录](platform_result.json) · [9 组校准结果](../optimization_v9/results.json) · [算法与训练报告](../research/双域模型与提交尺度修正_v8.md)
