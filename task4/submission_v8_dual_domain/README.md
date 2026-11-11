# 双域高分辨率模型：五折实验已完成

随机初始化的 561,899 参数网络完成五折，每折 3,000 步，共 15,000 步。训练同时监督最近邻缩放的细节标签和题目脚本转换后的提交标签，使用多尺度残差特征融合、连续旋转与缩放、EMA 和 D4 推理。

| 方案 | 同口径本地 OOF Dice | 用户确认的 Public Dice |
| --- | ---: | ---: |
| 修正导出后的 v3 | 0.849664 | 0.85697 |
| 新网络单独使用 | 0.836583 | 未测试 |
| 75% v3 + 25% 新网络，阈值 0.50 | 0.850762 | 0.85803 |
| 65% v3 + 35% 新网络，阈值 0.42 | **0.851748** | **0.86017** |

新网络单独使用不如原融合，因此不推荐本目录的单模型 CSV 作为最终提交；它在组合中提供了互补信息。当前最佳提交为 [submission_v9_calibrated.csv](../submission_v9_calibrated.csv)。上表的 v3 概率经过不读取标签的单调转换后才与新网络融合，两个成员均位于 512×512 提交空间。

折 0–4 的最佳 EMA 检查点依次为 3,000、2,500、2,500、2,500、1,500 步。五折训练总耗时约 18 分钟，训练峰值分配显存约 563 MiB（不含 CUDA 上下文及其他进程）。全部留出恒等推理指标从权重精确重放，随后生成 D4 OOF 与测试概率。独立进程从五个权重重新生成了相同 SHA-256 的单模型 CSV。

```powershell
python -m vessel_lab.dual_domain_study
python -m tools.reproduce_v8 --kind dual --output submission_v8_dual_domain/rebuilt
```

[实验结果](../runs/dual_domain_v8/results.json) · [训练曲线](../runs/dual_domain_v8/training_curves.png) · [融合全部候选](../runs/dual_domain_v8/fusion_results.json) · [完整方案](../research/双域模型与提交尺度修正_v8.md)
