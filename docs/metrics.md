# 指标说明

AI 生成文本为正类（`1`），人工文本为负类（`0`）。

| 符号 | 含义 |
|---|---|
| TP | AI 文本被正确判为 AI |
| FN | AI 文本被错误判为人工 |
| FP | 人工文本被错误判为 AI |
| TN | 人工文本被正确判为人工 |

## 核心指标

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
AI Detection Rate / Recall = TP / (TP + FN)
Human False Positive Rate = FP / (FP + TN)
Precision = TP / (TP + FP)
F1 = 2 × Precision × Recall / (Precision + Recall)
```

当分母为 0 时，对应指标记为“不适用”，不能用 0 代替。

## 适用范围

| 测试集 | Accuracy | AI 检出率 | 人工误检率 | Precision / F1 |
|---|---:|---:|---:|---:|
| 正负混合集 | 可用 | 可用 | 可用 | 可用 |
| 纯 AI 集 | 不建议表述为整体准确率 | 可用 | 不适用 | 不适用 |
| 纯人工集 | 不适用 | 不适用 | 可用 | 不适用 |

纯 AI 集中“预测正确比例”在数值上等于 AI 检出率，但报告中应称为检出率，避免误导。

## 扩展指标

若检测器能提供方向明确、跨样本可比较的连续 AI 分数，可进一步计算 ROC-AUC、PR-AUC、固定 FPR 下召回率、固定召回率下 FPR、Brier Score 和 ECE。

若接口返回的是“已预测标签的置信度”，不能直接将其当作 `P(AI)`，也不能通过简单反转构造 AI 概率。

## 性能指标

服务评测建议同时记录：

- 成功率与错误率；
- P50、P95、P99 响应耗时；
- 吞吐量；
- 截断样本数和截断比例；
- 输入字符数或 token 数；
- 推理实现与版本代号。

## 分组指标

启用增强分析参数后，评估脚本可以按以下维度重复计算同一组二分类指标：

- `source`：默认启用；
- 文本长度：通过 `--group-by-length` 启用；
- `generator`：通过 `--group-by-generator` 启用；
- 置信度区间：通过 `--confidence-buckets` 启用。

每个分组必须同时报告样本数。样本量过小的分组只能用于定位问题，不能单独形成稳定的模型能力结论。

### 长度分组

脚本按以下顺序自动选择第一个存在的长度字段：

1. `length`
2. `target_char_count`
3. `sent_length`
4. `original_length`

默认分界为：

- 短文本：长度小于或等于 200；
- 中等文本：长度大于 200 且小于或等于 500；
- 长文本：长度大于 500。

可通过 `--length-thresholds SHORT_MAX MEDIUM_MAX` 调整分界。

### 置信度分桶

`--confidence-buckets` 用于观察不同置信度区间内的样本数和准确率。默认桶宽为 0.2。

置信度分桶不要求分数一定是 `P(AI)`，但同一列中的数值必须具有一致且可比较的语义。

### 阈值曲线

只有当连续分数明确表示固定类别的概率或可比较得分时，才能使用 `--threshold-curve`。

- `--score-positive-class ai`：分数越高越倾向 AI；
- `--score-positive-class human`：分数越高越倾向人工，脚本内部转换为 AI 分数；
- `--threshold-step`：控制阈值采样间隔。

如果 `confidence` 只是“当前预测标签的置信度”，不能据此计算阈值曲线。

## 单类别数据的解释

脚本根据可用样本执行数学计算，因此部分单类别数据可能仍产生 Accuracy、Precision 或 F1 数值。但这些数值不代表完整二分类性能：

- 纯 AI 集只正式报告样本数、AI 检出率和漏检数；
- 纯人工集只正式报告样本数和人工文本误检率；
- 完整 Accuracy、Precision 和 F1 只在正负混合集上解释；
- JSON 中的 `null` 表示对应指标分母为 0。
