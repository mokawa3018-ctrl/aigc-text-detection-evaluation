# AIGC 文本检测评测框架

一个面向 AIGC 文本检测器的公开、脱敏评测工具集，提供数据校验、指标计算、分组分析和版本对比能力。

项目只处理匿名标签和合成示例数据，不包含真实接口、内部模型、业务文本、账号凭证或部署信息。

## 功能

- 检查 CSV 必需字段、空值和重复 ID；
- 对非结果数据执行基于哈希的重复文本检查；
- 统一中英文 AI/人工标签；
- 计算混淆矩阵、Accuracy、Recall、FPR、Precision 和 F1；
- 区分混合集、纯 AI 集和纯人工集；
- 默认按 `source` 输出分组指标；
- 可选按长度、生成器和置信度分组；
- 可选输出阈值-指标变化曲线；
- 按匿名样本 ID 比较两个检测运行；
- 通过单元测试和集成测试验证 JSON 契约。

## 环境要求

- Python 3.10 或更高版本；
- 不需要安装第三方依赖；
- CSV 使用 UTF-8 或 UTF-8 BOM 编码。

## 仓库结构

```text
.
├─ data/
│  ├─ demo_ai_only.csv
│  ├─ demo_mixed.csv
│  └─ demo_run_b.csv
├─ docs/
│  ├─ data_quality.md
│  ├─ dataset.md
│  ├─ evaluation.md
│  ├─ metrics.md
│  └─ privacy.md
├─ reports/
│  └─ example.md
├─ src/
│  ├─ compare.py
│  ├─ evaluate.py
│  ├─ label_utils.py
│  └─ validate.py
└─ tests/
   ├─ test_compare.py
   ├─ test_evaluate.py
   ├─ test_integration.py
   └─ test_validate.py
```

## 快速开始

### 1. 校验结果文件

```powershell
python src/validate.py data/demo_mixed.csv --result-file
```

输出示例：

```json
{
  "file": "demo_mixed.csv",
  "rows": 6,
  "missing_columns": [],
  "empty_required_values": {
    "api_label": 0,
    "id": 0,
    "true_label": 0
  },
  "duplicate_id_rows": 0,
  "valid": true,
  "errors": []
}
```

`valid` 为 `false` 时，命令退出码为 `2`。

### 2. 计算评估指标

```powershell
python src/evaluate.py data/demo_mixed.csv
```

输出示例：

```json
{
  "task_type": "mixed",
  "valid_rows": 6,
  "skipped_rows": [],
  "overall": {
    "samples": 6,
    "positive_samples": 3,
    "negative_samples": 3,
    "confusion_matrix": {
      "tp": 2,
      "fn": 1,
      "fp": 1,
      "tn": 2
    },
    "accuracy": 0.6666666666666666,
    "ai_detection_rate": 0.6666666666666666,
    "human_false_positive_rate": 0.3333333333333333,
    "precision": 0.6666666666666666,
    "f1": 0.6666666666666666
  },
  "by_source": {
    "domain_a": {
      "samples": 3,
      "accuracy": 1.0
    },
    "domain_b": {
      "samples": 3,
      "accuracy": 0.3333333333333333
    }
  },
  "interpretation": "正负混合集可解释全部核心二分类指标。"
}
```

### 3. 比较两个检测运行

```powershell
python src/compare.py `
  data/demo_mixed.csv `
  data/demo_run_b.csv
```

输出示例：

```json
{
  "run_a_rows": 6,
  "run_b_rows": 6,
  "common_ids": 6,
  "run_a_only_ids": 0,
  "run_b_only_ids": 0,
  "prediction_agreement": 0.8333333333333334,
  "a_ai_b_human": 0,
  "a_human_b_ai": 1
}
```

### 4. 运行测试

```powershell
python -m unittest discover -s tests -v
```

## 输入格式

评估文件至少需要：

| 字段 | 兼容字段 | 含义 |
|---|---|---|
| `true_label` | `label` | 真实标签 |
| `api_label` | `predicted_label` | 预测标签 |

校验和版本对比默认还需要 `id`。

支持的标签包括：

| 类别 | 支持值 |
|---|---|
| AI | `1`、`ai`、`aigc`、`generated`、`machine`、`AI生成`、`AI生成文本` |
| 人工 | `0`、`human`、`manual`、`人工`、`人类`、`人类文本`、`人工文本` |

无法识别的标签行不会进入评估分母，其 CSV 行号记录在 `skipped_rows` 中。

## 增强分析

所有增强分析默认关闭，不影响原有命令和 JSON 输出。

### 按文本长度分组

```powershell
python src/evaluate.py data/demo_mixed.csv `
  --group-by-length `
  --length-thresholds 200 400
```

长度字段自动识别顺序：

1. `length`
2. `target_char_count`
3. `sent_length`
4. `original_length`

### 按生成器分组

```powershell
python src/evaluate.py data/generator_results.csv `
  --group-by-generator
```

文件不存在 `generator` 时，脚本会输出 `analysis_warnings`，不会中断基础评估。

### 置信度分桶

```powershell
python src/evaluate.py data/demo_mixed.csv `
  --confidence-buckets `
  --confidence-bin-width 0.2
```

每个置信度桶包含样本数、Accuracy、检出率、误检率、Precision 和 F1。

### 阈值曲线

仅在连续分数明确代表固定类别时使用：

```powershell
python src/evaluate.py data/dev_results.csv `
  --threshold-curve `
  --score-column confidence `
  --score-positive-class ai `
  --threshold-step 0.05 `
  --output reports/threshold_curve.json
```

如果 `confidence` 只是当前预测标签自身的置信度，不能将其用于阈值搜索。

## 指标解释

AI 文本是正类，人工文本是负类。

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
AI Detection Rate = TP / (TP + FN)
Human False Positive Rate = FP / (FP + TN)
Precision = TP / (TP + FP)
F1 = 2 × Precision × Recall / (Precision + Recall)
```

分母为零时，JSON 中对应值为 `null`。

- 正负混合集可以解释全部核心二分类指标；
- 纯 AI 集只正式报告 AI 检出率和漏检数；
- 纯人工集只正式报告人工文本误检率；
- 小样本分组必须同时展示样本数；
- 检测结果只应视为风险信号，不应单独作为高风险自动决策依据。

## 常见问题

### 1. 如何添加自定义分组维度？

在 `evaluate.py` 中新增独立的 `analyze_by_<dimension>()` 函数，通过新的显式 CLI 开关启用，并将结果追加到报告顶层。不要改变 `overall`、`by_source` 等现有字段。

新的分组函数至少应返回：

```json
{
  "column": "字段名",
  "groups": {
    "分组值": {
      "samples": 10,
      "accuracy": 0.8
    }
  }
}
```

同时需要在 `tests/test_evaluate.py` 中增加正常值、空值和缺失字段测试。

### 2. 缺少可选字段会怎样？

基础评估不会受影响。请求某项增强分析但字段不存在时，脚本跳过该分析，并在 `analysis_warnings` 中说明原因。

### 3. 缺少必需字段会怎样？

评估脚本返回退出码 `2`，并向标准错误输出缺失字段说明。校验脚本会在 JSON 中列出 `missing_columns`。

### 4. 为什么有些指标是 null？

对应指标的分母为零。例如纯 AI 数据没有人工负样本，因此无法计算人工文本误检率。

### 5. 无效标签如何处理？

评估脚本跳过该行，并将 CSV 行号写入 `skipped_rows`。如果没有任何有效标签行，命令失败并返回退出码 `2`。

### 6. 如何保存 JSON 报告？

```powershell
python src/evaluate.py data/demo_mixed.csv `
  --output reports/demo.json
```

## 扩展开发

新增分析维度时遵循以下约定：

1. 默认关闭，通过显式参数启用；
2. 不删除或重命名现有 JSON 字段；
3. 可选字段缺失时给出警告，不影响基础评估；
4. 不在错误信息或报告中输出原始文本；
5. 使用 `calculate_metrics()` 保持指标口径一致；
6. 为新函数添加类型注解和 docstring；
7. 增加单元测试和至少一个集成测试断言；
8. 只使用 Python 标准库，除非项目依赖策略正式调整。

建议函数结构：

```python
def analyze_by_dimension(
    samples: list[PreparedSample],
    column: str,
) -> dict[str, object]:
    """Calculate grouped metrics for one optional dimension."""
    ...
```

## 贡献指南

提交修改前请运行：

```powershell
python -m unittest discover -s tests -v
```

贡献内容应满足：

- 不包含真实文本、接口地址、凭证或内部环境信息；
- 不改变 AI 为正类、人工文本为负类的定义；
- 不破坏现有命令行调用；
- 不删除现有 JSON 字段；
- 新增功能默认关闭；
- 文档、示例和测试同步更新；
- 提交信息简洁说明修改目的。

仓库当前未附带开源许可证。代码使用、分发和衍生授权应以仓库后续发布的许可证为准。

## 文档

- [数据设计](docs/dataset.md)
- [数据质量](docs/data_quality.md)
- [评测流程](docs/evaluation.md)
- [指标定义](docs/metrics.md)
- [脱敏说明](docs/privacy.md)
- [示例报告](reports/example.md)

## 数据与隐私

示例数据均为人工构造，不对应任何真实用户、接口、模型服务或内部测评结果。

公开前应检查：

- URL、IP、邮箱和内部路径；
- token、secret、password、cookie；
- 原始文本、问题池和真实预测；
- 模型权重、部署信息和运行日志；
- Git 历史中是否曾包含敏感文件。
