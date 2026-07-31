# AIGC 文本检测评测框架

## 背景与动机

AIGC 文本检测模型完成部署后，仍需通过系统化评测确认其在不同文本类型、长度和生成来源下的表现。实际评测常遇到重复数据、空标签、指标口径不一致，以及不同模型版本难以公平比较等问题。本框架提供数据质检、指标计算、结果对比和脱敏管理工具，用于建立可复用的评测流程。

项目源于一次真实的文本检测模型复现与评测实践。公开版本将检测服务统一抽象为“黑盒文本检测器”，不包含公司名称、接口地址、端口、账号凭证、服务器信息、内部代码、原始文本或逐条真实结果。

## 已完成的工作

| 功能模块 | 对应能力 |
|---|---|
| 数据质检 | 设计整段文本、分句文本和不同大模型生成文本三类评测任务；对万级样本执行数据完整性、重复文本、空标签和截断检查。 |
| 指标计算 | 统一 AI 为正类、人工文本为负类的标签与指标口径；支持 Accuracy、AI 检出率、人工文本误检率及分来源统计；将纯 AI 压力集与正负混合测试集分开解释，避免误报指标被误用。 |
| 对比分析 | 支持两个推理实现或两个版本的预测一致性比较。 |
| 结果管理 | 建立公开材料与内部资产的隔离规则。 |

## 关键设计决策

正负混合测试集同时包含 AI 与人工文本，可计算完整的二分类指标；纯 AI 测试集缺少负样本，只用于观察不同生成来源的检出率，不能用于计算人工文本误检率。

分句文本更短、上下文信息更少，检测难度和样本相关性均不同于整段文本。因此两者被定义为独立任务，分别统计结果，避免直接比较总体百分比造成误判。

## 仓库结构

```text
.
├─ docs/
│  ├─ dataset.md
│  ├─ data_quality.md
│  ├─ evaluation.md
│  ├─ metrics.md
│  └─ privacy.md
├─ data/
│  ├─ demo_mixed.csv
│  ├─ demo_ai_only.csv
│  └─ demo_run_b.csv
├─ src/
│  ├─ validate.py
│  ├─ evaluate.py
│  └─ compare.py
├─ reports/
│  └─ example.md
├─ tests/
│  └─ test_evaluate.py
└─ .gitignore
```

## 快速开始

本项目仅使用 Python 标准库，建议 Python 3.10 及以上版本。

| 命令 | 用途 |
|---|---|
| `python src/validate.py data/demo_mixed.csv --result-file` | 检查结果文件的字段、空值和重复 ID。 |
| `python src/evaluate.py data/demo_mixed.csv` | 计算整体指标及按来源分组的指标。 |
| `python src/compare.py data/demo_mixed.csv data/demo_run_b.csv` | 按样本 ID 对比两个运行版本的预测一致性。 |
| `python -m unittest discover -s tests -v` | 运行指标计算相关的单元测试。 |

示例数据均为人工构造，不对应任何真实接口、用户文本或内部测评结果。

## 输入数据格式要求

评测脚本读取 UTF-8 编码的 CSV 文件。结果文件至少需要以下字段：

| 列名 | 是否必需 | 含义 |
|---|---:|---|
| `id` | 是 | 样本唯一标识；也可在数据准备阶段使用 `sample_id`，进入脚本前需统一为 `id`。 |
| `true_label` | 是 | 真实标签，`1` 表示 AI 文本，`0` 表示人工文本；也兼容列名 `label`。 |
| `api_label` | 是 | 检测器预测标签；也兼容列名 `predicted_label`。 |
| `source` | 否 | 文本主题、来源或匿名分组，用于分组统计；缺失时记为“未分类”。 |
| `confidence` | 否 | 检测器返回的置信度。除非分数方向已经确认，否则不将其解释为 AI 概率。 |
| `original_length` | 否 | 原始文本字符数。 |
| `sent_length` | 否 | 实际送检字符数。 |
| `truncated` | 否 | 是否因长度限制发生截断。 |

最小输入示例：

```csv
id,true_label,api_label,confidence
demo-001,1,AI生成文本,0.91
demo-002,0,人类文本,0.88
```

## 输出示例

运行 `python src/evaluate.py data/demo_mixed.csv` 后，脚本输出 JSON。以下数值来自仓库中的人工构造示例：

```json
{
  "task_type": "mixed",
  "overall": {
    "samples": 6,
    "confusion_matrix": {
      "tp": 2,
      "fn": 1,
      "fp": 1,
      "tn": 2
    },
    "accuracy": 0.6667,
    "ai_detection_rate": 0.6667,
    "human_false_positive_rate": 0.3333
  }
}
```

其中：

- `accuracy`：整体准确率；
- `ai_detection_rate`：AI 文本检出率；
- `human_false_positive_rate`：人工文本误检率；
- `confusion_matrix`：由 TP、FN、FP、TN 组成的混淆矩阵。

## 结果解释原则

- 正负混合测试集可以计算 Accuracy、AI 检出率和人工文本误检率；
- 纯 AI 测试集只能计算检出率，不能据此声称整体准确率或人工文本误检率；
- 分句与整段文本属于不同任务，不能直接比较总体百分比；
- 阈值、规则或模型选择必须在开发集/验证集完成，冻结测试集仅用于最终报告；
- 检测结果是风险信号，不应单独作为高风险业务中的自动处置依据。

## 许可与数据

仓库不附带真实测评数据。使用者需自行确认数据授权、隐私、版权和模型许可。
