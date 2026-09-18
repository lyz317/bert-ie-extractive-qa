# BERT-based Information Extraction via Extractive QA

基于 BERT 的英文文本信息抽取项目，将信息抽取任务建模为**抽取式问答（Extractive QA）**，通过 HuggingFace Transformers 实现从数据处理、模型训练到预测输出的完整 Pipeline。

## 任务定义

输入一段英文文本与一个描述性问题，输出文本中所有符合问题要求的词汇列表。

**示例：**

```
输入文本：The great galleys of the Knights of St. John were sweeping slowly...
问题：Extract all adjectives that describe the appearance
输出：["great", "broad", "crimson", "stately"]
```

## 技术方案

| 模块 | 方案 |
|------|------|
| 模型 | `bert-base-uncased`（支持切换 RoBERTa / DeBERTa / ALBERT） |
| 框架 | PyTorch + HuggingFace Transformers（Trainer API） |
| 建模方式 | 抽取式 QA：问答头输出 start/end logits，offset mapping 映射回原文 span |
| 训练策略 | 学习率 2e-5，batch_size=8，max_length=512，warmup_ratio=0.1，EarlyStopping(patience=3) |
| 评估指标 | Exact Match (EM) + Token 级 F1 |
| 数据划分 | 训练集自动划分 20% 为验证集（seed=42） |

## 项目结构

```
├── main.py                  # 统一入口（train / evaluate / predict / create_val / info）
├── requirements.txt         # 依赖清单
├── data/                    # 数据目录
│   ├── train.jsonl          # 训练集（2416 条，每条含多个抽取任务）
│   └── test_a.jsonl         # 测试集（511 条）
├── results/
│   └── submit.jsonl        # 最终预测提交（1466 条）
├── src/                     # 源代码
│   ├── config.py            # 全局配置（数据路径、模型、超参数）
│   ├── data_utils.py        # 数据处理：JSONL 解析、QA 特征构造、答案 span 定位
│   ├── model_utils.py       # 模型加载与 Trainer 构建
│   ├── train.py             # 训练脚本
│   ├── evaluate.py          # 评估脚本（EM / F1）
│   └── predict.py           # 预测脚本
└── scripts/                 # 运行脚本
    ├── quick_start.sh       # 一键运行
    ├── run_train.sh         # 训练
    ├── run_evaluate.sh      # 评估
    └── run_predict.sh       # 预测
```

## 快速开始

### 环境准备

```bash
pip install -r requirements.txt
```

> 已内置 HuggingFace 国内镜像（hf-mirror.com），无需额外配置网络。

### 训练模型

```bash
python main.py train
```

可选参数：

```bash
python main.py train --model_name roberta-base --batch_size 16 --num_epochs 3
```

### 生成预测

```bash
python main.py predict
```

预测结果输出至 `outputs/predictions/predictions.jsonl`，格式为 `{id, question_id, answer}`。

### 评估模型

```bash
python main.py evaluate
```

## 核心方法

1. **数据处理**：解析 JSONL 多任务格式，将每条文本下的多个抽取任务展开为独立样本；使用 tokenizer 编码 question + text，通过 `return_offsets_mapping=True` 获取 token 到原文的字符映射。

2. **答案定位**：在原文中查找答案字符串的字符位置，再通过 offset mapping 反查对应的 token index，作为 start/end 训练标签。

3. **模型预测**：模型输出 start/end logits，选取概率最大的 token 区间，通过 offset mapping 映射回原文词汇，得到最终抽取结果。

## 数据集规模

| 数据集 | 样本数 | 说明 |
|--------|--------|------|
| 训练集 | 2416 条 | 每条文本含多个抽取任务 |
| 验证集 | 从训练集划分 20% | 用于 EarlyStopping |
| 测试集 | 511 条 | 输出 1466 条预测 |

## Author

李樱姿 · 首都师范大学人工智能专业
