# End2End-T5

基于 Mengzi-T5 的农业气象灾害事件端到端三元组抽取系统。

## 项目简介

本项目使用序列到序列（Seq2Seq）方法，从农业灾害文本中自动抽取 **原因-触发词-结果** 三元组。模型将输入文本转换为结构化格式 `<前件|触发词|后件>`，支持显性触发词和隐性直连两种模式。

训练数据来源于标注的农业灾害语料，标注类型包括 EVENT1（原因）、TRIG（触发词）、EVENT2（结果）及其关系。

## 目录结构

```
End2End-T5/
├── data/
│   ├── raw/              # 原始标注数据（Label Studio JSON）
│   └── processed/        # 处理后的训练数据（自动生成）
├── src/
│   ├── main.py           # 主入口（训练+评估）
│   ├── config.py         # 配置参数
│   ├── process.py        # 数据预处理
│   ├── data_load.py      # 数据加载与 Tokenizer
│   ├── model.py          # T5 模型封装
│   ├── train.py          # 训练流程
│   ├── evaluate.py       # 评估（实体级+三元组级 P/R/F1）
│   ├── predict.py        # 单句推理交互
│ 
├── models/               # 保存的模型权重（训练后生成）
└── cache/                # 预训练模型缓存
```

## 环境依赖

- Python >= 3.8
- PyTorch >= 1.12
- transformers >= 4.30
- datasets >= 2.14
- sentence-transformers >= 2.2
- tqdm
- numpy

```bash
pip install torch transformers datasets sentence-transformers tqdm numpy
```

## 预训练模型

- 基座模型：[Langboat/mengzi-t5-base](https://huggingface.co/Langboat/mengzi-t5-base)
- 语义评估模型：[shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese)（用于评估阶段实体软匹配）

## 使用方法

### 1. 训练与评估

```bash
cd src
python main.py --dataset example --epochs 20 --batch_size 12 --learning_rate 5e-5
```

可选参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | example | 数据集名称，对应 `data/raw/` 下的目录 |
| `--epochs` | 20 | 训练轮数 |
| `--batch_size` | 12 | 批大小 |
| `--learning_rate` | 5e-5 | 学习率 |
| `--device` | cuda/cpu | 训练设备 |
| `--save_path` | models/ | 模型保存路径 |

### 2. 单句推理

```bash
cd src
python predict.py
```

启动交互式抽取，输入句子即可获得结构化三元组。

### 3. 批量抽取

```bash
cd src
python extract.py
```

从文本文件中批量抽取三元组，输出为 CSV 格式。需先在 `extract.py` 中配置输入输出路径。

## 数据格式

**输入数据**：放置在 `data/raw/<dataset>/` 下，包含 `train.json`、`val.json`、`test.json`。

**处理后数据**：自动转换为 `{"input_text": "...", "target_text": "<前件|触发词|后件>"}` 格式。
