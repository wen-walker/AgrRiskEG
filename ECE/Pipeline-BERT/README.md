# Pipeline-BERT

基于 BERT 的农业气象灾害事件抽取 Pipeline，采用两阶段流水线架构：**命名实体识别（NER）** + **关系抽取（RE）**。

## 项目结构

```
Pipeline-BERT/
├── cache/              # 预训练模型本地缓存
├── data/
│   ├── raw/            # 原始标注数据
│   └── processed/      # 预处理后的数据
├── models/             # 训练好的模型权重（model_en.pth / model_re.pth）
├── output/             # 评估结果输出
└── src/
    ├── config.py       # 全局配置
    ├── model.py        # 模型定义（BIOBertNer / BIOBertCRFNer / RelationModel）
    ├── en_main.py      # NER 模型训练入口
    ├── re_main.py      # RE 模型训练入口
    ├── inference.py    # 完整 Pipeline 推理（NER → RE → 三元组）
    ├── predict.py      # 单独 NER 命令行预测
    ├── process.py      # 数据预处理
    ├── data_load.py    # 数据加载器
    ├── train.py        # 训练逻辑
    └── evaluate.py     # 评估逻辑
```

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0
- transformers >= 4.30
- pytorch-crf >= 0.7
- scikit-learn
- matplotlib

预训练模型：`hfl/chinese-roberta-wwm-ext`（首次运行自动下载至 `cache/` 目录）

## 快速开始

### 1. 训练 NER 模型

```bash
cd src
python en_main.py --dataset agriculture --model_name BIOBertNer --save_path model_en.pth --epochs 10 --batch_size 2
```

可选 `--model_name BIOBertCRFNer` 切换为 BERT+CRF 架构。

### 2. 训练 RE 模型

```bash
python re_main.py --dataset agriculture --model_name BIOBertNer --save_path model_re.pth --epochs 10 --batch_size 2
```

### 3. 运行完整 Pipeline 推理

```bash
python inference.py --model_name BIOBertNer
```

输入文本后，系统将依次完成实体识别与关系抽取，输出事件三元组：

```
< EVENT1 | TRIG | EVENT2 >
```

### 4. 单独 NER 预测

```bash
python predict.py --model_name BIOBertNer --save_path model_en.pth
```

## 实体标签

| 标签  | 说明         |
|-------|--------------|
| EVENT1 | 事件主体     |
| EVENT2 | 事件客体     |
| TRIG   | 触发词（关系）|
