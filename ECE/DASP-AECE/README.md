# DASP-AECE

**D**ual-**A**nchor **S**et **P**rediction for **A**gricultural **E**vent **C**ausality **E**xtraction

基于双锚点集合预测的农业事件因果关系联合抽取模型。

---

## 任务说明

从农业灾害文本中联合抽取因果三元组：

| 类型 | 结构 |
|------|------|
| 显式因果 (Explicit) | 原因事件 → 触发词 → 结果事件 |
| 隐式因果 (Implicit) | 原因事件 → 结果事件（无触发词） |

---

## 模型结构

```
输入文本
  │
  ▼
BERT Encoder (chinese-roberta-wwm-ext)
  │  前置 [EXP] [IMP] 双锚点 Token
  │
  ├─→ h_exp → 显式 Query 生成器 ─┐
  │                                ├─→ Transformer Decoder → Span Pointer Heads
  └─→ h_imp → 隐式 Query 生成器 ─┘        (cause / trigger / effect)
```

**关键设计：**
- **双锚点路由**：`[EXP]` / `[IMP]` 分别聚合显式与隐式语义，解耦两类查询
- **集合预测框架**：固定数量的 Query Slot，通过匈牙利算法进行二部图匹配
- **ISPD 损失**：约束两类锚点表征的余弦相似度 ≤ 0.1，强制语义解耦
- **辅助监督分支**：预测文本是否包含显式/隐式因果关系

---

## 项目结构

```
DASP-AECE/
├── src/
│   ├── config.py       # 路径与超参数配置
│   ├── process.py      # Label Studio JSON → JSONL 预处理
│   ├── dataload.py     # Dataset & DataLoader（含 char↔token 对齐）
│   ├── model.py        # CausalSetExtractor 模型定义
│   ├── train.py        # 训练器（分层学习率 + cosine warmup）
│   ├── evaluate.py     # 严格/宽松匹配评估 + 错例导出
│   ├── inference.py    # 端到端推理管道
│   └── main.py         # 主入口
├── data/
│   └── raw/            # train.json / val.json / test.json（Label Studio 格式）
├── cache/              # HuggingFace 模型缓存
├── models/             # 训练后权重保存
└── output/             # 评估结果与 bad case 输出
```

---

## 环境要求

| 依赖 | 版本 |
|------|------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.6.0（建议 CUDA 12.4） |
| Transformers | ≥ 4.57 |
| SciPy | ≥ 1.15 |
| NumPy | ≥ 2.2 |
| tqdm | ≥ 4.67 |

推荐使用 conda 创建独立环境：

```bash
conda create -n dasp-aece python=3.10 -y
conda activate dasp-aece
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers scipy numpy tqdm
```

---

## 使用方法

```bash
cd DASP-AECE/src

# 完整流程：数据处理 + 训练 + 评估
python main.py

# 可选参数覆盖
python main.py --epochs 50 --batch_size 8 --learning_rate 2e-5

# 单独推理
python inference.py
```

---

## 评估指标

- **三元组 F1**：Explicit / Implicit / All
- **Span F1**：cause / trigger / effect 
- 评估结果及错例自动导出至 `output/bad_cases.jsonl`
