# ECE — Agricultural Catastrophe Event Causality Extraction

Experiments on **joint extraction of causal relations** from agricultural disaster texts. The goal is to automatically identify cause–effect event pairs (with or without explicit trigger words) from unstructured Chinese agricultural risk narratives.

## Task

Given a sentence describing agricultural disaster events, extract causal triplets:

| Type | Structure |
|------|-----------|
| Explicit | Cause Event → Trigger Word → Effect Event |
| Implicit | Cause Event → Effect Event (no trigger) |

Example: *"持续干旱导致小麦减产"* → `<持续干旱 | 导致 | 小麦减产>`

## Sub-projects

Three different extraction paradigms are implemented and compared:

### 1. [Pipeline-BERT](Pipeline-BERT/)

A two-stage pipeline: **NER** (BIO tagging with BERT / BERT+CRF) followed by **Relation Extraction**. Entities (EVENT1, TRIG, EVENT2) are first identified, then paired to form triplets.

- Backbone: `hfl/chinese-roberta-wwm-ext`
- Simple and interpretable; serves as the baseline.

### 2. [End2End-T5](End2End-T5/)

A Seq2Seq generative approach using **Mengzi-T5**. The model directly converts input text into structured output `<Cause|Trigger|Effect>`, handling both explicit and implicit causality in a single pass.

- Backbone: `Langboat/mengzi-t5-base`
- Avoids error propagation from pipelined stages.

### 3. [DASP-AECE](DASP-AECE/)

**Dual-Anchor Set Prediction** — a set-based joint extraction model. Two special anchor tokens `[EXP]` and `[IMP]` are prepended to the input to separately route explicit and implicit queries through a Transformer decoder with span-pointer heads. Training uses bipartite matching (Hungarian algorithm).

- Backbone: `hfl/chinese-roberta-wwm-ext`
- Key innovations: dual-anchor routing, set prediction with Hungarian matching, ISPD loss for anchor disentanglement.


## Environment

- Python >= 3.10
- PyTorch >= 2.0
- transformers >= 4.30

See each sub-project's README for detailed dependencies and usage instructions.
